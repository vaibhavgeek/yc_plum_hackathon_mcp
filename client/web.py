"""Web client: FastAPI serves a voice UI, routes transcripts through OpenAI → MCP tools."""
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from server.conversation_log import log_conversation_turn

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEB_PORT = int(os.getenv("WEB_PORT", "5173"))
# Patient scope — every /voice call is logged under this patient's timeline.
CONVERSATION_PATIENT_ID = os.getenv(
    "CONVERSATION_PATIENT_ID", "8cde5a84-cc28-472a-a55f-4987eedee774"
)
# Text-to-speech settings.
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")
SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
# Skip TTS for absurdly long spoken summaries so we don't burn tokens.
MAX_SPEAK_CHARS = 800

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

STATIC_DIR = Path(__file__).parent / "static"


# Human-readable descriptions of what each tool does under the hood, so we can
# stream meaningful progress messages while the (opaque) MCP call is in flight.
TOOL_STEPS: dict[str, list[str]] = {
    "ask_doctor": [
        "Fetching the latest encounter from Medplum…",
        "Loading the transcript…",
    ],
    "visualize_diagnosis": [
        "Fetching the latest encounter from Medplum…",
        "Extracting clinical keywords with GPT…",
        "Generating an SVG diagnostic visualization…",
    ],
    "insurance_check": [
        "Fetching the latest encounter from Medplum…",
        "Extracting insurance-relevant context with GPT…",
        "Verifying your UnitedHealthcare coverage with Stedi…",
        "Summarizing your benefits (deductible, copay, coinsurance, OOP)…",
        "Personalizing the response for your plan…",
    ],
    "add": ["Running the calculation…"],
    "echo": ["Echoing your message…"],
}


def mcp_tools_to_openai_schema(tools: list[Any]) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)


class VoiceRequest(BaseModel):
    transcript: str


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _looks_like_html(text: str) -> bool:
    t = text.lstrip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html")


async def _generate_spoken_summary(
    transcript: str,
    tool: str,
    arguments: dict,
    result_text: str,
) -> str:
    """Produce a short, natural-sounding spoken response for the user.

    For HTML/visualization results, describes what was rendered rather than
    reading tags. For text results, condenses to 2-3 short sentences.
    """
    if _looks_like_html(result_text):
        # Give the summarizer only the meaningful body text (approx) so it can
        # describe what the visualization shows without reading markup.
        body_preview = result_text[:6000]
        user_content = (
            f"The user asked: {transcript!r}\n"
            f"I called the `{tool}` tool with arguments {arguments}. "
            f"The tool returned an HTML visualization. Summarize what the user "
            f"will see, in 2-3 short sentences, spoken naturally. Do NOT read "
            f"any HTML tags. Do NOT mention the word 'HTML'.\n\n"
            f"HTML body (for context only):\n{body_preview}"
        )
    else:
        trimmed = result_text if len(result_text) < 4000 else result_text[:4000] + "…"
        user_content = (
            f"The user asked: {transcript!r}\n"
            f"I called the `{tool}` tool with arguments {arguments}. "
            f"The tool returned this raw text:\n\n{trimmed}\n\n"
            f"Give a concise spoken response (2-4 short sentences) that directly "
            f"answers the user. Speak in natural conversational English. Do not "
            f"read raw JSON, IDs, timestamps, or code. If it's medical content, "
            f"be clear and calm."
        )

    completion = await openai_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You produce short, natural spoken responses for a voice "
                    "assistant. Output plain text only — no markdown, no lists, "
                    "no headings, no code. Keep it under 400 characters when "
                    "possible."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.4,
    )
    text = (completion.choices[0].message.content or "").strip()
    # Hard cap to protect TTS cost / latency.
    if len(text) > MAX_SPEAK_CHARS:
        text = text[:MAX_SPEAK_CHARS].rsplit(" ", 1)[0] + "…"
    return text


async def _synthesize_speech(text: str) -> str:
    """Call OpenAI TTS and return base64-encoded MP3 audio."""
    resp = await openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="mp3",
    )
    audio_bytes = resp.read() if hasattr(resp, "read") else resp.content
    return base64.b64encode(audio_bytes).decode("ascii")


async def _build_voice_response(
    transcript: str,
    tool: str,
    arguments: dict,
    result_text: str,
) -> tuple[str, str | None]:
    """Return (spoken_summary_text, base64_mp3_or_None). Never raises."""
    try:
        summary = await _generate_spoken_summary(transcript, tool, arguments, result_text)
    except Exception as e:  # noqa: BLE001
        summary = f"Here is the result from {tool}."
        print(f"[voice-summary] error: {e}")
    try:
        audio_b64 = await _synthesize_speech(summary)
    except Exception as e:  # noqa: BLE001
        print(f"[voice-tts] error: {e}")
        audio_b64 = None
    return summary, audio_b64


async def _run_pipeline(transcript: str) -> AsyncIterator[str]:
    """Run the voice → OpenAI → MCP pipeline, yielding SSE frames.

    Emits these events:
      progress    { step, phase }        — human-readable status update
      tool_selected { tool, arguments }  — LLM picked a tool
      tick        { elapsed_ms }         — keep-alive while waiting on slow tool
      done        { transcript, tool?, arguments?, result?, isError?, error? }
    """
    try:
        yield _sse("progress", {"step": "Connecting to MCP server…", "phase": "connect"})

        async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                yield _sse("progress", {"step": "Listing available tools…", "phase": "connect"})
                tools_result = await session.list_tools()
                openai_tools = mcp_tools_to_openai_schema(tools_result.tools)

                yield _sse(
                    "progress",
                    {"step": "Deciding which tool fits your request…", "phase": "route"},
                )

                completion = await openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You route user voice commands to MCP tools. "
                                "Always call exactly one tool matching the user's intent. "
                                "Do not answer directly."
                            ),
                        },
                        {"role": "user", "content": transcript},
                    ],
                    tools=openai_tools,
                    tool_choice="required",
                )

                msg = completion.choices[0].message
                if not msg.tool_calls:
                    log_status = await log_conversation_turn(
                        patient_id=CONVERSATION_PATIENT_ID,
                        transcript=transcript,
                        tool=None,
                        arguments=None,
                        result_text=msg.content or "",
                        is_error=True,
                    )
                    yield _sse(
                        "done",
                        {
                            "transcript": transcript,
                            "error": "No tool was selected by the LLM.",
                            "assistant": msg.content,
                            "logged": log_status,
                        },
                    )
                    return

                call = msg.tool_calls[0]
                tool_name = call.function.name
                tool_args = json.loads(call.function.arguments or "{}")

                yield _sse(
                    "tool_selected",
                    {"tool": tool_name, "arguments": tool_args},
                )

                # Fire the actual MCP call as a task and, in parallel, stream
                # per-tool sub-step messages on a timer so the user sees
                # meaningful progress during the (opaque) blocking call.
                steps = TOOL_STEPS.get(tool_name, ["Working…"])
                call_task = asyncio.create_task(session.call_tool(tool_name, tool_args))

                step_idx = 0
                elapsed_ms = 0
                # Emit first step immediately.
                if steps:
                    yield _sse("progress", {"step": steps[0], "phase": "tool"})
                    step_idx = 1

                # Advance every ~1.2s until the call resolves.
                while not call_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(call_task), timeout=1.2)
                    except asyncio.TimeoutError:
                        elapsed_ms += 1200
                        if step_idx < len(steps):
                            yield _sse(
                                "progress",
                                {"step": steps[step_idx], "phase": "tool"},
                            )
                            step_idx += 1
                        else:
                            # Keep-alive tick — stream stays warm, UI shows a
                            # subtle elapsed counter.
                            yield _sse("tick", {"elapsed_ms": elapsed_ms})

                try:
                    mcp_result = await call_task
                except Exception as e:
                    yield _sse(
                        "done",
                        {"transcript": transcript, "error": f"MCP call failed: {e}"},
                    )
                    return

                yield _sse("progress", {"step": "Finalizing response…", "phase": "finalize"})

                content_parts: list[str] = []
                for part in mcp_result.content:
                    text = getattr(part, "text", None)
                    if text is not None:
                        content_parts.append(text)
                    else:
                        content_parts.append(str(part))

                combined_result = "\n".join(content_parts)

                # Generate a short spoken summary + TTS audio in parallel with logging.
                yield _sse(
                    "progress",
                    {"step": "Composing spoken response…", "phase": "finalize"},
                )
                voice_task = asyncio.create_task(
                    _build_voice_response(transcript, tool_name, tool_args, combined_result)
                )

                yield _sse("progress", {"step": "Logging conversation to Medplum…", "phase": "finalize"})

                log_status = await log_conversation_turn(
                    patient_id=CONVERSATION_PATIENT_ID,
                    transcript=transcript,
                    tool=tool_name,
                    arguments=tool_args,
                    result_text=combined_result,
                    is_error=bool(mcp_result.isError),
                )

                spoken_text, audio_b64 = await voice_task

                yield _sse(
                    "done",
                    {
                        "transcript": transcript,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": content_parts,
                        "isError": mcp_result.isError,
                        "logged": log_status,
                        "spoken_text": spoken_text,
                        "audio_mp3_base64": audio_b64,
                    },
                )
    except Exception as e:
        yield _sse("done", {"transcript": transcript, "error": f"Pipeline error: {e}"})


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/voice")
async def voice(req: VoiceRequest) -> dict:
    """Non-streaming endpoint (kept for backwards compatibility).

    New clients should use /voice/stream to get live progress events while
    slow tools (insurance_check, visualize_diagnosis) are running.
    """
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(400, "empty transcript")

    final: dict = {"transcript": transcript, "error": "Pipeline produced no result."}
    async for frame in _run_pipeline(transcript):
        if frame.startswith("event: done\n"):
            data_line = frame.split("\ndata: ", 1)[1].rsplit("\n\n", 1)[0]
            final = json.loads(data_line)
    return final


@app.post("/voice/stream")
async def voice_stream(req: VoiceRequest) -> StreamingResponse:
    """Server-Sent Events stream of pipeline progress + final result.

    Event sequence (typical):
      progress   Connecting to MCP server…
      progress   Listing available tools…
      progress   Deciding which tool fits your request…
      tool_selected  { tool: "insurance_check", arguments: {…} }
      progress   Fetching the latest encounter from Medplum…
      progress   Extracting insurance-relevant context with GPT…
      progress   Querying Stedi eligibility for 6 payers in parallel…
      progress   Summarizing benefits…
      progress   Rendering the comparison UI…
      progress   Finalizing response…
      progress   Logging conversation to Medplum…
      done       { transcript, tool, arguments, result: […], isError, logged }
    """
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(400, "empty transcript")

    async def event_stream() -> AsyncIterator[bytes]:
        async for frame in _run_pipeline(transcript):
            yield frame.encode("utf-8")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering if present
            "Connection": "keep-alive",
        },
    )


@app.post("/speak")
async def speak(req: VoiceRequest) -> dict:
    """Standalone TTS endpoint. POST { transcript: "text to speak" } →
    { text, audio_mp3_base64 }. Useful for a replay button in the UI."""
    text = req.transcript.strip()
    if not text:
        raise HTTPException(400, "empty text")
    if len(text) > MAX_SPEAK_CHARS:
        text = text[:MAX_SPEAK_CHARS].rsplit(" ", 1)[0] + "…"
    try:
        audio_b64 = await _synthesize_speech(text)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"TTS failed: {e}") from e
    return {"text": text, "audio_mp3_base64": audio_b64}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT)
