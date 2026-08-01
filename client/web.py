"""Web client: FastAPI serves a voice UI, routes transcripts through OpenAI → MCP tools."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEB_PORT = int(os.getenv("WEB_PORT", "5173"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

STATIC_DIR = Path(__file__).parent / "static"


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


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/voice")
async def voice(req: VoiceRequest) -> dict:
    """Route transcript through OpenAI → MCP tool call → return raw MCP response."""
    transcript = req.transcript.strip()
    if not transcript:
        raise HTTPException(400, "empty transcript")

    async with streamablehttp_client(MCP_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai_schema(tools_result.tools)

            # Ask OpenAI to pick a tool + args
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
                return {
                    "transcript": transcript,
                    "error": "No tool was selected by the LLM.",
                    "assistant": msg.content,
                }

            call = msg.tool_calls[0]
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments or "{}")

            # Invoke the MCP tool
            mcp_result = await session.call_tool(tool_name, tool_args)

            # Extract text content from MCP response
            content_parts = []
            for part in mcp_result.content:
                text = getattr(part, "text", None)
                if text is not None:
                    content_parts.append(text)
                else:
                    content_parts.append(str(part))

            return {
                "transcript": transcript,
                "tool": tool_name,
                "arguments": tool_args,
                "result": content_parts,
                "isError": mcp_result.isError,
            }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT)
