"""MCP server: doctor-patient encounter tools backed by Medplum + OpenAI."""
import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import httpx
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from server.medplum import fetch_binary_text, fhir_get

mcp = FastMCP(
    "barebones-server",
    host=os.getenv("HOST", "127.0.0.1"),
    port=int(os.getenv("PORT", "8000")),
)

# Patient scope for ask_doctor / visualize_diagnosis — always this patient.
ASK_DOCTOR_PATIENT_ID = "8cde5a84-cc28-472a-a55f-4987eedee774"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

STEDI_API_KEY = os.getenv("STEDI_API_KEY")
if not STEDI_API_KEY:
    raise RuntimeError("STEDI_API_KEY not set in .env")
STEDI_ELIGIBILITY_URL = "https://healthcare.us.stedi.com/2024-04-01/change/medicalnetwork/eligibility/v3"

_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Use the strongest widely-available model for visualization. Override via env.
VIZ_MODEL = os.getenv("OPENAI_VIZ_MODEL", "gpt-4o")
EXTRACT_MODEL = os.getenv("OPENAI_EXTRACT_MODEL", "gpt-4o-mini")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return f"Echo: {message}"


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# Shared transcript fetcher
# ---------------------------------------------------------------------------


@dataclass
class TranscriptResult:
    encounter_id: str | None
    patient_ref: str
    when: str
    transcripts: list[str]
    error: str | None = None

    @property
    def combined_text(self) -> str:
        return "\n\n---\n\n".join(self.transcripts)


async def _fetch_latest_transcript(patient_id: str) -> TranscriptResult:
    """Fetch the raw transcript text(s) from the most recent Encounter for a patient."""
    patient_ref = f"Patient/{patient_id}"

    enc_bundle = await fhir_get(
        "Encounter",
        params={"patient": patient_ref, "_sort": "-date", "_count": "1"},
    )
    enc_entries = enc_bundle.get("entry", [])
    if not enc_entries:
        return TranscriptResult(
            encounter_id=None,
            patient_ref=patient_ref,
            when="",
            transcripts=[],
            error=f"No Encounter found for {patient_ref}.",
        )

    encounter = enc_entries[0]["resource"]
    enc_id = encounter["id"]
    period = encounter.get("period", {})
    when = period.get("start") or period.get("end") or "unknown time"

    doc_bundle = await fhir_get(
        "DocumentReference",
        params={"encounter": f"Encounter/{enc_id}", "_sort": "-date"},
    )
    doc_entries = doc_bundle.get("entry", [])
    if not doc_entries:
        doc_bundle = await fhir_get(
            "DocumentReference",
            params={"subject": patient_ref, "_sort": "-date", "_count": "5"},
        )
        doc_entries = doc_bundle.get("entry", [])

    if not doc_entries:
        return TranscriptResult(
            encounter_id=enc_id,
            patient_ref=patient_ref,
            when=when,
            transcripts=[],
            error=f"Encounter {enc_id} has no DocumentReference transcripts.",
        )

    transcripts: list[str] = []
    for entry in doc_entries:
        for content in entry["resource"].get("content", []):
            att = content.get("attachment", {})
            if att.get("data"):
                try:
                    transcripts.append(base64.b64decode(att["data"]).decode("utf-8", errors="replace"))
                except Exception as e:
                    transcripts.append(f"[failed to decode inline attachment: {e}]")
            elif att.get("url"):
                try:
                    transcripts.append(await fetch_binary_text(att["url"]))
                except Exception as e:
                    transcripts.append(f"[failed to fetch {att['url']}: {e}]")

    return TranscriptResult(
        encounter_id=enc_id,
        patient_ref=patient_ref,
        when=when,
        transcripts=transcripts,
    )


def _extract_transcript_plaintext(raw: str) -> str:
    """If the transcript is JSON (e.g. Medplum recording output), pull the
    `transcript` field. Otherwise return as-is."""
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("transcript"), str):
                return data["transcript"]
        except json.JSONDecodeError:
            pass
    return raw


# ---------------------------------------------------------------------------
# ask_doctor
# ---------------------------------------------------------------------------


@mcp.tool()
async def ask_doctor(question: str) -> str:
    """Fetch the transcript of the most recent doctor-patient Encounter for the
    fixed patient (8cde5a84-cc28-472a-a55f-4987eedee774) from Medplum and return
    it as raw text. The `question` argument is a passthrough label for context.
    """
    result = await _fetch_latest_transcript(ASK_DOCTOR_PATIENT_ID)
    if result.error and not result.transcripts:
        return f"{result.error}\n\nQuestion asked: {question}"

    header = (
        f"=== Encounter {result.encounter_id} ===\n"
        f"Patient: {result.patient_ref}\n"
        f"When:    {result.when}\n"
        f"Question: {question}\n"
        f"{'-' * 40}\n"
    )
    return header + result.combined_text


# ---------------------------------------------------------------------------
# visualize_diagnosis
# ---------------------------------------------------------------------------


KEYWORD_SYSTEM_PROMPT = """You are a clinical NLP extractor. Given a doctor-patient
encounter transcript, extract structured medical keywords for later use in a
diagnostic visualization.

Return JSON with exactly these fields:
{
  "ailment": "primary diagnosis or suspected condition (short phrase)",
  "affected_anatomy": ["specific organs / body parts / systems, most-affected first"],
  "symptoms": ["patient-reported symptoms, verbatim short phrases"],
  "clinical_findings": ["objective findings from exam or vitals"],
  "severity": "mild | moderate | severe | unclear",
  "treatment": ["medications, procedures, or actions the doctor prescribed"],
  "patient_context": "one-sentence summary of the patient (age/sex/history if mentioned, else empty)",
  "visualization_focus": "one short phrase describing what the diagram should center on (e.g. 'right lower lobe pneumonia in adult lung')"
}

Be specific and clinical. Do not invent details not present in the transcript.
If a field has no info, use an empty string or empty array."""


VIZ_SYSTEM_PROMPT = """You are a medical illustrator and front-end engineer. Produce
a single self-contained HTML document that visualizes a patient's diagnosis using
inline SVG. Requirements:

STRUCTURE
- Output ONLY raw HTML — no code fences, no commentary. Start with `<!doctype html>`.
- Embed all CSS in a <style> block. No external requests, no <script src>, no fonts
  from the internet, no images.
- Use inline <svg> for anatomy, not raster images. SVGs should be crisp, viewBox-based,
  scalable, and semantically labeled (<title>, <desc>).

VISUAL DESIGN
- Clean, editorial medical-poster aesthetic. Not childish, not cartoonish.
- Anatomically plausible SVG illustration of the affected organ/system, highlighting
  the specific region of pathology (use a distinct accent color for the affected area,
  softer tones for surrounding anatomy).
- Include labeled callouts / annotations pointing to the affected region with short
  clinical labels (e.g. "consolidation — right lower lobe").
- Header with the patient context and the primary diagnosis.
- A structured summary panel with sections: Symptoms, Findings, Treatment, Severity.
- Use a calm, professional palette: off-white background, deep slate text, one clear
  accent color for pathology (red/orange for acute, blue for chronic, amber for unclear).
- Typography: system-ui / -apple-system stack. Generous whitespace. No emojis.
- Layout: responsive, single column on mobile, two-column (illustration + summary) on
  screens ≥ 720px. Max width ~960px, centered.

CONTENT
- All text must be grounded in the provided keywords. Do NOT invent symptoms,
  findings, or treatments beyond what's given. If a field is empty, omit that
  section instead of fabricating.
- Include a small footer disclaimer: "Visualization generated from encounter
  transcript. Not a substitute for clinical judgment."

Return a single complete, valid HTML5 document."""


async def _extract_keywords(transcript_text: str) -> dict:
    """LLM call → structured keyword JSON."""
    resp = await _openai.chat.completions.create(
        model=EXTRACT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _generate_visualization_html(keywords: dict, patient_ref: str, when: str) -> str:
    """Generate HTML+SVG via the Claude Agent SDK.

    Routes through AWS Bedrock (claude-opus-4-6) when CLAUDE_CODE_USE_BEDROCK=1
    is set in the environment; otherwise uses the Anthropic API. Falls back to
    the OpenAI generator if the Claude Agent SDK call fails (import error, CLI
    missing, credentials issue, etc.).
    """
    prompt = f"""Generate a diagnostic visualization for this patient encounter.

PATIENT: {patient_ref}
ENCOUNTER TIME: {when}

EXTRACTED CLINICAL KEYWORDS (use these verbatim — do not invent):
{json.dumps(keywords, indent=2)}

Visualization focus: {keywords.get("visualization_focus", keywords.get("ailment", "general anatomy"))}

Build the HTML document per the system instructions. Prioritize a strong,
anatomically-plausible inline SVG illustration of the affected anatomy with
clearly highlighted pathology, plus a clean summary panel.

Output ONLY the raw HTML document. No commentary, no markdown fences."""

    try:
        html = await _claude_generate_html(system=VIZ_SYSTEM_PROMPT, user=prompt)
    except Exception as e:  # noqa: BLE001
        print(f"[viz] Claude Agent SDK failed, falling back to OpenAI: {e}")
        resp = await _openai.chat.completions.create(
            model=VIZ_MODEL,
            messages=[
                {"role": "system", "content": VIZ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        html = (resp.choices[0].message.content or "").strip()

    return _strip_fences(html)


def _strip_fences(html: str) -> str:
    """Strip accidental markdown code fences the model may have added."""
    html = html.strip()
    if html.startswith("```"):
        lines = html.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        html = "\n".join(lines).strip()
    return html


async def _claude_generate_html(system: str, user: str) -> str:
    """Call the Claude Agent SDK for a single-turn generation and return the
    concatenated assistant text output."""
    # Import lazily so the rest of the server still boots if the SDK isn't ready.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        system_prompt=system,
        # Disable Claude Code's default tools — we just want text generation.
        allowed_tools=[],
        # No tool calls, so a small turn budget is enough. Bump above 1 because
        # the CLI counts its own bookkeeping turns.
        max_turns=3,
        # Environment variables are inherited from the parent process, so
        # CLAUDE_CODE_USE_BEDROCK / AWS_* / ANTHROPIC_MODEL from .env already
        # take effect. Pass model explicitly as a belt-and-suspenders default.
        model=os.getenv("ANTHROPIC_MODEL"),
    )

    chunks: list[str] = []
    async for message in query(prompt=user, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)

    text = "".join(chunks).strip()
    if not text:
        raise RuntimeError("Claude Agent SDK produced no text output.")
    return text


# Hardcoded interactive ACL diagram used by visualize_diagnosis. Loaded once
# at import so the tool call is instant (after the deliberate 5s "thinking"
# pause used to keep the UX consistent with the previous LLM-backed version).
_ACL_DIAGRAM_PATH = Path(__file__).parent / "static" / "acl_diagram.html"
try:
    _ACL_DIAGRAM_HTML = _ACL_DIAGRAM_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    _ACL_DIAGRAM_HTML = (
        "<!doctype html><html><body>"
        "<p>ACL diagram asset missing at " + str(_ACL_DIAGRAM_PATH) + "</p>"
        "</body></html>"
    )


@mcp.tool()
async def visualize_diagnosis() -> str:
    """Show an interactive visual explainer of the patient's diagnosis (ACL
    tear of the right knee). Returns a self-contained HTML document with an
    inline SVG diagram of the knee, a Healthy ⇄ Torn ACL toggle, and tap-to-
    learn labels for the femur, tibia, patella, meniscus, ACL, and PCL.

    Use this tool WHENEVER the user asks to see, visualize, show, illustrate,
    or draw the diagnosis, the injury, the ACL, or the knee.

    Returns raw HTML (starting with `<!doctype html>`) that the client renders
    directly in an iframe.
    """
    # Simulate the ~5s "generating visualization" pause so the streaming UI
    # can flow through its progress steps as usual.
    await asyncio.sleep(5)
    return _ACL_DIAGRAM_HTML


# ---------------------------------------------------------------------------
# insurance_check
# ---------------------------------------------------------------------------


# The patient's insurance-on-file. In production this would come from a FHIR
# Coverage resource on the patient's record; for the demo we hardcode a single
# plan so responses are personalized (not a 6-payer marketplace comparison).
#
# Using the UHC mock subscriber because Stedi's test-mode UHC fixture returns
# the richest benefits payload (deductible, OOP, copay, coinsurance, network
# splits) which lets us render a realistic personalized plan view.
PATIENT_INSURANCE: dict = {
    "payer_label": "UnitedHealthcare",
    "plan_name": "Choice Plus",
    "plan_type": "Commercial PPO",
    "member": {
        "firstName": "John",
        "lastName": "Doe",
        "memberId": "UHC202649",
        "groupNumber": "186084",
        "relationship": "Self",
    },
    # Stedi test-mode requires this exact fixture for UHC — do not change.
    "stedi_request": {
        "tradingPartnerServiceId": "87726",
        "subscriber": {"firstName": "John", "lastName": "Doe", "memberId": "UHC202649"},
        "dependents": [
            {"firstName": "Jane", "lastName": "Doe", "dateOfBirth": "19521121"}
        ],
    },
}


# Legacy multi-payer fixtures — kept for reference / potential future
# "insurance discovery" tool, but no longer used by insurance_check.
STEDI_MOCK_PAYERS: list[dict] = [
    {
        "payer_label": "UnitedHealthcare",
        "tradingPartnerServiceId": "87726",
        "subscriber": {"firstName": "John", "lastName": "Doe", "memberId": "UHC202649"},
        "dependents": [
            {"firstName": "Jane", "lastName": "Doe", "dateOfBirth": "19521121"}
        ],
    },
    {
        "payer_label": "Aetna",
        "tradingPartnerServiceId": "60054",
        "subscriber": {
            "firstName": "Jane",
            "lastName": "Doe",
            "dateOfBirth": "20040404",
            "memberId": "AETNA12345",
        },
    },
    {
        "payer_label": "Cigna",
        "tradingPartnerServiceId": "62308",
        "subscriber": {
            "firstName": "James",
            "lastName": "Jones",
            "dateOfBirth": "19910202",
            "memberId": "23456789100",
        },
    },
    {
        "payer_label": "Anthem Blue Cross Blue Shield of California",
        "tradingPartnerServiceId": "040",
        "subscriber": {"firstName": "Jane", "lastName": "Doe", "memberId": "CGMBCBSCA123"},
        "dependents": [
            {"firstName": "John", "lastName": "Doe", "dateOfBirth": "19750101"}
        ],
    },
    {
        "payer_label": "Oscar Health",
        "tradingPartnerServiceId": "OSCAR",
        "subscriber": {"firstName": "John", "lastName": "Doe", "memberId": "OSCAR123456"},
        "dependents": [
            {"firstName": "Jane", "lastName": "Doe", "dateOfBirth": "20010101"}
        ],
    },
    {
        "payer_label": "Ambetter",
        "tradingPartnerServiceId": "68069",
        "subscriber": {
            "firstName": "John",
            "lastName": "Doe",
            "dateOfBirth": "19940404",
            "memberId": "AMBETTER123",
        },
    },
]


INSURANCE_STC_SYSTEM_PROMPT = """You are a clinical NLP extractor. Given a doctor-patient
encounter transcript, extract insurance-relevant context.

Return JSON with EXACTLY these fields:
{
  "primary_diagnosis": "short phrase (e.g. 'ACL tear, right knee')",
  "recommended_procedures": ["MRI right knee", "ACL reconstruction", ...],
  "provider_specialty": "e.g. orthopedic surgery",
  "urgency": "elective | urgent | emergency | unclear",
  "cost_drivers": ["surgery", "MRI", "physical therapy", "medication", ...]
}

Be specific and grounded in the transcript. Do not invent details."""


def _pick(entries: list, code: str) -> dict | None:
    """Return first benefits entry with matching `code`, or None."""
    for e in entries:
        if e.get("code") == code:
            return e
    return None


def _pick_all(entries: list, code: str) -> list:
    return [e for e in entries if e.get("code") == code]


def _summarize_benefits(stedi_resp: dict, payer_label: str) -> dict:
    """Distill Stedi's verbose response into a compact per-payer insurance option."""
    if "error" in stedi_resp:
        return {"payer": payer_label, "error": stedi_resp["error"]}

    benefits = stedi_resp.get("benefitsInformation", []) or []
    payer_name = (stedi_resp.get("payer") or {}).get("name") or payer_label
    plan_info = stedi_resp.get("planInformation") or {}

    active = _pick(benefits, "1")

    def money(entries: list) -> dict:
        """Build a summary of coverage-level + time-qualifier + amount."""
        out: dict = {}
        for e in entries:
            level = e.get("coverageLevel") or "Unknown"
            tq = e.get("timeQualifier") or ""
            amount = e.get("benefitAmount")
            net = e.get("inPlanNetworkIndicator")
            if amount is None:
                continue
            key_parts = [level, tq]
            if net and net != "Not Applicable":
                key_parts.append(f"network={net}")
            key = " · ".join(key_parts)
            out[key] = f"${amount}"
        return out

    deductible = money(_pick_all(benefits, "C"))
    out_of_pocket = money(_pick_all(benefits, "G"))

    copay_entry = _pick(benefits, "B")
    copay = copay_entry.get("benefitAmount") if copay_entry else None

    coinsurance_entry = None
    for e in _pick_all(benefits, "A"):
        if e.get("inPlanNetworkIndicator") == "Yes":
            coinsurance_entry = e
            break
    if coinsurance_entry is None:
        coinsurance_entry = _pick(benefits, "A")
    coinsurance_pct = None
    if coinsurance_entry and coinsurance_entry.get("benefitPercent") is not None:
        try:
            coinsurance_pct = f"{float(coinsurance_entry['benefitPercent']) * 100:.0f}%"
        except (TypeError, ValueError):
            coinsurance_pct = str(coinsurance_entry["benefitPercent"])

    # AAA-level errors returned by payer (e.g. member not found)
    aaa_errors = stedi_resp.get("errors") or []

    return {
        "payer": payer_name,
        "status": (active or {}).get("name") or ("Error" if aaa_errors else "Unknown"),
        "plan": (active or {}).get("planCoverage") or plan_info.get("planNumber"),
        "member": {
            "firstName": (stedi_resp.get("subscriber") or {}).get("firstName"),
            "lastName": (stedi_resp.get("subscriber") or {}).get("lastName"),
            "memberId": (stedi_resp.get("subscriber") or {}).get("memberId"),
            "groupNumber": (stedi_resp.get("subscriber") or {}).get("groupNumber"),
        },
        "deductible": deductible or None,
        "out_of_pocket": out_of_pocket or None,
        "copay": f"${copay}" if copay is not None else None,
        "coinsurance": coinsurance_pct,
        "payer_errors": [e.get("description") for e in aaa_errors] if aaa_errors else None,
    }


async def _extract_insurance_context(transcript_text: str) -> dict:
    """LLM call → structured insurance-relevant keywords."""
    resp = await _openai.chat.completions.create(
        model=EXTRACT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": INSURANCE_STC_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript_text}"},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


async def _call_stedi_eligibility(payer_fixture: dict) -> dict:
    """POST a single-payer eligibility check to Stedi and return parsed JSON.

    Uses one of Stedi's documented mock subscribers per payer. Each returns
    rich test-mode 271 benefits data for STC 30 (Health Benefit Plan Coverage).
    """
    body: dict = {
        "tradingPartnerServiceId": payer_fixture["tradingPartnerServiceId"],
        "encounter": {"serviceTypeCodes": ["30"]},
        "provider": {
            "organizationName": "Demo Clinic",
            "npi": "1999999984",
        },
        "subscriber": payer_fixture["subscriber"],
    }
    if payer_fixture.get("dependents"):
        body["dependents"] = payer_fixture["dependents"]

    headers = {
        "Authorization": f"Key {STEDI_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(STEDI_ELIGIBILITY_URL, json=body, headers=headers)
            if resp.status_code >= 400:
                return {"error": f"Stedi {resp.status_code}: {resp.text[:400]}"}
            data = resp.json()
            return data
    except Exception as e:
        return {"error": f"Stedi request failed: {e}"}


def _classify_intent(user_query: str) -> str:
    """Map the user's natural query to a UI focus mode.

    Returns one of: 'copay', 'deductible', 'out_of_pocket', 'coverage', 'compare'.
    Defaults to 'compare' (full side-by-side comparison table).
    """
    q = (user_query or "").lower()
    if any(w in q for w in ("copay", "co-pay", "co pay", "visit cost", "per visit")):
        return "copay"
    if any(w in q for w in ("deductible", "before insurance", "how much before")):
        return "deductible"
    if any(w in q for w in ("out of pocket", "out-of-pocket", "oop", "max cost", "maximum")):
        return "out_of_pocket"
    if any(w in q for w in ("cover", "coverage", "am i covered", "is it covered", "eligible", "eligibility")):
        return "coverage"
    return "compare"


def _pick_best_money(bucket: dict | None, prefer_level: str = "Family", prefer_time: str = "Calendar Year") -> str | None:
    """From a summarized money dict, pick the most representative value."""
    if not bucket:
        return None
    # First try: preferred level + time + in-network
    for k, v in bucket.items():
        if prefer_level in k and prefer_time in k and "network=Yes" in k:
            return v
    # Second: preferred level + time
    for k, v in bucket.items():
        if prefer_level in k and prefer_time in k:
            return v
    # Third: any calendar year
    for k, v in bucket.items():
        if prefer_time in k:
            return v
    # Fallback: first value
    return next(iter(bucket.values()), None)


def _esc(s) -> str:
    """HTML-escape helper."""
    if s is None:
        return "—"
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pick_money_slice(bucket: dict | None, level: str, time_q: str, network: str | None) -> str | None:
    """Pull the specific coverage-level + time-qualifier + network amount.

    `network`: 'Yes' (in-network), 'No' (out-of-network), or None (either).
    """
    if not bucket:
        return None
    for k, v in bucket.items():
        if level not in k:
            continue
        if time_q not in k:
            continue
        if network is None:
            if "network=" not in k:
                return v
        else:
            if f"network={network}" in k:
                return v
    return None


def _render_insurance_html(payload: dict, intent: str, user_query: str) -> str:
    """Build a personalized single-plan insurance HTML document.

    Shows the patient's own plan (hardcoded in PATIENT_INSURANCE) with an
    intent-focused hero answer up top, then key benefits and a network
    breakdown. Not a multi-payer marketplace comparison.
    """
    ctx = payload.get("diagnosis_context") or {}
    plan = payload.get("plan") or {}
    benefits = payload.get("benefits") or {}
    member = payload.get("member") or {}

    diagnosis = ctx.get("primary_diagnosis") or "your recent visit"
    procedures = ctx.get("recommended_procedures") or []
    specialty = ctx.get("provider_specialty")
    urgency = ctx.get("urgency")
    cost_drivers = ctx.get("cost_drivers") or []

    payer_name = plan.get("payer") or "Your insurer"
    plan_name = plan.get("plan_name") or "your plan"
    plan_type = plan.get("plan_type") or ""
    status = benefits.get("status") or "Unknown"
    active = status == "Active Coverage"

    ded = benefits.get("deductible") or {}
    oop = benefits.get("out_of_pocket") or {}

    # In-network family calendar-year deductible & OOP (most representative)
    ded_in = _pick_money_slice(ded, "Family", "Calendar Year", "Yes") or _pick_best_money(ded)
    ded_in_remaining = _pick_money_slice(ded, "Family", "Remaining", "Yes")
    ded_out = _pick_money_slice(ded, "Family", "Calendar Year", "No")
    ded_out_remaining = _pick_money_slice(ded, "Family", "Remaining", "No")

    oop_in = _pick_money_slice(oop, "Family", "Calendar Year", "Yes") or _pick_best_money(oop)
    oop_in_remaining = _pick_money_slice(oop, "Family", "Remaining", "Yes")
    oop_out = _pick_money_slice(oop, "Family", "Calendar Year", "No")
    oop_out_remaining = _pick_money_slice(oop, "Family", "Remaining", "No")

    copay = benefits.get("copay") or "—"
    coinsurance = benefits.get("coinsurance") or "—"

    # Intent → hero-card content
    intent_meta = {
        "copay": {
            "kicker": "Your copay",
            "value": copay,
            "label": "per office visit, in-network",
            "explain": f"With your {payer_name} {plan_name} plan, you pay this amount at each visit. The rest is covered by the plan.",
        },
        "deductible": {
            "kicker": "Your deductible",
            "value": ded_in or "—",
            "label": "family · calendar year · in-network",
            "explain": (
                f"You pay this out-of-pocket before {payer_name} starts covering costs. "
                + (f"You have {ded_in_remaining} remaining this year." if ded_in_remaining else "")
            ),
        },
        "out_of_pocket": {
            "kicker": "Your out-of-pocket max",
            "value": oop_in or "—",
            "label": "family · calendar year · in-network",
            "explain": (
                f"Once you hit this amount, {payer_name} covers 100% for the rest of the year. "
                + (f"You have {oop_in_remaining} remaining before the cap." if oop_in_remaining else "")
            ),
        },
        "coverage": {
            "kicker": "Your coverage status",
            "value": status,
            "label": f"{payer_name} · {plan_name}",
            "explain": (
                f"You have active coverage under {payer_name} {plan_name} ({plan_type}). "
                "Member ID " + (member.get("memberId") or "—") + "."
            ) if active else f"Coverage under {payer_name} could not be confirmed as active.",
        },
        "compare": {
            "kicker": "Your plan",
            "value": plan_name,
            "label": f"{payer_name} · {plan_type}",
            "explain": f"Here's a summary of what your {payer_name} plan covers for {diagnosis}.",
        },
    }
    hero = intent_meta.get(intent, intent_meta["compare"])

    # Diagnosis / procedures block
    proc_chips = "".join(f'<span class="chip">{_esc(p)}</span>' for p in procedures) if procedures else ""
    proc_html = f'<div class="chip-row">{proc_chips}</div>' if procedures else ""

    diag_meta = []
    if specialty:
        diag_meta.append(f"<b>Specialty:</b> {_esc(specialty)}")
    if urgency and urgency != "unclear":
        diag_meta.append(f"<b>Urgency:</b> {_esc(urgency)}")
    if cost_drivers:
        diag_meta.append("<b>Likely costs:</b> " + ", ".join(_esc(c) for c in cost_drivers))
    diag_meta_html = " · ".join(diag_meta)

    # Network breakdown table
    def row(label: str, in_v, in_rem, out_v, out_rem, is_focus: bool) -> str:
        cls = " focus-row" if is_focus else ""
        def fmt(v, rem):
            if not v and not rem:
                return "—"
            main = _esc(v) if v else "—"
            sub = f'<div class="sub">{_esc(rem)} remaining</div>' if rem else ""
            return main + sub
        return f"""
        <tr class="{cls}">
          <th scope="row">{_esc(label)}</th>
          <td>{fmt(in_v, in_rem)}</td>
          <td>{fmt(out_v, out_rem)}</td>
        </tr>"""

    network_table = "\n".join([
        row("Deductible (family, this year)", ded_in, ded_in_remaining, ded_out, ded_out_remaining, intent == "deductible"),
        row("Out-of-pocket maximum", oop_in, oop_in_remaining, oop_out, oop_out_remaining, intent == "out_of_pocket"),
    ])

    # Key benefits grid
    def kv(label: str, value: str, focus: bool = False) -> str:
        cls = "kv focus" if focus else "kv"
        return f'<div class="{cls}"><div class="kv-label">{_esc(label)}</div><div class="kv-value">{_esc(value)}</div></div>'

    kv_grid = "\n".join([
        kv("Copay per visit", copay, focus=intent == "copay"),
        kv("Coinsurance (in-network)", coinsurance, focus=False),
        kv("Deductible (in-network)", ded_in or "—", focus=intent == "deductible"),
        kv("Out-of-pocket max (in-network)", oop_in or "—", focus=intent == "out_of_pocket"),
    ])

    status_pill_cls = "pill ok" if active else "pill warn"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Your {_esc(payer_name)} coverage — {_esc(diagnosis)}</title>
<style>
  :root {{
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #e4e7ec;
    --text: #1a2230;
    --muted: #667085;
    --accent: #2563eb;
    --accent-soft: #eff4ff;
    --ok: #067647;
    --ok-bg: #ecfdf3;
    --warn: #b54708;
    --warn-bg: #fffaeb;
    --highlight: #fff7d6;
    --highlight-border: #f7d774;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    font-size: 14px;
    line-height: 1.45;
  }}
  .wrap {{ max-width: 780px; margin: 0 auto; }}

  /* Patient banner */
  .banner {{
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: #fff;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
  }}
  .banner-left .kicker {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    opacity: 0.75;
    margin-bottom: 4px;
  }}
  .banner-left .plan-name {{
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.01em;
  }}
  .banner-left .plan-sub {{ font-size: 13px; opacity: 0.8; margin-top: 2px; }}
  .banner-right {{ text-align: right; font-size: 12px; opacity: 0.85; }}
  .banner-right .member-id {{
    font-family: ui-monospace, Menlo, monospace;
    background: rgba(255,255,255,0.15);
    padding: 3px 8px;
    border-radius: 4px;
    margin-top: 4px;
    display: inline-block;
  }}

  /* Query pill */
  .query {{
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 12px;
  }}

  /* Hero answer card */
  .hero {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
  }}
  .hero .kicker {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .hero .value {{
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.05;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
    margin-bottom: 4px;
  }}
  .hero .value.text {{ font-size: 26px; }}
  .hero .label {{
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 12px;
  }}
  .hero .explain {{ color: var(--text); font-size: 14px; line-height: 1.55; }}

  /* Diagnosis strip */
  .diag {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 20px;
  }}
  .diag-label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
  }}
  .diag-value {{ font-size: 15px; font-weight: 600; }}
  .chip-row {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .chip {{
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 500;
  }}
  .diag-meta {{ margin-top: 8px; color: var(--muted); font-size: 12px; }}

  /* Key/value grid */
  .kv-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
  }}
  .kv {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .kv.focus {{ background: var(--highlight); border-color: var(--highlight-border); }}
  .kv-label {{
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    font-weight: 600;
  }}
  .kv-value {{ font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; }}

  /* Network breakdown table */
  .section-title {{
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 22px 0 8px;
    font-weight: 600;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }}
  th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
  thead th {{
    background: #f9fafb;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
  }}
  tbody th {{
    font-weight: 600;
    font-size: 13px;
    color: var(--text);
    background: #fafbfc;
    width: 45%;
  }}
  tbody td {{
    font-variant-numeric: tabular-nums;
    font-weight: 500;
  }}
  tbody td .sub {{
    font-size: 11px;
    color: var(--muted);
    font-weight: 400;
    margin-top: 2px;
  }}
  tr:last-child th, tr:last-child td {{ border-bottom: none; }}
  tr.focus-row th, tr.focus-row td {{ background: var(--highlight); }}

  .pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
  }}
  .pill.ok {{ background: var(--ok-bg); color: var(--ok); }}
  .pill.warn {{ background: var(--warn-bg); color: var(--warn); }}

  footer {{
    margin-top: 20px;
    color: var(--muted);
    font-size: 11px;
    text-align: center;
    line-height: 1.5;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <section class="banner">
      <div class="banner-left">
        <div class="kicker">Your insurance on file</div>
        <div class="plan-name">{_esc(payer_name)} · {_esc(plan_name)}</div>
        <div class="plan-sub">{_esc(plan_type)}</div>
      </div>
      <div class="banner-right">
        <div>{_esc(member.get("firstName") or "")} {_esc(member.get("lastName") or "")}</div>
        <div class="member-id">Member ID · {_esc(member.get("memberId") or "—")}</div>
        <div style="margin-top:6px;"><span class="{status_pill_cls}">{_esc(status)}</span></div>
      </div>
    </section>

    {'<div class="query">You asked: &ldquo;' + _esc(user_query) + '&rdquo;</div>' if user_query else ''}

    <section class="hero">
      <div class="kicker">{_esc(hero["kicker"])}</div>
      <div class="value {'' if str(hero['value']).startswith('$') or str(hero['value']).endswith('%') else 'text'}">{_esc(hero["value"])}</div>
      <div class="label">{_esc(hero["label"])}</div>
      <div class="explain">{_esc(hero["explain"])}</div>
    </section>

    <section class="diag">
      <div class="diag-label">For your visit</div>
      <div class="diag-value">{_esc(diagnosis)}</div>
      {proc_html}
      {f'<div class="diag-meta">{diag_meta_html}</div>' if diag_meta_html else ''}
    </section>

    <div class="section-title">Your plan at a glance</div>
    <div class="kv-grid">
      {kv_grid}
    </div>

    <div class="section-title">In-network vs. out-of-network</div>
    <table>
      <thead>
        <tr><th>Benefit</th><th>In-network</th><th>Out-of-network</th></tr>
      </thead>
      <tbody>{network_table}
      </tbody>
    </table>

    <footer>
      Eligibility verified via Stedi real-time 270/271 · {_esc(payer_name)} · retrieved just now.<br/>
      Amounts shown are family · calendar year. Individual amounts and specific procedure costs may vary.
    </footer>
  </div>
</body>
</html>"""


@mcp.tool()
async def insurance_check(user_query: str = "") -> str:
    """Fetch the latest doctor-patient encounter transcript and return a
    PERSONALIZED HTML document for the patient's insurance on file
    (UnitedHealthcare Choice Plus). Shows coverage status, deductible, copay,
    coinsurance, out-of-pocket max, and in-network vs. out-of-network breakdown
    — tailored to the patient's diagnosis.

    Use this tool WHENEVER the user asks about their insurance, coverage,
    benefits, deductible, copay, coinsurance, out-of-pocket cost, plan details,
    what their insurance pays for, or what a procedure will cost.

    IMPORTANT: Pass the user's ORIGINAL, VERBATIM question as `user_query` so
    the rendered UI can lead with the specific answer they want (copay,
    deductible, out-of-pocket, or coverage status).

    Returns a self-contained HTML document (starts with `<!doctype html>`) that
    the client renders directly in an iframe.
    """
    result = await _fetch_latest_transcript(ASK_DOCTOR_PATIENT_ID)
    if result.error and not result.transcripts:
        return f"<!doctype html><html><body><p>{_esc(result.error)}</p></body></html>"

    plain = _extract_transcript_plaintext(result.combined_text)

    # Kick off Stedi call and LLM extraction in parallel — they're independent.
    stedi_task = asyncio.create_task(_call_stedi_eligibility_for_patient())
    try:
        context = await _extract_insurance_context(plain)
    except Exception as e:
        context = {"error": f"context extraction failed: {e}"}

    stedi_resp = await stedi_task
    benefits = _summarize_benefits(stedi_resp, PATIENT_INSURANCE["payer_label"])

    payload = {
        "encounter_id": result.encounter_id,
        "patient_ref": result.patient_ref,
        "when": result.when,
        "diagnosis_context": {
            "primary_diagnosis": context.get("primary_diagnosis"),
            "recommended_procedures": context.get("recommended_procedures"),
            "provider_specialty": context.get("provider_specialty"),
            "urgency": context.get("urgency"),
            "cost_drivers": context.get("cost_drivers"),
        },
        "plan": {
            "payer": PATIENT_INSURANCE["payer_label"],
            "plan_name": PATIENT_INSURANCE["plan_name"],
            "plan_type": PATIENT_INSURANCE["plan_type"],
        },
        "member": PATIENT_INSURANCE["member"],
        "benefits": benefits,
    }

    intent = _classify_intent(user_query)
    return _render_insurance_html(payload, intent, user_query)


async def _call_stedi_eligibility_for_patient() -> dict:
    """Call Stedi eligibility for the patient's insurance on file."""
    req = PATIENT_INSURANCE["stedi_request"]
    fixture = {
        "payer_label": PATIENT_INSURANCE["payer_label"],
        "tradingPartnerServiceId": req["tradingPartnerServiceId"],
        "subscriber": req["subscriber"],
        "dependents": req.get("dependents"),
    }
    return await _call_stedi_eligibility(fixture)


if __name__ == "__main__":
    # Streamable HTTP transport (modern MCP HTTP transport)
    mcp.run(transport="streamable-http")
