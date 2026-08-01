"""Record MCP conversation turns as FHIR Communication resources in Medplum.

Each user↔assistant exchange becomes two Communications, linked to the patient
and (if available) the most recent Encounter. This makes MCP interactions visible
in the patient's Medplum timeline alongside their clinical record.
"""
from datetime import datetime, timezone
from typing import Any

from server.medplum import find_latest_encounter_id, fhir_post

# Cap payloads so we don't push megabytes of generated HTML per turn.
MAX_PAYLOAD_CHARS = 20_000


def _truncate(text: str, limit: int = MAX_PAYLOAD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n… [truncated, original was {len(text)} chars]"


def _base_communication(
    patient_ref: str,
    encounter_ref: str | None,
    category_code: str,
    category_display: str,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    resource: dict[str, Any] = {
        "resourceType": "Communication",
        "status": "completed",
        "subject": {"reference": patient_ref},
        "sent": now,
        "category": [
            {
                "coding": [
                    {
                        "system": "https://opencode.local/mcp-conversation",
                        "code": category_code,
                        "display": category_display,
                    }
                ],
                "text": category_display,
            }
        ],
        "topic": {"text": "MCP voice client conversation"},
        "meta": {"tag": [{"system": "https://opencode.local/tags", "code": "mcp-turn"}]},
    }
    if encounter_ref:
        resource["encounter"] = {"reference": encounter_ref}
    return resource


async def log_conversation_turn(
    *,
    patient_id: str,
    transcript: str,
    tool: str | None,
    arguments: dict | None,
    result_text: str,
    is_error: bool,
) -> dict[str, str | None]:
    """Persist a user question + assistant response as two Communication resources.

    Returns {"user_id": ..., "assistant_id": ...} on success, or {"error": "..."}
    on failure. Never raises — the caller can log turns opportunistically.
    """
    try:
        patient_ref = f"Patient/{patient_id}"
        enc_id = await find_latest_encounter_id(patient_id)
        encounter_ref = f"Encounter/{enc_id}" if enc_id else None

        # 1) User side — the transcript is what the user actually said/typed.
        user_comm = _base_communication(
            patient_ref, encounter_ref, "user-question", "MCP user question"
        )
        user_comm["sender"] = {"display": "MCP voice client (user)"}
        user_comm["recipient"] = [{"display": "MCP assistant"}]
        user_comm["payload"] = [{"contentString": _truncate(transcript)}]
        user_created = await fhir_post("Communication", user_comm)

        # 2) Assistant side — includes the tool call metadata and the result.
        assistant_note = ""
        if tool:
            assistant_note = f"[tool={tool} args={arguments or {}}]\n\n"
        assistant_body = assistant_note + result_text

        assistant_comm = _base_communication(
            patient_ref, encounter_ref, "assistant-response", "MCP assistant response"
        )
        assistant_comm["sender"] = {"display": f"MCP assistant ({tool or 'no-tool'})"}
        assistant_comm["recipient"] = [{"display": "MCP voice client (user)"}]
        assistant_comm["payload"] = [{"contentString": _truncate(assistant_body)}]
        if is_error:
            assistant_comm["statusReason"] = {"text": "tool returned error"}
        # Tag the assistant turn with the tool name so we can filter later.
        if tool:
            assistant_comm["meta"]["tag"].append(
                {"system": "https://opencode.local/mcp-tool", "code": tool}
            )
        assistant_created = await fhir_post("Communication", assistant_comm)

        return {
            "user_id": user_created.get("id"),
            "assistant_id": assistant_created.get("id"),
            "encounter": encounter_ref,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
