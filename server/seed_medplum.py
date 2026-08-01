"""Seed a demo Encounter + DocumentReference (transcript) into Medplum for testing.

Creates a clinical ACL tear consultation transcript attached to the fixed
patient used by ask_doctor / visualize_diagnosis.
"""
import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

from server.medplum import BASE_URL, get_access_token  # noqa: E402


PATIENT_ID = "8cde5a84-cc28-472a-a55f-4987eedee774"


TRANSCRIPT_TEXT = """Dr. Patel: Hi Marcus, come on in. I understand you hurt your knee playing soccer?

Patient: Yeah, three days ago. I was cutting to the left to change direction, my
right foot was planted, and I felt this loud pop in my knee. Went down immediately.
Sharp pain deep inside the knee.

Dr. Patel: Could you keep playing?

Patient: No way. My teammates helped me off. Within about twenty minutes it swelled
up like a balloon. I've been icing it and using crutches since.

Dr. Patel: Any prior injuries to that knee?

Patient: Sprained it in high school but nothing since. I'm 28, I play recreationally
twice a week.

Dr. Patel: Any locking, giving way, or clicking since the injury?

Patient: Yes — when I try to put weight on it, it feels loose, like it wants to shift
sideways. Very unstable. And there's this deep ache along the inside of the knee.

Dr. Patel: Let me examine you. There's significant effusion — swelling around the
joint. Range of motion is limited, about 10 to 90 degrees. I'm going to do a
Lachman test... yes, there's clear anterior translation of the tibia with a soft
endpoint. Positive Lachman. Pivot shift is also positive. McMurray's is negative,
though tenderness along the medial joint line makes me suspicious of a possible
medial meniscus involvement as well.

Patient: So what does that mean?

Dr. Patel: The mechanism — non-contact pivot with a pop, immediate swelling, and
now a positive Lachman and pivot shift — is classic for a complete anterior cruciate
ligament tear. That's the ACL, the main stabilizing ligament in the center of your
knee. Given the medial joint line tenderness, we should also rule out a meniscus tear,
which commonly occurs alongside ACL injuries.

Patient: How sure are you?

Dr. Patel: Clinically I'm quite confident, but we'll confirm with an MRI. I'll order
that today. For now, I want you to continue RICE — rest, ice, compression, elevation.
Keep using crutches, weight-bear only as tolerated. I'm prescribing naproxen 500mg
twice daily with food for the inflammation. Once swelling comes down we'll start
physical therapy to restore range of motion and quad activation.

Patient: Will I need surgery?

Dr. Patel: For someone your age who wants to return to cutting sports like soccer,
ACL reconstruction is usually recommended. Non-operative management leaves the knee
unstable and puts your meniscus and cartilage at risk for further damage. We'll
discuss graft options — patellar tendon, hamstring, or quad tendon autograft — after
the MRI. Typical recovery is 9 to 12 months back to sport.

Patient: That's a long road.

Dr. Patel: It is. But outcomes are excellent when patients commit to the rehab.
I'll have you back in after the MRI, probably later this week. Any questions?

Patient: Not right now. Thanks, doctor.

Dr. Patel: You're welcome. Front desk will schedule the MRI and your follow-up."""


# Wrap in the same JSON shape Medplum's recording pipeline emits
TRANSCRIPT_JSON = json.dumps(
    {
        "sourceFileName": f"visit-recording-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%fZ')}.webm",
        "savedAt": datetime.now(timezone.utc).isoformat(),
        "transcript": TRANSCRIPT_TEXT,
        "utterances": [],
        "duration": 480.0,
        "requestId": "seed-acl-demo",
    },
    indent=2,
)


async def fhir_post(client: httpx.AsyncClient, path: str, resource: dict, token: str) -> dict:
    resp = await client.post(
        f"{BASE_URL}/fhir/R4/{path}",
        json=resource,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json",
        },
    )
    resp.raise_for_status()
    return resp.json()


async def main() -> None:
    token = await get_access_token()
    now = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(timeout=30) as client:
        encounter = {
            "resourceType": "Encounter",
            "status": "finished",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "period": {"start": now, "end": now},
            "reasonCode": [{"text": "Right knee injury after soccer pivot — suspected ACL tear"}],
        }
        enc = await fhir_post(client, "Encounter", encounter, token)
        enc_id = enc["id"]
        print(f"Created Encounter/{enc_id}")

        b64 = base64.b64encode(TRANSCRIPT_JSON.encode("utf-8")).decode("ascii")
        doc_ref = {
            "resourceType": "DocumentReference",
            "status": "current",
            "docStatus": "final",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11488-4",
                        "display": "Consultation note",
                    }
                ],
                "text": "Doctor-patient encounter transcript",
            },
            "subject": {"reference": f"Patient/{PATIENT_ID}"},
            "date": now,
            "description": "Transcript of ambulatory visit — ACL tear workup",
            "content": [
                {
                    "attachment": {
                        "contentType": "application/json",
                        "data": b64,
                        "title": "encounter-transcript.json",
                    }
                }
            ],
            "context": {
                "encounter": [{"reference": f"Encounter/{enc_id}"}],
                "period": {"start": now, "end": now},
            },
        }
        dr = await fhir_post(client, "DocumentReference", doc_ref, token)
        print(f"Created DocumentReference/{dr['id']} attached to Encounter/{enc_id}")


if __name__ == "__main__":
    asyncio.run(main())
