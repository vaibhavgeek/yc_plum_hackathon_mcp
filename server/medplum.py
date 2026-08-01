"""Minimal Medplum FHIR client — OAuth2 client-credentials + FHIR search."""
import os
import time
from typing import Any

import httpx


def _env() -> tuple[str, str | None, str | None]:
    """Read Medplum env vars at call time (not import time) so .env can be loaded
    after this module is imported."""
    return (
        os.getenv("MEDPLUM_BASE_URL", "https://api.medplum.com").rstrip("/"),
        os.getenv("MEDPLUM_CLIENT_ID"),
        os.getenv("MEDPLUM_CLIENT_SECRET"),
    )


def _base_url() -> str:
    return _env()[0]


# Backwards-compat alias — used by seed_medplum.py etc. Evaluated lazily via a
# property-like module-level function is impossible without hackery, so callers
# that need the base URL should use _base_url(). We also expose a BASE_URL for
# code that reads it at call time (after load_dotenv).
def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "BASE_URL":
        return _base_url()
    raise AttributeError(name)


_token_cache: dict[str, Any] = {"access_token": None, "expires_at": 0.0}


async def get_access_token() -> str:
    """Fetch (and cache) an OAuth2 access token via client_credentials."""
    base_url, client_id, client_secret = _env()
    if not client_id or not client_secret:
        raise RuntimeError("MEDPLUM_CLIENT_ID / MEDPLUM_CLIENT_SECRET not set.")

    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"] - 30:
        return _token_cache["access_token"]

    token_url = f"{base_url}/oauth2/token"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()

    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return data["access_token"]


async def fhir_get(path: str, params: dict | None = None) -> dict:
    """GET a FHIR resource or search endpoint."""
    token = await get_access_token()
    url = f"{_base_url()}/fhir/R4/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/fhir+json",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_binary_text(url: str) -> str:
    """Fetch a Binary resource's raw content as text. URL can be relative or absolute."""
    token = await get_access_token()
    if url.startswith("http"):
        full_url = url
    else:
        full_url = f"{_base_url()}/fhir/R4/{url.lstrip('/')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            full_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        return resp.text


async def fhir_post(path: str, resource: dict) -> dict:
    """POST a FHIR resource. Returns the created resource with server-assigned id."""
    token = await get_access_token()
    url = f"{_base_url()}/fhir/R4/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            json=resource,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/fhir+json",
                "Accept": "application/fhir+json",
            },
        )
        resp.raise_for_status()
        return resp.json()


async def find_latest_encounter_id(patient_id: str) -> str | None:
    """Return the id of the most recent Encounter for this patient, or None."""
    bundle = await fhir_get(
        "Encounter",
        params={
            "patient": f"Patient/{patient_id}",
            "_sort": "-date",
            "_count": "1",
        },
    )
    entries = bundle.get("entry", [])
    if not entries:
        return None
    return entries[0]["resource"]["id"]
