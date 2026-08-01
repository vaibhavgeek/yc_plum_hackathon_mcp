"""Submit FHIR data to Medplum using client credentials OAuth2 flow."""
import requests

MEDPLUM_BASE_URL = "https://api.medplum.com"
CLIENT_ID = "6f3ab87a-2a28-4d8e-aeb0-e466002469c2"
CLIENT_SECRET = "f58130b7abc7ef8abb4b4d73162cf8d0fba2f80490bfdcff60d340e9fcc416f1"


def get_access_token():
    """Authenticate with Medplum using client_credentials grant."""
    resp = requests.post(
        f"{MEDPLUM_BASE_URL}/oauth2/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def create_patient(access_token: str):
    """Create a sample Patient resource."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json",
    }
    patient = {
        "resourceType": "Patient",
        "name": [{"given": ["Jane"], "family": "Doe"}],
        "gender": "female",
        "birthDate": "1990-05-15",
    }
    resp = requests.post(f"{MEDPLUM_BASE_URL}/fhir/R4/Patient", json=patient, headers=headers)
    resp.raise_for_status()
    return resp.json()


def create_observation(access_token: str, patient_id: str):
    """Create a sample Observation (blood pressure) linked to the patient."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/fhir+json",
    }
    observation = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Blood pressure panel",
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": "2026-08-01T10:00:00Z",
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8480-6",
                            "display": "Systolic blood pressure",
                        }
                    ]
                },
                "valueQuantity": {"value": 120, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8462-4",
                            "display": "Diastolic blood pressure",
                        }
                    ]
                },
                "valueQuantity": {"value": 80, "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            },
        ],
    }
    resp = requests.post(f"{MEDPLUM_BASE_URL}/fhir/R4/Observation", json=observation, headers=headers)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("1. Authenticating with Medplum...")
    token = get_access_token()
    print(f"   Access token obtained: {token[:20]}...")

    print("\n2. Creating Patient resource...")
    patient = create_patient(token)
    patient_id = patient["id"]
    print(f"   Patient created: id={patient_id}")
    print(f"   Name: {patient['name'][0]['given'][0]} {patient['name'][0]['family']}")

    print("\n3. Creating Observation (Blood Pressure) for patient...")
    obs = create_observation(token, patient_id)
    print(f"   Observation created: id={obs['id']}")
    print(f"   Status: {obs['status']}")
    print(f"   Systolic: {obs['component'][0]['valueQuantity']['value']} mmHg")
    print(f"   Diastolic: {obs['component'][1]['valueQuantity']['value']} mmHg")

    print("\nDone! Data successfully submitted to Medplum.")
