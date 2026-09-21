from fastapi.testclient import TestClient

from app.main import app, retriever


client = TestClient(app)


def setup_function() -> None:
    retriever.chunks.clear()


def test_ingest_and_assess_returns_grounded_citations() -> None:
    response = client.post(
        "/documents",
        json={
            "document_id": "policy-001",
            "title": "Commercial property policy",
            "text": "Covered property damage from accidental fire is covered subject to the policy limit. Theft from an unlocked vehicle is excluded and not covered.",
        },
    )
    assert response.status_code == 201
    assert response.json()["chunks_created"] == 1

    response = client.post(
        "/claims/assess",
        json={
            "claim_id": "claim-001",
            "policy_id": "policy-001",
            "incident_description": "Fire damaged covered property at the insured location.",
            "claimed_amount": 12500,
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["decision"] in {"potentially_covered", "potentially_excluded"}
    assert body["citations"][0]["document_id"] == "policy-001"


def test_claim_validation_rejects_blank_and_non_positive_values() -> None:
    response = client.post(
        "/claims/assess",
        json={"claim_id": " ", "policy_id": "p1", "incident_description": "short", "claimed_amount": 0},
    )
    assert response.status_code == 422


def test_assessment_without_evidence_is_explicit() -> None:
    response = client.post(
        "/claims/assess",
        json={
            "claim_id": "claim-002",
            "policy_id": "policy-002",
            "incident_description": "Water damage occurred after a sudden pipe rupture.",
            "claimed_amount": 9000,
        },
    )
    assert response.json()["decision"] == "insufficient_evidence"
    assert response.json()["citations"] == []