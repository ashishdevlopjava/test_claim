from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ClaimRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=100)
    policy_id: str = Field(min_length=1, max_length=100)
    incident_description: str = Field(min_length=10, max_length=10_000)
    claimed_amount: float = Field(gt=0, le=10_000_000)

    @field_validator("claim_id", "policy_id", "incident_description")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Citation(BaseModel):
    document_id: str
    title: str
    excerpt: str
    relevance: float = Field(ge=0, le=1)


class ClaimAssessment(BaseModel):
    claim_id: str
    policy_id: str
    decision: Literal["potentially_covered", "potentially_excluded", "insufficient_evidence"]
    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    citations: list[Citation] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=20, max_length=1_000_000)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("document_id", "title", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    chunks: int