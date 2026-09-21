from typing import Protocol

from app.models import Citation, ClaimAssessment, ClaimRequest


class GroundedLLM(Protocol):
    def generate(
        self, request: ClaimRequest, evidence: str, citations: list[Citation]
    ) -> ClaimAssessment: ...


class LocalGroundedLLM:
    """Offline structured-output substitute for an Azure OpenAI deployment."""

    def generate(
        self, request: ClaimRequest, evidence: str, citations: list[Citation]
    ) -> ClaimAssessment:
        exclusion_terms = ("excluded", "not covered", "exclusion", "does not cover")
        has_exclusion = any(term in evidence.lower() for term in exclusion_terms)
        decision = "potentially_excluded" if has_exclusion else "potentially_covered"
        conclusion = "may be excluded" if has_exclusion else "may be covered"
        return ClaimAssessment(
            claim_id=request.claim_id,
            policy_id=request.policy_id,
            decision=decision,
            answer=f"Based on the retrieved policy wording, this claim {conclusion}. A claims professional should confirm the facts and applicable limits.",
            confidence=round(min(0.95, 0.45 + citations[0].relevance * 0.5), 3),
            citations=citations,
            missing_information=["Loss date and supporting incident documentation"],
        )