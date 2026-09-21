from app.models import Citation, ClaimAssessment, ClaimRequest
from app.services.llm import GroundedLLM, LocalGroundedLLM
from app.services.retrieval import InMemoryRetriever


class ClaimsAssessmentService:
    def __init__(self, retriever: InMemoryRetriever, llm: GroundedLLM | None = None) -> None:
        self.retriever = retriever
        self.llm = llm or LocalGroundedLLM()

    def assess(self, request: ClaimRequest) -> ClaimAssessment:
        query = f"{request.incident_description} policy coverage exclusions claim amount {request.claimed_amount}"
        matches = [(chunk, score) for chunk, score in self.retriever.search(query) if score >= 0.08]
        citations = [
            Citation(
                document_id=chunk.document_id,
                title=chunk.title,
                excerpt=chunk.text[:500],
                relevance=round(score, 3),
            )
            for chunk, score in matches
        ]
        if not matches:
            return ClaimAssessment(
                claim_id=request.claim_id,
                policy_id=request.policy_id,
                decision="insufficient_evidence",
                answer="I could not find sufficiently relevant policy evidence to assess this claim.",
                confidence=0.0,
                missing_information=["Relevant policy wording or endorsement"],
            )

        evidence = " ".join(chunk.text for chunk, _ in matches)
        return self.llm.generate(request, evidence, citations)