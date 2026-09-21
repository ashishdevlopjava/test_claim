from fastapi import FastAPI

from app.models import ClaimAssessment, ClaimRequest, HealthResponse, IngestRequest, IngestResponse
from app.services.assessment import ClaimsAssessmentService
from app.services.embeddings import LocalEmbeddingService
from app.services.retrieval import InMemoryRetriever


embedding_service = LocalEmbeddingService()
retriever = InMemoryRetriever(embedding_service)
assessment_service = ClaimsAssessmentService(retriever)
app = FastAPI(title="Insurance Claims Intelligence API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", documents=retriever.document_count, chunks=len(retriever.chunks))


@app.post("/documents", response_model=IngestResponse, status_code=201)
def ingest_document(request: IngestRequest) -> IngestResponse:
    chunks_created = retriever.add(request.document_id, request.title, request.text, request.metadata)
    return IngestResponse(document_id=request.document_id, chunks_created=chunks_created)


@app.post("/claims/assess", response_model=ClaimAssessment)
def assess_claim(request: ClaimRequest) -> ClaimAssessment:
    return assessment_service.assess(request)