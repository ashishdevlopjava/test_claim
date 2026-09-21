from dataclasses import dataclass

from app.services.embeddings import LocalEmbeddingService, cosine_similarity


@dataclass(frozen=True)
class Chunk:
    document_id: str
    title: str
    text: str
    metadata: dict[str, str]
    vector: list[float]


class InMemoryRetriever:
    def __init__(self, embedding_service: LocalEmbeddingService) -> None:
        self.embedding_service = embedding_service
        self.chunks: list[Chunk] = []

    def add(self, document_id: str, title: str, text: str, metadata: dict[str, str]) -> int:
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id != document_id]
        document_chunks = self._chunk_text(text)
        self.chunks.extend(
            Chunk(document_id, title, chunk, metadata, self.embedding_service.embed(chunk))
            for chunk in document_chunks
        )
        return len(document_chunks)

    def search(self, query: str, limit: int = 4) -> list[tuple[Chunk, float]]:
        query_vector = self.embedding_service.embed(query)
        ranked = sorted(
            ((chunk, cosine_similarity(query_vector, chunk.vector)) for chunk in self.chunks),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:limit]

    @staticmethod
    def _chunk_text(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + max_words)
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap
        return chunks

    @property
    def document_count(self) -> int:
        return len({chunk.document_id for chunk in self.chunks})