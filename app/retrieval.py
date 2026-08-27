from app.embeddings import EmbeddingModel
from app.models import RetrievedChunk
from app.vector_store import VectorStore


class RetrievalService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.35
    ) -> list[RetrievedChunk]:

        query_embedding = self.embedding_model.model.encode(query)

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k
        )

        retrieved_chunks = []

        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for document, distance, metadata in zip(
            documents,
            distances,
            metadatas
        ):

            score = 1 / (1 + distance)

            chunk = RetrievedChunk(
                content=document,
                score=score,
                document_id=metadata["document_id"],
                position=metadata["position"],
                title=metadata.get("title", ""),
                source=metadata.get("source", "")
            )

            if chunk.score >= min_score:
                retrieved_chunks.append(chunk)

        return retrieved_chunks