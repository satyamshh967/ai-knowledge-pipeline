from app.embeddings import EmbeddingModel
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
        top_k: int = 3
    ):
        query_embedding = self.embedding_model.model.encode(query)

        return self.vector_store.search(
            query_embedding,
            top_k=top_k
        )