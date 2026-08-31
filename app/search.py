import numpy as np

from app.embeddings import EmbeddingModel
from app.models import Chunk


class SemanticSearch:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def search(
        self,
        query: str,
        chunks: list[Chunk],
        embeddings,
        top_k: int = 3
    ):
        query_embedding = self.embedding_model.model.encode(query)

        similarities = np.dot(
            embeddings,
            query_embedding
        ) / (
            np.linalg.norm(embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            results.append(
                {
                    "chunk": chunks[index],
                    "score": float(similarities[index])
                }
            )

        return results