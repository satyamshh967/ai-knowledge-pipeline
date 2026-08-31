from sentence_transformers import SentenceTransformer

from app.config import settings
from app.models import Chunk


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            settings.embedding_model
        )

    def embed_chunks(self, chunks: list[Chunk]):
        texts = [
            chunk.content
            for chunk in chunks
        ]

        return self.model.encode(
            texts,
            show_progress_bar=True
        )