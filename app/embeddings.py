from sentence_transformers import SentenceTransformer

from app.models import Chunk


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_chunks(self, chunks: list[Chunk]):
        texts = [chunk.content for chunk in chunks]

        return self.model.encode(
            texts,
            show_progress_bar=True
        )