from pathlib import Path

import chromadb

from app.models import Chunk


class VectorStore:
    def __init__(self, path: str = "./data/chroma"):
        Path(path).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=path)

        self.collection = self.client.get_or_create_collection(
            name="knowledge"
        )

    def add_chunks(self, chunks: list[Chunk], embeddings):
        self.collection.upsert(
            ids=[str(chunk.id) for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_id": str(chunk.document_id),
                    "position": chunk.position,
                }
                for chunk in chunks
            ],
        )

    def search(self, query_embedding, top_k: int = 3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )