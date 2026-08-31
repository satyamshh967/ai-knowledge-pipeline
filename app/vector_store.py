from pathlib import Path

import chromadb

from app.models import Chunk
from app.config import settings


class VectorStore:

    def __init__(self, path: str | None = None):

        if path is None:
            path = settings.vector_store_path

        Path(path).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = self.client.get_or_create_collection(
            name="knowledge"
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings
    ):

        self.collection.upsert(
            ids=[
                str(chunk.id)
                for chunk in chunks
            ],
            embeddings=embeddings.tolist(),
            documents=[
                chunk.content
                for chunk in chunks
            ],
            metadatas=[
                {
                    "document_id": str(chunk.document_id),
                    "position": chunk.position,
                    "title": chunk.metadata.get("title", ""),
                    "source": chunk.metadata.get("source", ""),
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding,
        top_k: int = 3
    ):

        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
        )

    def delete_by_document_id(
        self,
        document_id: str
    ):

        existing = self.collection.get(
            where={
                "document_id": document_id
            }
        )

        ids = existing["ids"]

        if not ids:
            return 0

        self.collection.delete(
            ids=ids
        )

        return len(ids)