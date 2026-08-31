from uuid import UUID

from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.models import Document
from app.vector_store import VectorStore
from app.document_repository import DocumentRepository


class DocumentService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        document_repository: DocumentRepository
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.document_repository = document_repository

    def ingest_url(
        self,
        url: str,
        title: str
    ):

        existing_document = (
            self.document_repository.get_by_source(url)
        )

        if existing_document is not None:
            return existing_document, []

        document = fetch_webpage(
            url,
            title
        )

        chunks = chunk_document(
            document
        )

        embeddings = self.embedding_model.embed_chunks(
            chunks
        )

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        self.document_repository.add(
            document
        )

        return document, chunks

    def update_document(
        self,
        document_id: UUID,
        title: str
    ):

        existing_document = (
            self.document_repository.get(document_id)
        )

        if existing_document is None:
            return None, []

        updated_document = fetch_webpage(
            str(existing_document.source),
            title
        )

        updated_document = updated_document.model_copy(
            update={
                "id": existing_document.id,
                "created_at": existing_document.created_at
            }
        )

        chunks = chunk_document(
            updated_document
        )

        embeddings = self.embedding_model.embed_chunks(
            chunks
        )

        self.vector_store.delete_by_document_id(
            str(document_id)
        )

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        self.document_repository.update(
            updated_document
        )

        return updated_document, chunks

    def delete_document(
        self,
        document_id: UUID
    ):

        document = self.document_repository.get(
            document_id
        )

        if document is None:
            return False

        self.vector_store.delete_by_document_id(
            str(document_id)
        )

        self.document_repository.delete(
            document_id
        )

        return True
