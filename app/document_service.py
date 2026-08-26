from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.vector_store import VectorStore
from app.document_repository import DocumentRepository


class DocumentService:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        repository: DocumentRepository
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.repository = repository

    def ingest_url(
        self,
        url: str,
        title: str
    ):
        document = fetch_webpage(
            url,
            title
        )

        chunks = chunk_document(
            document
        )

        for chunk in chunks:
            chunk.metadata["title"] = document.title
            chunk.metadata["source"] = str(document.source)

        embeddings = self.embedding_model.embed_chunks(
            chunks
        )

        self.vector_store.add_chunks(
            chunks,
            embeddings
        )

        self.repository.add(
            document
        )

        return document, chunks