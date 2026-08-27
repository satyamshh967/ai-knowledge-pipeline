from uuid import UUID
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.embeddings import EmbeddingModel
from app.llm import LLMService
from app.rag import RAGService
from app.retrieval import RetrievalService
from app.vector_store import VectorStore
from app.document_service import DocumentService
from app.document_repository import DocumentRepository


app = FastAPI(
    title="AI Knowledge Pipeline",
    description="RAG-powered knowledge API",
    version="0.1.0"
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    document_id: str
    chunk_position: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentRequest(BaseModel):
    url: str
    title: str


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    chunks_created: int


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    source: str
    created_at: str


# Initialize our RAG components once when the API starts.

embedding_model = EmbeddingModel()

vector_store = VectorStore()

retrieval_service = RetrievalService(
    embedding_model,
    vector_store
)

llm_service = LLMService()

rag_service = RAGService(
    retrieval_service,
    llm_service
)

document_repository = DocumentRepository()

document_service = DocumentService(
    embedding_model,
    vector_store,
    document_repository
)


@app.get("/")
def root():
    return {
        "message": "AI Knowledge Pipeline API is running"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    answer, retrieved_chunks = rag_service.answer(
        request.question,
        top_k=request.top_k
    )

    sources = [
        Source(
            document_id=str(chunk.document_id),
            chunk_position=chunk.position,
            score=chunk.score
        )
        for chunk in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources
    }


@app.post("/documents", response_model=DocumentResponse)
def create_document(request: DocumentRequest):

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    document, chunks = document_service.ingest_url(
        request.url,
        request.title
    )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "chunks_created": len(chunks)
    }


@app.get("/documents", response_model=list[DocumentSummary])
def get_documents():

    documents = document_repository.get_all()

    return [
        DocumentSummary(
            document_id=str(document.id),
            title=document.title,
            source=str(document.source),
            created_at=document.created_at.isoformat()
        )
        for document in documents
    ]


@app.get("/documents/{document_id}")
def get_document(document_id: str):

    from uuid import UUID

    try:
        document_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )

    document = document_repository.get(document_uuid)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "source": str(document.source),
        "content": document.content,
        "created_at": document.created_at.isoformat(),
        "metadata": document.metadata
    }
    
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    from uuid import UUID

    try:
        document_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )

    deleted = document_service.delete_document(
        document_uuid
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id
    }
    
@app.delete("/documents/{document_id}")
def delete_document(document_id: str):

    try:
        document_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID."
        )

    document = document_repository.get(document_uuid)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    deleted_chunks = vector_store.delete_by_document_id(
        str(document_uuid)
    )

    deleted = document_repository.delete(
        document_uuid
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": str(document_uuid),
        "chunks_deleted": deleted_chunks
    }