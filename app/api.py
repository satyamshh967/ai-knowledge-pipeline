import logging
import time
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.database import Base, engine
import app.database_models
Base.metadata.create_all(bind=engine)

from app.embeddings import EmbeddingModel
from app.llm import LLMService
from app.rag import RAGService
from app.retrieval import RetrievalService
from app.vector_store import VectorStore
from app.document_service import DocumentService
from app.document_repository import DocumentRepository


Base.metadata.create_all(bind=engine)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ai-knowledge-pipeline")


app = FastAPI(
    title="AI Knowledge Pipeline",
    description="RAG-powered knowledge API",
    version="0.2.0"
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )


class Source(BaseModel):
    document_id: str
    title: str
    source: str
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


logger.info("Initializing AI Knowledge Pipeline services")

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

logger.info("All services initialized")


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):
    request_id = str(uuid4())
    start_time = time.perf_counter()

    logger.info(
        "Request started | id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)

        duration = (
            time.perf_counter()
            - start_time
        )

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed | id=%s | status=%s | duration=%.3fs",
            request_id,
            response.status_code,
            duration
        )

        return response

    except Exception:
        duration = (
            time.perf_counter()
            - start_time
        )

        logger.exception(
            "Request failed | id=%s | duration=%.3fs",
            request_id,
            duration
        )

        raise


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):
    logger.exception(
        "Unhandled exception | method=%s | path=%s",
        request.method,
        request.url.path
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        }
    )


@app.get("/")
def root():
    return {
        "message": "AI Knowledge Pipeline API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/query",
    response_model=QueryResponse
)
def query(
    request: QueryRequest
):
    logger.info(
        "Query received | top_k=%s | min_score=%s",
        request.top_k,
        request.min_score
    )

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    answer, retrieved_chunks = rag_service.answer(
        request.question,
        top_k=request.top_k,
        min_score=request.min_score
    )

    sources = [
    Source(
        document_id=str(chunk.document_id),
        title=chunk.title,
        source=chunk.source,
        chunk_position=chunk.position,
        score=chunk.score
    )
    for chunk in retrieved_chunks
]

    logger.info(
        "Query completed | chunks_retrieved=%s",
        len(retrieved_chunks)
    )

    return {
        "answer": answer,
        "sources": sources
    }


@app.post(
    "/documents",
    response_model=DocumentResponse
)
def create_document(
    request: DocumentRequest
):
    logger.info(
        "Document ingestion requested | title=%s | url=%s",
        request.title,
        request.url
    )

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    document, chunks = document_service.ingest_url(
        request.url,
        request.title
    )

    logger.info(
        "Document ingestion completed | document_id=%s | chunks=%s",
        document.id,
        len(chunks)
    )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "chunks_created": len(chunks)
    }


@app.get(
    "/documents",
    response_model=list[DocumentSummary]
)
def get_documents():
    documents = document_repository.get_all()

    logger.info(
        "Documents listed | count=%s",
        len(documents)
    )

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
def get_document(
    document_id: UUID
):
    document = document_repository.get(
        document_id
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "source": str(document.source),
        "content": document.content,
        "created_at": document.created_at.isoformat(),
        "metadata": document.metadata
    }

@app.put(
    "/documents/{document_id}",
    response_model=DocumentResponse
)
def update_document(
    document_id: UUID,
    request: DocumentRequest
):
    logger.info(
        "Document update requested | document_id=%s | title=%s",
        document_id,
        request.title
    )

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    document, chunks = document_service.update_document(
        document_id,
        request.title
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    logger.info(
        "Document update completed | document_id=%s | chunks=%s",
        document.id,
        len(chunks)
    )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "chunks_created": len(chunks)
    }

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: UUID
):
    logger.info(
        "Document deletion requested | document_id=%s",
        document_id
    )

    deleted = document_service.delete_document(
        document_id
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    logger.info(
        "Document deleted | document_id=%s",
        document_id
    )

    return {
        "message": "Document deleted successfully.",
        "document_id": str(document_id)
    }