from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.embeddings import EmbeddingModel
from app.llm import LLMService
from app.rag import RAGService
from app.retrieval import RetrievalService
from app.vector_store import VectorStore
from app.document_service import DocumentService


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

document_service = DocumentService(
    embedding_model,
    vector_store
)


@app.get("/")
def root():
    return {
        "message": "AI Knowledge Pipeline API is running"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
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
    document, chunks = document_service.ingest_url(
        request.url,
        request.title
    )

    return {
        "document_id": str(document.id),
        "title": document.title,
        "chunks_created": len(chunks)
    }