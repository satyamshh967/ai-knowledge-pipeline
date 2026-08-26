from fastapi import FastAPI, HTTPException
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
    title: str = Field(min_length=1, max_length=200)


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

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        answer, retrieved_chunks = rag_service.answer(
            request.question,
            top_k=request.top_k
        )

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found."
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

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(exc)}"
        )

@app.post("/documents", response_model=DocumentResponse)
def create_document(request: DocumentRequest):

    if not request.url.strip():
        raise HTTPException(
            status_code=400,
            detail="URL cannot be empty."
        )

    if not request.title.strip():
        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    try:
        document, chunks = document_service.ingest_url(
            request.url,
            request.title
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Document was created but no chunks were generated."
            )

        return {
            "document_id": str(document.id),
            "title": document.title,
            "chunks_created": len(chunks)
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(exc)}"
        )