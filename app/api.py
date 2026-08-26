from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import EmbeddingModel
from app.llm import LLMService
from app.rag import RAGService
from app.retrieval import RetrievalService
from app.vector_store import VectorStore


app = FastAPI(
    title="AI Knowledge Pipeline",
    description="RAG-powered knowledge API",
    version="0.1.0"
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


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


@app.get("/")
def root():
    return {
        "message": "AI Knowledge Pipeline API is running"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = rag_service.answer(
        request.question,
        top_k=3
    )

    return {
        "answer": answer
    }