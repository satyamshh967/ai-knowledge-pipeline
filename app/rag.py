from app.context import build_context
from app.llm import LLMService
from app.retrieval import RetrievalService


class RAGService:

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService
    ):
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def answer(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.0
    ):

        retrieved_chunks = self.retrieval_service.retrieve(
            question,
            top_k=top_k,
            min_score=min_score
        )

        context = build_context(
            retrieved_chunks
        )

        answer = self.llm_service.generate(
            question,
            context
        )

        return answer, retrieved_chunks
