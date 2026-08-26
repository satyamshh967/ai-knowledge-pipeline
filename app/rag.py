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
        top_k: int = 3
    ) -> str:

        retrieved_chunks = self.retrieval_service.retrieve(
            question,
            top_k=top_k
        )

        context = build_context(retrieved_chunks)

        return self.llm_service.generate(
            question,
            context
        )