from app.rag import RAGService
from app.models import RetrievedChunk


class FakeRetrievalService:

    def retrieve(
        self,
        question,
        top_k=3,
        min_score=0.0
    ):
        return [
            RetrievedChunk(
                content="Machine learning is a subset of artificial intelligence.",
                score=0.95,
                document_id="12345678-1234-1234-1234-123456789012",
                position=0,
                title="Machine Learning",
                source="https://example.com"
            )
        ]


class FakeLLMService:

    def generate(self, question, context):
        return "Machine learning is a subset of artificial intelligence."


def test_rag_service_returns_answer():

    retrieval_service = FakeRetrievalService()
    llm_service = FakeLLMService()

    rag_service = RAGService(
        retrieval_service,
        llm_service
    )

    answer, chunks = rag_service.answer(
        "What is machine learning?",
        top_k=3
    )

    assert answer != ""
    assert len(chunks) == 1
    assert chunks[0].content.startswith("Machine learning")