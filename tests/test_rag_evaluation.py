from app.models import RetrievedChunk
from app.rag import RAGService


class FakeRetrievalService:

    def retrieve(
        self,
        question,
        top_k=3,
        min_score=0.0
    ):
        return [
            RetrievedChunk(
                content=(
                    "Machine learning is a subset of artificial "
                    "intelligence that enables computers to learn "
                    "from data without being explicitly programmed."
                ),
                score=0.91,
                document_id="12345678-1234-1234-1234-123456789012",
                position=0,
                title="Machine Learning",
                source="https://example.com/ml"
            )
        ]


class FakeLLMService:

    def generate(self, question, context):

        assert "Machine learning" in context

        return (
            "Machine learning is a subset of artificial "
            "intelligence that allows computers to learn "
            "from data."
        )


def test_rag_pipeline_answers_from_retrieved_context():

    retrieval_service = FakeRetrievalService()
    llm_service = FakeLLMService()

    rag_service = RAGService(
        retrieval_service,
        llm_service
    )

    answer, chunks = rag_service.answer(
        "What is machine learning?",
        top_k=3,
        min_score=0.5
    )

    assert answer != ""

    assert len(chunks) == 1

    assert chunks[0].title == "Machine Learning"

    assert chunks[0].source == "https://example.com/ml"

    assert chunks[0].score >= 0.5

    assert "machine learning" in answer.lower()