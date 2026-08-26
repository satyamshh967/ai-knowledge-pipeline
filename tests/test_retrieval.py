from uuid import uuid4

from app.models import RetrievedChunk
from app.retrieval import RetrievalService

class FakeEmbeddingModel:

    def __init__(self):
        self.model = self

    def encode(self, text):
        return [0.1, 0.2, 0.3]


class FakeVectorStore:

    def search(self, query_embedding, top_k=3):
        return {
            "documents": [
                ["Machine learning is a field of AI."]
            ],
            "distances": [
                [0.2]
            ],
            "metadatas": [
                [
                    {
                        "document_id": str(uuid4()),
                        "position": 0,
                        "title": "Machine Learning",
                        "source": "https://example.com/ml"
                    }
                ]
            ]
        }


def test_retrieval_returns_chunks():

    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    retrieval_service = RetrievalService(
        embedding_model,
        vector_store
    )

    results = retrieval_service.retrieve(
        "What is machine learning?",
        top_k=1
    )

    assert len(results) == 1

    assert isinstance(
        results[0],
        RetrievedChunk
    )

    assert results[0].title == "Machine Learning"

    assert results[0].source == "https://example.com/ml"

    assert results[0].position == 0

    assert results[0].score == 0.8