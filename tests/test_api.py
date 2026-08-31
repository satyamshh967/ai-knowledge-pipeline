from fastapi.testclient import TestClient

from app.api import app
from app.document_repository import DocumentRepository
from app.vector_store import VectorStore


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "AI Knowledge Pipeline API is running"


def test_query_validation():
    response = client.post(
        "/query",
        json={}
    )

    assert response.status_code == 422


def test_document_validation():
    response = client.post(
        "/documents",
        json={}
    )

    assert response.status_code == 422
    
def test_document_and_query_flow():

    repository = DocumentRepository()

    existing_document = repository.get_by_source(
        "https://en.wikipedia.org/wiki/Machine_learning"
    )

    if existing_document is not None:
        VectorStore().delete_by_document_id(
            str(existing_document.id)
        )

        repository.delete(
            existing_document.id
        )

    document_response = client.post(
        "/documents",
        json={
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "title": "Machine Learning"
        }
    )

    assert document_response.status_code == 200

    document_data = document_response.json()

    document_id = document_data["document_id"]

    try:
        assert document_id
        assert document_data["title"] == "Machine Learning"
        assert document_data["chunks_created"] > 0

        query_response = client.post(
            "/query",
            json={
                "question": "What is machine learning?"
            }
        )

        assert query_response.status_code == 200

        query_data = query_response.json()

        assert query_data["answer"]
        assert "sources" in query_data
        assert len(query_data["sources"]) > 0

    finally:
        client.delete(
            f"/documents/{document_id}"
        )
    
def test_document_and_query_flow():

    # Remove any existing copy from previous test runs.
    documents = client.get("/documents").json()

    for document in documents:

        if document["source"] == (
            "https://en.wikipedia.org/wiki/Machine_learning"
        ):
            client.delete(
                f"/documents/{document['document_id']}"
            )

    document_response = client.post(
        "/documents",
        json={
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "title": "Machine Learning"
        }
    )

    assert document_response.status_code == 200

    document_data = document_response.json()

    document_id = document_data["document_id"]

    try:

        assert document_id
        assert document_data["title"] == "Machine Learning"
        assert document_data["chunks_created"] > 0

        query_response = client.post(
            "/query",
            json={
                "question": "What is machine learning?"
            }
        )

        assert query_response.status_code == 200

        query_data = query_response.json()

        assert query_data["answer"]
        assert "sources" in query_data
        assert len(query_data["sources"]) > 0

    finally:

        client.delete(
            f"/documents/{document_id}"
        )

def test_query_empty_question():
    response = client.post(
        "/query",
        json={
            "question": "   ",
            "top_k": 3
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."
    
def test_document_empty_title():
    response = client.post(
        "/documents",
        json={
            "url": "https://example.com",
            "title": "   "
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Title cannot be empty."