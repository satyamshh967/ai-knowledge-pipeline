from fastapi.testclient import TestClient

from app.api import app


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
    document_response = client.post(
        "/documents",
        json={
            "url": "https://en.wikipedia.org/wiki/Machine_learning",
            "title": "Machine Learning"
        }
    )

    assert document_response.status_code == 200

    document_data = document_response.json()

    assert document_data["document_id"]
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