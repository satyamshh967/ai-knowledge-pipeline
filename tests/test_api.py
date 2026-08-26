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
    
def test_query_top_k_validation():
    response = client.post(
        "/query",
        json={
            "question": "What is machine learning?",
            "top_k": 0
        }
    )

    assert response.status_code == 422

    response = client.post(
        "/query",
        json={
            "question": "What is machine learning?",
            "top_k": 11
        }
    )

    assert response.status_code == 422

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