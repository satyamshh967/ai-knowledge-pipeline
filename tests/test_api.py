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