from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_get_documents():

    response = client.get("/documents")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_missing_document():
    document_id = uuid4()

    response = client.get(
        f"/documents/{document_id}"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."