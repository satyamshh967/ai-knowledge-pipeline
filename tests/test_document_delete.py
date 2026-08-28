from uuid import uuid4

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_delete_missing_document():

    document_id = uuid4()

    response = client.delete(
        f"/documents/{document_id}"
    )

    assert response.status_code == 404

    assert response.json()["detail"] == "Document not found"