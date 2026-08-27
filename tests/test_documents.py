from uuid import uuid4

from app.models import Document
from app.document_repository import DocumentRepository


def test_add_and_get_document(tmp_path):

    repository = DocumentRepository(
        path=str(tmp_path / "documents.json")
    )

    document = Document(
        title="Test Document",
        source="https://example.com",
        content="This is a test document."
    )

    repository.add(document)

    result = repository.get(document.id)

    assert result is not None
    assert result.id == document.id
    assert result.title == "Test Document"
    assert result.content == "This is a test document."


def test_get_all_documents(tmp_path):

    repository = DocumentRepository(
        path=str(tmp_path / "documents.json")
    )

    document_one = Document(
        title="Document One",
        source="https://example.com/one",
        content="First document."
    )

    document_two = Document(
        title="Document Two",
        source="https://example.com/two",
        content="Second document."
    )

    repository.add(document_one)
    repository.add(document_two)

    documents = repository.get_all()

    assert len(documents) == 2
    assert documents[0].title == "Document One"
    assert documents[1].title == "Document Two"


def test_delete_document(tmp_path):

    repository = DocumentRepository(
        path=str(tmp_path / "documents.json")
    )

    document = Document(
        title="Delete Me",
        source="https://example.com",
        content="This document will be deleted."
    )

    repository.add(document)

    result = repository.delete(document.id)

    assert result is True
    assert repository.get(document.id) is None


def test_delete_missing_document(tmp_path):

    repository = DocumentRepository(
        path=str(tmp_path / "documents.json")
    )

    document_id = uuid4()

    result = repository.delete(document_id)

    assert result is False