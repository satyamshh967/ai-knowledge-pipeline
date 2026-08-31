from app.chunking import chunk_document
from app.models import Document


def test_chunk_document_creates_chunks():
    document = Document(
        title="Test Document",
        source="https://example.com",
        content="This is a test document. " * 100
    )

    chunks = chunk_document(document)

    assert len(chunks) > 0
    assert all(chunk.document_id == document.id for chunk in chunks)