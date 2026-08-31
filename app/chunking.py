from app.models import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[Chunk]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = document.content.split()

    chunks = []
    start = 0
    position = 0

    while start < len(words):
        end = start + chunk_size

        content = " ".join(words[start:end])

        chunks.append(
            Chunk(
                document_id=document.id,
                content=content,
                position=position,
                metadata={
                    "title": document.title,
                    "source": str(document.source)
                }
            )
        )

        position += 1
        start += chunk_size - overlap

    return chunks