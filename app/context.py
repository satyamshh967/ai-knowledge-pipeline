from app.models import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:

    sections = []

    for chunk in chunks:

        sections.append(
            f"[Source: {chunk.title}]\n"
            f"[Chunk: {chunk.position}]\n"
            f"[Relevance Score: {chunk.score:.3f}]\n"
            f"{chunk.content}"
        )

    return "\n\n".join(sections)
