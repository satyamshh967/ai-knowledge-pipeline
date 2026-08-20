from app.models import RetrievedChunk


def build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []

    for chunk in chunks:
        sections.append(
            f"[Source chunk {chunk.position}]\n"
            f"{chunk.content}"
        )

    return "\n\n".join(sections)