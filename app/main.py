from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage


def main():
    document = fetch_webpage(
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "Artificial Intelligence"
    )

    chunks = chunk_document(document)

    print(f"Document: {document.title}")
    print(f"Total chunks: {len(chunks)}")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_chunks(chunks)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"First embedding:\n{embeddings[0]}")


if __name__ == "__main__":
    main()