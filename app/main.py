from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.search import SemanticSearch


def main():
    document = fetch_webpage(
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "Artificial Intelligence"
    )

    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_chunks(chunks)

    search_engine = SemanticSearch(embedding_model)

    query = "What is machine learning?"

    results = search_engine.search(
        query,
        chunks,
        embeddings,
        top_k=3
    )

    print(f"\nQuery: {query}")
    print("\nMost relevant chunks:\n")

    for result in results:
        chunk = result["chunk"]
        score = result["score"]

        print(f"--- Score: {score:.4f} ---")
        print(f"Chunk position: {chunk.position}")
        print(chunk.content[:500])
        print()


if __name__ == "__main__":
    main()