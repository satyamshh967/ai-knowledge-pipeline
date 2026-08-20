from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.vector_store import VectorStore


def main():
    document = fetch_webpage(
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "Artificial Intelligence"
    )

    chunks = chunk_document(document)

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.embed_chunks(chunks)

    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks,
        embeddings
    )

    query = "What is machine learning?"

    query_embedding = embedding_model.model.encode(query)

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    print(f"\nQuery: {query}")
    print("\nRetrieved chunks:\n")

    for i, document in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        print(f"--- Result {i + 1} ---")
        print(f"Distance: {distance:.4f}")
        print(document[:500])
        print()


if __name__ == "__main__":
    main()