from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.retrieval import RetrievalService
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

    retrieval_service = RetrievalService(
        embedding_model,
        vector_store
    )

    query = "What is machine learning?"

    results = retrieval_service.retrieve(
        query,
        top_k=3
    )

    print(f"\nQuery: {query}")
    print("\nRetrieved chunks:\n")

    for i, content in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]

        print(f"--- Result {i + 1} ---")
        print(f"Distance: {distance:.4f}")
        print(content[:500])
        print()


if __name__ == "__main__":
    main()