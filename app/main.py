from app.chunking import chunk_document
from app.context import build_context
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

    retrieved_chunks = retrieval_service.retrieve(
        query,
        top_k=3
    )

    context = build_context(retrieved_chunks)

    print("\n===== QUERY =====")
    print(query)

    print("\n===== RETRIEVED CONTEXT =====")
    print(context)


if __name__ == "__main__":
    main()