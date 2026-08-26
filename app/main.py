from app.chunking import chunk_document
from app.embeddings import EmbeddingModel
from app.ingestion import fetch_webpage
from app.llm import LLMService
from app.rag import RAGService
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

    llm_service = LLMService()

    rag = RAGService(
        retrieval_service,
        llm_service
    )

    question = "What is machine learning?"

    answer = rag.answer(
        question,
        top_k=3
    )

    print("\n===== QUESTION =====")
    print(question)

    print("\n===== ANSWER =====")
    print(answer)


if __name__ == "__main__":
    main()