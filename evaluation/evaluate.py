import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import json

from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore
from app.retrieval import RetrievalService
from app.llm import LLMService
from app.rag import RAGService
from app.document_repository import DocumentRepository
from app.document_service import DocumentService


EVALUATION_PATH = Path("evaluation/questions.json")
RESULTS_PATH = Path("evaluation/results.json")


EVALUATION_DOCUMENTS = {
    "Machine Learning": "https://en.wikipedia.org/wiki/Machine_learning",
    "Artificial Intelligence": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "Deep Learning": "https://en.wikipedia.org/wiki/Deep_learning",
}


def load_questions():
    return json.loads(
        EVALUATION_PATH.read_text(
            encoding="utf-8"
        )
    )


def ensure_evaluation_documents(
    document_service: DocumentService,
    document_repository: DocumentRepository
):
    print("\nChecking evaluation documents...")

    for title, url in EVALUATION_DOCUMENTS.items():

        existing = document_repository.get_by_source(
            url
        )

        if existing is not None:
            print(f"  ✓ {title}")
            continue

        print(f"  → Ingesting {title}...")

        document, chunks = document_service.ingest_url(
            url,
            title
        )

        print(
            f"    Created {len(chunks)} chunks"
        )


def evaluate():

    embedding_model = EmbeddingModel()

    vector_store = VectorStore()

    document_repository = DocumentRepository()

    document_service = DocumentService(
        embedding_model,
        vector_store,
        document_repository
    )

    ensure_evaluation_documents(
        document_service,
        document_repository
    )

    retrieval_service = RetrievalService(
        embedding_model,
        vector_store
    )

    llm_service = LLMService()

    rag_service = RAGService(
        retrieval_service,
        llm_service
    )

    questions = load_questions()

    results = []

    for item in questions:

        question = item["question"]

        answer, chunks = rag_service.answer(
            question,
            top_k=3,
            min_score=0.0
        )

        answer_lower = answer.lower()

        matched_keywords = [
            keyword
            for keyword in item["expected_keywords"]
            if keyword.lower() in answer_lower
        ]

        keyword_score = (
            len(matched_keywords)
            / len(item["expected_keywords"])
        )

        expected_source = item.get(
            "expected_source",
            ""
        )

        retrieved_sources = [
            chunk.title
            for chunk in chunks
        ]

        retrieval_scores = [
            round(chunk.score, 3)
            for chunk in chunks
        ]

        source_found = False

        if expected_source:

            source_found = any(
                expected_source.lower()
                == source.lower()
                for source in retrieved_sources
            )

        results.append(
            {
                "question": question,
                "answer": answer,
                "chunks_retrieved": len(chunks),
                "retrieved_sources": retrieved_sources,
                "retrieval_scores": retrieval_scores,
                "expected_source": expected_source,
                "source_found": source_found,
                "matched_keywords": matched_keywords,
                "keyword_score": keyword_score
            }
        )

    total_keyword_score = sum(
        result["keyword_score"]
        for result in results
    )

    total_source_hits = sum(
        1
        for result in results
        if result["source_found"]
    )

    average_keyword_score = (
        total_keyword_score
        / len(results)
    )

    retrieval_accuracy = (
        total_source_hits
        / len(results)
    )

    all_scores = [
        score
        for result in results
        for score in result["retrieval_scores"]
    ]

    average_retrieval_score = (
        sum(all_scores) / len(all_scores)
        if all_scores
        else 0.0
    )

    evaluation_report = {
        "questions_evaluated": len(results),
        "average_keyword_score": average_keyword_score,
        "retrieval_source_accuracy": retrieval_accuracy,
        "average_retrieval_score": average_retrieval_score,
        "source_hits": total_source_hits,
        "results": results
    }

    RESULTS_PATH.write_text(
        json.dumps(
            evaluation_report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nRAG Evaluation Results")
    print("=" * 70)

    for result in results:

        print(
            f"\nQuestion: "
            f"{result['question']}"
        )

        print(
            f"Expected source: "
            f"{result['expected_source']}"
        )

        print(
            f"Retrieved sources: "
            f"{result['retrieved_sources']}"
        )

        print(
            f"Retrieval scores: "
            f"{result['retrieval_scores']}"
        )

        print(
            f"Source retrieved: "
            f"{result['source_found']}"
        )

        print(
            f"Keyword score: "
            f"{result['keyword_score']:.2f}"
        )

        print(
            f"Matched keywords: "
            f"{result['matched_keywords']}"
        )

    print("\n" + "=" * 70)

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    print(
        f"Average keyword score: "
        f"{average_keyword_score:.2f}"
    )

    print(
        f"Retrieval source accuracy: "
        f"{retrieval_accuracy:.2f}"
    )

    print(
        f"Average retrieval score: "
        f"{average_retrieval_score:.3f}"
    )

    print(
        f"Source hits: "
        f"{total_source_hits}/{len(results)}"
    )

    print(
        f"\nDetailed results saved to: "
        f"{RESULTS_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate()
