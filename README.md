AI Knowledge Pipeline

A production-oriented Retrieval-Augmented Generation (RAG) API built
with Python, FastAPI, Sentence Transformers, ChromaDB, and an
OpenRouter-compatible LLM.

The project is designed as more than a simple chatbot. Its goal is to
demonstrate how a real knowledge system can ingest external documents,
transform them into searchable vector representations, retrieve relevant
knowledge for a user query, and generate an answer grounded in that
retrieved context.

Current status: Core document ingestion, chunking, embeddings,
persistent vector storage, semantic retrieval, RAG generation, source
attribution, document CRUD, validation, Dockerized deployment,
automated tests, and a basic RAG evaluation pipeline are implemented.

Why I Built This

I wanted to build a project that goes beyond basic CRUD APIs and
demonstrates how multiple backend and AI components work together as one
system.

The central problem is:

How can an application answer questions using a controlled knowledge
base instead of relying entirely on the model's pretrained knowledge?

This project addresses that problem through a RAG pipeline:

User Question
      |
      v
   FastAPI
      |
      v
 Query Embedding
      |
      v
   ChromaDB
      |
      v
Semantic Retrieval
      |
      v
Relevant Chunks
      |
      v
Context Builder
      |
      v
    LLM
      |
      v
Answer + Sources

For document ingestion, the flow is:

URL
 |
 v
Web Ingestion
 |
 v
Document
 |
 v
Chunking
 |
 v
Embeddings
 |
 v
ChromaDB
 |
 v
Persistent Knowledge Base

Project Goals

The project is being developed with several goals:

Build a complete RAG backend rather than a standalone AI demo.

Understand semantic search and vector databases through
implementation.

Practice backend architecture and separation of responsibilities.

Build APIs that are validated and testable.

Persist both documents and vector embeddings.

Provide source attribution for generated answers.

Add evaluation instead of judging the RAG system only by manually
testing it.

Containerize the application with Docker.

Gradually evolve the project toward a more production-like
architecture.

Core Features

1. Web Document Ingestion

The API accepts a webpage URL and a title.

The ingestion pipeline:

Downloads the webpage.

Extracts its textual content.

Creates a Document object.

Splits the document into overlapping chunks.

Generates embeddings for every chunk.

Stores the chunks and embeddings in ChromaDB.

Stores document metadata in the document repository.

Example:

POST /documents

{
  "url": "https://en.wikipedia.org/wiki/Machine_learning",
  "title": "Machine Learning"
}

2. Chunking

Large documents are divided into smaller overlapping pieces before
embedding.

The current implementation uses:

Chunk size: 500 words

Overlap: 50 words

The overlap helps preserve context between neighboring chunks.

For example:

Chunk 1
[--------------------]

             Chunk 2
             [--------------------]

                         Chunk 3
                         [--------------------]

This is important because embedding an entire webpage as one vector
would make precise retrieval much harder.

3. Embeddings

The project uses:

sentence-transformers/all-MiniLM-L6-v2

Each chunk is transformed into a numerical vector representing its
semantic meaning.

A query is embedded using the same model.

This allows the system to compare:

"What is machine learning?"

with stored chunks based on semantic similarity rather than simple
keyword matching.

4. Persistent Vector Search

ChromaDB is used as the vector database.

The vector store persists data under:

data/chroma/

Each stored chunk contains:

Chunk ID

Document ID

Chunk content

Chunk position

Document title

Source URL

The metadata is important because retrieval is not useful by itself
unless the system can also determine where the retrieved knowledge came
from.

5. Semantic Retrieval

When a question is submitted:

The question is embedded.

ChromaDB searches the vector collection.

The top k chunks are returned.

A relevance score is calculated.

Chunks below the configured minimum score can be filtered.

The API supports:

top_k
min_score

Current validation:

top_k: 1–10
min_score: 0.0–1.0

6. RAG Generation

The retrieved chunks are passed to an LLM as knowledge context.

The model is explicitly instructed to:

Use only the supplied knowledge.

Avoid inventing facts.

Admit when the available context is insufficient.

Conceptually:

Retrieved Knowledge
        +
     Question
        |
        v
       LLM
        |
        v
Grounded Answer

This separates the retrieval problem from the generation problem.

7. Source Attribution

Every retrieved chunk contains source information.

The query response therefore returns:

{
  "answer": "...",
  "sources": [
    {
      "document_id": "...",
      "chunk_position": 8,
      "score": 0.585
    }
  ]
}

This makes the system more transparent than returning an answer with no
indication of where the information came from.

API

GET /

Returns a basic API status message.

GET /health

Health check endpoint.

Example:

{
  "status": "healthy"
}

POST /query

Ask a question against the knowledge base.

Example:

{
  "question": "What is machine learning?",
  "top_k": 3,
  "min_score": 0.0
}

POST /documents

Ingest a webpage into the knowledge base.

Example:

{
  "url": "https://en.wikipedia.org/wiki/Machine_learning",
  "title": "Machine Learning"
}

GET /documents

List all ingested documents.

GET /documents/{document_id}

Retrieve a specific document and its metadata.

DELETE /documents/{document_id}

Deletes the document and its associated vector chunks.

Architecture

The project follows a service-oriented structure.

                         +----------------+
                         |     User       |
                         +-------+--------+
                                 |
                                 v
                         +-------+--------+
                         |    FastAPI     |
                         |    app/api.py  |
                         +---+--------+---+
                             |        |
                    Documents|        |Query
                             |        |
                             v        v
                    +--------+--+  +--+-------------+
                    | Document  |  | Retrieval      |
                    | Service   |  | Service        |
                    +-----+-----+  +-------+--------+
                          |                |
                          v                v
                    +-----+-----+    +----+-------+
                    | Ingestion |    | Embedding  |
                    | + Chunking|    | Model      |
                    +-----+-----+    +----+-------+
                          |                |
                          +-------+--------+
                                  |
                                  v
                           +------+------+
                           |  ChromaDB   |
                           | VectorStore |
                           +------+------+
                                  |
                                  v
                           Retrieved Chunks
                                  |
                                  v
                           +------+------+
                           | RAG Service |
                           +------+------+
                                  |
                                  v
                           +------+------+
                           | LLM Service |
                           +------+------+
                                  |
                                  v
                            Answer + Sources

Project Structure

ai-knowledge-pipeline/
│
├── app/
│   ├── api.py
│   ├── chunking.py
│   ├── config.py
│   ├── context.py
│   ├── document_repository.py
│   ├── document_service.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm.py
│   ├── models.py
│   ├── rag.py
│   ├── retrieval.py
│   └── vector_store.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── questions.json
│   └── results.json
│
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_document_delete.py
│   ├── test_documents.py
│   ├── test_rag.py
│   ├── test_rag_evaluation.py
│   ├── test_retrieval.py
│   └── test_vector_store.py
│
├── data/
│   ├── chroma/
│   └── documents.json
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md

Design Decisions

Why FastAPI?

FastAPI provides:

Typed request and response models.

Automatic validation.

OpenAPI documentation.

Easy testing through TestClient.

A clean foundation for building a backend API.

Why ChromaDB?

A traditional relational database is excellent for structured records,
but semantic retrieval requires vector similarity search.

ChromaDB provides persistent vector storage and similarity search
without requiring a separate infrastructure stack for this project.

Why Sentence Transformers?

A local embedding model allows the semantic retrieval layer to operate
without sending every embedding request to an external API.

This also makes the architecture:

Document -> Local Embedding Model -> Vector DB
Question -> Local Embedding Model -> Vector DB

Why Separate Services?

Instead of putting everything inside api.py, responsibilities are
separated:

DocumentService
RetrievalService
RAGService
LLMService
VectorStore
DocumentRepository
EmbeddingModel

This makes the system easier to test, replace, and extend.

For example, the LLM provider can eventually be replaced without
rewriting the retrieval layer.

Data Flow

Document ingestion

POST /documents
       |
       v
DocumentService
       |
       v
fetch_webpage()
       |
       v
Document
       |
       v
chunk_document()
       |
       v
Chunks
       |
       v
EmbeddingModel
       |
       v
Embeddings
       |
       +------------------+
       |                  |
       v                  v
   ChromaDB        DocumentRepository

Query

POST /query
       |
       v
RetrievalService
       |
       v
Query Embedding
       |
       v
ChromaDB
       |
       v
Top-K Chunks
       |
       v
RAGService
       |
       v
Context Builder
       |
       v
LLMService
       |
       v
Answer + Sources

Duplicate Document Handling

The system checks whether a document with the same source URL already
exists.

If it does, ingestion is skipped instead of creating another copy.

This prevents repeated ingestion of the same webpage from unnecessarily
increasing the vector collection.

Error Handling and Validation

The API validates request data through Pydantic.

Examples include:

Empty questions are rejected.

Empty document titles are rejected.

Invalid top_k values are rejected.

Invalid min_score values are rejected.

Missing documents return 404.

Invalid request bodies return 422.

This keeps invalid input away from the core services.

Testing

The project contains automated tests covering:

Root endpoint

Health endpoint

Query validation

Document validation

Document creation

Document retrieval

Document listing

Document deletion

Duplicate document behavior

Chunking

Vector storage

Retrieval

RAG behavior

Evaluation behavior

Tests are run inside the Docker container:

docker compose exec api python -m pytest -q

RAG Evaluation

The project includes a small evaluation pipeline rather than relying
exclusively on manual testing.

Evaluation questions currently cover:

Machine learning

Artificial intelligence

Deep learning

The evaluator measures:

Keyword score

Whether expected concepts appear in the generated answer.

Source retrieval accuracy

Whether the expected document appears among retrieved sources.

Retrieval score

The relevance score of retrieved chunks.

Run evaluation with:

docker compose exec api python evaluation/evaluate.py

Results are written to:

evaluation/results.json

The evaluation is intentionally simple at this stage. The future goal is
to replace the basic keyword metric with stronger automated evaluation.

Docker

The application runs as a Dockerized FastAPI service.

Start the project:

docker compose up -d

Check the container:

docker compose ps

View logs:

docker compose logs --tail=100 api

Stop the project:

docker compose down

The application is exposed on:

http://localhost:8000

FastAPI's interactive documentation is available through the
application's Swagger/OpenAPI interface.

Configuration

The application uses environment variables.

Create a .env file containing:

OPENROUTER_API_KEY=your_api_key_here

The .env file should never be committed to Git.

For sharing the project, use:

.env.example

instead.

Current Limitations

This is an intentionally evolving project.

Current limitations include:

The document repository uses a JSON file rather than a production
relational database.

The vector database runs inside the application environment.

Webpage extraction is still relatively basic.

Embedding generation is currently performed synchronously.

There is no authentication or authorization yet.

There is no user-specific knowledge isolation.

The evaluation dataset is small.

RAG evaluation currently relies on lightweight metrics.

There is no frontend yet.

There is no background job system for large ingestion workloads.

There is no production observability stack yet.

These are not accidental omissions; several of them are planned future
improvements.

Future Roadmap

The project will evolve in stages.

Phase 1 --- Core RAG Backend

Status: Mostly complete

FastAPI backend

Web document ingestion

Document chunking

Sentence Transformer embeddings

Persistent ChromaDB

Semantic retrieval

Relevance scoring

RAG answer generation

Source attribution

Document CRUD

Duplicate document detection

Input validation

Automated tests

Basic RAG evaluation

Dockerized deployment

Phase 2 --- Production API Improvements

Next priority

Centralized exception handling

Better structured logging

Request IDs

Improved API response schemas

Pagination for documents

Better URL validation

Rate limiting

Configuration improvements

More robust ingestion error handling

Goal:

Turn the current functional backend into a cleaner production-style
API.

Phase 3 --- Better Retrieval

The current retrieval system is intentionally simple.

Future improvements:

Hybrid keyword + semantic search

Better similarity metrics

Reranking

Metadata filtering

Diversity-aware retrieval

Query expansion

Retrieval evaluation with larger datasets

Experiment with different chunk sizes and overlaps

The objective is to improve the most important part of a RAG system:

retrieving the right knowledge.

Phase 4 --- Better RAG Generation

Planned improvements:

Stronger prompt design

Explicit citation formatting

Context compression

Answer confidence signals

Better handling of insufficient context

Model comparison

LLM evaluation

Hallucination testing

The target behavior is:

Good retrieval
      +
Relevant context
      +
Controlled generation
      =
Reliable answer

Phase 5 --- Database Upgrade

The current JSON document repository is suitable for learning and the
current scale.

The next backend upgrade will move document metadata to a real database.

Potential stack:

PostgreSQL
    +
SQLAlchemy
    +
Alembic

The architecture should become:

FastAPI
   |
   +---- PostgreSQL
   |
   +---- ChromaDB
   |
   +---- LLM

Phase 6 --- Authentication and Multi-User Knowledge Bases

Planned features:

User registration/login

JWT authentication

User ownership

Per-user documents

Authorization checks

Private knowledge bases

Document ownership enforcement

Eventually:

User A -> Knowledge Base A
User B -> Knowledge Base B

with strict isolation between them.

Phase 7 --- Background Processing

Large document ingestion should not block an HTTP request.

Future architecture:

POST /documents
       |
       v
Create ingestion job
       |
       v
Queue
       |
       v
Worker
       |
       +--> Download
       +--> Parse
       +--> Chunk
       +--> Embed
       +--> Store

Potential technologies:

Celery

Redis

Dramatiq

Background workers

The API could then return a job ID and expose job status.

Phase 8 --- Observability

Production systems need visibility into what is happening.

Planned additions:

Structured logs

Request latency

Retrieval latency

Embedding latency

LLM latency

Error rates

Token usage

Retrieval quality metrics

Health/readiness checks

Eventually the system should make it possible to answer:

Why was this answer slow?

and:

Why did the system retrieve the wrong document?

Phase 9 --- Frontend

After the backend is mature, a frontend can be added.

Possible interface:

+---------------------------------------+
|        AI Knowledge Assistant         |
+---------------------------------------+
|                                       |
|  Ask a question...              [Ask] |
|                                       |
+---------------------------------------+
| Answer                                |
|                                       |
| ...                                   |
|                                       |
+---------------------------------------+
| Sources                               |
|                                       |
| Machine Learning - Chunk 8            |
| Deep Learning - Chunk 21              |
+---------------------------------------+

The frontend would also provide:

Document management

Upload/URL ingestion

Search

Source inspection

Query history

Knowledge-base management

Phase 10 --- Deployment

Final deployment goals:

GitHub
   |
   v
CI/CD
   |
   v
Docker Image
   |
   v
Cloud Deployment

Potential infrastructure:

Cloud VM/container platform

Managed PostgreSQL

Persistent vector storage

HTTPS

Secret management

CI/CD

Monitoring

Long-Term Architecture

The long-term goal is to evolve the project toward:

                         Internet
                            |
                            v
                    +---------------+
                    |    Frontend   |
                    +-------+-------+
                            |
                            v
                    +---------------+
                    |    FastAPI    |
                    +-------+-------+
                            |
             +--------------+--------------+
             |              |              |
             v              v              v
        PostgreSQL       Redis          Auth
             |              |
             |              v
             |           Workers
             |              |
             |       +------+------+ 
             |       |             |
             |       v             v
             |    Ingestion     Embedding
             |       |             |
             |       +------+------+
             |              |
             v              v
        Metadata        ChromaDB
                            |
                            v
                       Retrieval
                            |
                            v
                       Reranking
                            |
                            v
                         Context
                            |
                            v
                           LLM
                            |
                            v
                    Answer + Citations

What I Am Learning From This Project

This project is intentionally being used as a learning vehicle.

The major concepts covered include:

Backend Engineering

REST APIs

Request validation

Response models

Service architecture

Repository pattern

Error handling

Testing

Docker

AI / ML Engineering

Embeddings

Semantic similarity

Vector databases

Retrieval

RAG

Prompt construction

LLM integration

Evaluation

Systems Thinking

The most important lesson is that an AI application is not simply:

Prompt -> LLM

A useful production-oriented AI system requires several interconnected
layers:

Data
 |
 v
Ingestion
 |
 v
Transformation
 |
 v
Representation
 |
 v
Storage
 |
 v
Retrieval
 |
 v
Context
 |
 v
Generation
 |
 v
Evaluation
 |
 v
Monitoring

Understanding how these pieces interact is one of the main purposes of
this project.

Development Philosophy

The project is being developed incrementally.

Rather than attempting to build a massive AI platform immediately, each
stage introduces one additional engineering challenge while keeping the
previous functionality working.

The progression is:

Working prototype
       |
       v
Better architecture
       |
       v
Better retrieval
       |
       v
Better evaluation
       |
       v
Production concerns
       |
       v
Scalability
       |
       v
Deployment

This approach makes it possible to understand why each technology is
being introduced instead of simply assembling a collection of
frameworks.

Current Technical Stack

Component          Technology

Language           Python
API                FastAPI
Validation         Pydantic
Embeddings         Sentence Transformers
Embedding Model    all-MiniLM-L6-v2
Vector Database    ChromaDB
LLM Gateway        OpenRouter
HTTP Client        OpenAI Python SDK
Document Storage   JSON
Testing            Pytest
Containerization   Docker
Orchestration      Docker Compose

Running the Project

From the project directory:

docker compose up -d

Check:

docker compose ps

Health check:

curl http://127.0.0.1:8000/health

Add a document:

curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"url":"https://en.wikipedia.org/wiki/Machine_learning","title":"Machine Learning"}'

List documents:

curl http://127.0.0.1:8000/documents

Ask a question:

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is machine learning?","top_k":3,"min_score":0.0}'

Run tests:

docker compose exec api python -m pytest -q

Run evaluation:

docker compose exec api python evaluation/evaluate.py

Stop:

docker compose down

Final Objective

The end goal is not merely to have an application that can answer
questions.

The goal is to build a system that demonstrates the engineering required
to make AI applications:

Grounded

Testable

Observable

Maintainable

Scalable

Secure

Deployable

The current implementation is the foundation.

The roadmap is to progressively transform that foundation into a
production-style AI knowledge platform.