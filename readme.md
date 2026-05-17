# 🏔️ Sherpa Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot designed for Himalayan travel, trekking, and tourism datasets.

This project combines:

* Semantic vector retrieval
* Metadata filtering
* Query understanding and normalization
* Hybrid search strategies
* Groq-powered LLM generation
* LlamaIndex orchestration
* ChromaDB vector storage

The architecture is designed to be extensible, modular, and production-oriented.

---

# 📌 Project Goals

The main objective of this project is to create a domain-specific intelligent assistant capable of:

* Understanding tourism and trekking-related queries
* Retrieving structured and unstructured information
* Performing metadata filtering and semantic search
* Generating contextual answers using LLMs
* Supporting scalable RAG pipelines
* Separating responsibilities using clean modular architecture

Example queries:

* “Best Annapurna trek under 10 days”
* “Compare Everest Base Camp and Langtang trek”
* “Shortest trek in Nepal”
* “Luxury trekking packages in Mustang”
* “Altitude information for Island Peak”

---

# 🧠 Core Architecture

The system follows a layered modular RAG architecture.

```text
┌─────────────────────────────┐
│        User Query           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Understanding Layer    │
│  - Normalization            │
│  - Classification           │
│  - Entity Resolution        │
│  - Query Enhancement        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Routing Layer         │
│  - Metadata Strategy        │
│  - Vector Strategy          │
│  - Hybrid Strategy          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Retrieval Layer        │
│  - Metadata Retriever       │
│  - Vector Retriever         │
│  - Hybrid Retriever         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Prompt Builder         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Generation Layer       │
│  - Groq LLM                 │
│  - Response Generation      │
│  - Token Counting           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Final Response        │
└─────────────────────────────┘
```

---

# 📂 Project Structure

```text
Sherpa-Chatbot/
│
├── data/
│   ├── cache/
│   ├── datasets/
│   ├── indexes/
│   └── logs/
│
├── scripts/
│   ├── indexing/
│   ├── ingestion/
│   ├── setup/
│   ├── testing/
│   └── verification/
│
├── src/
│   ├── api/
│   ├── config/
│   │   └── settings.py
│   │
│   ├── core/
│   │   ├── pipeline.py
│   │   ├── router.py
│   │   └── schema.py
│   │
│   ├── generation/
│   │   ├── generator.py
│   │   ├── prompt_builder.py
│   │   └── safe_summarizer.py
│   │
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   ├── metadata.py
│   │   └── vector.py
│   │
│   ├── services/
│   │
│   ├── understanding/
│   │   ├── classifier.py
│   │   ├── entity_resolver.py
│   │   ├── normalization.py
│   │   ├── normalizer.py
│   │   └── separator.py
│   │
│   └── chat.py
│
├── .env
├── .gitignore
├── requirements.txt
└── readme.md
```

---

# 🧩 Module Responsibilities

## 1. Understanding Layer

Location:

```text
src/understanding/
```

Purpose:

Transforms raw user input into structured execution plans.

### Components

| File               | Responsibility                              |
| ------------------ | ------------------------------------------- |
| classifier.py      | Query intent detection + execution planning |
| normalizer.py      | Query normalization                         |
| normalization.py   | Canonical normalization helpers             |
| entity_resolver.py | Entity extraction and matching              |
| separator.py       | Text splitting and chunk preparation        |

### Example

Input:

```text
best short trek near annapurna
```

Enhanced Query:

```text
Find the best trekking packages in the Annapurna region with minimum duration and high popularity.
```

The classifier converts vague user queries into semantically rich retrieval instructions.

---

# 🔀 Routing Layer

Location:

```text
src/core/router.py
```

Purpose:

Determines which retrieval strategy should be used.

### Supported Strategies

| Strategy | Use Case                                |
| -------- | --------------------------------------- |
| VECTOR   | Semantic similarity search              |
| METADATA | Exact filtering and sorting             |
| HYBRID   | Combination of metadata + vector search |
| FULL_DOC | Large summarization workflows           |

### Intent Mapping

| Intent    | Retrieval Mode |
| --------- | -------------- |
| Compare   | Metadata       |
| Filter    | Hybrid         |
| Aggregate | Metadata       |
| Summarize | Full Document  |
| General   | Vector         |

---

# 🔎 Retrieval Layer

Location:

```text
src/retrieval/
```

This layer is responsible for retrieving context data.

## Metadata Retrieval

Uses structured dataset filtering.

Supports:

* Sorting
* Aggregation
* Exact matching
* Field filtering

Example:

```python
region = "Everest"
duration_days < 10
```

---

## Vector Retrieval

Uses:

* ChromaDB
* Sentence Transformers
* Embedding similarity search

Purpose:

Retrieve semantically similar chunks.

Example:

```text
"Luxury trekking experience with helicopter return"
```

Can retrieve related packages even when wording differs.

---

## Hybrid Retrieval

Combines:

* Metadata filtering
* Semantic reranking

Flow:

```text
Metadata Filter
      ↓
Candidate Documents
      ↓
Vector Reranking
      ↓
Top Relevant Results
```

This is the most powerful retrieval mode in the system.

---

# 🤖 Generation Layer

Location:

```text
src/generation/
```

Responsible for:

* Prompt construction
* LLM communication
* Token counting
* Cost estimation
* Fallback handling

## Generator

File:

```text
generator.py
```

Uses:

* Groq API
* LlamaIndex LLM wrapper
* Tiktoken

### Features

* Token estimation
* Cost calculation
* Automatic fallback handling
* Reusable singleton-like initialization

### Current Model

```text
llama-3.3-70b-versatile
```

---

# 🧱 Pipeline Flow

Main pipeline file:

```text
src/core/pipeline.py
```

Execution Flow:

```text
1. Receive user query
2. Normalize query
3. Classify intent
4. Build execution plan
5. Select retrieval strategy
6. Retrieve documents
7. Build prompt
8. Generate answer
9. Return response with metadata
```

The pipeline is intentionally modular to support:

* Future APIs
* Agentic workflows
* Streaming responses
* Multi-LLM support
* MCP integration
* Tool calling

---

# 🗃️ Data Architecture

## Dataset Storage

Location:

```text
data/datasets/
```

Recommended format:

```json
{
  "id": "everest-base-camp",
  "title": "Everest Base Camp Trek",
  "region": "Everest",
  "duration_days": 14,
  "difficulty": "Hard",
  "altitude": 5364,
  "description": "Classic Himalayan trekking experience"
}
```

---

# 🧠 Embedding Architecture

Current embedding provider:

```text
sentence-transformers
```

Recommended model:

```text
BAAI/bge-small-en-v1.5
```

Alternative production models:

| Model            | Quality     | Speed     |
| ---------------- | ----------- | --------- |
| bge-small        | Good        | Fast      |
| bge-base         | Better      | Medium    |
| bge-large        | Best        | Slow      |
| all-MiniLM-L6-v2 | Lightweight | Very Fast |

---

# 🧮 Query Classification Design

The classifier creates structured execution plans.

Example output:

```json
{
  "intent": "filter",
  "filters": {
    "region": "Everest"
  },
  "sort_field": "duration_days",
  "sort_direction": "asc",
  "limit": 5,
  "needs_semantic_search": true,
  "search_query": "Best short Everest region treks with scenic mountain views"
}
```

This enables advanced RAG orchestration beyond simple vector search.

---

# ⚙️ Environment Setup

## 1. Clone Repository

```bash
git clone <repository-url>
cd Sherpa-Chatbot
```

---

## 2. Create Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create:

```text
.env
```

Example:

```env
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile
COMPANY_EMAIL=info@sherpahimalayas.com
```

---

# ▶️ Running the Chatbot

```bash
python -m src.chat
```

Or:

```bash
python src/chat.py
```

---

# 🚀 Running API Layer

If FastAPI endpoints are added:

```bash
uvicorn src.api.main:app --reload
```

Recommended production command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker src.api.main:app
```

---

# 📦 Updated requirements.txt

The current project imports additional libraries that should also exist in requirements.txt.

Recommended updated file:

```text
# =====================================================
# Core AI / RAG
# =====================================================
llama-index-core==0.11.23
llama-index-llms-groq==0.2.0
llama-index-embeddings-huggingface==0.3.0
llama-index-vector-stores-chroma==0.2.0
llama-index-readers-file==0.2.0

# =====================================================
# LLM Providers
# =====================================================
groq==0.11.0

# =====================================================
# Embeddings / Vector DB
# =====================================================
sentence-transformers==3.0.1
chromadb==0.5.5
transformers==4.44.2
torch==2.4.1

# =====================================================
# Tokenization
# =====================================================
tiktoken==0.7.0

# =====================================================
# API / Backend
# =====================================================
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2

# =====================================================
# Environment / Utilities
# =====================================================
python-dotenv==1.0.1
pandas==2.2.2
numpy==2.1.1

# =====================================================
# Scraping / Parsing
# =====================================================
requests==2.32.3
beautifulsoup4==4.12.3
trafilatura==1.12.0
lxml==5.3.0

# =====================================================
# Optional Production Utilities
# =====================================================
gunicorn==23.0.0
httpx==0.27.2
```

---

# 🛠️ Future Improvements

## 1. Add Structured Logging

Recommended:

```python
import logging
```

Use:

* Rotating file handlers
* JSON logs
* Request tracing
* Retrieval diagnostics

---

## 2. Add Streaming Responses

Recommended:

* FastAPI StreamingResponse
* Async generators
* Token streaming

---

## 3. Add Evaluation Framework

Possible metrics:

| Metric       | Purpose                 |
| ------------ | ----------------------- |
| Recall@K     | Retrieval quality       |
| MRR          | Ranking quality         |
| Faithfulness | Hallucination detection |
| Latency      | Performance             |
| Cost         | Token optimization      |

---

## 4. Add Caching

Recommended layers:

* Query cache
* Embedding cache
* Retrieval cache
* LLM response cache

Suggested tools:

* Redis
* DiskCache
* SQLite cache

---




---

# 📈 Scalability Roadmap

## Current Architecture

Suitable for:

* Local development
* Small production workloads
* Portfolio projects
* Research prototypes

---



# 📚 Technologies Used

| Technology            | Purpose           |
| --------------------- | ----------------- |
| Python                | Core language     |
| Groq                  | LLM provider      |
| LlamaIndex            | RAG orchestration |
| ChromaDB              | Vector storage    |
| Sentence Transformers | Embeddings        |
| FastAPI               | API framework     |
| Tiktoken              | Token counting    |
| Pandas                | Data processing   |

---

#