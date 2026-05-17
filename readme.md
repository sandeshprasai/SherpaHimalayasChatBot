# 🤖 RAG Chatbot

A modular Retrieval-Augmented Generation (RAG) chatbot that works with any domain-specific structured data. Swap in your own dataset and branding — no code changes required.

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

The main objective of this project is to create a domain-agnostic intelligent assistant capable of:

* Understanding natural language queries about any structured dataset
* Retrieving structured and unstructured information
* Performing metadata filtering and semantic search
* Generating contextual answers using LLMs
* Supporting scalable RAG pipelines
* Separating responsibilities using clean modular architecture

The bot works out of the box for any domain — e-commerce, education, travel, healthcare, real estate, hospitality, and more. Example queries depend entirely on what data you feed it:

* "Show me options under $500"
* "Compare the beginner and advanced plans"
* "What is included in the premium package?"
* "Summarize your refund policy"
* "Who is the team lead for product?"

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
RAG-Chatbot/
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
cheapest beginner option under $300
```

Enhanced Query:

```text
Find the most affordable beginner-level options with a price below $300, sorted by price ascending.
```

The classifier converts vague user queries into semantically rich retrieval instructions, regardless of domain.

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
category = "Electronics"
price < 500
```

---

## Vector Retrieval

Uses:

* ChromaDB
* Sentence Transformers
* Embedding similarity search

Purpose:

Retrieve semantically similar chunks even when the user's wording differs from the source content.

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
data/datasets/rag_ready/
```

Every JSON document in `rag_ready` is the authoritative source for what the chatbot knows. The structure is flexible — any domain-specific fields in `meta_tags` are automatically discovered and made available for filtering and sorting.

Example document (e-commerce domain):

```json
{
  "source_url": "https://yourstore.com/products/wireless-headphones",
  "filename": "products-wireless-headphones.md",
  "category": "Products",
  "meta_tags": {
    "summary": "Over-ear wireless headphones with 30-hour battery life.",
    "keywords": ["headphones", "wireless", "audio"],
    "price": 149,
    "brand": "SoundWave",
    "in_stock": true
  },
  "content": "Full product description goes here..."
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
    "category": "Electronics"
  },
  "sort_field": "price",
  "sort_direction": "asc",
  "limit": 5,
  "needs_semantic_search": true,
  "search_query": "Affordable electronics with good reviews under $200"
}
```

This enables advanced RAG orchestration beyond simple vector search — the same logic works for any domain.

---

# 🌐 Switching Domains

This chatbot is fully domain-agnostic. To point it at a completely different business, product line, or subject area, you only need to do three things.

## 1. Update Branding in `.env`

The bot's identity and tone are controlled by three variables in your `.env` file:

```env
COMPANY_NAME=Your Company Name
SYSTEM_NAME=Your Bot Name
COMPANY_EMAIL=support@yourcompany.com
```

These values flow into the prompt builder and API responses automatically. No code changes are needed.

## 2. Swap the Dataset

Replace the contents of `data/datasets/rag_ready/` with JSON files for your new domain. Organise them into subfolders by category — the folder name becomes the document's `category` value used for filtering.

```text
data/datasets/rag_ready/
├── Products/
├── Policies/
├── Blog/
├── Team/
└── FAQs/
```

Any `meta_tags` fields you include are automatically indexed. There is no schema to register or configure — the index builder discovers all fields from the data itself.

## 3. Rebuild the Indexes

```bash
python scripts/indexing/index_builder.py
python scripts/indexing/build_vector_index.py
```

That's it. The bot is now answering questions about your new domain with no other changes to the codebase.

---

# 📥 Adding Data to `datasets/rag_ready`

The `data/datasets/rag_ready/` directory is the single source of truth for everything the chatbot can answer. Both the metadata index and the vector store are built from these JSON files.

---

## Overview of the Data Pipeline

Raw content must pass through several stages before it reaches `rag_ready`:

```text
Raw HTML / Website / CMS Export
      ↓
  Scraping / Export
      ↓
  data/separated/         ← raw Markdown files, one per page
      ↓
  scripts/ingestion/cleaner.py
      ↓
  data/cleaned_separated/ ← cleaned Markdown files
      ↓
  scripts/ingestion/universal_converter.py   (uses Groq LLM)
      ↓
  data/datasets/rag_ready/<Category>/        ← structured JSON ✅
      ↓
  scripts/indexing/index_builder.py
      ↓
  data/indexes/metadata/metadata_index.json
      ↓
  scripts/indexing/build_vector_index.py
      ↓
  data/indexes/chroma/vector_db/
```

---

## Step 1 — Prepare Raw Content

Place one Markdown (`.md`) file per page inside `data/separated/`, organised into subfolders by content type:

```text
data/separated/
├── blogs/
├── company/
├── information/
├── team/
└── insights/
```

Each file should contain the raw scraped or exported text of a single page. No special formatting is required at this stage.

---

## Step 2 — Clean the Raw Files

Run the cleaner to strip HTML tags, navigation noise, duplicate lines, and junk blocks:

```bash
python scripts/ingestion/cleaner.py
```

Output is written to `data/cleaned_separated/` with the same folder structure. The original files in `data/separated/` are not modified.

---

## Step 3 — Convert to Structured JSON (LLM-powered)

Run the universal converter. This sends each cleaned Markdown file to Groq and returns a structured JSON document:

```bash
python scripts/ingestion/universal_converter.py
```

Output files are saved to `data/datasets/rag_ready/<folder>/` as `.json`.

The converter automatically skips files that already have a corresponding output file, so it is safe to re-run after adding new source files.

### What the JSON looks like

Every document in `rag_ready` follows this schema:

```json
{
  "id": "lowercase_slug_with_underscores",
  "type": "blog|team|company|info",
  "title": "Page or person title",
  "summary": "1–2 sentence overview",
  "attributes": {
    "key": "value"
  },
  "content_sections": [
    { "heading": "Section Title", "text": "Cleaned paragraph text" }
  ],
  "tags": ["keyword1", "keyword2", "keyword3"],
  "embedding_chunks": [
    { "type": "main_content", "text": "Most important content for semantic search." }
  ]
}
```

---

## Step 4 — Add Files Directly (Manual / Custom Data)

If you already have structured data (e.g. exported from a CMS, spreadsheet, or API), you can skip Steps 1–3 and place JSON files directly into the correct category folder.

**Rules for manual files:**

- Each file must be a valid `.json` file.
- The filename should be a URL-style slug, e.g. `products-my-new-item.json`.
- The file must contain at least a `content` field (plain text or Markdown) for the vector index to embed it.
- Structured `meta_tags` fields are used by the metadata retriever for filtering and sorting — include as many as are applicable to your domain.
- Place the file in the folder that best matches its category. The folder name becomes the document's `category` value.

### Minimal valid example

```json
{
  "source_url": "https://yoursite.com/my-page",
  "filename": "my-page.md",
  "category": "Products",
  "meta_tags": {
    "summary": "A short description of this item.",
    "keywords": ["tag1", "tag2"],
    "price": 199,
    "duration": "6 weeks"
  },
  "content": "Full text content of the page goes here. This is what gets embedded and returned to the LLM."
}
```

---

## Step 5 — Rebuild the Metadata Index

After adding or updating any files in `rag_ready`, rebuild the metadata index:

```bash
python scripts/indexing/index_builder.py
```

Output: `data/indexes/metadata/metadata_index.json`

This index powers the metadata retriever (exact filtering, sorting, aggregation). It is built automatically from whatever JSON files exist in `rag_ready` — no configuration changes needed when you add new categories or fields.

---

## Step 6 — Rebuild the Vector Index

Rebuild the ChromaDB vector store so new documents are searchable by semantic similarity:

```bash
python scripts/indexing/build_vector_index.py
```

Output: `data/indexes/chroma/vector_db/`

> ⚠️ This step re-embeds all documents and may take several minutes depending on dataset size and hardware. The embedding model (`BAAI/bge-small-en-v1.5`) runs locally via `sentence-transformers`.

---

## Step 7 — Verify

Start the chatbot and ask a question about the newly added content:

```bash
python -m src.chat
```

Or test via the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about [your new content]"}'
```

---

## Quick Reference

| Task | Command |
| --- | --- |
| Clean raw Markdown files | `python scripts/ingestion/cleaner.py` |
| Convert cleaned files to JSON via LLM | `python scripts/ingestion/universal_converter.py` |
| Rebuild metadata index | `python scripts/indexing/index_builder.py` |
| Rebuild vector index | `python scripts/indexing/build_vector_index.py` |
| Run chatbot | `python -m src.chat` |
| Run API | `uvicorn src.api.main:app --reload` |

---

## Troubleshooting

**Groq rate limit during conversion**

The converter sleeps 60 seconds automatically when it hits a rate limit, then retries. For large datasets, run conversion in batches by adjusting `target_folders` inside `universal_converter.py`.

**New documents not appearing in search results**

Always run both `index_builder.py` and `build_vector_index.py` after adding files. Updating only one index will cause the two retrievers to go out of sync.

**Short or stub files being skipped by the index builder**

The index builder ignores documents whose `content` field is shorter than 500 characters (configurable via the `MIN_CONTENT_LEN` environment variable). Ensure your JSON files contain meaningful content.

**Wrong category assigned**

The `category` value in each document comes from the subfolder it lives in inside `rag_ready`. Moving a file to the correct subfolder and rebuilding the indexes is all that is needed to fix it.

---

# ⚙️ Environment Setup

## 1. Clone Repository

```bash
git clone <repository-url>
cd RAG-Chatbot
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

Create a `.env` file in the project root. All branding, model, and path configuration lives here.

```env
# ── App ──────────────────────────────────────────
ENVIRONMENT=development
DEBUG=true

# ── Server ───────────────────────────────────────
HOST=0.0.0.0
PORT=8000

# ── Security ─────────────────────────────────────
API_KEY=your_api_key

# ── LLM ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1024

# ── Embeddings ────────────────────────────────────
EMBED_MODEL=BAAI/bge-small-en-v1.5

# ── Vector DB ─────────────────────────────────────
CHROMA_DB_PATH=data/indexes/chroma/vector_db
CHROMA_COLLECTION=rag_collection

# ── Metadata Index ────────────────────────────────
METADATA_INDEX_PATH=data/indexes/metadata/metadata_index.json

# ── Retrieval ─────────────────────────────────────
SIMILARITY_TOP_K=5

# ── Branding (change these for your domain) ───────
COMPANY_NAME=Your Company Name
SYSTEM_NAME=Your Bot Name
COMPANY_EMAIL=support@yourcompany.com

# ── Logging ───────────────────────────────────────
LOG_LEVEL=INFO

# ── CORS ──────────────────────────────────────────
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# ── Rate Limiting ─────────────────────────────────
DAILY_TOKEN_LIMIT=100000
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

```bash
uvicorn src.api.main:app --reload
```

Recommended production command:

```bash
gunicorn -k uvicorn.workers.UvicornWorker src.api.main:app
```

---

# 📦 Requirements

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