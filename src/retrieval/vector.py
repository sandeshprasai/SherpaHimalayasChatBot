"""
retrieval/vector.py
────────────────────
Vector retrieval layer. Wraps ChromaDB + LlamaIndex.

Responsibilities:
  - Load and index documents from rag_ready/ (once, then cached)
  - Run semantic similarity search for a query string
  - Return a list of SourceNode objects

Usage:
    from retrieval.vector import VectorRetriever
    retriever = VectorRetriever()
    nodes = retriever.search("What is included in Ghorepani Circuit Trek?")
"""

import os
import sys
import json

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import SourceNode


class VectorRetriever:
    """
    Lazy-initialised vector retriever.
    First call to search() builds/loads the ChromaDB index.
    """

    def __init__(self):
        self._index = None      # LlamaIndex VectorStoreIndex
        self._engine = None     # LlamaIndex retriever engine

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> list[SourceNode]:
        """
        Run semantic similarity search.
        Returns up to top_k SourceNode objects ordered by relevance.
        """
        self._ensure_ready()
        k = top_k or settings.similarity_top_k

        raw_nodes = self._engine.retrieve(query)[:k]

        return [
            SourceNode(
                text=n.node.get_content(),
                score=round(n.score, 4) if n.score else None,
                source_id=n.node.metadata.get("id"),
                title=self._extract_title(n.node.get_content()),
            )
            for n in raw_nodes
        ]

    def get_index(self):
        """Expose the raw LlamaIndex index (used by chat engine if needed)."""
        self._ensure_ready()
        return self._index

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_ready(self):
        if self._index is not None:
            return

        # Import heavy deps only when needed
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext, Settings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # ── Embedding model ────────────────────────────────────────────────
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embed_model
        )

        # ── ChromaDB ───────────────────────────────────────────────────────
        os.makedirs(settings.vector_db_dir, exist_ok=True)
        db = chromadb.PersistentClient(path=settings.vector_db_dir)
        collection = db.get_or_create_collection(settings.chroma_collection)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        # ── Load or build index ────────────────────────────────────────────
        if collection.count() == 0:
            print("📥 Building new ChromaDB index...")
            docs = self._load_documents()
            self._index = VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                show_progress=True,
            )
        else:
            print(f"⚡ Loading existing index ({collection.count()} chunks)")
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )

        # ── Build retriever engine ─────────────────────────────────────────
        self._engine = self._index.as_retriever(
            similarity_top_k=settings.similarity_top_k
        )

    def _load_documents(self):
        """
        Build LlamaIndex Document objects from all rag_ready JSONs.
        Mirrors your existing loader.py logic, with additions:
          - richer metadata (title, category, region, duration, difficulty)
          - skips metadata_index.json and .bak files
        """
        from llama_index.core import Document

        documents = []
        skipped = []

        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in sorted(files):
                if not fname.endswith(".json"):
                    continue
                if fname in ("metadata_index.json",):
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)

                    # Handle list-rooted files gracefully
                    if isinstance(data, list):
                        data = data[0] if data and isinstance(data[0], dict) else None
                    if not isinstance(data, dict):
                        skipped.append(fname)
                        continue

                    text = self._build_text(data)
                    if not text.strip():
                        skipped.append(fname)
                        continue

                    doc = Document(
                        text=text,
                        metadata={
                            "id":           data.get("id", ""),
                            "title":        data.get("title", ""),
                            "category":     data.get("category", ""),
                            "region":       data.get("region", ""),
                            "duration_days":data.get("duration_days"),
                            "difficulty":   data.get("difficulty", ""),
                            "max_altitude_m":data.get("max_altitude_m"),
                        },
                    )
                    documents.append(doc)

                except Exception as e:
                    skipped.append(fname)
                    print(f"  ⚠️  Skipping {fname}: {e}")

        print(f"✅ Loaded {len(documents)} documents"
              + (f" ({len(skipped)} skipped)" if skipped else ""))
        return documents

    @staticmethod
    def _build_text(data: dict) -> str:
        """
        Build a rich, LLM-readable text block from a rag_ready JSON record.
        Keeps your existing loader.py structure and extends it.
        """
        lines = []

        def add(label: str, value):
            if value:
                lines.append(f"{label}: {value}")

        add("TITLE",      data.get("title"))
        add("CATEGORY",   data.get("category"))
        add("REGION",     data.get("region"))
        # Proper singular/plural for duration
        duration_val = data.get("duration_days")
        if duration_val is not None:
            try:
                days = int(duration_val)
            except Exception:
                days = duration_val
            if isinstance(days, int):
                dur_str = f"{days} day" if days == 1 else f"{days} days"
            else:
                dur_str = f"{days} days"
        else:
            dur_str = None
        add("DURATION",   dur_str)
        add("DIFFICULTY", data.get("difficulty"))
        add("MAX ALTITUDE", f"{data['max_altitude_m']} m" if data.get("max_altitude_m") else None)
        add("SUMMARY",    data.get("summary"))

        highlights = data.get("highlights", [])
        if highlights:
            add("TRIP HIGHLIGHTS", ", ".join(h for h in highlights if h))

        itinerary = data.get("itinerary", [])
        if itinerary:
            itin_lines = []
            for day in itinerary:
                if day.get("title") or day.get("description"):
                    itin_lines.append(
                        f"Day {day.get('day', '?')}: "
                        f"{day.get('title', '')} — {day.get('description', '')}"
                    )
            if itin_lines:
                lines.append("FULL ITINERARY:\n" + "\n".join(itin_lines))

        includes = data.get("includes", [])
        if includes:
            add("WHAT IS INCLUDED (INCLUSIONS)",
                ", ".join(i for i in includes if i))

        excludes = data.get("excludes", [])
        if excludes:
            add("WHAT IS EXCLUDED (EXCLUSIONS)",
                ", ".join(e for e in excludes if e))

        # embedding_chunks as supplementary searchable text
        for chunk in data.get("embedding_chunks", []):
            chunk_text = chunk.get("text", "").strip()
            if chunk_text and chunk_text not in "\n".join(lines):
                lines.append(chunk_text)

        return "\n".join(lines)

    @staticmethod
    def _extract_title(text: str) -> str | None:
        """Pull the TITLE line from a document text block."""
        for line in text.splitlines():
            if line.startswith("TITLE:"):
                return line.replace("TITLE:", "").strip()
        return None