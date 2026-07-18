import os
import json
import time
import asyncio
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from src.config.settings import settings
from src.core.schema import SourceNode
from src.services.logger import logger

class VectorRetriever:
    def __init__(self):
        self._index = None
        self._lock = asyncio.Lock()

    async def search(self, query: str, top_k: int = None, prefer_types: list = None) -> list[SourceNode]:
        t_start = time.monotonic()
        await self._ensure_ready()

        k = top_k or settings.similarity_top_k
        # Fetch a much wider candidate pool than we need so that after
        # deduplicating to one chunk per source document, we still have
        # k *distinct documents* left -- not just k chunks that might all
        # belong to the same 1-2 highly-matching documents.
        fetch_k = max(k * 3, k + 15) if prefer_types else max(k * 6, 30)

        retriever = self._index.as_retriever(similarity_top_k=fetch_k)
        raw_nodes = await retriever.aretrieve(query)

        if prefer_types:
            preferred = [n for n in raw_nodes if n.node.metadata.get("type", "").lower() in prefer_types]
            others = [n for n in raw_nodes if n.node.metadata.get("type", "").lower() not in prefer_types]
            raw_nodes = (preferred + others)

        # Deduplicate: keep only the best-scoring chunk per source document,
        # preserving rank order, so a document with many matching chunks
        # doesn't crowd out other relevant documents.
        seen_sources = set()
        deduped_nodes = []
        for n in raw_nodes:
            source_id = n.node.metadata.get("source_id")
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            deduped_nodes.append(n)

        results = [
            SourceNode(
                text=n.node.get_content(),
                score=round(n.score, 4) if n.score else None,
                source_id=n.node.metadata.get("source_id"),
                title=n.node.metadata.get("title") or "Unknown Record"
            )
            for n in deduped_nodes[:k]
        ]

        logger.info("vector_search_completed", extra={
            "query": query[:200],
            "fetch_k": fetch_k,
            "k": k,
            "raw_candidates": len(raw_nodes),
            "deduped_candidates": len(deduped_nodes),
            "returned": len(results),
            "top_score": results[0].score if results else None,
            "duration_s": round(time.monotonic() - t_start, 2),
        })

        return results

    def _load_documents(self):
        documents = []
        skipped = 0
        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in files:
                if not fname.endswith(".json") or "metadata_index" in fname: continue
                try:
                    with open(os.path.join(root, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        text = self._build_text(item)
                        title = None
                        for key in ["title", "name", "id", "canonical_title"]:
                            if item.get(key):
                                title = item[key]
                                break

                        doc = Document(
                            text=text,
                            metadata={"source_id": fname, "title": str(title or fname), "type": str(item.get("type", ""))}
                        )
                        doc.excluded_embed_metadata_keys = ["source_id", "title", "type"]
                        documents.append(doc)
                except Exception as e:
                    skipped += 1
                    logger.error("document_load_failed", extra={
                        "file": fname,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    })
        logger.info("document_load_completed", extra={
            "documents_loaded": len(documents),
            "files_skipped": skipped,
            "source_dir": settings.rag_ready_dir,
        })
        return documents

    async def _ensure_ready(self):
        if self._index:
            return

        async with self._lock:
            if self._index:
                return

            Settings.embed_model = OpenAIEmbedding(model_name=settings.embed_model)
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

            os.makedirs(settings.vector_db_dir, exist_ok=True)
            db = chromadb.PersistentClient(path=settings.vector_db_dir)
            collection = db.get_or_create_collection(settings.chroma_collection)

            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            existing_count = collection.count()
            if existing_count == 0:
                logger.info("vector_index_build_started", extra={"source_dir": settings.rag_ready_dir})
                t_start = time.monotonic()
                docs = await asyncio.to_thread(self._load_documents)
                self._index = await asyncio.to_thread(
                    VectorStoreIndex.from_documents,
                    docs,
                    storage_context=storage_context,
                    show_progress=True
                )
                logger.info("vector_index_build_completed", extra={
                    "documents_indexed": len(docs),
                    "chunk_count": collection.count(),
                    "duration_s": round(time.monotonic() - t_start, 2),
                })
            else:
                self._index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
                logger.info("vector_index_loaded_from_disk", extra={
                    "existing_chunk_count": existing_count,
                })

    @staticmethod
    def _build_text(data: dict) -> str:
        """Recursive flattener for indexing."""
        lines = []
        def walk(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ["embedding_chunks", "_source"]: continue
                    label = k.replace("_", " ").upper()
                    if isinstance(v, (dict, list)):
                        lines.append(f"{prefix}{label}:")
                        walk(v, prefix + "  ")
                    else:
                        lines.append(f"{prefix}{label}: {v}")
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)): walk(item, prefix + "- ")
                    else: lines.append(f"{prefix}- {item}")
        walk(data)
        return "\n".join(lines)