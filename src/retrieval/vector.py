import os
import json
import asyncio  # Added
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from src.config.settings import settings
from src.core.schema import SourceNode

class VectorRetriever:
    def __init__(self):
        self._index = None
        # --- ADDED LOCK FOR THREAD-SAFETY ---
        self._lock = asyncio.Lock()

    async def search(self, query: str, top_k: int = None, prefer_types: list = None) -> list[SourceNode]:
        # Await the initialization safely
        await self._ensure_ready()
        
        k = top_k or settings.similarity_top_k
        fetch_k = k * 3 if prefer_types else k
        
        # Fresh retriever per-call to ensure correct top_k
        retriever = self._index.as_retriever(similarity_top_k=fetch_k)
        raw_nodes = await retriever.aretrieve(query)

        # Logic for type-based promotion (Agnostic)
        if prefer_types:
            preferred = [n for n in raw_nodes if n.node.metadata.get("type", "").lower() in prefer_types]
            others = [n for n in raw_nodes if n.node.metadata.get("type", "").lower() not in prefer_types]
            raw_nodes = (preferred + others)

        return [
            SourceNode(
                text=n.node.get_content(),
                score=round(n.score, 4) if n.score else None,
                source_id=n.node.metadata.get("source_id"),
                title=n.node.metadata.get("title") or "Unknown Record"
            )
            for n in raw_nodes[:k]
        ]

    def _load_documents(self):
        documents = []
        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in files:
                if not fname.endswith(".json") or "metadata_index" in fname: continue
                try:
                    with open(os.path.join(root, fname), encoding="utf-8") as f:
                        data = json.load(f)
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        text = self._build_text(item)
                        # Generic Title Discovery
                        title = None
                        for key in ["title", "name", "id", "canonical_title"]:
                            if item.get(key):
                                title = item[key]
                                break
                        
                        doc = Document(
                            text=text, 
                            metadata={"source_id": fname, "title": str(title or fname), "type": str(item.get("type", ""))}
                        )
                        # Only exclude from the vector itself, keep metadata visible to generating LLM
                        doc.excluded_embed_metadata_keys = ["source_id", "title", "type"]
                        documents.append(doc)
                except Exception: continue
        return documents

    # --- UPDATED TO BE ASYNC AND THREAD-SAFE ---
    async def _ensure_ready(self):
        if self._index: 
            return
            
        async with self._lock:
            # Double-check pattern: another request might have finished initializing 
            # while this request was waiting for the lock.
            if self._index: 
                return
                
            Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
            # --- FIX: Changed chunk_overlap from 50 to 200 to prevent semantic context splitting ---
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
            
            os.makedirs(settings.vector_db_dir, exist_ok=True)
            db = chromadb.PersistentClient(path=settings.vector_db_dir)
            collection = db.get_or_create_collection(settings.chroma_collection)
            
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            if collection.count() == 0:
                # Wrap document loading & indexing in a separate thread so it doesn't freeze the event loop
                docs = await asyncio.to_thread(self._load_documents)
                self._index = await asyncio.to_thread(
                    VectorStoreIndex.from_documents, 
                    docs, 
                    storage_context=storage_context, 
                    show_progress=True
                )
            else:
                self._index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

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