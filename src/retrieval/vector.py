
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

from src.config.settings import settings
from src.core.schema import SourceNode


class VectorRetriever:
   

    def __init__(self):
        self._index  = None   # LlamaIndex VectorStoreIndex
        self._engine = None   # LlamaIndex retriever engine

    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        prefer_types: list[str] | None = None,
    ) -> list[SourceNode]:
        
        self._ensure_ready()
        k = top_k or settings.similarity_top_k

       
        fetch_k = k * 3 if prefer_types else k
        raw_nodes = self._engine.retrieve(query)[:fetch_k]

        # Promote preferred doc types to the front
        if prefer_types:
            preferred = [
                n for n in raw_nodes
                if n.node.metadata.get("type", "").lower() in prefer_types
            ]
            others = [
                n for n in raw_nodes
                if n.node.metadata.get("type", "").lower() not in prefer_types
            ]
            raw_nodes = (preferred + others)[:k]
        else:
            raw_nodes = raw_nodes[:k]

        return [
            SourceNode(
                text=n.node.get_content(),
                score=round(n.score, 4) if n.score else None,
                source_id=n.node.metadata.get("id"),
                title=n.node.metadata.get("title") or "Unknown",
                metadata=n.node.metadata.get("raw_data", {})
            )
            for n in raw_nodes
        ]

    def get_index(self):
        
        self._ensure_ready()
        return self._index

    

    def _ensure_ready(self):
        if self._index is not None:
            return

        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext, Settings
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=settings.embed_model
        )

        os.makedirs(settings.vector_db_dir, exist_ok=True)
        db         = chromadb.PersistentClient(path=settings.vector_db_dir)
        collection = db.get_or_create_collection(settings.chroma_collection)
        vector_store    = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

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

        self._engine = self._index.as_retriever(
            similarity_top_k=settings.similarity_top_k
        )

    def _load_documents(self):
        from llama_index.core import Document
        documents = []

        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in sorted(files):
                if not fname.endswith(".json") or "metadata_index" in fname:
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)

                    
                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        if not isinstance(item, dict) or not item.get("title"):
                            continue
                        
                        text = self._build_text(item)
                        doc = Document(
                            text=text,
                            metadata={
                                "id":             item.get("id", ""),
                                "title":          item.get("title", ""),
                                "type":           item.get("type", "trek"),
                                "category":       item.get("category", ""),
                                "duration_days":  item.get("duration_days"),
                                "raw_data": item
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"  ⚠️  Skipping {fname}: {e}")
        return documents

    @staticmethod
    def _build_text(data: dict) -> str:
        lines = []
        
        # System keys that shouldn't be indexed directly as raw text
        ignore_keys = {"id", "embedding_chunks", "raw_data", "type"}

        def flatten(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k not in ignore_keys:
                        clean_key = k.replace("_", " ").upper()
                        flatten(v, prefix=f"{clean_key}: ")
            elif isinstance(obj, list):
                for item in obj:
                    flatten(item, prefix=f"{prefix}- ")
            else:
                if obj is not None and str(obj).strip():
                    lines.append(f"{prefix}{str(obj).strip()}")

        flatten(data)
        
        # Append pre-computed embedding chunks if they exist
        joined = "\n".join(lines)
        for chunk in data.get("embedding_chunks", []):
            if not isinstance(chunk, dict):
                continue
            chunk_text = chunk.get("text", "").strip()
            if chunk_text and chunk_text not in joined:
                lines.append(chunk_text)

        return "\n".join(lines)

    @staticmethod
    def _extract_title(text: str) -> str | None:
     
        for line in text.splitlines():
            if line.startswith("TITLE:"):
                return line.replace("TITLE:", "").strip()
        return None