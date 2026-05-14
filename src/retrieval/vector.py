
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
                title=self._extract_title(n.node.get_content()),
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
                            },
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"  ⚠️  Skipping {fname}: {e}")
        return documents

    @staticmethod
    def _build_text(data: dict) -> str:
        
        lines = []

        def add(label, value):
            if value:
                lines.append(f"{label}: {value}")

        add("TITLE",    data.get("title"))
        add("TYPE",     data.get("type"))
        add("CATEGORY", data.get("category"))
        add("REGION",   data.get("region"))

       
        duration_val = data.get("duration_days")
        if duration_val is not None:
            try:
                days   = int(duration_val)
                suffix = "day" if days == 1 else "days"
            except Exception:
                days, suffix = duration_val, "days"
            lines.append(f"DURATION: {days} {suffix}")

        add("DIFFICULTY",   data.get("difficulty"))
        add("MAX ALTITUDE", f"{data['max_altitude_m']} m" if data.get("max_altitude_m") else None)
        add("SUMMARY",      data.get("summary"))

       
        for section in data.get("content_sections", []):
            if not isinstance(section, dict):
                continue
            heading = section.get("heading", "")
            text    = section.get("text", "").strip()
            if heading and text:
                lines.append(f"{heading.upper()}:\n{text}")
            elif text:
                lines.append(text)

       
        raw_attrs = data.get("attributes")
        if isinstance(raw_attrs, dict):
            for k, v in raw_attrs.items():
                if k and v:
                    lines.append(f"{k}: {v}")
        elif isinstance(raw_attrs, list):
            for attr in raw_attrs:
                if not isinstance(attr, dict):
                    continue
                k = attr.get("label") or attr.get("key", "")
                v = attr.get("value", "")
                if k and v:
                    lines.append(f"{k}: {v}")

       
        highlights = data.get("highlights", [])
        if highlights:
            add("TRIP HIGHLIGHTS", ", ".join(h for h in highlights if h))

        itinerary = data.get("itinerary", [])
        if itinerary:
            itin_lines = [
                f"Day {d.get('day', '?')}: {d.get('title', '')} — {d.get('description', '')}"
                for d in itinerary
                if isinstance(d, dict) and (d.get("title") or d.get("description"))
            ]
            if itin_lines:
                lines.append("FULL ITINERARY:\n" + "\n".join(itin_lines))

        includes = data.get("includes", [])
        if includes:
            add("INCLUSIONS", ", ".join(i for i in includes if i))

        excludes = data.get("excludes", [])
        if excludes:
            add("EXCLUSIONS", ", ".join(e for e in excludes if e))

       
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