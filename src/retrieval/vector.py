
import os
import sys
import json
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

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

    
    async def search(
        self,
        query: str,
        top_k: int | None = None,
        prefer_types: list[str] | None = None,
    ) -> list[SourceNode]:
        
        self._ensure_ready()
        k = top_k or settings.similarity_top_k

       
        fetch_k = k * 3 if prefer_types else k
        raw_nodes = await self._engine.aretrieve(query)
        raw_nodes = raw_nodes[:k]

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
                source_id=n.node.metadata.get("source_id"),
                title=n.node.metadata.get("title") or "Unknown"
            )
            for n in raw_nodes
        ]
    def get_index(self):
        
        self._ensure_ready()
        return self._index

    

    def _ensure_ready(self):
        if self._index is not None:
            return
        Settings.embed_model = HuggingFaceEmbedding(model_name=settings.embed_model)
        Settings.node_parser = SentenceSplitter(
            chunk_size=1024, 
            chunk_overlap=50,
            include_metadata=False 
        )

       
        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host:
            db = chromadb.HttpClient(host=chroma_host, port=8000)
        else:            
            os.makedirs(settings.vector_db_dir, exist_ok=True)
            db = chromadb.PersistentClient(path=settings.vector_db_dir)
            print(f" Using local ChromaDB at {settings.vector_db_dir}")
        
        collection = db.get_or_create_collection(settings.chroma_collection)
        
       
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        
        if collection.count() == 0:
            print(f"🔍 Vector store is empty. Scanning {settings.rag_ready_dir}...")
            docs = self._load_documents()
            
            if not docs:
                print(" No valid documents found to index.")
               
                self._index = VectorStoreIndex.from_documents(
                    [], storage_context=storage_context
                )
            else:
                print(f" Generating embeddings for {len(docs)} documents...")
               
                self._index = VectorStoreIndex.from_documents(
                    docs,
                    storage_context=storage_context,
                    show_progress=True,
                    insert_batch_size=100 
                )
                print(" Vector Index built and saved.")
        else:
            
            print(f"⚡ Loading existing Vector Index ({collection.count()} nodes)")
            self._index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
            )

        self._engine = self._index.as_retriever(
            similarity_top_k=settings.similarity_top_k
        )
    def _load_documents(self):
        from llama_index.core import Document
        import json
        documents = []

        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in files:
                if not fname.endswith(".json") or "metadata_index" in fname:
                    continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)

                    items = data if isinstance(data, list) else [data]

                    for item in items:
                        # Fully agnostic text builder
                        text = self._build_text(item)
                        if not text: continue

                        # Smart title extraction
                        meta = item.get("meta_tags", {})
                        title = meta.get("title") or item.get("title") or item.get("name") or fname.replace(".json", "")

                        print(f"📄 Indexing: {title} ({len(text)} chars)")
                        
                        doc = Document(
                            text=text,
                            metadata={
                                "source_id": str(fname),
                                "title": str(title)
                            }
                        )
                        # Ensure metadata is never embedded to avoid length errors
                        doc.excluded_embed_metadata_keys = ["source_id", "title"]
                        doc.excluded_llm_metadata_keys = ["source_id", "title"]
                        
                        documents.append(doc)
                except Exception as e:
                    print(f"  ⚠️ Skipping {fname}: {e}")
        return documents

    @staticmethod
    def _build_text(data: dict) -> str:
        """
        100% Agnostic: Converts any JSON dictionary into readable text.
        It doesn't care if it's a Trek, a Laptop, or a Yoga Course.
        """
        lines = []
        
        def extract_values(d: dict, prefix=""):
            for key, val in d.items():
                # Skip massive system fields if they exist
                if key in ["embedding_chunks", "embedding", "_source"]:
                    continue
                
                clean_key = key.replace("_", " ").upper()
                
                if isinstance(val, (str, int, float, bool)):
                    lines.append(f"{prefix}{clean_key}: {val}")
                elif isinstance(val, list):
                    # Handle lists (like tags, highlights, or itineraries)
                    valid_items = [str(i) for i in val if isinstance(i, (str, int, float))]
                    if valid_items:
                        lines.append(f"{prefix}{clean_key}: {', '.join(valid_items)}")
                    elif val and isinstance(val[0], dict):
                        # Handle nested lists of dictionaries (like itineraries)
                        lines.append(f"{prefix}{clean_key}:")
                        for item in val[:20]: # Cap at 20 items to prevent overflow
                            extract_values(item, prefix="  - ")
                elif isinstance(val, dict):
                    # Handle nested dictionaries
                    lines.append(f"{prefix}{clean_key}:")
                    extract_values(val, prefix="  - ")

        extract_values(data)
        return "\n".join(lines)

    @staticmethod
    def _extract_title(text: str) -> str | None:
     
        for line in text.splitlines():
            if line.startswith("TITLE:"):
                return line.replace("TITLE:", "").strip()
        return None