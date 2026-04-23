import os
import json
import numpy as np
from dotenv import load_dotenv

from src.data_pipeline.loader_v2 import load_docs
from src.brain.embeddings import get_embedding_model

# Optional: use faiss when available
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
DATA_DIR = os.path.join(project_root, "data/rag_ready")
VECTOR_DIR = os.path.join(project_root, "data/vector_db")
FAISS_PATH = os.path.join(VECTOR_DIR, "faiss.index")
DOCSTORE_PATH = os.path.join(VECTOR_DIR, "docstore.json")

load_dotenv(os.path.join(project_root, ".env"))


def build_index_if_missing():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    if FAISS_AVAILABLE and not os.path.exists(FAISS_PATH):
        print("Building FAISS index...")
        docs = load_docs(DATA_DIR)
        if not docs:
            raise RuntimeError("No documents found to index in data/cleaned_separated")

        embed_model = get_embedding_model()
        # get text list
        texts = [getattr(d, "text", str(d)) for d in docs]
        vectors = embed_model.embed_documents(texts)
        vecs = np.array(vectors).astype("float32")

        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        faiss.write_index(index, FAISS_PATH)

        # Save docstore mapping
        docstore = [{"id": i, "text": texts[i], "metadata": getattr(docs[i], "metadata", {})} for i in range(len(texts))]
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False, indent=2)
        print("FAISS index built and persisted.")


def load_index():
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss is not installed. Install faiss-cpu to use FAISS retriever.")
    if not os.path.exists(FAISS_PATH) or not os.path.exists(DOCSTORE_PATH):
        build_index_if_missing()
    index = faiss.read_index(FAISS_PATH)
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    return index, docstore


def retrieve(query, top_k: int = 5):
    index, docstore = load_index()
    embed_model = get_embedding_model()
    qvec = np.array([embed_model.embed_query(query)]).astype("float32")
    D, I = index.search(qvec, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docstore):
            continue
        results.append(docstore[idx])
    return results
