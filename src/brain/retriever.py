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
        print("Building FAISS index from data/rag_ready...")
        docs = load_docs(DATA_DIR)
        if not docs:
            raise RuntimeError("No documents found to index in data/rag_ready")

        embed_model = get_embedding_model()
        # get text list
        texts = [getattr(d, "text", str(d)) for d in docs]
        vectors = embed_model.embed_documents(texts)
        vecs = np.array(vectors).astype("float32")

        # Normalize vectors for cosine checks
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs_norm = vecs / norms

        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        faiss.write_index(index, FAISS_PATH)

        # Persist normalized vectors for later similarity checks
        np.save(os.path.join(VECTOR_DIR, "vectors.npy"), vecs_norm)

        # Save docstore mapping
        docstore = [{"id": i, "text": texts[i], "metadata": getattr(docs[i], "metadata", {})} for i in range(len(texts))]
        with open(DOCSTORE_PATH, "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False, indent=2)
        print("FAISS index and vector store built and persisted.")


def load_index():
    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss is not installed. Install faiss-cpu to use FAISS retriever.")
    if not os.path.exists(FAISS_PATH) or not os.path.exists(DOCSTORE_PATH):
        build_index_if_missing()
    index = faiss.read_index(FAISS_PATH)
    with open(DOCSTORE_PATH, "r", encoding="utf-8") as f:
        docstore = json.load(f)
    vectors_path = os.path.join(VECTOR_DIR, "vectors.npy")
    vectors = None
    if os.path.exists(vectors_path):
        vectors = np.load(vectors_path)
    return index, docstore, vectors


def retrieve(query, top_k: int = 5, similarity_threshold: float = 0.15):
    index, docstore, vectors = load_index()
    embed_model = get_embedding_model()
    qvec = np.array([embed_model.embed_query(query)]).astype("float32")

    # If normalized vectors are available, compute cosine similarities and filter
    if vectors is not None:
        qnorm = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        sims = (vectors @ qnorm.T).flatten()  # cosine similarities
        # get top_k by similarity
        order = np.argsort(-sims)[:top_k]
        results = []
        for idx in order:
            if idx < 0 or idx >= len(docstore):
                continue
            if sims[idx] < similarity_threshold:
                continue
            entry = docstore[idx].copy()
            entry['score'] = float(sims[idx])
            results.append(entry)
        return results

    # fallback to FAISS nearest neighbors if no vectors.npy
    D, I = index.search(qvec, top_k)
    results = []
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(docstore):
            continue
        entry = docstore[idx].copy()
        entry['score'] = float(D[0][i])
        results.append(entry)
    return results
