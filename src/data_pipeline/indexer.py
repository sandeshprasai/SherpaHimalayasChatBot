import chromadb
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

def get_vector_index(documents, persist_dir):
    # Ensure the directory exists
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize ChromaDB
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection("sherpa_knowledge")
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # If the collection is empty, index the documents
    if chroma_collection.count() == 0:
        print("📥 Indexing new documents into ChromaDB...")
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            show_progress=True
        )
    else:
        print("⚡ Loading existing index from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context
        )
    return index