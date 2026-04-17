import os
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Setup paths relative to this script's location
base_path = os.path.dirname(os.path.abspath(__file__)) # src/brain
project_root = os.path.abspath(os.path.join(base_path, "../../"))
DATA_DIR = os.path.join(project_root, "data/processed")
INDEX_DIR = os.path.join(project_root, "data/vector_db")
env_path = os.path.join(project_root, ".env")

load_dotenv(env_path)

def get_chat_engine():
    # Get current date for seasonal awareness
    current_date = datetime.now().strftime("%B %d, %Y")

    # 1. Setup Groq (Llama 3.3 70B is highly recommended for reasoning)
    llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    # 2. Local Embedding (Search logic)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 3. Handle Index Persistence
    index_exists = os.path.exists(INDEX_DIR) and os.path.isfile(os.path.join(INDEX_DIR, "docstore.json"))

    if not index_exists:
        print(f"🔍 Index not found. Creating new index from {DATA_DIR}...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        print("✅ Index created and saved.")
    else:
        print("⚡ Loading existing high-speed index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)

    # 4. Create Chat Engine with Strict System Prompt
    return index.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            f"Today's date is {current_date}. "
            "You are the official Sherpa Himalaya Travel Assistant. "
            "### STRICT OPERATIONAL RULES: "
            "1. ONLY answer questions using the provided context from the company's website data. "
            "2. Identify the current season based on today's date and suggest appropriate treks from the data. "
            "3. If the user's question is NOT about trekking, Nepal, or the provided data, politely refuse to answer. "
            "4. NEVER provide recipes, general life advice, or non-trekking information. "
            "5. If the answer is not found in the context, say: 'Namaste. I'm sorry, but I don't have information on that specific topic in our records. Please contact our team at info@sherpahimalaya.com for more details.' "
            "6. Do not explain that you are an AI. Just be the expert assistant."
        )
    )

if __name__ == "__main__":
    try:
        engine = get_chat_engine()
        print(f"\n🏔️ Sherpa Himalaya AI is ready! (System Date: {datetime.now().strftime('%Y-%m-%d')})")
        print("Type 'exit' to stop.")
        while True:
            q = input("\nYou: ")
            if q.lower() in ["exit", "quit"]: break
            response = engine.chat(q)
            print(f"\nAI: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")