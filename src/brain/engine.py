import os
import tiktoken
from datetime import datetime
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ==============================
# PATH SETUP
# ==============================
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))

DATA_DIR = os.path.join(project_root, "data/rag_ready")
INDEX_DIR = os.path.join(project_root, "data/vector_db")

load_dotenv(os.path.join(project_root, ".env"), override=True)
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is completely missing from environment variables!")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ==============================
# CONFIG (OPTIMIZED)
# ==============================
MODEL_NAME = "llama-3.3-70b-versatile"   # 🔥 more capable model for better reasoning and RAG extraction
DAILY_LIMIT = 100000                  # your plan limit
ESTIMATED_RESPONSE_TOKENS = 800       # rough buffer

# ==============================
# TOKEN COUNTER
# ==============================
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ==============================
# SYSTEM PROMPT
# ==============================
def get_system_prompt():
    current_date = datetime.now().strftime("%B %d, %Y")

    return (
        f"Today's date is {current_date}. "
        "You are the official Sherpa Himalaya Travel Assistant. "
        "ONLY answer using provided context. "
        "If answer is not in context, respond with fallback message."
    )


# ==============================
# LOAD INDEX (FIXED)
# ==============================
def load_index():
    # 🔥 MUST be set BEFORE loading index
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-en-v1.5"
    )

    Settings.llm = Groq(
        model=MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY")
    )

    if not os.path.exists(os.path.join(INDEX_DIR, "docstore.json")):
        from src.data_pipeline.loader import load_json_docs
        docs = load_json_docs(DATA_DIR)

        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)

    return index


# ==============================
# CHAT ENGINE
# ==============================
def get_chat_engine(index):
    # LLM is already set in Settings globally during initial setup
    return index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=8,   # 🔥 increased to 8 to gather enough context for full itineraries & comparison
        system_prompt=get_system_prompt(),
        response_mode="compact"
    )


# ==============================
# MAIN LOOP
# ==============================
if __name__ == "__main__":
    index = load_index()
    engine = get_chat_engine(index)

    print("\n🏔️ Sherpa Himalaya AI Ready (Optimized Mode)\n")

    used_tokens = 0   # track locally

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # ==============================
        # TOKEN BUDGET CHECK
        # ==============================
        estimated_tokens = count_tokens(user_input) + ESTIMATED_RESPONSE_TOKENS

        if used_tokens + estimated_tokens > DAILY_LIMIT:
            print("\n🚫 Daily token limit reached. Try again later.\n")
            break

        # ==============================
        # SAFE REQUEST
        # ==============================
        try:
            response = engine.chat(user_input)
        except Exception as e:
            err = str(e).lower()

            if "tokens per day" in err:
                print("\n🚫 Daily token limit reached. Please wait for reset.\n")
                break
            elif "rate limit" in err or "429" in err:
                print("\n⚠️ Rate limited. Please wait and try again.\n")
                continue
            else:
                raise e

        # ==============================
        # EXTRACT CONTEXT (TRUNCATED)
        # ==============================
        context_text = "".join(node.text[:300] for node in response.source_nodes)

        if not context_text.strip():
            fallback = (
                "Namaste. I'm sorry, but I don't have information on that specific topic in our records. "
                "Please contact our team at info@sherpahimalaya.com for more details."
            )
            print(f"\nAI: {fallback}\n")
            continue

        # ==============================
        # TOKEN CALCULATION
        # ==============================
        total_tokens = (
            count_tokens(user_input)
            + count_tokens(context_text)
            + count_tokens(str(response))
        )

        used_tokens += total_tokens
        cost = (total_tokens / 1_000_000) * 0.59

        # ==============================
        # OUTPUT
        # ==============================
        print(f"\nAI: {response}")
        print(f"\n📊 Tokens Used: {total_tokens}")
        print(f"💰 Cost: ${cost:.6f}\n")