import os
from dotenv import load_dotenv
from src.brain.retriever import retrieve

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
load_dotenv(os.path.join(project_root, ".env"))

SYSTEM_PROMPT = None
try:
    from src.brain.engine import get_system_prompt, MODEL_NAME
    SYSTEM_PROMPT = get_system_prompt()
    MODEL = MODEL_NAME
except Exception:
    SYSTEM_PROMPT = "You are a helpful assistant constrained to provided context."
    MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def answer(query, top_k: int = 5):
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client is not installed or not available in environment.")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    results = retrieve(query, top_k=top_k)
    context = "\n\n".join([r.get("text", "") for r in results])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    resp = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        response_format={"type": "text"}
    )

    return resp.choices[0].message.content
