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


def answer(query, top_k: int = 8):
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client is not installed or not available in environment.")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Quick aggregation for comparative queries
    qlow = query.lower()
    try:
        from src.brain import aggregator
    except Exception:
        aggregator = None

    if aggregator:
        import re
        m = re.search(r"\btop\s+(\d+)\b", qlow)
        top_k = int(m.group(1)) if m else 1
        detailed = any(x in qlow for x in ['detail', 'detailed', 'provide the detailed', 'detailed information', 'provide details'])

        if any(k in qlow for k in ['shortest', 'shorter', 'shortest trek']):
            items = aggregator.shortest_treks(top_k=top_k)
            if not items:
                return "No treks with duration metadata found."
            if top_k == 1:
                r = items[0]
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                days = r.get('metadata', {}).get('duration_days')
                if detailed:
                    return f"{title} — {days} days\n\n{r.get('text','')}"
                return f"{title} — {days} days"
            # multiple
            lines = []
            for r in items:
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                days = r.get('metadata', {}).get('duration_days')
                if detailed:
                    lines.append(f"{title} — {days} days\n{r.get('text','')}")
                else:
                    lines.append(f"{title} — {days} days")
            return "\n\n".join(lines)

        if any(k in qlow for k in ['longest', 'longer', 'longest trek']):
            items = aggregator.longest_treks(top_k=top_k)
            if not items:
                return "No treks with duration metadata found."
            if top_k == 1:
                r = items[0]
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                days = r.get('metadata', {}).get('duration_days')
                if detailed:
                    return f"{title} — {days} days\n\n{r.get('text','')}"
                return f"{title} — {days} days"
            lines = []
            for r in items:
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                days = r.get('metadata', {}).get('duration_days')
                if detailed:
                    lines.append(f"{title} — {days} days\n{r.get('text','')}")
                else:
                    lines.append(f"{title} — {days} days")
            return "\n\n".join(lines)

        if any(k in qlow for k in ['highest altitude', 'highest', 'highest peak', 'highest altitude trek']):
            items = aggregator.highest_altitude(top_k=top_k)
            if not items:
                return "No altitude metadata found."
            if top_k == 1:
                r = items[0]
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                alt = r.get('metadata', {}).get('max_altitude_m')
                if detailed:
                    return f"{title} — {alt} meters\n\n{r.get('text','')}"
                return f"{title} — {alt} meters"
            lines = []
            for r in items:
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                alt = r.get('metadata', {}).get('max_altitude_m')
                if detailed:
                    lines.append(f"{title} — {alt} meters\n{r.get('text','')}")
                else:
                    lines.append(f"{title} — {alt} meters")
            return "\n\n".join(lines)

        if any(k in qlow for k in ['easiest', 'easy', 'beginner']):
            r = aggregator.easiest_trek()
            if r:
                title = r.get('metadata', {}).get('title') or r.get('text','')[:80]
                if detailed:
                    return f"{title} — difficulty: {r.get('metadata', {}).get('difficulty')}\n\n{r.get('text','')}"
                return f"{title} — difficulty: {r.get('metadata', {}).get('difficulty')}"

    # Otherwise perform retrieval + Groq LLM
    results = retrieve(query, top_k=top_k)
    if not results:
        # No relevant context found — return strict fallback
        fallback = (
            "Namaste. I'm sorry, but I don't have information on that specific topic in our records. "
            "Please contact our team at info@sherpahimalaya.com for more details."
        )
        return fallback

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
