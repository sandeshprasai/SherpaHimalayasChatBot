"""
understanding/classifier.py
────────────────────────────
Classifies a normalized query into a QueryIntent and suggests
a RetrievalStrategy. Populates the QueryRequest in-place.

NEW: Uses the Groq LLM for classification instead of brittle regex patterns.
     This means ANY phrasing of a question is correctly classified —
     no more missed intents due to missing regex patterns.

     The LLM also decides whether entity resolution is needed, preventing
     false positive entity matches on general/filter/aggregate queries.

     Falls back to the original regex classifier if the LLM call fails.

Usage:
    from understanding.classifier import classify
    from core.schema import QueryRequest

    req = QueryRequest(text="How do I book a trek?")
    classify(req)
    # req.intent   → QueryIntent.GENERAL
    # req.strategy → RetrievalStrategy.VECTOR
"""

import os
import sys
import re
import json

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.schema import QueryRequest, QueryIntent, RetrievalStrategy


# ──────────────────────────────────────────────────────────────────────────────
# INTENT MAP  (LLM string → QueryIntent enum)
# ──────────────────────────────────────────────────────────────────────────────

_INTENT_MAP = {
    "lookup":    QueryIntent.LOOKUP,
    "aggregate": QueryIntent.AGGREGATE,
    "compare":   QueryIntent.COMPARE,
    "filter":    QueryIntent.FILTER,
    "summarize": QueryIntent.SUMMARIZE,
    "general":   QueryIntent.GENERAL,
}


# ──────────────────────────────────────────────────────────────────────────────
# LLM CLASSIFIER  (primary)
# ──────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """\
You are a query classifier for a Himalayan travel chatbot (Sherpa Himalaya).
Classify the user query into exactly ONE intent and extract structured fields.

User query: "{query}"

Respond ONLY with a valid JSON object — no markdown, no explanation:
{{
  "intent": "lookup|aggregate|compare|filter|summarize|general",
  "confidence": <float 0.0-1.0>,
  "agg_op": "max|min|null",
  "agg_field": "max_altitude_m|duration_days|difficulty|null",
  "filters": {{
    "region": "<string or null>",
    "duration_days": <number or null>,
    "beginner_friendly": <true or null>
  }},
  "top_k": <number or null>,
  "needs_entity": <true or false>,
  "reasoning": "<one sentence>"
}}

Intent definitions:
- aggregate : superlative questions — highest, shortest, easiest, longest, most popular
- compare   : two specific destinations being compared (vs, versus, difference between)
- filter    : recommendation or listing — suggest, find me N, best for X, show me options
- summarize : request for a full itinerary or complete step-by-step description of ONE destination
- lookup    : specific factual question about ONE known destination (duration, altitude, inclusions, day N)
- general   : everything else — company info, booking process, permits, visas, health/altitude sickness,
              Nepal facts, team/staff questions, contact info, pricing, seasonal advice, activities

needs_entity rules:
- true  → for lookup, compare, summarize (user is asking about a specific named destination)
- false → for aggregate, filter, general (no specific entity needed; running entity resolver
          causes false positive matches like "views of Himalayas" → company page)
"""


def _llm_classify(req: QueryRequest) -> bool:
    """
    Call Groq LLM to classify the query.
    Returns True on success, False on any failure (caller falls back to regex).
    """
    try:
        from groq import Groq
        from dotenv import load_dotenv

        load_dotenv(os.path.join(_ROOT, ".env"))
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False

        client = Groq(api_key=api_key)
        text   = req.normalized or req.text

        response = client.chat.completions.create(
            model    = "llama-3.3-70b-versatile",
            messages = [{"role": "user", "content": _CLASSIFICATION_PROMPT.format(query=text)}],
            response_format = {"type": "json_object"},
            temperature = 0.0,
            max_tokens  = 250,
        )

        raw  = response.choices[0].message.content
        data = json.loads(raw)

        # ── Populate QueryRequest from LLM response ──────────────────────
        intent_str = str(data.get("intent", "general")).lower().strip()
        req.intent            = _INTENT_MAP.get(intent_str, QueryIntent.GENERAL)
        req.intent_confidence = float(data.get("confidence", 0.85))
        req.needs_entity      = bool(data.get("needs_entity", False))

        agg_op    = data.get("agg_op")
        agg_field = data.get("agg_field")
        req.agg_op    = agg_op    if agg_op    not in (None, "null")  else None
        req.agg_field = agg_field if agg_field not in (None, "null")  else None

        top_k = data.get("top_k")
        req.top_k = int(top_k) if top_k and str(top_k).isdigit() else None

        filters = data.get("filters") or {}
        if filters.get("region"):
            req.filters["region"] = filters["region"]
        if filters.get("duration_days"):
            try:
                req.filters["duration_days"] = int(filters["duration_days"])
            except (TypeError, ValueError):
                pass
        if filters.get("beginner_friendly"):
            req.filters["beginner_friendly"] = True

        # Strategy is always VECTOR now — the pipeline handles routing
        req.strategy = RetrievalStrategy.VECTOR

        return True

    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# REGEX CLASSIFIER  (fallback — preserved from original)
# ──────────────────────────────────────────────────────────────────────────────

_COMPARE_PATTERNS = [
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference between\b",
    r"\bbetter\b.{0,20}\bor\b",
]

_SUMMARIZE_PATTERNS = [
    r"\bfull itinerary\b",
    r"\bcomplete itinerary\b",
    r"\bday[\s-]by[\s-]day\b",
    r"\bdescribe\b.{0,20}\bitinerary\b",
    r"\bgive me.{0,20}\bitinerary\b",
    r"\bwalk me through\b",
    r"\ball days?\b",
]

_AGGREGATE_PATTERNS = [
    r"\bwhich\b.{0,30}\b(shortest|longest|highest|lowest|easiest|hardest|most popular)\b",
    r"\b(shortest|longest|highest|lowest|easiest|hardest)\b.{0,20}\b(trek|expedition|peak|climb|tour)\b",
    r"\bmost popular\b.{0,20}\b(trek|expedition|peak)\b",
    r"\brank\b.{0,20}\b(trek|peak|expedition)\b",
    r"\btop\s+\d+\b.{0,20}\b(trek|expedition|peak)\b",
]

_FILTER_PATTERNS = [
    r"\bsuggest\b",
    r"\brecommend\b",
    r"\bbest for beginners?\b",
    r"\bgood for beginners?\b",
    r"\bsuitable for beginners?\b",
    r"\bany\b.{0,20}\b(trek|expedition|peak)\b.{0,20}\b(for|under|around)\b",
    r"\bunder\s+\d+\s+days?\b",
    r"\b\d+[\s-]day\b.{0,20}\b(trek|expedition|peak|tour)\b",
    r"\bfor\s+(beginners?|intermediate|advanced|experienced|families?)\b",
    r"\bin\s+(annapurna|everest|langtang|manaslu|mustang|kanchenjunga|dolpa|makalu|rolwaling|helambu)\b",
    r"\bfamily[\s-]friendly\b",
    r"\beasy\b.{0,20}\b(trek|hike|walk)\b",
    r"\bwhat.{0,20}(trek|expedition).{0,20}(can|should|would)\b",
    r"\bbest\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?|expeditions?|peaks?)\b",
    r"\btop\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?)\b",
    r"\bpopular\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?)\b",
    r"\bwhat.{0,30}(treks?|trekking|destinations?|expeditions?).{0,20}(available|offer|have)\b",
    r"\blist.{0,30}(treks?|trekking|destinations?|expeditions?|peaks?)\b",
    r"\ball\b.{0,30}\b(treks?|trekking|destinations?|expeditions?|peaks?)\b",
    r"\b(trekking|hiking)\b.{0,30}\b(destinations?|places?|options?|available)\b",
    r"\bfind\b.{0,20}\b(me|us)\b.{0,30}\b(treks?|expeditions?|peaks?|tours?)\b",
    r"\bgive\b.{0,20}\b(me|us)\b.{0,30}\b(treks?|expeditions?|peaks?|tours?)\b",
    r"\bshow\b.{0,20}\b(me|us)\b.{0,30}\b(treks?|expeditions?|peaks?|tours?)\b",
    r"\b\d+\b.{0,20}\b(treks?|expeditions?|peaks?|tours?)\b.{0,20}\b(beginner|easy|short|long|hard)\b",
    r"\b(beginner|easy|short|long).{0,20}\b(treks?|expeditions?|peaks?)\b",
]

_GENERAL_INFO_PATTERNS = [
    r"\bhow (to|do i|can i) book\b",
    r"\bbooking (process|procedure|steps?|deposit|payment)\b",
    r"\b(wire transfer|bank transfer|payment method)\b",
    r"\b(cancel|refund|cancellation policy)\b",
    r"\b(tims|tims card|trekking permit|national park fee)\b",
    r"\b(visa|nepal visa|visa on arrival)\b",
    r"\b(altitude sickness|ams|acclimatization|diamox)\b",
    r"\b(travel insurance|emergency evacuation)\b",
    r"\b(about|who are|tell me about).{0,20}(sherpa|company|team)\b",
    r"\bwhy (choose|book with) sherpa\b",
    r"\b(who is|meet).{0,20}(ceo|founder|guide|team|sherpa)\b",
    r"\b(contact|office|address|phone|email|reach you)\b",
    r"\b(best time|trekking season|when to (go|visit|trek))\b",
    r"\b(paragliding|rafting|jungle safari|chitwan|pokhara city)\b",
]

_FIELD_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(altitude|height|high|elevation)\b",       "max_altitude_m"),
    (r"\b(duration|days|long|short|length)\b",      "duration_days"),
    (r"\b(difficult|easy|hard|difficulty|grade)\b", "difficulty"),
    (r"\b(region|area|location|zone)\b",            "region"),
]

_OP_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(highest|tallest|longest|hardest|most|maximum|max)\b", "max"),
    (r"\b(lowest|shortest|easiest|least|minimum|min|cheapest)\b", "min"),
]

_REGION_KEYWORDS = [
    "annapurna", "everest", "langtang", "manaslu",
    "mustang", "kanchenjunga", "dolpa", "makalu",
    "rolwaling", "helambu", "tsum",
]


def _match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _infer_field(text: str) -> str | None:
    for pattern, field in _FIELD_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return field
    return None


def _infer_op(text: str) -> str | None:
    for pattern, op in _OP_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            return op
    return None


def _extract_duration(text: str) -> int | None:
    m = re.search(r"\b(\d+)\s*[-\s]?\s*days?\b", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _extract_region(text: str) -> str | None:
    tl = text.lower()
    return next(
        (r.capitalize() for r in _REGION_KEYWORDS if r in tl),
        None
    )


def _regex_classify(req: QueryRequest) -> QueryRequest:
    """
    Original regex-based classifier — used as fallback when LLM is unavailable.
    """
    text = req.normalized or req.text

    # 1. COMPARE
    if _match(_COMPARE_PATTERNS, text):
        req.intent        = QueryIntent.COMPARE
        req.strategy      = RetrievalStrategy.METADATA
        req.agg_field     = _infer_field(text)
        req.needs_entity  = True
        return req

    # 2. SUMMARIZE
    if _match(_SUMMARIZE_PATTERNS, text):
        req.intent        = QueryIntent.SUMMARIZE
        req.strategy      = RetrievalStrategy.FULL_DOC
        req.needs_entity  = True
        return req

    # 3. AGGREGATE
    if _match(_AGGREGATE_PATTERNS, text):
        req.intent        = QueryIntent.AGGREGATE
        req.strategy      = RetrievalStrategy.METADATA
        req.agg_op        = _infer_op(text) or "max"
        req.agg_field     = _infer_field(text) or "duration_days"
        req.needs_entity  = False

        if re.search(r"\b(easiest|beginner)\b", text, re.IGNORECASE):
            req.agg_op    = "min"
            req.agg_field = "difficulty"

        if re.search(r"\bmost popular\b", text, re.IGNORECASE):
            req.strategy  = RetrievalStrategy.VECTOR
            req.agg_op    = None
            req.agg_field = None

        return req

    # 4. FILTER
    if _match(_FILTER_PATTERNS, text):
        req.intent       = QueryIntent.FILTER
        req.strategy     = RetrievalStrategy.HYBRID
        req.needs_entity = False

        duration = _extract_duration(text)
        region   = _extract_region(text)

        if duration:
            req.filters["duration_days"] = duration
        if region:
            req.filters["region"] = region
        if re.search(r"\bbeginners?\b", text, re.IGNORECASE):
            req.filters["beginner_friendly"] = True
        if re.search(r"\bfamil(y|ies)\b", text, re.IGNORECASE):
            req.filters["beginner_friendly"] = True
        if re.search(r"\beasy\b", text, re.IGNORECASE):
            req.filters["beginner_friendly"] = True

        count_match = re.search(r"\b(\d+)\b.{0,20}\b(treks?|expeditions?|peaks?)\b", text, re.IGNORECASE)
        if count_match:
            req.top_k = int(count_match.group(1))

        return req

    # 4b. GENERAL INFO
    if _match(_GENERAL_INFO_PATTERNS, text):
        req.intent            = QueryIntent.GENERAL
        req.strategy          = RetrievalStrategy.VECTOR
        req.intent_confidence = 0.85
        req.needs_entity      = False
        return req

    # 5. LOOKUP (default)
    req.intent        = QueryIntent.LOOKUP
    req.strategy      = RetrievalStrategy.VECTOR
    req.needs_entity  = True
    return req


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def classify(req: QueryRequest) -> QueryRequest:
    """
    Classifies the query using the Groq LLM.
    Falls back to regex classification if the LLM call fails.

    Populates req.intent, req.strategy, req.agg_op, req.agg_field,
    req.filters, req.top_k, req.needs_entity in-place.
    Returns the same request for chaining.
    """
    success = _llm_classify(req)

    if not success:
        # LLM unavailable (no API key, network issue, etc.) — use regex
        _regex_classify(req)

    return req


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "What is the duration of Ghorepani Circuit Trek?",
        "Which trek has the highest altitude?",
        "Which trek is the shortest?",
        "Suggest a 7-day trek in Annapurna region",
        "Compare Everest Base Camp Trek vs Annapurna Base Camp Trek",
        "Describe the full itinerary of Langtang Valley Trek",
        "How do I book a trek?",
        "What is a TIMS card?",
        "Find me 5 trekking destinations with beautiful Himalaya views",
        "Tell me about altitude sickness",
        "Who are your guides?",
        "What is the best time to visit Nepal?",
        "Does any trek have views of Himalayas?",
        "ABC vs EBC",
    ]

    print("\n🔍 Classifier Test\n" + "─" * 60)
    for q in tests:
        req = QueryRequest(text=q, normalized=q)
        classify(req)
        print(f"Q: {q}")
        print(f"   intent={req.intent.value:<12} confidence={req.intent_confidence:.2f} needs_entity={req.needs_entity}", end="")
        if req.agg_op:    print(f" op={req.agg_op}", end="")
        if req.agg_field: print(f" field={req.agg_field}", end="")
        if req.filters:   print(f" filters={req.filters}", end="")
        if req.top_k:     print(f" top_k={req.top_k}", end="")
        print()