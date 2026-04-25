"""
understanding/classifier.py
────────────────────────────
Classifies a normalized query into a QueryIntent and suggests
a RetrievalStrategy. Populates the QueryRequest in-place.

Usage:
    from understanding.classifier import classify
    from core.schema import QueryRequest

    req = QueryRequest(text="Which trek has the highest altitude?")
    classify(req)
    # req.intent   → QueryIntent.AGGREGATE
    # req.strategy → RetrievalStrategy.METADATA
    # req.agg_op   → "max"
    # req.agg_field→ "max_altitude_m"
"""

import os
import sys
import re

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.schema import QueryRequest, QueryIntent, RetrievalStrategy


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN BANKS
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
    # Open-ended listing queries
    r"\bbest\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?|expeditions?|peaks?)\b",
    r"\btop\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?)\b",
    r"\bpopular\b.{0,30}\b(treks?|trekking|destinations?|places?|regions?)\b",
    r"\bwhat.{0,30}(treks?|trekking|destinations?|expeditions?).{0,20}(available|offer|have)\b",
    r"\blist.{0,30}(treks?|trekking|destinations?|expeditions?|peaks?)\b",
    r"\ball\b.{0,30}\b(treks?|trekking|destinations?|expeditions?|peaks?)\b",
    r"\b(trekking|hiking)\b.{0,30}\b(destinations?|places?|options?|available)\b",
]

# ── field inference ───────────────────────────────────────────────────────────

_FIELD_KEYWORDS: list[tuple[str, str]] = [
    # (keyword_pattern, json_field_name)
    (r"\b(altitude|height|high|elevation)\b",       "max_altitude_m"),
    (r"\b(duration|days|long|short|length)\b",      "duration_days"),
    (r"\b(difficult|easy|hard|difficulty|grade)\b", "difficulty"),
    (r"\b(region|area|location|zone)\b",            "region"),
]

_OP_KEYWORDS: list[tuple[str, str]] = [
    # (keyword_pattern, operation)
    (r"\b(highest|tallest|longest|hardest|most|maximum|max)\b", "max"),
    (r"\b(lowest|shortest|easiest|least|minimum|min|cheapest)\b", "min"),
]

_REGION_KEYWORDS = [
    "annapurna", "everest", "langtang", "manaslu",
    "mustang", "kanchenjunga", "dolpa", "makalu",
    "rolwaling", "helambu", "tsum",
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def classify(req: QueryRequest) -> QueryRequest:
    """
    Populates req.intent, req.strategy, req.agg_op, req.agg_field,
    and req.filters in-place. Returns the same request for chaining.

    Uses req.normalized if set, otherwise req.text.
    """
    text = req.normalized or req.text

    # ── 1. COMPARE ──────────────────────────────────────────────────────────
    if _match(_COMPARE_PATTERNS, text):
        req.intent   = QueryIntent.COMPARE
        req.strategy = RetrievalStrategy.METADATA
        req.agg_field = _infer_field(text)
        return req

    # ── 2. SUMMARIZE ─────────────────────────────────────────────────────────
    if _match(_SUMMARIZE_PATTERNS, text):
        req.intent   = QueryIntent.SUMMARIZE
        req.strategy = RetrievalStrategy.FULL_DOC
        return req

    # ── 3. AGGREGATE ─────────────────────────────────────────────────────────
    if _match(_AGGREGATE_PATTERNS, text):
        req.intent    = QueryIntent.AGGREGATE
        req.strategy  = RetrievalStrategy.METADATA
        req.agg_op    = _infer_op(text) or "max"
        req.agg_field = _infer_field(text) or "duration_days"

        # Special case: "easiest" → min difficulty
        if re.search(r"\b(easiest|beginner)\b", text, re.IGNORECASE):
            req.agg_op    = "min"
            req.agg_field = "difficulty"

        # Special case: "most popular" → no scalar field, use vector
        if re.search(r"\bmost popular\b", text, re.IGNORECASE):
            req.strategy  = RetrievalStrategy.VECTOR
            req.agg_op    = None
            req.agg_field = None

        return req

    # ── 4. FILTER / RECOMMENDATION ───────────────────────────────────────────
    if _match(_FILTER_PATTERNS, text):
        req.intent   = QueryIntent.FILTER
        req.strategy = RetrievalStrategy.HYBRID

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

        return req

    # ── 5. LOOKUP (default) ──────────────────────────────────────────────────
    req.intent   = QueryIntent.LOOKUP
    req.strategy = RetrievalStrategy.VECTOR
    return req


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.schema import QueryRequest

    tests = [
        "What is the duration of Ghorepani Circuit Trek?",
        "Which trek has the highest altitude?",
        "Which trek is the shortest?",
        "Which trek is easiest based on difficulty?",
        "Suggest a 7-day trek in Annapurna region",
        "Recommend a trek for beginners",
        "Compare Everest Base Camp Trek vs Annapurna Base Camp Trek in terms of duration",
        "Describe the full itinerary of Langtang Valley Trek",
        "What is included in the Ghorepani Circuit Trek package?",
        "Which trek has the longest duration?",
        "What is the most popular trek in Nepal?",
        "Give me a family-friendly trek under 10 days",
    ]

    print("\n🔍 Classifier Test\n" + "─" * 60)
    for q in tests:
        req = QueryRequest(text=q)
        classify(req)
        print(f"Q: {q}")
        print(f"   intent={req.intent.value:<12} strategy={req.strategy.value:<10}", end="")
        if req.agg_op:    print(f" op={req.agg_op}", end="")
        if req.agg_field: print(f" field={req.agg_field}", end="")
        if req.filters:   print(f" filters={req.filters}", end="")
        print()