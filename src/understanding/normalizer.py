"""
understanding/normalizer.py
────────────────────────────
Normalizes raw user query text before classification and retrieval.

Handles:
  - Abbreviation expansion  ("EBC" → "Everest Base Camp Trek")
  - Case / whitespace cleanup
  - Common misspellings
  - Colloquial phrasings    ("himalayas trip" → "himalaya trek")

Usage:
    from understanding.normalizer import normalize
    clean = normalize("whats included in EBC trek?")
    # → "what is included in Everest Base Camp Trek?"
"""

import re


# ──────────────────────────────────────────────────────────────────────────────
# ABBREVIATION + ALIAS MAP
# Order matters: longer/more-specific patterns first
# ──────────────────────────────────────────────────────────────────────────────

_ALIASES: list[tuple[str, str]] = [

    # ── Trek abbreviations ─────────────────────────────────────────────────
    (r"\bEBC\b",            "Everest Base Camp Trek"),
    (r"\bABC\b",            "Annapurna Base Camp Trek"),
    (r"\bACT\b",            "Annapurna Circuit Trek"),
    (r"\bMCT\b",            "Manaslu Circuit Trek"),
    (r"\bLVT\b",            "Langtang Valley Trek"),
    (r"\bGHT\b",            "Great Himalaya Trail"),

    # ── Peak abbreviations ─────────────────────────────────────────────────
    (r"\bIsland Peak\b",    "Island Peak Climbing"),
    (r"\bMera\b",           "Mera Peak Climbing"),
    (r"\bLobuche\b",        "Lobuche East Peak Climbing"),
    (r"\bAma Dablam\b",     "Ama Dablam Expedition"),

    # ── Region shorthands ─────────────────────────────────────────────────
    (r"\bEverest region\b", "Everest Region"),
    (r"\bAnnapurna region\b","Annapurna Region"),
    (r"\bLangtang region\b","Langtang Region"),

    # ── Common colloquial phrasings ────────────────────────────────────────
    (r"\bhimalaya[s]?\s+trip\b",   "himalaya trek"),
    (r"\bnepal\s+hiking\b",        "trekking in Nepal"),
    (r"\bpoon\s*hill\b",           "Poon Hill"),
    (r"\bghorepani\s+poon\s*hill\b","Ghorepani Poonhill Trek"),

    # ── Contraction expansions ─────────────────────────────────────────────
    (r"\bwhats\b",    "what is"),
    (r"\bwhat's\b",   "what is"),
    (r"\bwhich's\b",  "which is"),
    (r"\bwhen's\b",   "when is"),
    (r"\bdon't\b",    "do not"),
    (r"\bcan't\b",    "cannot"),
    (r"\bi'm\b",      "I am"),
    (r"\bI've\b",     "I have"),
]


# ──────────────────────────────────────────────────────────────────────────────
# COMMON MISSPELLINGS
# ──────────────────────────────────────────────────────────────────────────────

_MISSPELLINGS: list[tuple[str, str]] = [
    (r"\btreking\b",        "trekking"),
    (r"\bexpedtion\b",      "expedition"),
    (r"\biterary\b",        "itinerary"),
    (r"\bitinarary\b",      "itinerary"),
    (r"\bkathamndu\b",      "Kathmandu"),
    (r"\bkathamandu\b",     "Kathmandu"),
    (r"\bpokhara\b",        "Pokhara"),       # already correct, force capitalize
    (r"\bnamche\s+bazar\b", "Namche Bazaar"),
    (r"\bnamche\s+bazzar\b","Namche Bazaar"),
    (r"\bannapurna\b",      "Annapurna"),
    (r"\beverest\b",        "Everest"),
    (r"\blangtang\b",       "Langtang"),
    (r"\bmanaslu\b",        "Manaslu"),
    (r"\bmustang\b",        "Mustang"),
]


# ──────────────────────────────────────────────────────────────────────────────
# NORMALIZER
# ──────────────────────────────────────────────────────────────────────────────

def normalize(query: str) -> str:
    """
    Returns a cleaned, expanded version of the raw query string.
    The original casing intent is preserved where possible.
    """
    if not query or not query.strip():
        return query

    result = query.strip()

    # 1. Collapse multiple spaces / tabs
    result = re.sub(r"[ \t]+", " ", result)

    # 2. Fix misspellings (case-insensitive)
    for pattern, replacement in _MISSPELLINGS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 3. Expand abbreviations and aliases (case-sensitive patterns)
    for pattern, replacement in _ALIASES:
        result = re.sub(pattern, replacement, result)

    # 4. Remove redundant punctuation sequences (e.g. "??", "!!")
    result = re.sub(r"([?!])\1+", r"\1", result)

    # 5. Collapse duplicate consecutive words  ("Trek trek" → "Trek")
    result = re.sub(r"\b(\w+)\s+\1\b", r"\1", result, flags=re.IGNORECASE)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "whats included in EBC trek?",
        "Which trek has highest altitude in ABC region",
        "Suggest a treking route for beginners in Langtang region",
        "Compare EBC vs ABC in terms of duration",
        "Tell me about Ama Dablam",
        "What happens on day 3 of ghorepani poon hill trek",
        "I want a himalaya trip under 10 days",
        "namche bazzar altitude",
        "kathamandu to lukla flight",
    ]

    print("\n🔤 Normalizer Test\n" + "─" * 55)
    for q in tests:
        n = normalize(q)
        changed = " ✏️" if n != q else ""
        print(f"IN : {q}")
        print(f"OUT: {n}{changed}")
        print()