#!/usr/bin/env python3
"""
Shared normalization utilities for title matching, abbreviation
extraction, and light scalar inference used across the pipeline.

This module centralizes stop-words and parsing helpers so entity
resolution and metadata cleaning behave consistently.
"""

import re
from typing import Optional

# Stop-words used for normalizing titles (for grouping / matching)
STOPWORDS_NORM = {
    "trek", "trekking", "the", "with", "and", "via", "to",
    "from", "by", "in", "of", "a", "an", "days", "day",
    "base", "camp", "valley", "pass", "lake", "peak",
    "expedition", "helicopter", "fly", "tour", "shortest",
    "longest", "short", "long",
}

# Words to exclude when generating abbreviations (we want meaningful initials)
ABBR_EXCLUDE = {"the", "and", "of", "in", "to", "by", "from", "a", "an", "trek", "trekking"}

REGION_KEYWORDS = [
    "annapurna", "everest", "langtang", "manaslu", "mustang",
    "kanchenjunga", "dolpa", "makalu", "rolwaling", "helambu",
    "pokhara", "nepal", "tsum",
]


def parse_int(x) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
    try:
        return int(x)
    except Exception:
        return None


def canonical_difficulty(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).lower()
    if "easy/moderate" in s or ("easy" in s and "moderate" in s):
        return "Easy/Moderate"
    if "easy" in s and "moderate" not in s:
        return "Easy"
    if "moderate" in s:
        return "Moderate"
    if "challeng" in s or "hard" in s:
        return "Challenging"
    if "stren" in s:
        return "Strenuous"
    if "extreme" in s:
        return "Extreme"
    return s.capitalize()


def normalize_title(text: str) -> str:
    """Lowercase, strip punctuation, remove common stop-words used in titles."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    parts = [w for w in t.split() if w and w not in STOPWORDS_NORM]
    return " ".join(parts).strip()


def abbrev(title: str) -> Optional[str]:
    if not title:
        return None
    toks = [w for w in re.split(r"\W+", title) if w and w.lower() not in ABBR_EXCLUDE]
    if not toks:
        return None
    a = ''.join(w[0].upper() for w in toks)
    return a if 2 <= len(a) <= 6 else None


def find_duration(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{1,2})\s*(?:-\s*)?(?:days|day)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"duration[:\s]*(\d{1,2})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def find_altitude(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(\d{3,4})\s*(?:m|meters|metres)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"max\s+altitude[:\s]*(\d{3,4})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def infer_region(text: str) -> Optional[str]:
    if not text:
        return None
    tl = text.lower()
    for r in REGION_KEYWORDS:
        if r in tl:
            return " ".join(w.capitalize() for w in r.split())
    return None
