"""
understanding/entity_resolver.py
──────────────────────────────────
Extracts and resolves trek/peak/expedition names mentioned in a query
against the known records in metadata_index.json.

Why this matters:
  "ghorepani circuit"  →  "Ghorepani Circuit Trek – 10 Days"
  "EBC"               →  "Everest Base Camp Trek"  (after normalizer)
  "ama dablam"        →  "Ama Dablam Expedition"

Usage:
    from understanding.entity_resolver import EntityResolver
    resolver = EntityResolver()
    entities = resolver.resolve("Compare Ghorepani trek vs EBC trek")
    # → ["Ghorepani Circuit Trek – 10 Days", "Everest Base Camp Trek"]
"""

import os
import sys
import json
import re

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings


class EntityResolver:
    """
    Loads all known titles from metadata_index.json and performs
    fuzzy matching against a query string.
    """

    def __init__(self):
        self._records: list[dict] = self._load()
        # Pre-build a list of (normalized_title, original_title, record_meta) for matching
        self._title_map: list[tuple[str, str, dict]] = [
            (self._normalize_title(r["title"]), r["title"], r)
            for r in self._records
            if r.get("title")
        ]

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def resolve(self, query: str) -> list[str]:
        """
        Returns a list of canonical titles found in the query.
        Ordered by position of match in the query (left to right).
        """
        query_norm = self._normalize_title(query)
        # found entries will hold tuples: (position_in_query, key_token, canonical_title, record, norm_title)
        found: list[tuple[int, str, str, dict, str]] = []

        for norm_title, canonical, record in self._title_map:
            # Prefer exact phrase matches first
            if norm_title in query_norm:
                pos = query_norm.find(norm_title)
                key_token = (norm_title.split()[0] if norm_title else norm_title)
                found.append((pos, key_token, canonical, record, norm_title))
                continue
            # Build a set of significant words from the title (length > 3)
            title_words = [w for w in norm_title.split() if len(w) > 3]
            if not title_words:
                continue

            # Score: how many title words appear in the query
            matched_tokens = [w for w in title_words if w in query_norm]
            matches = len(matched_tokens)
            coverage = matches / len(title_words) if title_words else 0.0

            # Significant query words (used for single-word queries)
            query_words = [w for w in query_norm.split() if len(w) > 3]

            key_token = None
            pos = 0

            if coverage >= 0.5 and matches >= 2:
                # position of first matched token in the query
                pos = min(query_norm.find(w) for w in matched_tokens)
                key_token = min(matched_tokens, key=lambda w: query_norm.find(w))
                found.append((pos, key_token, canonical, record, norm_title))
            else:
                # Accept single-word queries that match a key title token
                if len(query_words) == 1 and query_words[0] in title_words:
                    key_token = query_words[0]
                    pos = query_norm.find(key_token)
                    found.append((pos, key_token, canonical, record, norm_title))
                # Accept when title is very short (<=2 significant words) and at least 1 match
                elif matches >= 1 and len(title_words) <= 2:
                    pos = min(query_norm.find(w) for w in matched_tokens) if matched_tokens else 0
                    key_token = min(matched_tokens, key=lambda w: query_norm.find(w)) if matched_tokens else None
                    found.append((pos, key_token or norm_title, canonical, record, norm_title))
                # Direct substring match as a final fallback
                elif norm_title in query_norm or query_norm in norm_title:
                    pos = query_norm.find(norm_title) if norm_title in query_norm else 0
                    # choose a token from the norm_title as key
                    key_token = (norm_title.split()[0] if norm_title else norm_title)
                    found.append((pos, key_token, canonical, record, norm_title))

        # Group candidates by key_token and pick the best-scoring record per token
        candidates: dict[str, list[tuple[int, str, dict, str]]] = {}
        for pos, key_token, canonical, record, norm_title in found:
            candidates.setdefault(key_token, []).append((pos, canonical, record, norm_title))

        def score_record(rec: dict, norm_title: str) -> int:
            s = 0
            if norm_title and norm_title in query_norm:
                s += 100
            cat = (rec.get('category') or '').lower()
            if 'trek' in cat or 'expedition' in cat or 'tour' in cat:
                s += 50
            if rec.get('duration_days'):
                s += 10
            if rec.get('max_altitude_m'):
                s += 10
            if rec.get('region'):
                s += 5
            if rec.get('summary'):
                s += 2
            return s

        selected: list[tuple[int, str]] = []  # (pos, canonical)
        for key, items in candidates.items():
            # pick best by score, tie-breaker by earliest position
            best = max(items, key=lambda it: (score_record(it[2], it[3]), -it[0]))
            selected.append((best[0], best[1]))

        # sort selected by position and return canonicals
        selected.sort(key=lambda x: x[0])
        return [c for _, c in selected]

    def find_record(self, entity_name: str) -> dict | None:
        """
        Find the full metadata record for a canonical entity name.
        """
        name_norm = self._normalize_title(entity_name)
        best_score = 0
        best_record = None

        for record in self._records:
            title_norm = self._normalize_title(record.get("title", ""))
            title_words = [w for w in title_norm.split() if len(w) > 3]
            if not title_words:
                continue

            matches = sum(1 for w in title_words if w in name_norm)
            score = matches / len(title_words)

            if score > best_score:
                best_score = score
                best_record = record

        # Accept slightly lower score for short titles, or direct substring match
        if best_score >= 0.5:
            return best_record
        if best_score >= 0.4:
            # if title is short, accept it
            title_len = len([w for w in self._normalize_title(best_record.get("title","")).split() if len(w)>3]) if best_record else 0
            if title_len <= 2:
                return best_record
        # direct substring fallback
        if best_record and self._normalize_title(best_record.get("title","")) in name_norm:
            return best_record
        return None

    def all_titles(self) -> list[str]:
        """Return all known canonical titles."""
        return [canonical for _, canonical in self._title_map]

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not os.path.exists(settings.metadata_index):
            print(
                f"⚠️  metadata_index.json not found — entity resolution disabled."
            )
            return []
        with open(settings.metadata_index, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize_title(text: str) -> str:
        """Lowercase, remove special chars, collapse spaces."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
        text = re.sub(r"\s+", " ", text).strip()
        # Remove generic stop words that appear in many titles
        stop = {"trek", "trekking", "the", "with", "and", "via", "to",
                "from", "by", "in", "of", "a", "an", "days", "day"}
        words = [w for w in text.split() if w not in stop]
        return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resolver = EntityResolver()

    tests = [
        "What is the duration of Ghorepani Circuit Trek?",
        "Compare Everest Base Camp Trek vs Annapurna Base Camp Trek",
        "Tell me about Ama Dablam Expedition",
        "Which trek is best for beginners?",
        "What is included in Langtang Valley Trek package?",
        "Manaslu Circuit highlights",
    ]

    print("\n🔍 Entity Resolver Test\n" + "─" * 55)
    for q in tests:
        entities = resolver.resolve(q)
        print(f"Q: {q}")
        print(f"   → {entities if entities else '(none found)'}")
        print()