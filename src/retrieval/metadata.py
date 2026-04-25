"""
retrieval/metadata.py
──────────────────────
Answers aggregate, filter, and compare queries directly from
metadata_index.json — no LLM or vector search needed.

Usage:
    from retrieval.metadata import MetadataRetriever
    from core.schema import QueryRequest, QueryIntent

    retriever = MetadataRetriever()
    answer = retriever.answer(req)   # returns str | None
"""

import os
import sys
import json

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import QueryRequest, QueryIntent


class MetadataRetriever:

    def __init__(self):
        # Prefer a cleaned metadata index when available (produced by metadata_cleaner)
        meta_path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        with open(meta_path, encoding="utf-8") as f:
            all_records: list[dict] = json.load(f)

        # Heuristic: determine which records represent destinations (treks/tours/expeditions)
        def _looks_like_destination(rec: dict) -> bool:
            src = (rec.get("_source") or "").lower()
            cat = (rec.get("category") or "").lower()
            title = (rec.get("title") or "").lower()
            if src.startswith("destinations/"):
                return True
            if any(k in cat for k in ("trek", "expedition", "tour", "peak")):
                return True
            if any(k in title for k in ("trek", "expedition", "peak", "tour")):
                return True
            return False

        self._records = [r for r in all_records if _looks_like_destination(r)]

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────

    def answer(self, req: QueryRequest) -> str | None:
        """
        Returns a formatted answer string, or None if this retriever
        cannot handle the query (caller falls back to vector search).
        """
        if req.intent == QueryIntent.AGGREGATE:
            return self._aggregate(req)

        if req.intent == QueryIntent.FILTER:
            return self._filter(req)

        if req.intent == QueryIntent.COMPARE:
            return self._compare(req)

        return None

    # ──────────────────────────────────────────────────────────────────────
    # AGGREGATE
    # ──────────────────────────────────────────────────────────────────────

    def _aggregate(self, req: QueryRequest) -> str | None:
        field = req.agg_field or "duration_days"
        op    = req.agg_op   or "max"

        if field == "difficulty":
            return self._aggregate_difficulty(op)

        # Exclude records with zero/null values (region pages, stubs)
        valid = [
            r for r in self._records
            if r.get(field) is not None and r.get(field) != 0
        ]
        if not valid:
            return None

        winner = max(valid, key=lambda r: r[field]) if op == "max" \
            else min(valid, key=lambda r: r[field])

        adj   = _op_adjective(op, field)
        label = _field_label(field)
        unit  = _field_unit(field)
        val   = winner[field]

        return (
            f"**{winner['title']}** has the {adj} {label} "
            f"among all available destinations.\n\n"
            f"- **{label.capitalize()}:** {val}{unit}\n"
            f"- **Region:** {winner.get('region', 'N/A')}\n"
            f"- **Difficulty:** {winner.get('difficulty', 'N/A')}\n\n"
            f"{winner.get('summary', '')}"
        )

    def _aggregate_difficulty(self, op: str) -> str | None:
        rank = settings.difficulty_rank
        valid = [
            r for r in self._records
            if r.get("difficulty") and r["difficulty"].lower() in rank
        ]
        if not valid:
            return None

        key_fn = lambda r: rank.get(r["difficulty"].lower(), 99)
        winner = min(valid, key=key_fn) if op == "min" \
            else max(valid, key=key_fn)
        adj = "easiest" if op == "min" else "most difficult"

        return (
            f"**{winner['title']}** is the {adj} option available.\n\n"
            f"- **Difficulty:** {winner.get('difficulty', 'N/A')}\n"
            f"- **Duration:** {winner.get('duration_days', 'N/A')} days\n"
            f"- **Region:** {winner.get('region', 'N/A')}\n\n"
            f"{winner.get('summary', '')}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # FILTER
    # ──────────────────────────────────────────────────────────────────────

    def _filter(self, req: QueryRequest) -> str:
        records = list(self._records)
        filters = req.filters

        # Region
        if region := filters.get("region"):
            records = [
                r for r in records
                if region.lower() in (r.get("region") or "").lower()
            ]

        # Duration  (±3 day tolerance, and exclude 0-day records)
        if days := filters.get("duration_days"):
            records = [
                r for r in records
                if r.get("duration_days") is not None
                and r["duration_days"] > 0
                and abs(r["duration_days"] - days) <= 3
            ]

        # Beginner-friendly
        if filters.get("beginner_friendly"):
            friendly = {"easy", "moderate", "easy/moderate"}
            records = [
                r for r in records
                if (r.get("difficulty") or "").lower() in friendly
            ]

        if not records:
            return (
                f"I couldn't find any trips matching those exact criteria. "
                f"Try adjusting the duration or region, or contact us at "
                f"**{settings.company_email}** for a custom itinerary."
            )

        records.sort(key=lambda r: r.get("duration_days") or 999)
        top = records[:5]

        lines = [
            f"Here are {len(top)} option{'s' if len(top) > 1 else ''} "
            f"matching your criteria:\n"
        ]
        for i, r in enumerate(top, 1):
            lines.append(
                f"**{i}. {r['title']}**\n"
                f"   - Duration: {r.get('duration_days', 'N/A')} days\n"
                f"   - Difficulty: {r.get('difficulty', 'N/A')}\n"
                f"   - Region: {r.get('region', 'N/A')}\n"
                f"   - Max Altitude: "
                f"{str(r['max_altitude_m']) + ' m' if r.get('max_altitude_m') else 'N/A'}\n"
                f"   - {(r.get('summary') or '')[:120]}…\n"
            )

        if len(records) > 5:
            lines.append(
                f"\n*{len(records) - 5} more option(s) available — "
                f"ask for more details or narrow your criteria.*"
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # COMPARE
    # ──────────────────────────────────────────────────────────────────────

    def _compare(self, req: QueryRequest) -> str | None:
        if len(req.entities) < 2:
            return None

        matched = [self._fuzzy_find(e) for e in req.entities]
        matched = [m for m in matched if m]

        if len(matched) < 2:
            return None

        a, b = matched[0], matched[1]

        compare_fields = [
            ("Duration",     "duration_days",   " days"),
            ("Difficulty",   "difficulty",       ""),
            ("Max Altitude", "max_altitude_m",   " m"),
            ("Region",       "region",           ""),
        ]

        lines = [f"**{a['title']} vs {b['title']}**\n"]
        lines.append(f"| Attribute | {a['title']} | {b['title']} |")
        lines.append("|-----------|" + "---|" * 2)

        for label, field, unit in compare_fields:
            va = str(a.get(field) or "N/A") + (unit if a.get(field) else "")
            vb = str(b.get(field) or "N/A") + (unit if b.get(field) else "")
            lines.append(f"| {label} | {va} | {vb} |")

        lines.append("\n**Summaries:**")
        for r in (a, b):
            lines.append(f"\n*{r['title']}:* {r.get('summary', '')}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _fuzzy_find(self, name: str) -> dict | None:
        name_lower = name.lower().strip()
        best, best_score = None, 0.0

        for r in self._records:
            title = (r.get("title") or "").lower()
            # Word overlap score
            name_words  = set(w for w in name_lower.split() if len(w) > 3)
            title_words = set(w for w in title.split()      if len(w) > 3)
            if not title_words:
                continue
            overlap = len(name_words & title_words) / len(title_words)
            if overlap > best_score:
                best_score = overlap
                best = r

        return best if best_score >= 0.4 else None


# ──────────────────────────────────────────────────────────────────────────────
# FORMAT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _field_label(field: str) -> str:
    return {
        "duration_days":  "duration",
        "max_altitude_m": "maximum altitude",
        "difficulty":     "difficulty",
        "region":         "region",
    }.get(field, field)


def _field_unit(field: str) -> str:
    return {
        "duration_days":  " days",
        "max_altitude_m": " m",
    }.get(field, "")


def _op_adjective(op: str, field: str) -> str:
    if field == "duration_days":
        return "longest" if op == "max" else "shortest"
    if field == "max_altitude_m":
        return "highest" if op == "max" else "lowest"
    return "highest" if op == "max" else "lowest"