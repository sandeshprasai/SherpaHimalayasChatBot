"""
retrieval/metadata.py
──────────────────────
Answers aggregate, filter, and compare queries directly from
metadata_index.json — no LLM or vector search needed.

KEY CHANGES vs original:
  - _aggregate() enforces per-field sanity bounds (e.g. altitude 100–9000m)
    to prevent bad JSON data values (e.g. 19175m) from surfacing as answers.
  - _filter() is now called only as fallback; the primary filter path goes
    through HybridRetriever which calls _format_filter_results() directly
    after vector reranking.
  - _format_filter_results() is extracted as a public method so HybridRetriever
    can pass its own already-ranked list.

Usage:
    from retrieval.metadata import MetadataRetriever
    from core.schema import QueryRequest, QueryIntent

    retriever = MetadataRetriever()
    answer = retriever.answer(req)   # returns str | None
"""

import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import QueryRequest, QueryIntent


# Sanity bounds for known numeric fields.
# Any value outside these bounds is treated as a data error and excluded.
_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "max_altitude_m": (100,  9000),   # Everest summit is 8849m
    "duration_days":  (1,    365),
}


class MetadataRetriever:

    def __init__(self):
        meta_path = (
            settings.cleaned_metadata_index
            if os.path.exists(settings.cleaned_metadata_index)
            else settings.metadata_index
        )
        with open(meta_path, encoding="utf-8") as f:
            all_records: list[dict] = json.load(f)

        def _looks_like_destination(rec: dict) -> bool:
            src   = (rec.get("_source") or "").lower()
            cat   = (rec.get("category") or "").lower()
            title = (rec.get("title") or "").lower()
            typ   = (rec.get("type") or "").lower()

            # Explicitly exclude non-destination types
            if typ in ("team", "company", "blog", "info"):
                return False
            if src.startswith(("team/", "company/", "blogs/", "information/")):
                return False

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

        # Filter to records with a valid scalar for this field
        candidates = [
            r for r in self._records
            if r.get(field) is not None and r.get(field) != 0
        ]

        # Apply sanity bounds — reject clearly wrong values
        lo, hi = _FIELD_BOUNDS.get(field, (float("-inf"), float("inf")))
        valid_candidates = []
        for r in candidates:
            try:
                val = float(r[field])
                if lo <= val <= hi:
                    valid_candidates.append(r)
            except (TypeError, ValueError):
                continue
        candidates = valid_candidates

        # Apply any additional filters from the request
        filters = getattr(req, "filters", {}) or {}
        if "category" in filters:
            cat = str(filters["category"]).lower()
            candidates = [r for r in candidates if (r.get("category") or "").lower() == cat]
        if "region" in filters:
            region = str(filters["region"]).lower()
            candidates = [r for r in candidates if region in (r.get("region") or "").lower()]

        if not candidates:
            return None

        top_k = getattr(req, "top_k", None) or None

        if top_k and isinstance(top_k, int) and top_k > 1:
            sorted_list = sorted(
                candidates,
                key=lambda r: (r.get(field) is None, r.get(field)),
                reverse=(op == "max"),
            )
            top_list = sorted_list[:top_k]
            label    = _field_label(field)
            unit     = _field_unit(field)
            lines    = [f"Top {len(top_list)} by {label} ({'highest' if op == 'max' else 'lowest'}):\n"]
            for i, r in enumerate(top_list, 1):
                val     = r.get(field)
                val_str = str(val) + unit if val is not None else "N/A"
                lines.append(
                    f"{i}. **{r.get('title')}** — {val_str}\n"
                    f"   - Region: {r.get('region', 'N/A')}\n"
                    f"   - Difficulty: {r.get('difficulty', 'N/A')}\n"
                    f"   - {(r.get('summary') or '')[:140]}…\n"
                )
            return "\n".join(lines)

        # Single winner
        winner = (
            max(candidates, key=lambda r: r[field])
            if op == "max"
            else min(candidates, key=lambda r: r[field])
        )

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
        rank  = settings.difficulty_rank
        valid = [
            r for r in self._records
            if r.get("difficulty") and r["difficulty"].lower() in rank
        ]
        if not valid:
            return None

        key_fn = lambda r: rank.get(r["difficulty"].lower(), 99)
        winner = min(valid, key=key_fn) if op == "min" else max(valid, key=key_fn)
        adj    = "easiest" if op == "min" else "most difficult"

        return (
            f"**{winner['title']}** is the {adj} option available.\n\n"
            f"- **Difficulty:** {winner.get('difficulty', 'N/A')}\n"
            f"- **Duration:** {winner.get('duration_days', 'N/A')} days\n"
            f"- **Region:** {winner.get('region', 'N/A')}\n\n"
            f"{winner.get('summary', '')}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # FILTER  (used as fallback when HybridRetriever is unavailable)
    # ──────────────────────────────────────────────────────────────────────

    def _filter(self, req: QueryRequest) -> str:
        records = list(self._records)
        filters = req.filters

        if cat := filters.get("category"):
            records = [r for r in records if (r.get("category") or "").lower() == str(cat).lower()]

        if filters.get("is_trek") is not None:
            val     = bool(filters.get("is_trek"))
            records = [r for r in records if bool(r.get("is_trek")) == val]

        if region := filters.get("region"):
            records = [r for r in records if region.lower() in (r.get("region") or "").lower()]

        if days := filters.get("duration_days"):
            records = [
                r for r in records
                if r.get("duration_days") is not None
                and r["duration_days"] > 0
                and abs(r["duration_days"] - days) <= 3
            ]

        if filters.get("beginner_friendly"):
            friendly = {"easy", "moderate", "easy/moderate"}
            records  = [r for r in records if (r.get("difficulty") or "").lower() in friendly]

        if not records:
            return (
                f"I couldn't find any trips matching those exact criteria. "
                f"Try adjusting the duration or region, or contact us at "
                f"**{settings.company_email}** for a custom itinerary."
            )

        records.sort(key=lambda r: r.get("duration_days") or 999)
        limit = req.top_k or 5
        return self._format_filter_results(records[:limit], len(records), limit)

    # ──────────────────────────────────────────────────────────────────────
    # FORMAT FILTER RESULTS  (called by both _filter and HybridRetriever)
    # ──────────────────────────────────────────────────────────────────────

    def _format_filter_results(
        self,
        top: list[dict],
        total: int,
        limit: int,
    ) -> str:
        """
        Format a list of already-ranked records into a display string.
        Called by _filter() (metadata-only path) and HybridRetriever (after
        vector reranking).
        """
        if not top:
            return (
                f"I couldn't find any trips matching those criteria. "
                f"Contact us at **{settings.company_email}** for a custom itinerary."
            )

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

        remaining = total - len(top)
        if remaining > 0:
            lines.append(
                f"\n*{remaining} more option(s) available — "
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
        name_lower  = name.lower().strip()
        best, best_score = None, 0.0

        for r in self._records:
            title       = (r.get("title") or "").lower()
            name_words  = set(w for w in name_lower.split() if len(w) > 3)
            title_words = set(w for w in title.split()      if len(w) > 3)
            if not title_words:
                continue
            overlap = len(name_words & title_words) / len(title_words)
            if overlap > best_score:
                best_score = overlap
                best       = r

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