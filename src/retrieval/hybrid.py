"""
retrieval/hybrid.py
────────────────────
Hybrid retrieval: metadata pre-filter → vector rerank.

KEY CHANGES vs original:
  - Filter results are reranked by vector semantic score, not sorted by
    duration. So "5 treks with beautiful Himalaya views" returns semantically
    matched results, not just the 5 shortest treks alphabetically.
  - req.top_k is honored (extracted by LLM classifier from "find me 5" etc.)
  - Falls back to metadata-only listing if vector retriever is unavailable.

Usage:
    from retrieval.hybrid import HybridRetriever
    from core.schema import QueryRequest

    retriever = HybridRetriever()
    nodes, answer = retriever.search(req)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import QueryRequest, SourceNode


class HybridRetriever:
    """
    Combines metadata filtering with vector similarity reranking.
    Instantiate once and reuse.
    """

    def __init__(self):
        self._meta = None
        self._vec  = None

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        req: QueryRequest,
        top_k: int | None = None,
    ) -> tuple[list[SourceNode], str | None]:
        """
        Returns (source_nodes, metadata_answer).

        Flow:
          1. Apply scalar filters from req.filters (region, duration, difficulty)
          2. Run vector search on the full query
          3. Rerank filtered candidates by vector semantic score
          4. Format the top-k reranked results as the metadata_answer string
        """
        self._ensure_ready()

        # Honor top_k extracted by classifier ("find me 5", "top 3") with
        # a default of 5 when not specified.
        limit = req.top_k or top_k or 5

        # ── Step 1: Apply scalar filters from req.filters ─────────────────
        candidates = self._apply_filters(req)

        # ── Step 2: Vector-rerank filtered candidates ─────────────────────
        query_text = req.normalized or req.text
        try:
            # Fetch enough candidates for reranking
            fetch_k   = max(limit * 4, 20)
            vec_nodes = self._vec.search(query_text, top_k=fetch_k)

            # Build title → score map from vector results
            score_map: dict[str, float] = {}
            for n in vec_nodes:
                title = (n.title or "").lower().strip()
                if title and title not in score_map:
                    score_map[title] = n.score or 0.0

            # Sort candidates by vector semantic score (descending)
            candidates.sort(
                key=lambda r: score_map.get((r.get("title") or "").lower().strip(), 0.0),
                reverse=True,
            )
        except Exception:
            # Vector unavailable — fall back to duration sort (original behavior)
            candidates.sort(key=lambda r: r.get("duration_days") or 999)

        # ── Step 3: Take top-k reranked and format answer ─────────────────
        top          = candidates[:limit]
        meta_answer  = self._meta._format_filter_results(top, len(candidates), limit)

        return [], meta_answer

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_ready(self):
        if self._meta is not None:
            return
        from retrieval.metadata import MetadataRetriever
        from retrieval.vector import VectorRetriever
        self._meta = MetadataRetriever()
        self._vec  = VectorRetriever()

    def _apply_filters(self, req: QueryRequest) -> list[dict]:
        """
        Apply scalar filters from req.filters to the metadata records.
        Returns all matching records (no limit applied here).
        """
        records = list(self._meta._records)
        filters = req.filters

        if region := filters.get("region"):
            records = [
                r for r in records
                if region.lower() in (r.get("region") or "").lower()
            ]

        if days := filters.get("duration_days"):
            records = [
                r for r in records
                if r.get("duration_days") is not None
                and r["duration_days"] > 0
                and abs(r["duration_days"] - days) <= 3
            ]

        if filters.get("beginner_friendly"):
            friendly = {"easy", "moderate", "easy/moderate"}
            records = [
                r for r in records
                if (r.get("difficulty") or "").lower() in friendly
            ]

        if cat := filters.get("category"):
            records = [
                r for r in records
                if (r.get("category") or "").lower() == str(cat).lower()
            ]

        return records