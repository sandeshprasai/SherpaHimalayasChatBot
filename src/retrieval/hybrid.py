"""
retrieval/hybrid.py
────────────────────
Hybrid retrieval: metadata pre-filter → vector rerank.

Used for FILTER / recommendation queries where the user specifies
constraints (region, duration, difficulty) AND wants semantically
relevant results within those constraints.

Flow:
  1. MetadataRetriever filters the flat index by scalar constraints
     → returns candidate record IDs
  2. VectorRetriever runs semantic search on the full query
     → returns top-k chunks
  3. Hybrid merges: only keep vector results whose doc ID
     appears in the metadata candidates
  4. Falls back to metadata-only results if vector finds nothing

Usage:
    from retrieval.hybrid import HybridRetriever
    from core.schema import QueryRequest, QueryIntent, RetrievalStrategy

    retriever = HybridRetriever()
    nodes, answer = retriever.search(req)
"""

import os
import sys

# ── path setup ───────────────────────────────────────────────────────────────
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
        # Lazy imports — heavy deps loaded on first search()
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

        - source_nodes: vector chunks filtered to metadata candidates
          (used as LLM context if metadata_answer is None)
        - metadata_answer: pre-formatted string from metadata layer
          (not None when metadata alone is sufficient — e.g. filter listing)
        """
        self._ensure_ready()
        k = top_k or settings.similarity_top_k

        # ── Step 1: metadata filter ────────────────────────────────────────
        candidate_ids = self._get_candidate_ids(req)
        meta_answer   = self._meta.answer(req)    # may be a formatted listing

        # ── Step 2: vector search ──────────────────────────────────────────
        vec_nodes = self._vec.search(req.normalized or req.text, top_k=k * 2)

        # ── Step 3: rerank — prefer nodes whose doc is in candidates ───────
        if candidate_ids:
            prioritized = [n for n in vec_nodes if n.source_id in candidate_ids]
            remainder   = [n for n in vec_nodes if n.source_id not in candidate_ids]
            merged = (prioritized + remainder)[:k]
        else:
            # No scalar constraints → just return top-k vector results
            merged = vec_nodes[:k]

        return merged, meta_answer

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

    def _get_candidate_ids(self, req: QueryRequest) -> set[str]:
        """
        Run metadata filters and return matching document IDs.
        Returns empty set if no scalar filters are set.
        """
        filters = req.filters
        if not filters:
            return set()

        records = list(self._meta._records)

        if region := filters.get("region"):
            records = [
                r for r in records
                if region.lower() in (r.get("region") or "").lower()
            ]

        if days := filters.get("duration_days"):
            records = [
                r for r in records
                if r.get("duration_days") is not None
                and abs(r["duration_days"] - days) <= 2
            ]

        if filters.get("beginner_friendly"):
            friendly = {"easy", "moderate", "easy/moderate"}
            records = [
                r for r in records
                if (r.get("difficulty") or "").lower() in friendly
            ]

        return {r["id"] for r in records if r.get("id")}