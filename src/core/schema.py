"""
core/schema.py
──────────────
Dataclasses that flow through the entire pipeline.

    QueryRequest  →  pipeline  →  QueryResponse

Nothing here imports from other src modules — this is the
foundation every other module imports from.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# QUERY INTENT  (what kind of question is being asked?)
# ──────────────────────────────────────────────────────────────────────────────

class QueryIntent(Enum):
    LOOKUP    = "lookup"     # "What is the duration of X?"
    AGGREGATE = "aggregate"  # "Which trek has the highest altitude?"
    FILTER    = "filter"     # "Suggest a 5-day trek in Annapurna"
    COMPARE   = "compare"    # "Compare X vs Y"
    SUMMARIZE = "summarize"  # "Describe the full itinerary of X"
    GENERAL   = "general"    # Catch-all → vector search


# ──────────────────────────────────────────────────────────────────────────────
# RETRIEVAL STRATEGY  (which retriever should answer it?)
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalStrategy(Enum):
    VECTOR   = "vector"    # semantic similarity search
    METADATA = "metadata"  # flat index filter/sort
    HYBRID   = "hybrid"    # metadata filter → vector rerank
    FULL_DOC = "full_doc"  # fetch entire document by entity name


# ──────────────────────────────────────────────────────────────────────────────
# QUERY REQUEST  (input to the pipeline)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryRequest:
    """
    Raw user query plus all understanding-layer annotations.
    Populated progressively as the query passes through the pipeline.
    """

    # ── Raw input ──────────────────────────────────────────────────────────
    text:        str                        # original user message

    # ── Understanding layer fills these in ────────────────────────────────
    intent:      QueryIntent       = QueryIntent.GENERAL
    strategy:    RetrievalStrategy = RetrievalStrategy.VECTOR

    # Normalised query text (e.g. "EBC" → "Everest Base Camp Trek")
    normalized:  str               = ""

    # Resolved entity names extracted from the query
    entities:    list[str]         = field(default_factory=list)

    # Scalar filters for metadata retrieval
    # e.g. {"region": "Annapurna", "duration_days": 7, "beginner_friendly": True}
    filters:     dict[str, Any]    = field(default_factory=dict)

    # Aggregate operation hint  ("max" | "min")
    agg_op:      str | None        = None

    # Field to aggregate on  ("duration_days" | "max_altitude_m" | "difficulty")
    agg_field:   str | None        = None

    def __post_init__(self):
        # Default normalized to original text if not set
        if not self.normalized:
            self.normalized = self.text


# ──────────────────────────────────────────────────────────────────────────────
# SOURCE NODE  (one retrieved chunk / record)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SourceNode:
    """A single piece of retrieved context."""
    text:       str
    score:      float | None  = None   # similarity score (vector) or None (metadata)
    source_id:  str | None    = None   # document id
    title:      str | None    = None   # human-readable title


# ──────────────────────────────────────────────────────────────────────────────
# QUERY RESPONSE  (output of the pipeline)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResponse:
    """
    Final response returned to the API / chat loop.
    """
    answer:    str                        # text shown to the user
    intent:    QueryIntent = QueryIntent.GENERAL
    strategy:  RetrievalStrategy = RetrievalStrategy.VECTOR
    sources:   list[SourceNode] = field(default_factory=list)

    # Token accounting
    tokens_used: int   = 0
    cost_usd:    float = 0.0

    # Was this answered by metadata (no LLM call) or required LLM?
    used_llm:  bool = True

    # True if we hit the fallback (LLM couldn't answer from context)
    is_fallback: bool = False