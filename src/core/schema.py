# --- FILE: src/core/schema.py ---
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

class QueryIntent(Enum):
    LOOKUP    = "lookup"
    AGGREGATE = "aggregate"
    FILTER    = "filter"
    COMPARE   = "compare"
    SUMMARIZE = "summarize"
    GENERAL   = "general"

class RetrievalStrategy(Enum):
    VECTOR   = "vector"
    METADATA = "metadata"
    HYBRID   = "hybrid"
    FULL_DOC = "full_doc"

@dataclass
class QueryRequest:
    text:        str
    intent:      QueryIntent       = QueryIntent.GENERAL
    strategy:    RetrievalStrategy = RetrievalStrategy.VECTOR
    normalized:  str               = ""
    entities:    list[Any]         = field(default_factory=list) # Changed to Any to support dicts
    intent_confidence: float       = 1.0
    filters:     dict[str, Any]    = field(default_factory=dict)
    agg_op:      str | None        = None
    agg_field:   str | None        = None
    top_k:       int | None        = None
    needs_entity: bool             = False

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text

# --- ADD THIS CLASS ---
@dataclass
class RoutingDecision:
    strategy:         RetrievalStrategy
    use_metadata:     bool = False
    use_full_doc:     bool = False
    use_vector:       bool = True
    metadata_only:    bool = False
    reason:           str  = ""

@dataclass
class SourceNode:
    text:       str
    score:      float | None  = None
    source_id:  str | None    = None
    title:      str | None    = None

@dataclass
class QueryResponse:
    answer:    str
    intent:    QueryIntent = QueryIntent.GENERAL
    strategy:  RetrievalStrategy = RetrievalStrategy.VECTOR
    sources:   list[SourceNode] = field(default_factory=list)
    tokens_used: int   = 0
    cost_usd:    float = 0.0
    used_llm:  bool = True
    is_fallback: bool = False