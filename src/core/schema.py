from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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
class DataExecutionPlan:
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_field: Optional[str] = None
    sort_direction: str = "asc"
    limit: int = 5
    needs_semantic_search: bool = True
    intent: QueryIntent = QueryIntent.GENERAL
    reasoning: str = ""

@dataclass
class QueryRequest:
    text: str
    normalized: str = ""
    plan: DataExecutionPlan = field(default_factory=DataExecutionPlan)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    # Compatibility properties
    @property
    def intent(self): return self.plan.intent

@dataclass
class SourceNode:
    text: str
    title: Optional[str] = None
    score: Optional[float] = None
    source_id: Optional[str] = None

@dataclass
class QueryResponse:
    answer: str
    sources: List[SourceNode] = field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    used_llm: bool = True
    is_fallback: bool = False
    intent: QueryIntent = QueryIntent.GENERAL
    strategy: RetrievalStrategy = RetrievalStrategy.VECTOR