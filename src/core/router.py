"""
core/router.py
───────────────
Takes a classified QueryRequest and decides:
  1. Which retriever to use
  2. Whether to skip the LLM entirely (metadata-only answer)
  3. Whether to fetch a full document (summarize)

This is the single decision point for the entire pipeline.
Nothing here does any actual retrieval or generation.

Usage:
    from core.router import Router
    from core.schema import QueryRequest, RoutingDecision

    router = Router()
    decision = router.route(req)
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

from core.schema import QueryRequest, QueryIntent, RetrievalStrategy
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING DECISION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    Output of the router. Tells the pipeline exactly what to do.
    """
    strategy:         RetrievalStrategy

    # True  → run metadata retriever, skip LLM if it returns an answer
    use_metadata:     bool = False

    # True  → fetch the full document JSON by entity name
    use_full_doc:     bool = False

    # True  → run vector search (may be combined with metadata)
    use_vector:       bool = True

    # True  → metadata answer is sufficient; do not call LLM
    metadata_only:    bool = False

    # Human-readable explanation (shown in debug mode)
    reason:           str  = ""


# ──────────────────────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────────────────────

class Router:
    """
    Stateless router — no retrieval happens here.
    Instantiate once and reuse.
    """

    def route(self, req: QueryRequest) -> RoutingDecision:
        """
        Inspect the classified QueryRequest and return a RoutingDecision.
        """
        intent   = req.intent
        strategy = req.strategy

        # ── AGGREGATE ────────────────────────────────────────────────────────
        # "Which trek has the highest altitude?"
        # Metadata can answer this entirely — no LLM needed.
        if intent == QueryIntent.AGGREGATE:
            if strategy == RetrievalStrategy.VECTOR:
                # "most popular" — no scalar field, use vector + LLM
                return RoutingDecision(
                    strategy     = RetrievalStrategy.VECTOR,
                    use_vector   = True,
                    use_metadata = False,
                    metadata_only= False,
                    reason       = "aggregate/vector: no scalar field available",
                )
            return RoutingDecision(
                strategy     = RetrievalStrategy.METADATA,
                use_metadata = True,
                use_vector   = False,
                metadata_only= True,
                reason       = "aggregate: answered directly from metadata index",
            )

        # ── COMPARE ──────────────────────────────────────────────────────────
        # "Compare EBC vs Annapurna Base Camp Trek"
        # Metadata can build a comparison table — no LLM needed
        # IF both entities were resolved. Otherwise fall back to vector.
        if intent == QueryIntent.COMPARE:
            if len(req.entities) >= 2:
                return RoutingDecision(
                    strategy     = RetrievalStrategy.METADATA,
                    use_metadata = True,
                    use_vector   = False,
                    metadata_only= True,
                    reason       = "compare: both entities resolved, metadata sufficient",
                )
            return RoutingDecision(
                strategy     = RetrievalStrategy.VECTOR,
                use_metadata = False,
                use_vector   = True,
                metadata_only= False,
                reason       = "compare: entities not resolved, falling back to vector",
            )

        # ── SUMMARIZE ────────────────────────────────────────────────────────
        # "Describe the full itinerary of Langtang Valley Trek"
        # Fetch the full document, then let LLM format it.
        if intent == QueryIntent.SUMMARIZE:
            return RoutingDecision(
                strategy     = RetrievalStrategy.FULL_DOC,
                use_full_doc = True,
                use_vector   = False,
                use_metadata = False,
                metadata_only= False,
                reason       = "summarize: full document fetch for itinerary formatting",
            )

        # ── FILTER ───────────────────────────────────────────────────────────
        # "Suggest a 7-day trek in Annapurna"
        # Hybrid: metadata filters candidates, vector reranks by relevance.
        # Metadata listing is shown directly; LLM adds nuance if needed.
        if intent == QueryIntent.FILTER:
            return RoutingDecision(
                strategy     = RetrievalStrategy.HYBRID,
                use_metadata = True,
                use_vector   = True,
                metadata_only= True,   # metadata listing is usually sufficient
                reason       = "filter: hybrid retrieval, metadata listing shown directly",
            )

        # ── LOOKUP / GENERAL ─────────────────────────────────────────────────
        # "What is included in Ghorepani Circuit Trek?"
        # Pure vector search + LLM generation.
        return RoutingDecision(
            strategy     = RetrievalStrategy.VECTOR,
            use_metadata = False,
            use_vector   = True,
            metadata_only= False,
            reason       = f"{intent.value}: standard vector search + LLM",
        )


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from core.schema import QueryRequest, QueryIntent, RetrievalStrategy

    router = Router()

    cases = [
        # (description, intent, strategy, entities)
        ("Highest altitude trek",
         QueryIntent.AGGREGATE, RetrievalStrategy.METADATA, []),

        ("Most popular trek (no scalar field)",
         QueryIntent.AGGREGATE, RetrievalStrategy.VECTOR, []),

        ("Compare EBC vs ABC (entities resolved)",
         QueryIntent.COMPARE, RetrievalStrategy.METADATA,
         ["Everest Base Camp Trek", "Annapurna Base Camp Trek"]),

        ("Compare vague (entities not resolved)",
         QueryIntent.COMPARE, RetrievalStrategy.METADATA, []),

        ("Full itinerary summarize",
         QueryIntent.SUMMARIZE, RetrievalStrategy.FULL_DOC, []),

        ("Filter 7-day Annapurna",
         QueryIntent.FILTER, RetrievalStrategy.HYBRID, []),

        ("Lookup what is included",
         QueryIntent.LOOKUP, RetrievalStrategy.VECTOR, []),

        ("General question",
         QueryIntent.GENERAL, RetrievalStrategy.VECTOR, []),
    ]

    print("\n🗺️  Router Test\n" + "─" * 65)
    for desc, intent, strategy, entities in cases:
        req          = QueryRequest(text=desc, intent=intent, strategy=strategy)
        req.entities = entities
        decision     = router.route(req)
        print(f"Q : {desc}")
        print(f"   strategy={decision.strategy.value:<10} "
              f"meta_only={decision.metadata_only}  "
              f"full_doc={decision.use_full_doc}")
        print(f"   reason: {decision.reason}")
        print()