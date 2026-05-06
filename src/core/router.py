# --- FILE: src/core/router.py ---
import os
import sys

from core.schema import QueryRequest, QueryIntent, RetrievalStrategy, RoutingDecision

class Router:
    """
    Stateless router — decides which tool to use based on intent.
    """
    def route(self, req: QueryRequest) -> RoutingDecision:
        intent = req.intent

        # 1. Comparison: Metadata for facts + LLM for narration
        if intent == QueryIntent.COMPARE:
            return RoutingDecision(
                strategy=RetrievalStrategy.METADATA,
                use_metadata=True, 
                use_vector=False,
                metadata_only=False,
                reason="Comparison: Fetching metadata for both entities"
            )

        # 2. Recommendation: Hybrid filtering + LLM narration
        if intent == QueryIntent.FILTER:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID,
                use_metadata=True, 
                use_vector=True,
                metadata_only=False,
                reason="Recommendation: Hybrid filtering + LLM explanation"
            )

        # 3. Superlatives (Math): Direct metadata answer
        if intent == QueryIntent.AGGREGATE:
            return RoutingDecision(
                strategy=RetrievalStrategy.METADATA,
                use_metadata=True, 
                use_vector=False,
                metadata_only=True,
                reason="Aggregate: Direct calculation from metadata"
            )

        # 4. Itinerary requests: Full Document
        if intent == QueryIntent.SUMMARIZE:
            return RoutingDecision(
                strategy=RetrievalStrategy.FULL_DOC,
                use_full_doc=True,
                use_vector=False,
                reason="Summarize: Full document fetch for itinerary"
            )

        # 5. Default: Standard Vector RAG
        return RoutingDecision(
            strategy=RetrievalStrategy.VECTOR,
            use_vector=True, 
            use_metadata=False,
            metadata_only=False,
            reason="General: Standard semantic search"
        )