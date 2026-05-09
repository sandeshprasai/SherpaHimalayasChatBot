import os
from typing import Any
from core.schema import QueryRequest, QueryResponse, SourceNode, RetrievalStrategy
from retrieval.metadata import MetadataRetriever
from understanding.classifier import classify
from understanding.normalizer import normalize
from generation.generator import Generator

class Pipeline:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.meta_engine = MetadataRetriever()
        self.generator = Generator()
        self._vec = None

    def _recursive_flatten(self, data: Any, depth: int = 0) -> str:
        """Recursive flattener: Handles any data depth (itineraries, specs, etc)"""
        prefix = "  " * depth
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                if k.startswith("_") or k in ["embedding_chunks", "id"]: continue
                label = k.replace("_", " ").upper()
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{label}:")
                    lines.append(self._recursive_flatten(v, depth + 1))
                else:
                    lines.append(f"{prefix}{label}: {v}")
            return "\n".join(lines)
        elif isinstance(data, list):
            items = []
            for i in data:
                if isinstance(i, (dict, list)):
                    items.append(self._recursive_flatten(i, depth + 1))
                else:
                    items.append(f"{prefix}- {i}")
            return "\n".join(items)
        return str(data)

    def run(self, user_query: str) -> QueryResponse:
        norm_text = normalize(user_query, self.meta_engine._records)
        req = QueryRequest(text=user_query, normalized=norm_text)
        classify(req, self.meta_engine.schema_keys)
        
        source_nodes = []
        strategy = RetrievalStrategy.VECTOR

        # 1. Structured Data Execution
        if req.plan.filters or req.plan.sort_field:
            strategy = RetrievalStrategy.METADATA
            raw_data = self.meta_engine.execute_plan(req.plan)
            source_nodes = [
                SourceNode(text=self._recursive_flatten(d), title=d.get("title"), source_id=d.get("id"))
                for d in raw_data
            ]

        # 2. Semantic Fallback
        if req.plan.needs_semantic_search or not source_nodes:
            if strategy == RetrievalStrategy.METADATA: strategy = RetrievalStrategy.HYBRID
            try:
                if not self._vec:
                    from retrieval.vector import VectorRetriever
                    self._vec = VectorRetriever()
                vec_results = self._vec.search(req.normalized)
                source_nodes.extend(vec_results)
            except Exception as e:
                if self.debug: print(f"  [Error] Vector search: {e}")

        if not source_nodes:
            return QueryResponse(answer="I couldn't find any relevant records.", is_fallback=True, used_llm=False)

        # 3. LLM Generation
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        prompt = builder.build(req, source_nodes)
        answer, tokens = self.generator.generate(prompt)
        
        return QueryResponse(
            answer=answer,
            sources=source_nodes,
            tokens_used=tokens,
            cost_usd=(tokens / 1_000_000) * 0.59,
            used_llm=True,
            intent=req.plan.intent,
            strategy=strategy
        )