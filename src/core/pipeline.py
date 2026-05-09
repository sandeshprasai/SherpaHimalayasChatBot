import os
import json
from typing import Any, List, Optional # <--- FIXED: Added Any here
from config.settings import settings
from core.schema import QueryRequest, QueryResponse, SourceNode, RoutingDecision
from core.router import Router
from understanding.normalizer import normalize
from understanding.classifier import classify
from understanding.entity_resolver import EntityResolver
from generation.generator import Generator, FALLBACK_MESSAGE
from generation.prompt_builder import PromptBuilder

class Pipeline:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._router = Router()
        self._builder = PromptBuilder()
        self._resolver = EntityResolver()
        self._generator = None
        self._vec = None
        self._meta = None
        self._hybrid = None
        self._used_tokens = 0

    def run(self, user_query: str) -> QueryResponse:
        # 1. Normalize and Classify (Generates data_plan)
        normalized = normalize(user_query)
        req = QueryRequest(text=user_query, normalized=normalized)
        classify(req)
        
        # 2. Resolve entities if needed
        if req.needs_entity:
            req.entities = self._resolver.resolve(normalized)
        
        # 3. Route the request
        decision = self._router.route(req)
        self._log(f"Routing Decision: {decision.reason}")

        source_nodes = []
        
        # 4. Retrieval based on the Plan
        # If the LLM data_plan has filters or sorts, use Metadata Engine
        plan = getattr(req, 'data_plan', {})
        
        if plan.get("filters") or plan.get("sort"):
            try:
                raw_results = self._get_meta().execute_plan(plan)
                source_nodes = [
                    SourceNode(
                        text=self._universal_formatter(r), 
                        title=r.get("title")
                    ) for r in raw_results
                ]
            except Exception as e:
                self._log(f"Metadata execution failed: {e}")

        # 5. Semantic Fallback (Vector Search)
        # If metadata found nothing or it's a general question, use Vector
        if not source_nodes:
            try:
                source_nodes = self._get_vec().search(normalized)
            except Exception as e:
                self._log(f"Vector search failed: {e}")

        # 6. Entity Injection (Guarantee specific data is present)
        if req.entities:
            for ent in req.entities:
                if not any(n.title == ent.get("title") for n in source_nodes):
                    source_nodes.insert(0, SourceNode(
                        text=self._universal_formatter(ent),
                        title=ent.get("title")
                    ))

        # 7. LLM Generation
        if not source_nodes:
            return QueryResponse(answer=FALLBACK_MESSAGE, is_fallback=True)

        prompt = self._builder.build(req, source_nodes)
        gen = self._get_generator()
        answer, tokens = gen.generate(prompt)
        self._used_tokens += tokens

        return QueryResponse(
            answer=answer,
            sources=source_nodes,
            tokens_used=tokens,
            is_fallback=gen.is_fallback(answer)
        )

    @staticmethod
    def _universal_formatter(data: Any, depth=0) -> str:
        """Recursively flattens ANY dictionary or list into a readable string."""
        if not isinstance(data, (dict, list)):
            return str(data)

        lines = []
        prefix = "  " * depth
        
        if isinstance(data, dict):
            for k, v in data.items():
                # Skip heavy/internal fields
                if k in ["_source", "id", "embedding_chunks", "aliases"]: 
                    continue
                
                label = f"{prefix}{k.replace('_', ' ').upper()}:"
                
                if isinstance(v, (dict, list)):
                    lines.append(f"{label}")
                    lines.append(Pipeline._universal_formatter(v, depth + 1))
                else:
                    lines.append(f"{label} {v}")
        
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(Pipeline._universal_formatter(item, depth + 1))
                else:
                    lines.append(f"{prefix}- {item}")
                    
        return "\n".join(lines)

    # ── Lazy Loaders ───────────────────────────────────────────────────
    def _get_generator(self):
        if not self._generator: self._generator = Generator()
        return self._generator
    def _get_vec(self):
        if not self._vec: 
            from retrieval.vector import VectorRetriever
            self._vec = VectorRetriever()
        return self._vec
    def _get_meta(self):
        if not self._meta: 
            from retrieval.metadata import MetadataRetriever
            self._meta = MetadataRetriever()
        return self._meta

    def _log(self, msg: str):
        if self.debug: print(f"  [pipeline] {msg}")