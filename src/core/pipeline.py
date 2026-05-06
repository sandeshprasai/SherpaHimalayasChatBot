import os
import json
from config.settings import settings
from core.schema import QueryRequest, QueryResponse, QueryIntent, RetrievalStrategy, SourceNode
from core.router import Router
from understanding.normalizer import normalize
from understanding.classifier import classify
from understanding.entity_resolver import EntityResolver
from generation.generator import Generator, FALLBACK_MESSAGE
from generation.prompt_builder import PromptBuilder

from core.schema import (
    QueryRequest, QueryResponse,
    QueryIntent, RetrievalStrategy, SourceNode, RoutingDecision # Added here
)

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
        normalized = normalize(user_query)
        req = QueryRequest(text=user_query, normalized=normalized)
        classify(req)
        
        if req.needs_entity:
            req.entities = self._resolver.resolve(normalized)
        else:
            req.entities = []

        decision = self._router.route(req)
        self._log(f"Routing: {decision.reason}")

        source_nodes = []
        meta_answer = None

        # ── 5. Generalized Retrieval ───────────────────────────────────────
        if decision.use_metadata and not decision.use_vector:
            try:
                meta_answer = self._get_meta().answer(req)
            except Exception as e: self._log(f"Meta failed: {e}")

        elif decision.use_metadata and decision.use_vector:
            try:
                source_nodes, _ = self._get_hybrid().search(req)
            except Exception as e: self._log(f"Hybrid failed: {e}")

        elif decision.use_full_doc:
            doc_text = self._fetch_full_doc(req)
            if doc_text:
                source_nodes = [SourceNode(text=doc_text, title="Itinerary")]

        else:
            try:
                source_nodes = self._get_vec().search(normalized)
            except Exception as e: self._log(f"Vector failed: {e}")

        # ── 5b. Entity Injection (Solves "No information found") ───────────
        if req.entities and (not source_nodes or req.intent in [QueryIntent.COMPARE, QueryIntent.LOOKUP]):
            for ent_dict in req.entities:
                if not any(n.title == ent_dict.get("title") for n in source_nodes):
                    source_nodes.append(SourceNode(
                        text=self._format_full_doc(ent_dict),
                        title=ent_dict.get("title")
                    ))

        # ── 6. LLM Generation ──────────────────────────────────────────────
        if decision.metadata_only and meta_answer:
            return QueryResponse(answer=meta_answer, intent=req.intent, used_llm=False)

        if not source_nodes and not meta_answer:
            return QueryResponse(answer=FALLBACK_MESSAGE, intent=req.intent, is_fallback=True, used_llm=False)

        prompt = self._builder.build(req, source_nodes)
        gen = self._get_generator()
        answer, tokens = gen.generate(prompt)
        self._used_tokens += tokens

        return QueryResponse(
            answer=answer, 
            intent=req.intent, 
            sources=source_nodes, 
            tokens_used=tokens, 
            used_llm=True, 
            is_fallback=gen.is_fallback(answer)
        )

    def _fetch_full_doc(self, req: QueryRequest) -> str | None:
        if not req.entities: return None
        return self._format_full_doc(req.entities[0])

    @staticmethod
    def _format_full_doc(data: dict) -> str:
        if not isinstance(data, dict): return "Data unavailable."
        lines = [f"TITLE: {data.get('title', 'Unknown')}"]
        for f, l in [("duration_days", "DURATION"), ("max_altitude_m", "ALTITUDE"), ("difficulty", "DIFFICULTY"), ("summary", "SUMMARY")]:
            if data.get(f): lines.append(f"{l}: {data[f]}")
        
        attrs = data.get("attributes", {})
        if isinstance(attrs, dict):
            for k, v in attrs.items(): lines.append(f"{k.upper()}: {v}")
            
        highlights = data.get("highlights", [])
        if highlights: lines.append("HIGHLIGHTS: " + " | ".join(map(str, highlights)))
        
        itin = data.get("itinerary", [])
        if itin:
            lines.append("ITINERARY:")
            for d in itin:
                if isinstance(d, dict): lines.append(f"  Day {d.get('day')}: {d.get('title')} - {d.get('description')}")
        return "\n".join(lines)

    # ... keep lazy loaders and _log as they are ...
    def _get_generator(self):
        if not self._generator: self._generator = Generator()
        return self._generator
    def _get_vec(self):
        if not self._vec: from retrieval.vector import VectorRetriever; self._vec = VectorRetriever()
        return self._vec
    def _get_meta(self):
        if not self._meta: from retrieval.metadata import MetadataRetriever; self._meta = MetadataRetriever()
        return self._meta
    def _get_hybrid(self):
        if not self._hybrid: from retrieval.hybrid import HybridRetriever; self._hybrid = HybridRetriever()
        return self._hybrid
    def _log(self, msg):
        if self.debug: print(f"  [pipeline] {msg}")