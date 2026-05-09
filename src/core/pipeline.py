import os
import json
from config.settings import settings
from core.schema import (
    QueryRequest, QueryResponse,
    QueryIntent, RetrievalStrategy, SourceNode, RoutingDecision 
)
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
        """
        The single, unified entry point for the pipeline.
        """
        # 1. Normalize and Classify
        normalized = normalize(user_query)
        req = QueryRequest(text=user_query, normalized=normalized)
        classify(req)
        
        # 2. Resolve entities if the classifier says we need them
        if req.needs_entity:
            req.entities = self._resolver.resolve(normalized)
        else:
            req.entities = []

        # 3. Route the request
        decision = self._router.route(req)
        self._log(f"Routing: {decision.reason}")

        source_nodes = []
        meta_answer = None

        # 4. Retrieval logic based on Router decision
        if decision.use_metadata and not decision.use_vector:
            try:
                meta_answer = self._get_meta().answer(req)
            except Exception as e: 
                self._log(f"Meta failed: {e}")

        elif decision.use_metadata and decision.use_vector:
            try:
                # Hybrid search
                source_nodes, _ = self._get_hybrid().search(req)
            except Exception as e: 
                self._log(f"Hybrid failed: {e}")

        elif decision.use_full_doc:
            doc_text = self._fetch_full_doc(req)
            if doc_text:
                source_nodes = [SourceNode(text=doc_text, title="Itinerary")]

        else:
            try:
                source_nodes = self._get_vec().search(normalized)
            except Exception as e: 
                self._log(f"Vector failed: {e}")

        # 5. Entity Injection Fallback (Fix 5b)
        # If we have specific entities, we inject their full data sheets into the LLM context.
        # This ensures the LLM has all the facts even if vector search was fuzzy.
        if req.entities:
            injected_nodes = []
            for ent in req.entities:
                # Check if this entity's title is already represented in source_nodes
                if not any(n.title == ent.get("title") for n in source_nodes):
                    injected_nodes.append(SourceNode(
                        text=self._format_full_doc(ent),
                        title=ent.get("title")
                    ))
            # Put injected high-quality metadata at the front of the context
            source_nodes = injected_nodes + source_nodes
            self._log(f"Injected {len(injected_nodes)} entity records into context.")

        # 6. Check for Metadata-Only exit
        if decision.metadata_only and meta_answer:
            return QueryResponse(answer=meta_answer, intent=req.intent, used_llm=False)

        # 7. Check for Fallback (No context found)
        if not source_nodes and not meta_answer:
            return QueryResponse(answer=FALLBACK_MESSAGE, intent=req.intent, is_fallback=True, used_llm=False)

        # 8. LLM Generation
        prompt = self._builder.build(req, source_nodes)
        gen = self._get_generator()
        
        # Token Budget Check
        estimated = gen.count_tokens(user_query) + settings.estimated_resp_tokens
        if self._used_tokens + estimated > settings.daily_token_limit:
            return QueryResponse(answer="🚫 Daily token limit reached.", is_fallback=True, used_llm=False)

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
        """Helper to get the first resolved entity as a full text block."""
        if not req.entities: 
            return None
        return self._format_full_doc(req.entities[0])

    @staticmethod
    def _format_full_doc(data: dict) -> str:
        """
        Generalized formatter: Loops through all JSON keys to build LLM context.
        """
        if not isinstance(data, dict): 
            return "Data unavailable."
            
        lines = [f"TITLE: {data.get('title', 'Unknown')}"]
        
        # Internal keys to skip
        blacklist = ["_source", "id", "aliases", "embedding_chunks"]

        for key, value in data.items():
            if key in blacklist or not value:
                continue
            
            label = key.replace("_", " ").upper()
            
            if isinstance(value, list):
                # Highlights, Inclusions, Tags
                lines.append(f"{label}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                continue # Nested dicts handled below for itinerary
            else:
                lines.append(f"{label}: {value}")

        # Specific formatting for Itinerary
        if "itinerary" in data:
            lines.append("ITINERARY:")
            for d in data["itinerary"]:
                if isinstance(d, dict):
                    lines.append(f"  Day {d.get('day', '?')}: {d.get('title', '')} - {d.get('description', '')}")

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
    def _get_hybrid(self):
        if not self._hybrid: 
            from retrieval.hybrid import HybridRetriever
            self._hybrid = HybridRetriever()
        return self._hybrid

    def _log(self, msg: str):
        if self.debug: print(f"  [pipeline] {msg}")