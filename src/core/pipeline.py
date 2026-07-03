import os
import re
import time
from typing import Any
from src.core.schema import QueryRequest, QueryResponse, SourceNode, RetrievalStrategy, QueryIntent

from src.retrieval.metadata import MetadataRetriever
from src.understanding.classifier import UniversalClassifier
from src.understanding.normalizer import DynamicNormalizer
from src.generation.generator import Generator
from src.config.settings import settings
from src.generation.response_normalizer import ResponseNormalizer
from src.services.logger import logger
from src.services.memory import get_history, save_history

class Pipeline:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.meta_engine = MetadataRetriever()
        self.generator = Generator()
        self.response_normalizer = ResponseNormalizer()
        self._vec = None
        
        self.normalizer = DynamicNormalizer(self.meta_engine._records)
        self.classifier = UniversalClassifier(self.meta_engine.schema_keys)

    def _recursive_flatten(self, data: Any, depth: int = 0) -> str:
        """Flattens dicts into text, truncating massive fields to save tokens."""
        prefix = "  " * depth
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                if k.startswith("_") or k in ["embedding_chunks", "embedding", "content"]:
                    continue
                label = k.replace("_", " ").upper()
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{label}:")
                    lines.append(self._recursive_flatten(v, depth + 1))
                else:
                    
                    val_str = str(v)
                    if len(val_str) > 600 and k not in ["id", "title"]:
                        val_str = val_str[:600] + "... [truncated]"
                    lines.append(f"{prefix}{label}: {val_str}")
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

    def _sanitize_query(self, query: str) -> str:
        if not query: return ""
        sanitized = query[:1000].strip().replace("`", "'").replace("---", " - ")
        sanitized = re.sub(r"<[^>]*>", "", sanitized)
        suspicious_patterns = [
            r"ignore\s+all\s+previous", r"ignore\s+above\s+instructions",
            r"you\s+are\s+now\s+a", r"system\s+prompt"
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning(f"SECURITY_ALERT: Blocked prompt injection.")
                return "suspicious_query_blocked"
        return sanitized

    async def run(self, user_query: str, session_id: str = None) -> QueryResponse:
        t_start = time.time()
        clean_query = self._sanitize_query(user_query)
        
        if clean_query == "suspicious_query_blocked":
            return QueryResponse(answer="Request blocked.", is_fallback=True, used_llm=False)

        history = await get_history(session_id)
        norm_text = self.normalizer.normalize(clean_query)
        req = QueryRequest(text=clean_query, normalized=norm_text)
    
        is_exact_match = False
        norm_lower = norm_text.lower()
        
        for rec in self.meta_engine._records:
            rec_id = str(rec.get("id", "")).lower()
            if rec_id and rec_id in norm_lower and len(rec_id) > 4:
                req.plan.intent = QueryIntent.LOOKUP
                req.plan.filters = {"id": rec.get("id")}
                req.plan.needs_semantic_search = False
                is_exact_match = True
                break

        if not is_exact_match:
            await self.classifier.classify(req, history=history)

        t_class = time.time()
        if self.debug: print(f"CLASSIFICATION: {round(t_class - t_start, 2)}s")

        source_nodes = []
        strategy = RetrievalStrategy.VECTOR

       
        if req.plan.intent in [QueryIntent.COMPARE, QueryIntent.AGGREGATE, QueryIntent.LOOKUP] or req.plan.filters:
            strategy = RetrievalStrategy.METADATA
            raw_data = self.meta_engine.execute_plan(req.plan)
            
           
            raw_data = raw_data[:2] 
            
            source_nodes = [
                SourceNode(
                    text=self._recursive_flatten(d),
                    metadata=d,
                    title=str(d.get("title") or d.get("name") or d.get("id")),
                    source_id=str(d.get("id") or d.get("source_id"))
                ) for d in raw_data
            ]

        if not source_nodes or (req.plan.needs_semantic_search and not is_exact_match):
            strategy = RetrievalStrategy.HYBRID if source_nodes else RetrievalStrategy.VECTOR
            try:
                if not self._vec:
                    from src.retrieval.vector import VectorRetriever
                    self._vec = VectorRetriever()
                
                vec_results = await self._vec.search(req.plan.search_query)
                existing_titles = [str(n.title).lower() for n in source_nodes if n.title]
                
                for vn in vec_results:
                    if str(vn.title).lower() not in existing_titles:
                        source_nodes.append(vn)
            except Exception as e:
                logger.error(f"Vector search error: {e}")

        t_ret = time.time()
        if self.debug: print(f"RETRIEVAL: {round(t_ret - t_class, 2)}s")

        if not source_nodes:
            return QueryResponse(
                answer=f"I couldn't find matching info. Please contact {settings.company_email}.", 
                is_fallback=True, used_llm=False
            )

    
        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        prompt = builder.build(req, source_nodes, history=history)
        
        raw_output, tokens = await self.generator.generate(prompt)

        t_gen = time.time()
        if self.debug: print(f"GENERATION: {round(t_gen - t_ret, 2)}s")

        normalized = self.response_normalizer.normalize(raw_output)

        if normalized.get("is_fallback", False):
            final_answer = f"I don't have enough data to answer that. Please contact {settings.company_email}."
            is_fb = True
        else:
            final_answer = normalized["answer"]
            is_fb = False
            
            if session_id:
                history.append({"role": "user", "content": clean_query})
                history.append({"role": "assistant", "content": final_answer})
                await save_history(session_id, history)
        
        return QueryResponse(
            answer=final_answer, sections=normalized.get("sections", []),
            suggestions=normalized.get("suggestions", []), sources=source_nodes,
            strategy=strategy, used_llm=True, is_fallback=is_fb,
            tokens_used=tokens, intent=req.plan.intent
        )