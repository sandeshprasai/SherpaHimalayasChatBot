import os
import re
import json
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

        prefix = "  " * depth
        if isinstance(data, dict):
            lines = []
            for k, v in data.items():
                if k.startswith("_") or k in ["embedding_chunks", "embedding"]:
                    continue
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


    def _sanitize_query(self, query: str) -> str:

        if not query:
            return ""


        sanitized = query[:1000].strip()

        sanitized = sanitized.replace("`", "'").replace("---", " - ")


        sanitized = re.sub(r"<[^>]*>", "", sanitized)


        suspicious_patterns = [
            r"ignore\s+all\s+previous",
            r"ignore\s+above\s+instructions",
            r"you\s+are\s+now\s+a",
            r"system\s+prompt",
            r"reveal\s+your\s+instructions",
            r"forget\s+what\s+you"
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                logger.warning("prompt_injection_blocked", extra={
                    "query_preview": sanitized[:100],
                })
                return "suspicious_query_blocked"

        return sanitized

    # ─────────────────────────────────────────────────────────────────────────

    async def run(self, user_query: str, session_id: str = None) -> QueryResponse:
        t_start = time.time()

        logger.info("pipeline_run_started", extra={
            "session_id": session_id,
            "query": user_query[:200] if user_query else "",
        })

        clean_query = self._sanitize_query(user_query)

        if clean_query == "suspicious_query_blocked":
            return QueryResponse(
                answer=(
                    f"I'm sorry, but we could not process your request because it contained "
                    f"unsupported terms. If you have questions, please contact **{settings.company_email}**."
                ),
                is_fallback=True,
                used_llm=False
            )


        history = await get_history(session_id)
        norm_text = self.normalizer.normalize(clean_query)
        req = QueryRequest(text=clean_query, normalized=norm_text)


        await self.classifier.classify(req, history=history)

        # --- STAGE 1: CLASSIFICATION ---
        t_class = time.time()
        logger.info("classification_stage_completed", extra={
            "session_id": session_id,
            "duration_s": round(t_class - t_start, 2),
            "intent": req.plan.intent.value,
            "search_query": req.plan.search_query,
        })

        logger.debug("classification_reasoning", extra={
            "session_id": session_id,
            "original_query": clean_query,
            "enhanced_query": req.plan.search_query,
            "reasoning": req.plan.reasoning,
        })

        source_nodes = []
        strategy = RetrievalStrategy.VECTOR


        if req.plan.intent in [QueryIntent.COMPARE, QueryIntent.AGGREGATE] or req.plan.filters:
            strategy = RetrievalStrategy.METADATA
            raw_data = self.meta_engine.execute_plan(req.plan)
            source_nodes = [
                SourceNode(
                    text=self._recursive_flatten(d),
                    metadata=d,
                    title=str(d.get("title") or d.get("name") or d.get("id")),
                    source_id=str(d.get("id") or d.get("source_id"))
                ) for d in raw_data
            ]


        if not source_nodes or req.plan.needs_semantic_search:
            reason = "no_metadata_results" if not source_nodes else "needs_semantic_search_flag"
            if source_nodes:
                strategy = RetrievalStrategy.HYBRID
            logger.info("vector_fallback_triggered", extra={
                "session_id": session_id,
                "reason": reason,
                "metadata_result_count": len(source_nodes),
            })
            try:
                if not self._vec:
                    from src.retrieval.vector import VectorRetriever
                    self._vec = VectorRetriever()

                vec_results = await self._vec.search(req.plan.search_query)

                existing_titles = [str(n.title).lower() for n in source_nodes if n.title]
                added = 0
                for vn in vec_results:
                    if str(vn.title).lower() not in existing_titles:
                        source_nodes.append(vn)
                        added += 1
                logger.info("vector_results_merged", extra={
                    "session_id": session_id,
                    "vector_candidates": len(vec_results),
                    "added_after_dedup": added,
                    "total_source_nodes": len(source_nodes),
                })
            except Exception as e:
                # Always log this -- a failed vector search should never be
                # invisible in production just because debug=False.
                logger.error("vector_search_failed", extra={
                    "session_id": session_id,
                    "search_query": req.plan.search_query,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }, exc_info=True)

        # --- STAGE 2: RETRIEVAL ---
        t_ret = time.time()
        logger.info("retrieval_stage_completed", extra={
            "session_id": session_id,
            "duration_s": round(t_ret - t_class, 2),
            "strategy": strategy.value if hasattr(strategy, "value") else str(strategy),
            "source_node_count": len(source_nodes),
        })

        if not source_nodes:
            logger.warning("no_sources_found", extra={
                "session_id": session_id,
                "query": clean_query,
                "search_query": req.plan.search_query,
            })
            return QueryResponse(
                answer=f"I'm sorry, I couldn't find any information matching that request in our system. Please contact {settings.company_email}.",
                is_fallback=True,
                used_llm=False
            )


        from src.generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        prompt = builder.build(req, source_nodes, history=history)

        raw_output, tokens = await self.generator.generate(prompt)

        # --- STAGE 3: GENERATION ---
        t_gen = time.time()
        logger.info("generation_stage_completed", extra={
            "session_id": session_id,
            "duration_s": round(t_gen - t_ret, 2),
            "tokens": tokens,
        })


        normalized = self.response_normalizer.normalize(raw_output)

        if normalized.get("is_fallback", False):
            final_answer = f"I'm sorry, I don't have enough data to answer that. Please contact {settings.company_email}."
            is_fb = True
        else:
            final_answer = normalized["answer"]
            is_fb = False

            if session_id:

                is_duplicate = False
                if len(history) >= 2 and history[-2].get("role") == "user":
                    if history[-2].get("content") == clean_query:
                        is_duplicate = True

                if not is_duplicate:
                    history.append({"role": "user", "content": clean_query})
                    history.append({"role": "assistant", "content": final_answer})
                    await save_history(session_id, history)
                else:
                    logger.info("duplicate_turn_skipped", extra={"session_id": session_id})

        total_duration = round(time.time() - t_start, 2)
        logger.info("pipeline_run_completed", extra={
            "session_id": session_id,
            "duration_s": total_duration,
            "strategy": strategy.value if hasattr(strategy, "value") else str(strategy),
            "intent": req.plan.intent.value,
            "tokens": tokens,
            "is_fallback": is_fb,
            "source_node_count": len(source_nodes),
        })

        return QueryResponse(
            answer=final_answer,
            sections=normalized.get("sections", []),
            suggestions=normalized.get("suggestions", []),
            sources=source_nodes,
            strategy=strategy,
            used_llm=True,
            is_fallback=is_fb,
            tokens_used=tokens,
            intent=req.plan.intent
        )