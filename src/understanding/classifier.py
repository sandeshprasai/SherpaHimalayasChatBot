import json
import os
import time
from openai import AsyncOpenAI
from dotenv import load_dotenv
from src.services.logger import logger
from src.core.schema import QueryRequest, DataExecutionPlan, QueryIntent
from src.config.settings import settings
import asyncio

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing in .env")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client

class UniversalClassifier:
    def __init__(self, available_keys: list[str]):
        self.client = get_openai_client()
        self.available_keys = available_keys

    async def classify(self, req: QueryRequest, history: list = None) -> QueryRequest:
        allowed_intents = [i.value for i in QueryIntent]
        history_text = "No previous history."
        if history:
            history_text = "RECENT CHAT HISTORY:\n"
            for msg in history[-8:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('content', '')[:300]}\n"

        prompt = f"""
        ACT AS: A Logical Data Architect.
        SCHEMA_FIELDS: {self.available_keys}
        {history_text}
        LATEST USER QUERY: "{req.text}"

        TASK: Generate a JSON Data Execution Plan.
        
        CRITICAL RULES:
        1. CONTEXT RESOLUTION: Replace pronouns with exact entity names from history.
        2. SEARCH QUERY: Expand the input into a professional search string.
        3. INTENT: Must be exactly one of these values, no others: {allowed_intents}
        4. LIMIT: Default to 5. If the user asks for "all", "every", "complete list",
           "how many", or otherwise implies they want an exhaustive answer rather than
           a few examples, set limit to 50 instead.
        5. NUMERIC FILTERS: For threshold conditions (e.g. "over 7000m", "under $2000",
           "at least 10 days"), use this structured shape for the filter value:
           {{"operator": ">"|">="|"<"|"<="|"=="|"!=", "value": <number>}}
           Only use a plain string/number value for exact/substring matches.

        Return ONLY raw JSON:
        {{
          "intent": "string",
          "filters": {{}},
          "sort_field": "string" or null,
          "sort_direction": "asc"|"desc",
          "limit": 5,
          "needs_semantic_search": boolean,
          "search_query": "string",
          "reasoning": "string"
        }}
        """

        logger.info("classification_started", extra={
            "query": req.text[:200],
            "history_turns": len(history) if history else 0,
        })

        for attempt in range(3):
            attempt_start = time.monotonic()
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.logic_model,
                    messages=[
                        {"role": "system", "content": "You are a precise data architect."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    timeout=8.0,
                )
                d = json.loads(resp.choices[0].message.content)
                req.plan = DataExecutionPlan(
                    filters=d.get("filters", {}),
                    sort_field=d.get("sort_field"),
                    sort_direction=d.get("sort_direction", "asc"),
                    limit=d.get("limit", 5),
                    needs_semantic_search=d.get("needs_semantic_search", True),
                    search_query=d.get("search_query", req.text),
                    intent=QueryIntent(d.get("intent", "general")),
                    reasoning=d.get("reasoning", "")
                )
                usage = getattr(resp, "usage", None)
                logger.info("classification_succeeded", extra={
                    "attempt": attempt + 1,
                    "duration_s": round(time.monotonic() - attempt_start, 2),
                    "model": settings.logic_model,
                    "intent": req.plan.intent.value,
                    "filters": req.plan.filters,
                    "sort_field": req.plan.sort_field,
                    "limit": req.plan.limit,
                    "needs_semantic_search": req.plan.needs_semantic_search,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                })
                break
            except Exception as e:
                elapsed = time.monotonic() - attempt_start
                logger.error("classification_attempt_failed", extra={
                    "attempt": attempt + 1,
                    "duration_s": round(elapsed, 2),
                    "model": settings.logic_model,
                    "error_type": type(e).__name__,
                    "error": str(e),
                })
                if attempt == 2:
                    req.plan = DataExecutionPlan(search_query=req.text, intent=QueryIntent.GENERAL)
                    logger.warning("classification_exhausted_retries", extra={
                        "query": req.text[:200],
                    })
                else:
                    await asyncio.sleep(1)

        return req

async def classify(req: QueryRequest, keys: list[str], history: list = None) -> QueryRequest:
    return await UniversalClassifier(keys).classify(req, history)