import json
import os
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
        
        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=settings.logic_model,
                    messages=[
                        {"role": "system", "content": "You are a precise data architect."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
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
                break
            except Exception as e:
                if attempt == 2:
                    req.plan = DataExecutionPlan(search_query=req.text, intent=QueryIntent.GENERAL)
                await asyncio.sleep(1)
        
        return req

async def classify(req: QueryRequest, keys: list[str], history: list = None) -> QueryRequest:
    return await UniversalClassifier(keys).classify(req, history)