import json
import os
from groq import AsyncGroq
from dotenv import load_dotenv
from src.core.schema import QueryRequest, DataExecutionPlan, QueryIntent

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

class UniversalClassifier:
    def __init__(self, available_keys: list[str]):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY missing in .env")
        self.client = AsyncGroq(api_key=api_key)
        self.available_keys = available_keys

    async def classify(self, req: QueryRequest) -> QueryRequest:
        # 1. Define allowed intents clearly for the LLM
        allowed_intents = ["lookup", "aggregate", "filter", "compare", "summarize", "general"]
        
        prompt = f"""
        FIELDS: {self.available_keys}
        QUERY: "{req.text}"
        
        TASK: Generate a Data Execution Plan and an ENHANCED search query.
        VALID INTENTS: {allowed_intents}

        Return ONLY valid JSON:
        {{
          "intent": "general",
          "filters": {{}},
          "sort_field": null,
          "sort_direction": "asc",
          "limit": 5,
          "needs_semantic_search": true,
          "search_query": "A highly descriptive search query",
          "reasoning": "logic"
        }}
        """
        try:
            resp = await self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a precise data architect. Use only the provided intents."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            d = json.loads(resp.choices[0].message.content)
            
            # 2. SAFETY CHECK: Fallback to 'general' if LLM hallucinates an intent
            intent_str = d.get("intent", "general").lower()
            if intent_str not in allowed_intents:
                intent_str = "general"

            req.plan = DataExecutionPlan(
                filters=d.get("filters", {}),
                sort_field=d.get("sort_field"),
                sort_direction=d.get("sort_direction", "asc"),
                limit=d.get("limit", 5),
                needs_semantic_search=d.get("needs_semantic_search", True),
                search_query=d.get("search_query", req.text),
                intent=QueryIntent(intent_str), # Now this is guaranteed to work
                reasoning=d.get("reasoning", "")
            )
        except Exception as e:
            print(f"⚠️ Classifier Error: {e}")
            req.plan = DataExecutionPlan(search_query=req.text, intent=QueryIntent.GENERAL)
        return req

# Use this wrapper function at the bottom
async def classify(req: QueryRequest, keys: list[str]) -> QueryRequest:
    classifier = UniversalClassifier(keys)
    return await classifier.classify(req)