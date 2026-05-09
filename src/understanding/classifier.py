import json
import os
from groq import Groq
from dotenv import load_dotenv
from core.schema import QueryRequest, DataExecutionPlan, QueryIntent

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

class UniversalClassifier:
    def __init__(self, available_keys: list[str]):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY missing in .env")
        self.client = Groq(api_key=api_key)
        self.available_keys = available_keys

    def classify(self, req: QueryRequest) -> QueryRequest:
        prompt = f"""
        Database Fields: {self.available_keys}
        Query: "{req.text}"
        
        Task: Create a JSON Data Execution Plan.
        Intents: "compare", "filter", "summarize", "lookup", "aggregate", "general"

        Return ONLY JSON:
        {{
          "intent": "string",
          "filters": {{}},
          "sort_field": "string or null",
          "sort_direction": "asc"|"desc",
          "limit": 5,
          "needs_semantic_search": boolean,
          "reasoning": "string"
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are a data architect."},
                          {"role": "user", "content": prompt}],
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
                intent=QueryIntent(d.get("intent", "general")),
                reasoning=d.get("reasoning", "")
            )
            return req
        except Exception:
            req.plan = DataExecutionPlan()
            return req

def classify(req: QueryRequest, keys: list[str]) -> QueryRequest:
    return UniversalClassifier(keys).classify(req)