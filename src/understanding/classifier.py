import json
import os
from groq import Groq
from dotenv import load_dotenv
from src.core.schema import QueryRequest, DataExecutionPlan, QueryIntent

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
        FIELDS: {self.available_keys}
        QUERY: "{req.text}"
        
        TASK: Generate a Data Execution Plan and an ENHANCED search query.

        RULES for 'search_query':
        - Rewrite the user query into a highly descriptive, professional expansion.
        - If the user is vague ("Shortest"), explain what is being measured ("Find the record with the minimum value in the duration field").
        - If entities are mentioned, use their full canonical names.
        - This string will be used for Vector Semantic Search.

        Return ONLY JSON:
        {{
          "intent": "string",
          "filters": {{ "field": "value" }},
          "sort_field": "field_name" or null,
          "sort_direction": "asc"|"desc",
          "limit": 5,
          "needs_semantic_search": boolean,
          "search_query": "The professionally rewritten and expanded search string",
          "reasoning": "logical explanation"
        }}
        """
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are a precise data architect and query optimizer."},
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
                search_query=d.get("search_query", req.text), # <--- NEW
                intent=QueryIntent(d.get("intent", "general")),
                reasoning=d.get("reasoning", "")
            )
            return req
        except Exception:
            req.plan = DataExecutionPlan(search_query=req.text)
            return req

def classify(req: QueryRequest, keys: list[str]) -> QueryRequest:
    return UniversalClassifier(keys).classify(req)