import os
import json
import re
from typing import Any
from groq import Groq
from dotenv import load_dotenv
from core.schema import QueryRequest, QueryIntent, RetrievalStrategy

# Load environment variables
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

_INTENT_MAP = {
    "lookup":    QueryIntent.LOOKUP,
    "aggregate": QueryIntent.AGGREGATE,
    "compare":   QueryIntent.COMPARE,
    "filter":    QueryIntent.FILTER,
    "summarize": QueryIntent.SUMMARIZE,
    "general":   QueryIntent.GENERAL,
}

class QueryClassifier:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # We pre-load valid labels to guide the LLM
        self.valid_regions = ["Annapurna", "Everest", "Langtang", "Manaslu", "Mustang", "Dolpa", "Makalu", "Kanchenjunga"]
        self.valid_categories = ["Trek", "Expedition", "Peak", "Tour"]

    def _get_system_prompt(self):
        return f"""
You are a high-precision query classifier for Sherpa Himalaya.
Your goal is to extract intent and filters from a user's trekking query.

VALID REGIONS: {self.valid_regions}
VALID CATEGORIES: {self.valid_categories}

INTENT DEFINITIONS:
- aggregate: Superlatives (Highest, Shortest, Easiest). User wants the "best" or "most" of something.
- filter: Recommendations (Suggest me..., Find me 3...). User wants a list based on criteria.
- compare: Comparison between two or more specific treks.
- summarize: Request for a full day-by-day itinerary.
- lookup: Factual questions about a specific trek (How high is X? What is included in Y?).
- general: General travel info (Visa, Altitude Sickness, About the company, Booking process).

EXTRACTION RULES:
1. agg_field: Use "max_altitude_m", "duration_days", or "difficulty".
2. agg_op: Use "max" (highest/longest) or "min" (shortest/easiest).
3. needs_entity: Set to TRUE only if the user names a specific destination (e.g., "Everest", "ABC").
4. top_k: Extract the number if the user asks for a specific count (e.g., "Top 3" -> 3).

Respond ONLY with JSON.
"""

    def classify(self, req: QueryRequest) -> QueryRequest:
        try:
            text = req.normalized or req.text
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": f"Query: {text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            data = json.loads(response.choices[0].message.content)

            # Map results to QueryRequest
            req.intent = _INTENT_MAP.get(data.get("intent"), QueryIntent.GENERAL)
            req.agg_field = data.get("agg_field")
            req.agg_op = data.get("agg_op")
            req.top_k = data.get("top_k")
            req.needs_entity = data.get("needs_entity", False)
            
            # Extract filters
            llm_filters = data.get("filters", {})
            if llm_filters.get("region"): req.filters["region"] = llm_filters["region"]
            if llm_filters.get("duration_days"): req.filters["duration_days"] = llm_filters["duration_days"]
            if llm_filters.get("beginner_friendly"): req.filters["beginner_friendly"] = True

            # Logic for routing
            if req.intent in [QueryIntent.AGGREGATE, QueryIntent.COMPARE]:
                req.strategy = RetrievalStrategy.METADATA
            elif req.intent == QueryIntent.FILTER:
                req.strategy = RetrievalStrategy.HYBRID
            else:
                req.strategy = RetrievalStrategy.VECTOR

            return req

        except Exception as e:
            print(f"Classifier Error: {e}. Falling back to Regex.")
            # Fallback logic would go here
            return req

# Global instance for easy importing
_classifier_instance = QueryClassifier()

def classify(req: QueryRequest) -> QueryRequest:
    return _classifier_instance.classify(req)