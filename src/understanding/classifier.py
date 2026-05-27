import json
import os
from groq import AsyncGroq
from dotenv import load_dotenv
from src.core.schema import QueryRequest, DataExecutionPlan, QueryIntent

# Find .env in project root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

class UniversalClassifier:
    def __init__(self, available_keys: list[str]):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY missing in .env")
        self.client = AsyncGroq(api_key=api_key)
        self.available_keys = available_keys

    async def classify(self, req: QueryRequest, history: list = None) -> QueryRequest:
        """
        Logic-only classification. 
        Matches user intent to discovered schema fields without domain knowledge.
        """
        allowed_intents = [i.value for i in QueryIntent]
        
        # Format history for the logic model
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
        1. CONTEXT RESOLUTION: If the LATEST QUERY uses pronouns (it, that, there) or refers to a previous subject, replace them in 'search_query' with the actual entity name from history.
        2. DATA SORTING: If the user wants the "best/worst/most/least", pick the EXACT field from SCHEMA_FIELDS.
        3. SEARCH QUERY: Expand the user input into a professional, standalone search string for a vector database.

        Return ONLY raw JSON:
        {{
          "intent": "{'|'.join(allowed_intents)}",
          "filters": {{ "field": "value" }},
          "sort_field": "exact_field_from_schema" or null,
          "sort_direction": "asc"|"desc",
          "limit": 5,
          "needs_semantic_search": boolean,
          "search_query": "Professional expansion of query with specific entities",
          "reasoning": "How context was resolved"
        }}
        """
        try:
            resp = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant", # Faster/cheaper logic tier
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
        except Exception as e:
            print(f"⚠️ Classifier Error: {e}")
            req.plan = DataExecutionPlan(search_query=req.text, intent=QueryIntent.GENERAL)
        
        return req

async def classify(req: QueryRequest, keys: list[str], history: list = None) -> QueryRequest:
 
    classifier = UniversalClassifier(keys)
    return await classifier.classify(req, history)