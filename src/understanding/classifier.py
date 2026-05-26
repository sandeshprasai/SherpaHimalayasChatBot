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

    # FIX: Added history parameter
    async def classify(self, req: QueryRequest, history: list = None) -> QueryRequest:
        allowed_intents = ["lookup", "aggregate", "filter", "compare", "summarize", "general"]
        
        # Format history for the classifier
        history_text = ""
        if history:
            history_text = "RECENT CHAT HISTORY:\n"
            for msg in history[-4:]:
                role = "User" if msg.get("role") == "user" else "Bot"
                history_text += f"{role}: {msg.get('content')[:100]}...\n"

        prompt = f"""
        FIELDS: {self.available_keys}
        {history_text}
        LATEST QUERY: "{req.text}"
        
        TASK: Generate a Data Execution Plan and an ENHANCED search query.
        VALID INTENTS: {allowed_intents}
        
        CRITICAL: If the LATEST QUERY uses pronouns (it, this, that, there) or implies a subject from the RECENT CHAT HISTORY, the `search_query` MUST be rewritten to include the actual name of the subject.
        EXAMPLE: Instead of "How long does it take?", write "[Subject Name] duration".

        Return ONLY valid JSON:
        {{
          "intent": "general",
          "filters": {{}},
          "sort_field": null,
          "sort_direction": "asc",
          "limit": 5,
          "needs_semantic_search": true,
          "search_query": "A highly descriptive search query replacing pronouns with the exact subject name",
          "reasoning": "logic"
        }}
        """
        try:
            # Using 8B here is fast and cheap for this task
            resp = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "system", "content": "You are a precise data architect."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            d = json.loads(resp.choices[0].message.content)
            
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
                intent=QueryIntent(intent_str),
                reasoning=d.get("reasoning", "")
            )
        except Exception as e:
            print(f"⚠️ Classifier Error: {e}")
            req.plan = DataExecutionPlan(search_query=req.text, intent=QueryIntent.GENERAL)
        return req

# FIX: Update the wrapper function to accept history
async def classify(req: QueryRequest, keys: list[str], history: list = None) -> QueryRequest:
    classifier = UniversalClassifier(keys)
    return await classifier.classify(req, history)