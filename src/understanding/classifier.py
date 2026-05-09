import json
import os
from groq import Groq
from dotenv import load_dotenv
from core.schema import QueryRequest, RetrievalStrategy
from config.settings import settings

# Load environment variables
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
load_dotenv(os.path.join(_ROOT, ".env"))

class UniversalClassifier:
    def __init__(self):
        # Initialize Groq client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._schema_keys = self._load_schema()

    def _load_schema(self):
        """
        Dynamically detects what fields are available in your database.
        This makes the bot 'Schema-Agnostic'.
        """
        if os.path.exists(settings.metadata_index):
            try:
                with open(settings.metadata_index, "r") as f:
                    data = json.load(f)
                    if data and isinstance(data, list):
                        # Get keys from the first record
                        return list(data[0].keys())
            except Exception as e:
                print(f"Classifier Warning: Could not detect schema: {e}")
        
        # Fallback keys if index isn't built yet
        return ["title", "summary", "region", "duration_days", "max_altitude_m", "difficulty"]

    def classify(self, req: QueryRequest) -> QueryRequest:
        """
        Uses LLM reasoning to generate a Data Execution Plan.
        """
        prompt = f"""
        Analyze the trekking query: "{req.text}"
        Available Data Fields: {self._schema_keys}

        Your task is to generate a JSON Execution Plan.
        
        RULES:
        1. Set "needs_entity" to true ONLY if a specific trek/place name is mentioned.
        2. Use "filters" for specific constraints (e.g., region, difficulty).
        3. Use "sort" for superlatives (shortest, highest, longest, easiest).
        4. "field" in sort must match an Available Data Field.
        
        Return ONLY valid JSON:
        {{
          "needs_entity": bool,
          "filters": {{ "field_name": "value" or null }},
          "sort": {{ "field": "field_name", "direction": "asc|desc" }} or null,
          "limit": int (default 5),
          "logic_reasoning": "brief explanation"
        }}
        """
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a data execution architect for a Himalayan travel agency."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            # Store the plan in the request object
            req.data_plan = json.loads(response.choices[0].message.content)
            
            # Sync metadata for the Pipeline's logic branches
            req.needs_entity = req.data_plan.get("needs_entity", False)
            
            return req

        except Exception as e:
            print(f"Classification Error: {e}")
            # Fallback empty plan
            req.data_plan = {"limit": 5}
            return req

# --- CRITICAL FIX: Add these lines at the bottom ---

# 1. Create a single internal instance
_classifier_inst = UniversalClassifier()

# 2. Define the 'classify' function that Pipeline.py is looking for
def classify(req: QueryRequest) -> QueryRequest:
    """
    Standalone function to satisfy the Pipeline import.
    Delegates to the UniversalClassifier instance.
    """
    return _classifier_inst.classify(req)