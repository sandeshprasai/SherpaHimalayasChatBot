import os
from datetime import datetime
from src.config.settings import settings
from src.api.schemas.chat import ChatResponse
import json




class PromptBuilder:
    def __init__(self):
        self.company = settings.company_name
        self.sys_name = settings.system_name
        self.email = settings.company_email

    def build(self, req, source_nodes):
        schema_dict = ChatResponse.model_json_schema()
        schema_text = json.dumps({
            k: v for k, v in schema_dict["properties"].items() 
            if k in ["answer", "sections", "suggestions", "is_fallback"]
        }, indent=2)
        system_prompt_header = f"""
Today's date is {datetime.now().strftime("%B %d, %Y")}.

You are the {self.company} {self.sys_name}.

### CRITICAL INSTRUCTIONS

1. Speak naturally.
2. Do NOT mention:
   - source records
   - chunks
   - database
   - provided context
   - retrieved data
3. Do NOT say:
   - "Based on the records"
   - "According to the data"
4. Use ONLY the provided records below.
5. If information is unavailable, politely suggest contacting:
   {self.email}

### REQUIRED OUTPUT FORMAT

Return ONLY valid JSON.

Schema:
"""

        system_prompt_footer = """
### RULES

- Do NOT return markdown
- Do NOT use <DATA> tags
- Do NOT include explanations outside JSON
- Do NOT invent fields
- Use only information available in records
- Group related entities naturally
- sections must always be an array
- items must always be an array
"""
        system_prompt = system_prompt_header + "\n" + schema_text + "\n" + system_prompt_footer

        context_data = ""

        for i, node in enumerate(source_nodes, 1):
            context_data += f"""

---
RECORD {i}

{node.text}
"""

        full_prompt = f"""
{system_prompt}

### INTERNAL DATA RECORDS

{context_data}

### USER REQUEST

{req.text}

### JSON RESPONSE
"""

        return full_prompt