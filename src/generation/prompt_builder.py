import json
import tiktoken
from datetime import datetime
from src.config.settings import settings

class PromptBuilder:
    def __init__(self):
        self.company = settings.company_name
        self.sys_name = settings.system_name
        self.email = settings.company_email
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = 9000 

    def build(self, req, source_nodes):
        schema_text = json.dumps({
            "answer": "A clear summary...",
            "sections": [{"type": "cards|list|table|text", "title": "...", "items": []}],
            "suggestions": ["Follow-up?"]
        }, indent=2)

        system_prompt = f"""Today is {datetime.now().strftime("%B %d, %Y")}.
You are the {self.company} {self.sys_name}. Use ONLY the provided records.
Speak naturally. Do NOT mention 'database' or 'records'.
If info is missing, suggest: {self.email}

REQUIRED JSON FORMAT:
{schema_text}
"""

        context_data = ""
        current_tokens = len(self.encoder.encode(system_prompt + req.text))
        
        for i, node in enumerate(source_nodes, 1):
            node_text = f"\n---\nRECORD {i}\n{node.text}\n"
            node_tokens = len(self.encoder.encode(node_text))
            
            if current_tokens + node_tokens > self.max_context_tokens:
                break 
            
            context_data += node_text
            current_tokens += node_tokens

       
        full_prompt = f"""
{system_prompt}

### INTERNAL DATA RECORDS
{context_data}

### USER REQUEST
The user is asking: 
<user_input>
{req.text}
</user_input>

### JSON RESPONSE
"""
        return full_prompt