import os
from datetime import datetime
from src.config.settings import settings


class PromptBuilder:
    def __init__(self):
        # We pull these from settings to keep the code 0% domain-hardcoded
        self.company = settings.company_name
        self.sys_name = settings.system_name
        self.email = settings.company_email

    def build(self, req, source_nodes):
        """
        Constructs a prompt that forces a natural, human-like response 
        without technical 'meta-talk' about chunks or inference.
        """
        
        # 1. SYSTEM IDENTITY & STYLE RULES
        system_prompt = f"""
        Today's date is {datetime.now().strftime("%B %d, %Y")}.
        You are the {self.company} {self.sys_name}.

        ### CRITICAL INSTRUCTIONS:
        1. INTEGRATION: Speak naturally. Do NOT mention "context chunks," "source records," "provided text," or "database" in your reply.
        2. NO SELF-REFERENCING: Do not say "Based on the info provided" or "According to the records." Just state the facts.
        3. NO TECHNICAL TAGS: Never use the word "(inferred)" or similar markers. If you are suggesting something likely but not 100% stated, use natural language like "Typically," or "We suggest."
        4. SOURCE OF TRUTH: Use the provided records below as your ONLY knowledge. If the answer is not there, politely say you don't have that specific detail and suggest contacting {self.email}.
        5. FORMATTING: Use clear, clean formatting (bullet points or bold text) to make information easy to read.
            6. JSON EXTRACTION RULES:
                - Identify the primary attributes (e.g., name, status, metrics) from the records.
                - Create a list of objects containing these key attributes.
                - Wrap this list in <DATA> [...] </DATA> tags at the very end of your response.
                - Do not invent fields; only use what is present in the data.
        """

        # 2. DATA PREPARATION (Internal only, LLM shouldn't repeat these labels)
        context_data = ""
        for i, node in enumerate(source_nodes, 1):
            # We use a clean separator that doesn't look like a "Chunk ID"
            context_data += f"\n---\n{node.text}\n"

        # 3. FINAL ASSEMBLY
        full_prompt = f"""
        {system_prompt}

        ### INTERNAL DATA RECORDS:
        {context_data}

        ### USER REQUEST:
        {req.text}

        ### RESPONSE:
        """
        return full_prompt