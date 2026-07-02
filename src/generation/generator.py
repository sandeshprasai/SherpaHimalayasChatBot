import os
import json
import re
from typing import Dict, Any
from src.services.logger import logger
import tiktoken
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from src.config.settings import settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))

FALLBACK_MESSAGE = (
    f"I'm sorry, but I don't have that specific information in our records. "
    f"Please contact our team at **{settings.company_email}** for further assistance."
)

class Generator:
    def __init__(self):
        self._llm = None
        self._enc = tiktoken.get_encoding("cl100k_base")

    async def generate(self, prompt: str) -> tuple[Dict[str, Any], int]:
        try:
            self._ensure_ready()
        except Exception as e:
            logger.error(f"Generator initialization failed: {e}")
            return {"answer": FALLBACK_MESSAGE, "is_fallback": True}, 0
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Using OpenAI via LlamaIndex
                response = await self._llm.acomplete(prompt)
                raw_text = str(response).strip()
                
                clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.MULTILINE).strip()

                try:
                    parsed = json.loads(clean_json)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean_json, re.DOTALL)
                    parsed = json.loads(match.group()) if match else {"answer": raw_text}

                tokens = self.count_tokens(prompt) + self.count_tokens(str(parsed))
                return parsed, tokens

            except Exception as e:
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2.0 ** attempt)
                else:
                    logger.error(f"OpenAI synthesis failed: {e}")
                    return {"answer": FALLBACK_MESSAGE, "is_fallback": True}, 0

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(str(text)))

    def _ensure_ready(self):
        if self._llm is not None:
            return
        
        load_dotenv(os.path.join(_ROOT, ".env"))
        self._llm = OpenAI(
            model=settings.llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        Settings.llm = self._llm