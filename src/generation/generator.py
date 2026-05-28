import os
import sys
import json
import re
from typing import Dict, Any
from src.services.logger import logger
import tiktoken
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import Settings

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config.settings import settings

FALLBACK_MESSAGE = (
    f"I'm sorry, but I don't have that specific information in our records. "
    f"Please contact our team at **{settings.company_email}** for further assistance."
)
class Generator:


    def __init__(self):
        self._llm = None
        self._enc = None

   

    async def generate(self, prompt: str) -> tuple[Dict[str, Any], int]:
        try:
            self._ensure_ready()
        except Exception as e:
            logger.error(f"Generator initialization failed: {e}")
            return {
                "answer": FALLBACK_MESSAGE, 
                "sections": [], 
                "suggestions": [], 
                "is_fallback": True
            }, 0
        
        max_retries = 3
        backoff_delay = 1.0  
        for attempt in range(max_retries):
            try:
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
                    logger.warning(
                        f"Groq Generation call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {backoff_delay}s..."
                    )
                    import asyncio
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2.0  
                else:
                    logger.error(f"LLM synthesis call failed completely after {max_retries} attempts: {e}")
                    return {"answer": FALLBACK_MESSAGE, "is_fallback": True}, 0


    def is_fallback(self, parsed_json: dict) -> bool:
        
        if isinstance(parsed_json, dict):
            return parsed_json.get("is_fallback", False)
        return False

    def count_tokens(self, text: str) -> int:
        try:
            self._ensure_ready()
            return len(self._enc.encode(str(text)))
        except Exception:
            return max(1, len(str(text)) // 4)

    def estimate_cost(self, tokens: int) -> float:
        return (tokens / 1_000_000) * 0.59

    def get_llm(self):
        self._ensure_ready()
        return self._llm


    def _ensure_ready(self):
        if self._llm is not None:
            return

        

        load_dotenv(os.path.join(_ROOT, ".env"))

        self._llm = Groq(
            model=settings.llm_model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        Settings.llm = self._llm

        self._enc = tiktoken.get_encoding("cl100k_base")