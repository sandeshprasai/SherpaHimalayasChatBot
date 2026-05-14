import os
import sys
import json
import re
from typing import Dict, Any
from src.services.logger import logger


# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.config.settings import settings

FALLBACK_PHRASES = [
    "i don't have that information",
    "i do not have that information",
    "not specified in the provided context",
    "i cannot find",
    "no information available",
    "i'm sorry, but i don't",
    "i am sorry, but i don't",
    "i'm sorry, but i do not",
    "i am sorry, but i do not",
    "namaste. i'm sorry",
    "namaste. i am sorry",
    "don't have information on that",
    "do not have information on that",
    "not in our records",
    "not available in the context",
]

FALLBACK_MESSAGE = (
    f"Namaste! I'm sorry, but I don't have that specific information "
    f"in our records. Please contact our team at "
    f"**{settings.company_email}** — we'd love to help you plan "
    f"your Himalayan adventure! 🏔️"
)


class Generator:


    def __init__(self):
        self._llm = None
        self._enc = None

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> tuple[Dict[str, Any], int]:
        
        try:
            self._ensure_ready()
        except Exception as e:
            
            return FALLBACK_MESSAGE, 0
        
        response = self._llm.complete(prompt)
        raw_text = str(response).strip()
        clean_json = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.IGNORECASE | re.MULTILINE).strip()

        try:
            parsed = json.loads(clean_json)
        except Exception:
            match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError("No JSON block found")

        except Exception as e:
            logger.error(f"LLM Generation/Parsing failed: {e}")
            parsed = {
            "answer": raw_text, # Fallback to raw text if all parsing fails
            "sections": [],
            "suggestions": []
        }

        tokens = self.count_tokens(prompt) + self.count_tokens(raw_text)
        return parsed, tokens


    def is_fallback(self, text: str) -> bool:
        tl = text.lower()
        return any(p in tl for p in FALLBACK_PHRASES)

    def count_tokens(self, text: str) -> int:
        try:
            self._ensure_ready()
            return len(self._enc.encode(str(text)))
        except Exception:
            
            return max(1, len(str(text)) // 4)

    def is_fallback(self, text: str) -> bool:
        tl = text.lower()
        return any(p in tl for p in FALLBACK_PHRASES)

    def count_tokens(self, text: str) -> int:
        self._ensure_ready()
        return len(self._enc.encode(str(text)))

    def estimate_cost(self, tokens: int) -> float:
        return (tokens / 1_000_000) * 0.59

    def get_llm(self):
        self._ensure_ready()
        return self._llm


    def _ensure_ready(self):
        if self._llm is not None:
            return

        import tiktoken
        from dotenv import load_dotenv
        from llama_index.llms.groq import Groq
        from llama_index.core import Settings

        load_dotenv(os.path.join(_ROOT, ".env"))

        self._llm = Groq(
            model=settings.llm_model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        Settings.llm = self._llm

        self._enc = tiktoken.get_encoding("cl100k_base")