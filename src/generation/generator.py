"""
generation/generator.py
────────────────────────
LLM interaction layer. Wraps Groq via LlamaIndex.

Responsibilities:
  - Initialize and return the LLM (once)
  - Generate a response given a prompt string
  - Count tokens and compute cost
  - Detect fallback responses

Usage:
    from generation.generator import Generator

    gen = Generator()
    answer, tokens = gen.generate(prompt)
"""

import os
import sys

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings


# ── Fallback phrases — if LLM output contains any of these,
#    the pipeline treats the response as a fallback ─────────────────────────
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
    """
    Wraps the Groq LLM. Instantiate once and reuse across requests.
    """

    def __init__(self):
        self._llm = None
        self._enc = None

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def generate(self, prompt: str) -> tuple[str, int]:
        """
        Send prompt to LLM and return (answer_text, token_count).
        token_count is an estimate based on tiktoken.
        """
        self._ensure_ready()

        response = self._llm.complete(prompt)
        answer   = str(response).strip()

        tokens = self.count_tokens(prompt) + self.count_tokens(answer)
        return answer, tokens

    def is_fallback(self, text: str) -> bool:
        """Return True if the LLM admitted it could not answer."""
        tl = text.lower()
        return any(p in tl for p in FALLBACK_PHRASES)

    def count_tokens(self, text: str) -> int:
        self._ensure_ready()
        return len(self._enc.encode(str(text)))

    def estimate_cost(self, tokens: int) -> float:
        """
        Rough cost estimate in USD.
        Groq llama-3.3-70b pricing as of 2025: ~$0.59 / 1M tokens.
        """
        return (tokens / 1_000_000) * 0.59

    def get_llm(self):
        """Expose raw LLM for LlamaIndex chat engine if needed."""
        self._ensure_ready()
        return self._llm

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────

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