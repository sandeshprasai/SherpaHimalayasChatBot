"""
generation/prompt_builder.py
─────────────────────────────
Builds the final prompt sent to the LLM, tailored to the query intent.

Different intents need different instructions:
  - LOOKUP    → strict factual, copy from context
  - SUMMARIZE → structured day-by-day format
  - COMPARE   → table format preferred
  - FILTER    → recommendation tone
  - GENERAL   → balanced

Usage:
    from generation.prompt_builder import PromptBuilder
    from core.schema import QueryRequest, SourceNode

    builder = PromptBuilder()
    prompt  = builder.build(req, source_nodes)
"""

import os
import sys
from datetime import datetime

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import QueryRequest, QueryIntent, SourceNode


# ──────────────────────────────────────────────────────────────────────────────
# BASE SYSTEM PROMPT  (your original, preserved exactly)
# ──────────────────────────────────────────────────────────────────────────────

_BASE_SYSTEM = """\
Today's date is {date}.

You are the {company} Travel Information System.

### GUIDING PRINCIPLES

1. Use the provided context chunks as the PRIMARY source of truth.
2. Prefer factual answers drawn from context; you MAY synthesize or infer when
   context is partial, but always indicate when a statement is inferred.
3. If key facts are missing from context, provide a best-effort answer labeled
   with "(inferred)" and include a suggestion to contact {email} for
   confirmation when appropriate.
4. Avoid inventing policies, prices, or procedures not supported by context.
5. When multiple documents exist, merge only when clearly related and cite
   sources (include chunk titles or relevance scores).
6. If absolutely no relevant context exists, respond:
   "Namaste. I couldn't find information on that topic in our records — please
   contact {email}."

### RESPONSE STYLE
- Be factual and clear.
- Prefer quoting or lightly rephrasing retrieved text, and mark inferred
  statements with "(inferred)".
- Offer next steps (e.g., ask a clarifying question, suggest contacting
  support) when appropriate.
"""

# ──────────────────────────────────────────────────────────────────────────────
# INTENT-SPECIFIC ADDENDUMS
# ──────────────────────────────────────────────────────────────────────────────

_ADDENDUM = {

    QueryIntent.LOOKUP: """
### ADDITIONAL RULE FOR GENERAL QUERIES
- If the context contains partial but relevant information, use it to give a helpful answer even if it does not fully address the question.
- For procedural questions (how to book, how to apply, how to reach), summarize the steps present in context clearly.
- If context mentions a contact, website, or email relevant to the question, always include it in your answer.
""",

    QueryIntent.SUMMARIZE: """
### ADDITIONAL RULE FOR ITINERARY REQUESTS
- Present the itinerary as a numbered day-by-day list.
- Use the format: "**Day N:** Title — Description"
- Do not skip any days present in the context.
- If a day description is empty, write "Details not available."
""",

    QueryIntent.COMPARE: """
### ADDITIONAL RULE FOR COMPARISON REQUESTS
- Present comparisons as a markdown table where possible.
- Columns: Attribute | Trip A | Trip B
- Include: Duration, Difficulty, Max Altitude, Region.
- After the table, add a one-line summary for each trip.
- Do NOT declare a winner unless the context explicitly states one.
""",

    QueryIntent.FILTER: """
### ADDITIONAL RULE FOR RECOMMENDATION REQUESTS
- List each matching trip as a numbered item.
- For each trip include: name, duration, difficulty, region.
- Add a one-sentence description from the context summary.
- If fewer than 3 results match, acknowledge the limited options.
""",

    QueryIntent.AGGREGATE: """
### ADDITIONAL RULE FOR RANKING/AGGREGATE REQUESTS
- State the answer directly in the first sentence.
- Then provide supporting details (duration, region, difficulty).
- Do not list other trips unless asked.
""",

    QueryIntent.GENERAL: """
### ADDITIONAL RULE FOR OPEN-ENDED QUERIES
- If context contains a list of destinations, treks, or services, summarize the most relevant ones clearly.
- Do not refuse to answer just because the question is broad — use whatever relevant context is available.
- Always end with an invitation to ask more specific questions.
""",
}


class PromptBuilder:

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def build(
        self,
        req: QueryRequest,
        source_nodes: list[SourceNode],
    ) -> str:
        """
        Returns the complete prompt string to send to the LLM.
        Combines: system rules + intent addendum + context + user query.
        """
        system   = self._system_block(req.intent)
        context  = self._context_block(source_nodes)
        question = self._question_block(req)

        return f"{system}\n\n{context}\n\n{question}"

    def build_system_only(self) -> str:
        """
        Returns the base system prompt (no context, no question).
        Used when initialising a LlamaIndex chat engine.
        """
        return _BASE_SYSTEM.format(
            date    = datetime.now().strftime("%B %d, %Y"),
            company = settings.company_name,
            email   = settings.company_email,
        )

    # ──────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────

    def _system_block(self, intent: QueryIntent) -> str:
        base = _BASE_SYSTEM.format(
            date    = datetime.now().strftime("%B %d, %Y"),
            company = settings.company_name,
            email   = settings.company_email,
        )
        addendum = _ADDENDUM.get(intent, "")
        return base + addendum

    def _context_block(self, nodes: list[SourceNode]) -> str:
        if not nodes:
            return "### CONTEXT\n(No relevant context found.)"

        sections = []
        for i, node in enumerate(nodes, 1):
            header = f"### CONTEXT CHUNK {i}"
            if node.title:
                header += f" — {node.title}"
            if node.score is not None:
                header += f" (relevance: {node.score:.2f})"
            sections.append(f"{header}\n{node.text.strip()}")

        return "\n\n".join(sections)

    def _question_block(self, req: QueryRequest) -> str:
        lines = ["### QUESTION"]

        # Show normalized query if it differs meaningfully from original
        if req.normalized and req.normalized.lower() != req.text.lower():
            lines.append(f"(Interpreted as: {req.normalized})")

        lines.append(req.text)
        lines.append("\n### ANSWER")
        return "\n".join(lines)