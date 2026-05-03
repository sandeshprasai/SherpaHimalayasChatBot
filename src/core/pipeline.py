"""
core/pipeline.py
─────────────────
Main query pipeline. Wires all layers together in order:

  1. Normalize   (understanding/normalizer)
  2. Classify    (understanding/classifier)   ← now LLM-based
  3. Resolve     (understanding/entity_resolver) ← now opt-in via needs_entity
  4. Route       (core/router)
  5. Retrieve    (retrieval/vector | metadata | hybrid | full_doc)
               ↑ vector search now type-boosted per intent
  6. Generate    (generation/prompt_builder + generator)
  7. Return      QueryResponse

KEY CHANGES vs original:
  - Entity resolver only runs when classifier sets needs_entity=True
    → Fixes false matches like "Himalaya views" → "Why Sherpa Himalaya"
  - Vector search passes prefer_types for GENERAL queries
    → Company/team/info/blog pages surface above trek pages
  - Filter path uses vector scores to rerank candidates
    → "5 best treks with Himalaya views" returns semantically matched results
  - Metadata aggregate enforces field sanity bounds
    → Prevents bad JSON data (e.g. 19175m altitude) from surfacing

Usage:
    from core.pipeline import Pipeline

    pipeline = Pipeline()
    response = pipeline.run("How do I book a trek?")
    print(response.answer)
"""

import os
import sys
import json

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import settings
from core.schema import (
    QueryRequest, QueryResponse,
    QueryIntent, RetrievalStrategy, SourceNode,
)
from core.router import Router
from understanding.normalizer import normalize
from understanding.classifier import classify
from understanding.entity_resolver import EntityResolver
from generation.generator import Generator, FALLBACK_MESSAGE
from generation.prompt_builder import PromptBuilder


# ──────────────────────────────────────────────────────────────────────────────
# TYPE BOOST MAP
# For GENERAL queries, prefer these doc types so company/team/info/blog pages
# surface above the 60+ trek pages that otherwise dominate vector rankings.
# ──────────────────────────────────────────────────────────────────────────────

_TYPE_BOOST: dict[QueryIntent, list[str] | None] = {
    QueryIntent.GENERAL:   ["info", "company", "team", "blog"],
    QueryIntent.LOOKUP:    None,   # no boost — let vector decide
    QueryIntent.SUMMARIZE: None,   # full doc fetch anyway
    QueryIntent.COMPARE:   None,
    QueryIntent.AGGREGATE: None,
    QueryIntent.FILTER:    None,   # handled by hybrid reranker
}


class Pipeline:
    """
    Main query pipeline. Instantiate once — all sub-components
    are lazy-loaded on first use.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

        # ── Stateless / lightweight — init immediately ──────────────────────
        self._router   = Router()
        self._builder  = PromptBuilder()
        self._resolver = EntityResolver()

        # ── Heavy deps — lazy loaded on first query ─────────────────────────
        self._generator: Generator | None = None
        self._vec       = None   # VectorRetriever
        self._meta      = None   # MetadataRetriever
        self._hybrid    = None   # HybridRetriever

        # ── Token budget tracking ───────────────────────────────────────────
        self._used_tokens: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def run(self, user_query: str) -> QueryResponse:
        """
        Full pipeline: raw text → QueryResponse.
        """

        # ── 1. Normalize ───────────────────────────────────────────────────
        normalized = normalize(user_query)
        req = QueryRequest(text=user_query, normalized=normalized)
        self._log(f"normalized: {normalized}")

        # ── 2. Classify ────────────────────────────────────────────────────
        classify(req)
        self._log(
            f"intent={req.intent.value}  "
            f"strategy={req.strategy.value}  "
            f"confidence={req.intent_confidence:.2f}  "
            f"needs_entity={req.needs_entity}"
        )

        # ── 3. Resolve entities (only when classifier requests it) ─────────
        # Guard prevents false-positive entity matches on general/filter/
        # aggregate queries (e.g. "views of Himalayas" → "Why Sherpa Himalaya")
        if req.needs_entity:
            req.entities = self._resolver.resolve(normalized)
            self._log(f"entities: {req.entities}")
        else:
            req.entities = []
            self._log("entities: skipped (needs_entity=False)")

        # COMPARE requires two resolved entities
        if req.intent == QueryIntent.COMPARE and len(req.entities) < 2:
            return QueryResponse(
                answer      = FALLBACK_MESSAGE,
                intent      = req.intent,
                is_fallback = True,
                used_llm    = False,
            )

        # ── 4. Route ───────────────────────────────────────────────────────
        decision = self._router.route(req)
        self._log(f"routing: {decision.reason}")

        source_nodes: list[SourceNode] = []
        meta_answer:  str | None       = None

        # ── 5. Retrieve ────────────────────────────────────────────────────

        # A) Metadata only (aggregate / compare)
        if decision.use_metadata and not decision.use_vector:
            try:
                meta_answer = self._get_meta().answer(req)
            except Exception as e:
                self._log(f"metadata retrieval failed: {e}")
                meta_answer = None

        # B) Hybrid (filter: metadata pre-filter → vector rerank)
        elif decision.use_metadata and decision.use_vector:
            try:
                source_nodes, meta_answer = self._get_hybrid().search(req)
            except Exception as e:
                self._log(f"hybrid retrieval failed: {e}")
                try:
                    meta_answer  = self._get_meta().answer(req)
                    source_nodes = []
                except Exception:
                    meta_answer  = None
                    source_nodes = []

        # C) Full document fetch (summarize)
        elif decision.use_full_doc:
            try:
                full_text = self._fetch_full_doc(req)
                if full_text:
                    source_nodes = [SourceNode(
                        text=full_text,
                        title=req.entities[0] if req.entities else None,
                    )]
            except Exception as e:
                self._log(f"full doc fetch failed: {e}")
                source_nodes = []

        # D) Vector search (lookup / general)
        else:
            prefer = _TYPE_BOOST.get(req.intent)
            try:
                source_nodes = self._get_vec().search(
                    normalized,
                    prefer_types=prefer,
                )
            except Exception as e:
                self._log(f"vector retrieval failed: {e}")
                source_nodes = []

        # ── 5b. Entity-based full-doc fallback ─────────────────────────────
        # If retrieval returned nothing but entities were resolved, fetch
        # the full document for the first entity as LLM context.
        if not source_nodes and req.entities:
            try:
                full_text = self._fetch_full_doc(req)
                if full_text:
                    source_nodes = [SourceNode(
                        text=full_text,
                        title=req.entities[0] if req.entities else None,
                    )]
                    self._log("fallback: fetched full document for entity")
            except Exception as e:
                self._log(f"full doc fetch failed: {e}")

        # ── 6. Generate ────────────────────────────────────────────────────

        # Metadata-only path: skip LLM entirely (aggregate / compare / filter)
        if decision.metadata_only and meta_answer:
            self._log("metadata_only: skipping LLM")
            return QueryResponse(
                answer   = meta_answer,
                intent   = req.intent,
                strategy = decision.strategy,
                sources  = source_nodes,
                used_llm = False,
            )

        # Nothing retrieved at all
        if not source_nodes and not meta_answer:
            return QueryResponse(
                answer      = FALLBACK_MESSAGE,
                intent      = req.intent,
                strategy    = decision.strategy,
                is_fallback = True,
                used_llm    = False,
            )

        # Inject metadata listing as a source node when available
        if meta_answer and not source_nodes:
            source_nodes = [SourceNode(text=meta_answer)]

        prompt = self._builder.build(req, source_nodes)

        # ── Initialize LLM ─────────────────────────────────────────────────
        gen = self._get_generator()

        try:
            gen._ensure_ready()
        except Exception as e:
            self._log(f"LLM init failed: {e}")
            try:
                from generation.safe_summarizer import safe_summarize_nodes
                summary = safe_summarize_nodes(source_nodes, req.text)
                return QueryResponse(
                    answer      = summary,
                    intent      = req.intent,
                    strategy    = decision.strategy,
                    sources     = source_nodes,
                    used_llm    = False,
                    is_fallback = False,
                )
            except Exception as e2:
                self._log(f"safe summarizer failed: {e2}")
                if source_nodes and getattr(source_nodes[0], "text", None):
                    content = source_nodes[0].text.strip()
                    if len(content) > 2000:
                        content = content[:2000] + "..."
                    return QueryResponse(
                        answer      = content,
                        intent      = req.intent,
                        strategy    = decision.strategy,
                        sources     = source_nodes,
                        used_llm    = False,
                        is_fallback = False,
                    )
                return QueryResponse(
                    answer      = FALLBACK_MESSAGE,
                    is_fallback = True,
                    used_llm    = False,
                )

        estimated = gen.count_tokens(user_query) + settings.estimated_resp_tokens
        if self._used_tokens + estimated > settings.daily_token_limit:
            return QueryResponse(
                answer      = "🚫 Daily token limit reached. Please try again later.",
                is_fallback = True,
                used_llm    = False,
            )

        answer, tokens = gen.generate(prompt)
        self._used_tokens += tokens
        cost = gen.estimate_cost(tokens)

        is_fallback = gen.is_fallback(answer)
        if is_fallback:
            if source_nodes and getattr(source_nodes[0], "text", None):
                content = source_nodes[0].text.strip()
                if len(content) > 2000:
                    content = content[:2000] + "..."
                return QueryResponse(
                    answer      = content,
                    intent      = req.intent,
                    strategy    = decision.strategy,
                    sources     = source_nodes,
                    tokens_used = tokens,
                    cost_usd    = cost,
                    used_llm    = False,
                    is_fallback = False,
                )
            answer = FALLBACK_MESSAGE

        return QueryResponse(
            answer      = answer,
            intent      = req.intent,
            strategy    = decision.strategy,
            sources     = source_nodes,
            tokens_used = tokens,
            cost_usd    = cost,
            used_llm    = True,
            is_fallback = is_fallback,
        )

    @property
    def tokens_used(self) -> int:
        return self._used_tokens

    # ──────────────────────────────────────────────────────────────────────
    # FULL DOCUMENT FETCH
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_full_doc(self, req: QueryRequest) -> str | None:
        """
        Find the rag_ready JSON for the first resolved entity.
        """
        search_terms: list[str] = []
        if req.entities:
            search_terms.extend(e.lower() for e in req.entities)
        search_terms.append((req.normalized or req.text).lower())

        candidates: list[tuple[str, dict]] = []
        for root, _, files in os.walk(settings.rag_ready_dir):
            for fname in files:
                if not fname.endswith(".json") or fname == "metadata_index.json":
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        data = next((item for item in data if isinstance(item, dict)), None)
                    if isinstance(data, dict) and data.get("title"):
                        candidates.append((fpath, data))
                except Exception:
                    continue

        query_noise = {"describe", "full", "itinerary", "what", "tell",
                       "give", "complete", "schedule", "plan", "about"}

        for search in search_terms:
            search_words = [w for w in search.split()
                            if len(w) > 3 and w not in query_noise]
            if not search_words:
                continue

            best_match, best_score = None, 0.0
            for _, data in candidates:
                title = (data.get("title") or "").lower()
                title_words = [w for w in title.split() if len(w) > 3]
                if not title_words:
                    continue
                matched = sum(1 for w in title_words if w in search)
                score   = matched / len(title_words)
                if score > best_score:
                    best_score = score
                    best_match = data

            if best_match and best_score >= 0.5:
                return self._format_full_doc(best_match)

        return None

    @staticmethod
    def _format_full_doc(data: dict) -> str:
        lines = [f"TITLE: {data.get('title', 'Unknown')}"]

        for fld, label in [
            ("duration_days",  "DURATION"),
            ("difficulty",     "DIFFICULTY"),
            ("max_altitude_m", "MAX ALTITUDE"),
            ("region",         "REGION"),
            ("summary",        "SUMMARY"),
        ]:
            val = data.get(fld)
            if val is not None and val != "":
                suffix = ""
                if fld == "duration_days":
                    try:
                        days = int(val)
                        suffix = " day" if days == 1 else " days"
                    except Exception:
                        suffix = " days"
                elif fld == "max_altitude_m":
                    suffix = " m"
                lines.append(f"{label}: {val}{suffix}")

        for section in data.get("content_sections", []):
            if not isinstance(section, dict):
                continue
            heading = section.get("heading", "")
            text    = section.get("text", "").strip()
            if heading and text:
                lines.append(f"{heading.upper()}:\n{text}")
            elif text:
                lines.append(text)

        itinerary = data.get("itinerary", [])
        if itinerary:
            lines.append("FULL ITINERARY:")
            for day in itinerary:
                if day.get("title") or day.get("description"):
                    lines.append(
                        f"  Day {day.get('day', '?')}: "
                        f"{day.get('title', '')} — "
                        f"{day.get('description', '')}"
                    )

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # LAZY LOADERS
    # ──────────────────────────────────────────────────────────────────────

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator()
        return self._generator

    def _get_vec(self):
        if self._vec is None:
            from retrieval.vector import VectorRetriever
            self._vec = VectorRetriever()
        return self._vec

    def _get_meta(self):
        if self._meta is None:
            from retrieval.metadata import MetadataRetriever
            self._meta = MetadataRetriever()
        return self._meta

    def _get_hybrid(self):
        if self._hybrid is None:
            from retrieval.hybrid import HybridRetriever
            self._hybrid = HybridRetriever()
        return self._hybrid

    # ──────────────────────────────────────────────────────────────────────
    # DEBUG
    # ──────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.debug:
            print(f"  [pipeline] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = Pipeline(debug=True)

    test_queries = [
        "How do I book a trek?",
        "What is a TIMS card?",
        "Tell me about altitude sickness",
        "Who are your guides?",
        "What is the duration of Ghorepani Circuit Trek?",
        "Which trek has the highest altitude?",
        "Find me 5 trekking destinations with beautiful views of Himalayas",
        "Compare EBC vs Langtang Valley Trek",
        "Describe the full itinerary of Langtang Valley Trek",
        "Suggest a 7-day trek in Annapurna region",
        "Does any trek provide views of Himalayas?",
        "Explain Rafting in Nepal",
        "Why should I choose Sherpa Himalaya?",
    ]

    print("\n🏔️  Sherpa Himalaya — Pipeline Test\n" + "═" * 60)

    for q in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        resp = pipeline.run(q)
        print(f"\nA: {resp.answer[:300]}{'...' if len(resp.answer) > 300 else ''}")
        print(
            f"\n   intent={resp.intent.value}  "
            f"strategy={resp.strategy.value}  "
            f"llm={resp.used_llm}  "
            f"fallback={resp.is_fallback}  "
            f"tokens={resp.tokens_used}"
        )

    print(f"\n{'═' * 60}")
    print(f"Total tokens used: {pipeline.tokens_used}")