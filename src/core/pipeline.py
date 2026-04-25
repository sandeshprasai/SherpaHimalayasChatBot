"""
core/pipeline.py
─────────────────
Main query pipeline. Wires all layers together in order:

  1. Normalize   (understanding/normalizer)
  2. Classify    (understanding/classifier)
  3. Resolve     (understanding/entity_resolver)
  4. Route       (core/router)
  5. Retrieve    (retrieval/vector | metadata | hybrid | full_doc)
  6. Generate    (generation/prompt_builder + generator)
  7. Return      QueryResponse

Usage:
    from core.pipeline import Pipeline

    pipeline = Pipeline()
    response = pipeline.run("Which trek has the highest altitude?")
    print(response.answer)

CLI (for testing):
    python3 -m core.pipeline
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
        self._generator : Generator | None = None
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

        # ── 0. Budget check (deferred until LLM needed) ────────────────────
        # Generator/LLM initialization is deferred to avoid heavy dependency
        # loading for metadata-only queries. Token-budget will be enforced
        # immediately before any LLM call.

        # ── 1. Normalize ───────────────────────────────────────────────────
        normalized = normalize(user_query)
        req = QueryRequest(text=user_query, normalized=normalized)

        self._log(f"normalized: {normalized}")

        # ── 2. Classify ────────────────────────────────────────────────────
        classify(req)
        self._log(f"intent={req.intent.value}  strategy={req.strategy.value}  confidence={getattr(req, 'intent_confidence', 1.0)}")

        # ── 3. Resolve entities ────────────────────────────────────────────
        req.entities = self._resolver.resolve(normalized)
        # If user asked for a comparison but we couldn't resolve two entities, avoid substituting another trip
        if req.intent == QueryIntent.COMPARE and len(req.entities) < 2:
            return QueryResponse(
                answer      = FALLBACK_MESSAGE,
                intent      = req.intent,
                is_fallback = True,
                used_llm    = False,
            )
        self._log(f"entities: {req.entities}")

        # If classifier is low-confidence, run multiple retrieval strategies
        # (metadata + vector) and prefer metadata when it yields an answer.
        low_confidence = getattr(req, 'intent_confidence', 1.0) < 0.75

        source_nodes: list[SourceNode] = []
        meta_answer:  str | None       = None

        if low_confidence:
            self._log("low classifier confidence — running metadata + vector retrieval")
            # Try metadata answer first (fast)
            try:
                meta_answer = self._get_meta().answer(req)
            except Exception:
                meta_answer = None

            # Always run vector search as backup
            try:
                vec_nodes = self._get_vec().search(normalized)
            except Exception:
                vec_nodes = []

            # If metadata gave a solid answer, prefer it (skip LLM)
            if meta_answer:
                source_nodes = []
            else:
                source_nodes = vec_nodes

            # Continue to generation/routing logic below using whichever we have
            decision = self._router.route(req)
            self._log(f"routing (post-confidence) : {decision.reason}")

        else:
            # ── 4. Route ───────────────────────────────────────────────────────
            decision = self._router.route(req)
            self._log(f"routing: {decision.reason}")

            # ── 5. Retrieve ────────────────────────────────────────────────────
            # A) Metadata only (aggregate / compare / filter)
            if decision.use_metadata and not decision.use_vector:
                try:
                    meta_answer = self._get_meta().answer(req)
                except Exception as e:
                    self._log(f"metadata retrieval failed: {e}")
                    meta_answer = None

            # B) Hybrid (filter + vector rerank)
            elif decision.use_metadata and decision.use_vector:
                try:
                    source_nodes, meta_answer = self._get_hybrid().search(req)
                except Exception as e:
                    self._log(f"hybrid retrieval failed: {e}")
                    # fallback to metadata-only
                    try:
                        meta_answer = self._get_meta().answer(req)
                        source_nodes = []
                    except Exception:
                        meta_answer = None
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

            # D) Pure vector (lookup / general)
            else:
                try:
                    source_nodes = self._get_vec().search(normalized)
                except Exception as e:
                    self._log(f"vector retrieval failed: {e}")
                    source_nodes = []

        # ── 6. Generate ────────────────────────────────────────────────────

        # If retrieval returned nothing but entities were resolved, try
        # fetching the full document for the first resolved entity and
        # use it as context for the LLM. This helps with "how to book" or
        # other procedural lookups where a dedicated page exists.
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

        # Metadata-only path: skip LLM entirely
        if decision.metadata_only and meta_answer:
            self._log("metadata_only: skipping LLM")
            return QueryResponse(
                answer   = meta_answer,
                intent   = req.intent,
                strategy = decision.strategy,
                sources  = source_nodes,
                used_llm = False,
            )

        # LLM path
        if not source_nodes and not meta_answer:
            # Nothing retrieved at all
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

        # ── Initialize LLM and enforce token budget (deferred) ───────────
        gen = self._get_generator()

        # Try to initialize the LLM; if initialization fails, fall back to
        # returning the first retrieved document content (no LLM used).
        try:
            gen._ensure_ready()
        except Exception as e:
            self._log(f"LLM init failed: {e}")
            if source_nodes and getattr(source_nodes[0], 'text', None):
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
                answer     = FALLBACK_MESSAGE,
                is_fallback= True,
                used_llm   = False,
            )

        estimated = gen.count_tokens(user_query) + settings.estimated_resp_tokens
        if self._used_tokens + estimated > settings.daily_token_limit:
            return QueryResponse(
                answer     = "🚫 Daily token limit reached. Please try again later.",
                is_fallback= True,
                used_llm   = False,
            )

        answer, tokens = gen.generate(prompt)

        self._used_tokens += tokens
        cost = gen.estimate_cost(tokens)

        # Detect LLM fallback. If the LLM couldn't be initialized or
        # explicitly declined to answer, but a full document source is
        # available, return the document content as a safe, factual answer
        # (no LLM used).
        is_fallback = gen.is_fallback(answer)
        if is_fallback:
            # Prefer returning the first (most relevant) source node's text
            if source_nodes and getattr(source_nodes[0], 'text', None):
                content = source_nodes[0].text.strip()
                # Trim to a reasonable length
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
            # otherwise return the standard fallback message
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
        Search priority:
          1. Resolved entity names (most accurate)
          2. Normalized query text (fallback)
        """
        # Entity names first, then query text as fallback
        search_terms: list[str] = []
        if req.entities:
            search_terms.extend(e.lower() for e in req.entities)
        search_terms.append((req.normalized or req.text).lower())

        # Load all candidates once
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
                        data = data[0] if data else None
                    if isinstance(data, dict) and data.get("title"):
                        candidates.append((fpath, data))
                except Exception:
                    continue

        # Query noise words that are not part of entity names
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

        for field, label in [
            ("duration_days",  "DURATION"),
            ("difficulty",     "DIFFICULTY"),
            ("max_altitude_m", "MAX ALTITUDE"),
            ("region",         "REGION"),
            ("summary",        "SUMMARY"),
        ]:
            val = data.get(field)
            if val is not None and val != "":
                suffix = ""
                if field == "duration_days":
                    try:
                        days = int(val)
                        suffix = " day" if days == 1 else " days"
                    except Exception:
                        suffix = " days"
                elif field == "max_altitude_m":
                    suffix = " m"
                lines.append(f"{label}: {val}{suffix}")

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
# CLI  — run the full test query set from the original chat log
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = Pipeline(debug=True)

    test_queries = [
        # Single-entity lookups (were working before)
        "What is the duration of Ghorepani Circuit Trek?",
        "What is the maximum altitude of Ghorepani Circuit Trek?",
        "What are the highlights of Ghorepani Circuit Trek?",
        "What is included in the Ghorepani Circuit Trek package?",
        "What happens on Day 3 of Ghorepani Circuit Trek?",
        "Describe the full itinerary of Ghorepani Circuit Trek",

        # Comparative / aggregate (were failing before)
        "Which trek is the shortest among all available treks?",
        "Which trek has the highest altitude?",
        "Which trek is the easiest based on difficulty level?",
        "Which trek has the longest duration?",

        # Filter / recommendation (were failing before)
        "Suggest a 7-day trek in Annapurna region",
        "Which trek is best for beginners?",

        # Compare (were failing before)
        "Compare Ghorepani Circuit Trek vs Everest Base Camp Trek in terms of duration",
    ]

    print("\n🏔️  Sherpa Himalaya — Pipeline Test\n" + "═" * 60)

    for q in test_queries:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        resp = pipeline.run(q)
        print(f"\nA: {resp.answer}")
        print(
            f"\n   intent={resp.intent.value}  "
            f"strategy={resp.strategy.value}  "
            f"llm={resp.used_llm}  "
            f"fallback={resp.is_fallback}  "
            f"tokens={resp.tokens_used}  "
            f"cost=${resp.cost_usd:.6f}"
        )

    print(f"\n{'═' * 60}")
    print(f"Total tokens used: {pipeline.tokens_used}")