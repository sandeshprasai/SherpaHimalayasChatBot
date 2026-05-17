"""
scripts/indexing/index_builder.py
──────────────────────────────────
Builds metadata_index.json from any rag_ready dataset.

Zero domain assumptions. Works for trekking, yoga, e-commerce,
real estate, or any other site — no code changes needed when you
swap datasets. Everything is discovered from the data itself.

Output written to: data/indexes/metadata/metadata_index.json
(mirrors the path in settings.py → settings.metadata_index)

Usage:
    python scripts/indexing/index_builder.py

Optional env overrides (all have sensible defaults):
    RAG_READY_DIR   — override the input folder
    METADATA_OUT    — override the output file path
    MIN_CONTENT_LEN — skip documents shorter than N chars (default 100)
"""

import os
import re
import sys
import json
import glob

# ── Path setup ────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))

RAG_READY_DIR = os.environ.get(
    "RAG_READY_DIR",
    os.path.join(PROJECT_ROOT, "data/datasets/rag_ready"),
)

METADATA_OUT = os.environ.get(
    "METADATA_OUT",
    os.path.join(PROJECT_ROOT, "data/indexes/metadata/metadata_index.json"),
)

# Documents shorter than this (content field, chars) are skipped as stubs/junk.
MIN_CONTENT_LEN = int(os.environ.get("MIN_CONTENT_LEN", 500))

# These filenames are always skipped — they are index artefacts, not source docs.
SKIP_FILENAMES = {"metadata_index.json", "metadata_index.cleaned.json"}

# meta_tags keys whose values are too large or structurally useless to index.
# Only exclude what is actively harmful (embedding blobs, deeply nested data).
# Everything else is kept — the classifier LLM will decide what to filter on.
SKIP_META_KEYS = {"embedding_chunks", "embedding"}


# ── Value coercion helpers ────────────────────────────────────────────────────

def _to_str(v) -> str:
    """Coerce any scalar to a clean string."""
    return str(v).strip()


def _try_numeric(v):
    """
    If a value looks numeric, return int or float.
    Otherwise return the original value untouched.
    This lets the classifier sort/compare duration_days, price_usd, etc.
    """
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        clean = v.replace(",", "").replace("$", "").strip()
        try:
            return int(clean)
        except ValueError:
            pass
        try:
            return float(clean)
        except ValueError:
            pass
    return v


def _flatten_list(lst: list) -> list:
    """
    Turn a mixed list into a flat list of strings.

    Handles all three shapes found in real datasets:
      - ["string", "string"]                  → kept as-is
      - [{"key": "val", ...}, ...]            → each dict → "key: val; key: val"
      - ["string", {"key": "val"}, "string"]  → mixed, both handled
    """
    result = []
    for item in lst:
        if isinstance(item, str):
            s = item.strip()
            if s:
                result.append(s)
        elif isinstance(item, dict):
            parts = [
                f"{k}: {v}"
                for k, v in item.items()
                if isinstance(v, (str, int, float, bool)) and str(v).strip()
            ]
            if parts:
                result.append("; ".join(parts))
        # Skip nested lists — they don't appear in our data and add noise
    return result


def _flatten_dict(d: dict) -> str:
    """
    Turn a shallow dict into a single readable string.
    e.g. {"Shared Room": "$150", "Private": "$170"} → "Shared Room: $150, Private: $170"
    Only scalar values are included.
    """
    parts = [
        f"{k}: {v}"
        for k, v in d.items()
        if isinstance(v, (str, int, float, bool)) and str(v).strip()
    ]
    return ", ".join(parts)


# ── Title discovery ───────────────────────────────────────────────────────────

def _derive_title(data: dict) -> str | None:
    """
    Find the best human-readable title for a document.

    Priority order:
      1. meta_tags.title          — explicit, highest quality
      2. data["title"]            — top-level title field
      3. First H1 in content      — extracted from markdown
      4. filename slug            — last resort, always succeeds

    Returns None only if the document has no content AND no filename,
    which would mean it's an empty/corrupt record — caller should skip it.
    """
    meta = data.get("meta_tags") or {}

    # 1. meta_tags.title
    if isinstance(meta, dict):
        t = meta.get("title", "")
        if t and str(t).strip():
            return str(t).strip()

    # 2. top-level title
    t = data.get("title", "")
    if t and str(t).strip():
        return str(t).strip()

    # 3. First H1/H2 in the content field that looks like a real title
    content = data.get("content", "") or ""
    if content:
        for match in re.finditer(r"^#{1,2}\s+(.+)$", content, re.MULTILINE):
            candidate = match.group(1).strip()
            # Skip: URL lines ("# Source: https://...")
            if re.search(r"https?://", candidate):
                continue
            # Skip: generic/useless headings
            if candidate.lower() in {"home", "index", "menu", "page", "source", "contents"}:
                continue
            # Skip: site-name-only lines (single word or very short)
            if len(candidate) <= 4:
                continue
            return candidate

    # 4. filename slug → human-readable
    filename = data.get("filename", "") or ""
    if not filename:
        return None
    slug = filename.replace(".md", "").replace(".json", "").replace(".html", "")
    # "courses-200-hours-yoga-teacher-training" → "Courses 200 Hours Yoga Teacher Training"
    return slug.replace("-", " ").replace("_", " ").title()


# ── Core record builder ───────────────────────────────────────────────────────

def process_document(data: dict, filepath: str) -> dict | None:
    """
    Convert one JSON document into a flat metadata record.

    The record always has these guaranteed keys:
        id, title, category, _source, _content_length

    Every scalar/string-list meta_tags value is then added on top —
    whatever keys exist in *this* document's data. No assumptions made
    about what those keys will be.

    Returns None for documents that should be skipped (stubs, junk).
    """

    # ── Guard: skip documents with negligible content ──────────────────────
    content = data.get("content", "") or ""
    if len(content.strip()) < MIN_CONTENT_LEN:
        return None

    # ── Guard: must have a usable title ───────────────────────────────────
    title = _derive_title(data)
    if not title:
        return None

    # ── Core identity fields ───────────────────────────────────────────────
    filename = data.get("filename", os.path.basename(filepath))
    rel_path = os.path.relpath(filepath, RAG_READY_DIR).replace("\\", "/")

    record: dict = {
        "id":              filename.replace(".md", "").replace(".json", ""),
        "title":           title,
        "category":        data.get("category", "General"),
        "source_url":      data.get("source_url", ""),
        "_source":         rel_path,
        "_content_length": len(content),
    }

    # ── Generic meta_tags extraction ───────────────────────────────────────
    #
    # Rules:
    #   scalar (str/int/float/bool) → stored directly; numbers coerced
    #   list of strings             → stored as list (filterable)
    #   list of dicts               → flattened to list of strings
    #   dict (shallow)              → flattened to single string
    #   dict of dicts               → skipped (too deep, not filterable)
    #   SKIP_META_KEYS              → always skipped
    #
    meta = data.get("meta_tags") or {}
    if isinstance(meta, dict):
        for key, value in meta.items():
            if key in SKIP_META_KEYS:
                continue
            if key == "title":
                # Already captured above; don't duplicate
                continue

            if isinstance(value, (str, int, float, bool)):
                coerced = _try_numeric(value) if not isinstance(value, bool) else value
                if _to_str(value):  # skip empty strings
                    record[key] = coerced

            elif isinstance(value, list) and len(value) > 0:
                flat = _flatten_list(value)
                if flat:
                    record[key] = flat

            elif isinstance(value, dict) and len(value) > 0:
                # Only flatten if values are scalars — skip dict-of-dicts
                if all(isinstance(v, (str, int, float, bool)) for v in value.values()):
                    flat_str = _flatten_dict(value)
                    if flat_str:
                        record[key] = flat_str
                # else: nested dict — skip silently

    return record


# ── File loader ───────────────────────────────────────────────────────────────

def load_file(filepath: str) -> list[dict]:
    """
    Load a JSON file and return a list of document dicts.
    Handles both single-document ({...}) and multi-document ([{...}, ...]) files.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return [raw]
    elif isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


# ── Main build routine ────────────────────────────────────────────────────────

def build_index() -> None:
    print(f"\n📁  Scanning : {RAG_READY_DIR}")
    print(f"📤  Output   : {METADATA_OUT}")
    print(f"🔍  Min content length: {MIN_CONTENT_LEN} chars\n")

    if not os.path.isdir(RAG_READY_DIR):
        print(f"❌  RAG_READY_DIR not found: {RAG_READY_DIR}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)

    all_json = sorted(glob.glob(os.path.join(RAG_READY_DIR, "**", "*.json"), recursive=True))
    all_json = [p for p in all_json if os.path.basename(p) not in SKIP_FILENAMES]

    print(f"📄  Found {len(all_json)} JSON file(s) to process...\n")

    records      = []
    skipped_stub = 0
    skipped_err  = 0
    categories   = {}

    for filepath in all_json:
        try:
            documents = load_file(filepath)
        except Exception as e:
            print(f"  ⚠️  Parse error  — {os.path.basename(filepath)}: {e}")
            skipped_err += 1
            continue

        for data in documents:
            rec = process_document(data, filepath)
            if rec is None:
                skipped_stub += 1
                continue

            records.append(rec)
            cat = rec.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1

    # ── Write output ───────────────────────────────────────────────────────
    if not records:
        print("❌  No valid records produced. Check your data format.", file=sys.stderr)
        sys.exit(1)

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────────────
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    schema_keys = sorted(k for k in all_keys if not k.startswith("_"))

    print(f"✅  {len(records)} records written → {METADATA_OUT}")
    print(f"⏭️   Skipped: {skipped_stub} stubs/junk  |  {skipped_err} parse errors\n")
    print("📂  Records per category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {count:4d}  {cat}")
    print(f"\n🔑  Discovered schema keys ({len(schema_keys)} total):")
    print(f"    {schema_keys}")
    print("\n💡  These keys are what the classifier LLM can filter and sort on.")


if __name__ == "__main__":
    build_index()