"""
index_builder.py
Run this once (and re-run whenever rag_ready data changes):
    python src/utils/index_builder.py

Produces: data/rag_ready/metadata_index.json
"""

import os
import json
import glob

# ──────────────────────────────────────────
# PATHS  (adjust if you move this file)
# ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
RAG_READY_DIR = os.path.join(PROJECT_ROOT, "data/rag_ready")
OUTPUT_FILE = os.path.join(RAG_READY_DIR, "metadata_index.json")


# ──────────────────────────────────────────
# FIELDS WE CARE ABOUT FOR FILTERING/SORTING
# ──────────────────────────────────────────
SCALAR_FIELDS = [
    "id", "title", "category", "region",
    "duration_days", "difficulty", "max_altitude_m",
    "start_location", "summary",
]

# Difficulty → numeric for sorting
DIFFICULTY_RANK = {
    "easy": 1,
    "moderate": 2,
    "moderately challenging": 3,
    "challenging": 4,
    "strenuous": 5,
    "extreme": 6,
}


def extract_record(filepath: str) -> dict | None:
    """Pull only the lightweight scalar fields from one JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Could not read {filepath}: {e}")
        return None

    record = {}

    # ── scalar fields ──────────────────────────────────────────────────────
    for field in SCALAR_FIELDS:
        record[field] = data.get(field)          # None if missing

    # ── derived: difficulty rank for easy sorting ──────────────────────────
    diff_str = (record.get("difficulty") or "").lower().strip()
    record["difficulty_rank"] = DIFFICULTY_RANK.get(diff_str, 99)

    # ── source file info ───────────────────────────────────────────────────
    rel_path = os.path.relpath(filepath, RAG_READY_DIR)
    record["_source"] = rel_path.replace("\\", "/")   # windows-safe

    # ── infer category from folder structure if not set ────────────────────
    if not record.get("category"):
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            record["category"] = parts[-2].capitalize()

    # ── tags passthrough (useful for recommendation) ───────────────────────
    record["tags"] = data.get("tags", [])

    # ── highlights count (quick proxy for richness) ───────────────────────
    record["highlights"] = data.get("highlights", [])

    return record


def build_index():
    print(f"\n📁  Scanning: {RAG_READY_DIR}")

    all_json = glob.glob(
        os.path.join(RAG_READY_DIR, "**", "*.json"),
        recursive=True
    )

    # Exclude the output file itself (if it already exists)
    all_json = [p for p in all_json if os.path.basename(p) != "metadata_index.json"]

    print(f"   Found {len(all_json)} JSON files\n")

    records = []
    skipped = 0

    for filepath in sorted(all_json):
        record = extract_record(filepath)
        if record is None:
            skipped += 1
            continue

        # Only keep records that have at least a title
        if not record.get("title"):
            skipped += 1
            continue

        records.append(record)
        print(f"  ✅  {record.get('title', record.get('id', '?'))[:60]}")

    # ── write output ───────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✨  Done! {len(records)} records written → {OUTPUT_FILE}")
    print(f"   Skipped: {skipped} files\n")

    # ── quick sanity: show fields present in first record ──────────────────
    if records:
        print("📋  Sample record (first):")
        sample = {k: v for k, v in records[0].items() if k not in ("tags", "highlights")}
        print(json.dumps(sample, indent=4))


if __name__ == "__main__":
    build_index()