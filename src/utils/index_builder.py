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


def process_single_dict(data: dict, filepath: str, RAG_READY_DIR: str) -> dict | None:
    """Helper to process a single dictionary record."""
    if not data.get("title"):
        return None

    record = {}
    for field in SCALAR_FIELDS:
        record[field] = data.get(field)

    # derived: difficulty rank
    diff_str = (record.get("difficulty") or "").lower().strip()
    record["difficulty_rank"] = DIFFICULTY_RANK.get(diff_str, 99)

    # source file info
    rel_path = os.path.relpath(filepath, RAG_READY_DIR)
    record["_source"] = rel_path.replace("\\", "/")

    # tags and highlights
    record["tags"] = data.get("tags", [])
    record["highlights"] = data.get("highlights", [])
    
    return record

def build_index():
    print(f"\n📁  Scanning: {RAG_READY_DIR}")
    all_json = glob.glob(os.path.join(RAG_READY_DIR, "**", "*.json"), recursive=True)
    all_json = [p for p in all_json if os.path.basename(p) not in ("metadata_index.json", "metadata_index.cleaned.json")]

    records = []
    for filepath in sorted(all_json):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # GENERALIZED HANDLING: Handle both {} and [{}]
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        rec = process_single_dict(item, filepath, RAG_READY_DIR)
                        if rec: records.append(rec)
            elif isinstance(data, dict):
                rec = process_single_dict(data, filepath, RAG_READY_DIR)
                if rec: records.append(rec)

        except Exception as e:
            print(f"  ⚠️  Error processing {filepath}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✨  Done! {len(records)} records written → {OUTPUT_FILE}")

if __name__ == "__main__":
    build_index()