import os
import json
import glob
import shutil

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
RAG_READY    = os.path.join(PROJECT_ROOT, "data/rag_ready")


def build_chunks(data: dict) -> list[dict]:
   
    existing_types = {c["type"] for c in data.get("embedding_chunks", [])}
    chunks = list(data.get("embedding_chunks", []))  

    title    = data.get("title", "")
    region   = data.get("region", "")
    diff     = data.get("difficulty", "")
    dur      = data.get("duration_days")
    alt      = data.get("max_altitude_m")
    summary  = data.get("summary", "")

    
    if "difficulty_region" not in existing_types:
        parts = []
        if diff:    parts.append(f"Difficulty: {diff}")
        if region:  parts.append(f"Region: {region}")
        if dur:     parts.append(f"Duration: {dur} days")
        if alt:     parts.append(f"Maximum altitude: {alt} m")
        if parts:
            chunks.append({
                "type": "difficulty_region",
                "text": f"{title}. {'. '.join(parts)}."
            })

   
    highlights = data.get("highlights", [])
    if highlights and "highlights" not in existing_types:
        text = f"Highlights of {title}: " + "; ".join(highlights) + "."
        chunks.append({"type": "highlights", "text": text})

   
    includes = data.get("includes", [])
    if includes and "includes_summary" not in existing_types:
        
        sample = includes[:8]
        text   = f"What is included in {title}: " + ". ".join(sample) + "."
        chunks.append({"type": "includes_summary", "text": text})

  
    itinerary = data.get("itinerary", [])
    if itinerary and "full_itinerary" not in existing_types:
        day_lines = []
        for day in itinerary:
            day_lines.append(
                f"Day {day['day']}: {day.get('title', '')} — {day.get('description', '')}"
            )
        text = f"Complete itinerary for {title}:\n" + "\n".join(day_lines)
        chunks.append({"type": "full_itinerary", "text": text})

    
    tags = data.get("tags", [])
    cat  = data.get("category", "")
    if tags and "tags" not in existing_types:
        text = f"{title} ({cat}) tags: " + ", ".join(tags) + "."
        chunks.append({"type": "tags", "text": text})

    return chunks


def enrich_all():
    json_files = glob.glob(os.path.join(RAG_READY, "**", "*.json"), recursive=True)
    json_files = [p for p in json_files if "metadata_index" not in p]

    print(f"\n📦 Found {len(json_files)} rag_ready JSON files\n")

    updated = 0
    for fpath in sorted(json_files):
        with open(fpath, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"  ⚠️  Skip (parse error): {fpath}  — {e}")
                continue

        old_count = len(data.get("embedding_chunks", []))
        new_chunks = build_chunks(data)

        if len(new_chunks) == old_count:
            continue   

        
        shutil.copy2(fpath, fpath + ".bak")

        data["embedding_chunks"] = new_chunks

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        title = data.get("title", os.path.basename(fpath))
        print(f"  ✅  {title[:55]:<55}  chunks: {old_count} → {len(new_chunks)}")
        updated += 1

    print(f"\n✨  Done. {updated} files enriched.\n")
    print("⚠️  You must re-index (delete data/vector_db/ and restart) for")
    print("    the new chunks to take effect.\n")


if __name__ == "__main__":
    enrich_all()