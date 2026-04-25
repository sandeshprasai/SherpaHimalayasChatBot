#!/usr/bin/env python3
"""
metadata_cleaner.py - Rebuild/clean metadata_index from data/rag_ready JSONs
Usage:
  python src/utils/metadata_cleaner.py [--apply]
Outputs: data/rag_ready/metadata_index.cleaned.json
"""

import os
import json
import re
import argparse
from collections import defaultdict

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAG_DIR = os.path.join(BASE, "data", "rag_ready")
OUT = os.path.join(RAG_DIR, "metadata_index.cleaned.json")
ORIG = os.path.join(RAG_DIR, "metadata_index.json")

# Stop-words used for normalizing titles (for grouping)
STOPWORDS_NORM = {
    "trek", "trekking", "the", "with", "and", "via", "to",
    "from", "by", "in", "of", "a", "an", "days", "day",
    "base", "camp", "valley", "pass", "lake", "peak",
    "expedition", "helicopter", "fly", "tour", "shortest",
    "longest", "short", "long",
}

# Words to exclude when generating abbreviations (we want meaningful initials)
ABBR_EXCLUDE = {"the", "and", "of", "in", "to", "by", "from", "a", "an", "trek", "trekking"}

REGION_KEYWORDS = [
    "annapurna", "everest", "langtang", "manaslu", "mustang",
    "kanchenjunga", "dolpa", "makalu", "rolwaling", "helambu",
    "pokhara", "nepal", "tsum",
]


def parse_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        m = re.search(r"(\d+)", x)
        if m:
            return int(m.group(1))
    try:
        return int(x)
    except Exception:
        return None


def canonical_difficulty(s):
    if not s:
        return None
    s = str(s).lower()
    if "easy/moderate" in s or ("easy" in s and "moderate" in s):
        return "Easy/Moderate"
    if "easy" in s and "moderate" not in s:
        return "Easy"
    if "moderate" in s:
        return "Moderate"
    if "challeng" in s or "hard" in s:
        return "Challenging"
    if "stren" in s:
        return "Strenuous"
    if "extreme" in s:
        return "Extreme"
    return s.capitalize()


def normalize_title(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    parts = [w for w in t.split() if w not in STOPWORDS_NORM]
    return " ".join(parts).strip()


def abbrev(title: str) -> str | None:
    if not title:
        return None
    toks = [w for w in re.split(r"\W+", title) if w and w.lower() not in ABBR_EXCLUDE]
    if not toks:
        return None
    a = ''.join(w[0].upper() for w in toks)
    return a if 2 <= len(a) <= 6 else None


def find_duration(text: str) -> int | None:
    if not text:
        return None
    # look for patterns like '16 days', '4-day', 'duration: 10 days'
    m = re.search(r"(\d{1,2})\s*(?:-\s*)?(?:days|day)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"duration[:\s]*(\d{1,2})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def find_altitude(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d{3,4})\s*(?:m|meters|metres)\b", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"max\s+altitude[:\s]*(\d{3,4})", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def infer_region(text: str) -> str | None:
    if not text:
        return None
    tl = text.lower()
    for r in REGION_KEYWORDS:
        if r in tl:
            return " ".join(w.capitalize() for w in r.split())
    return None


def scan_rag_ready():
    records = []
    for root, _, files in os.walk(RAG_DIR):
        for fname in sorted(files):
            if not fname.endswith('.json'):
                continue
            if fname in ("metadata_index.json", "metadata_index.cleaned.json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"skip {fpath}: {e}")
                continue

            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                continue

            title = data.get('title') or (data.get('metadata') or {}).get('title') or os.path.splitext(fname)[0]
            tdata = {}
            tdata['title'] = title.strip()
            tdata['id'] = data.get('id') or os.path.splitext(fname)[0]

            # duration
            dur = data.get('duration_days') or data.get('duration') or (data.get('metadata') or {}).get('duration_days') or (data.get('metadata') or {}).get('duration')
            tdata['duration_days'] = parse_int(dur)

            # altitude
            alt = data.get('max_altitude_m') or data.get('max_altitude') or (data.get('metadata') or {}).get('max_altitude_m') or (data.get('metadata') or {}).get('max_altitude')
            tdata['max_altitude_m'] = parse_int(alt)

            # difficulty
            diff = data.get('difficulty') or (data.get('metadata') or {}).get('difficulty')
            tdata['difficulty'] = canonical_difficulty(diff)

            # region
            region = data.get('region') or (data.get('metadata') or {}).get('region')
            if not region:
                region = infer_region(' '.join(filter(None, [tdata['title'], data.get('summary') or ''])))
            tdata['region'] = region

            # summary / itinerary
            tdata['summary'] = data.get('summary') or (data.get('metadata') or {}).get('summary') or ''
            tdata['itinerary'] = data.get('itinerary') or (data.get('metadata') or {}).get('itinerary') or []

            # category detection
            typ = (data.get('category') or (data.get('metadata') or {}).get('category') or '')
            typ_lower = str(typ).lower()
            title_lower = (tdata['title'] or '').lower()
            if 'helicopter' in title_lower or 'helicopter' in typ_lower or 'fly' in typ_lower or 'tour' in typ_lower:
                cat = 'tour'
            elif 'expedition' in typ_lower or 'peak' in typ_lower or 'climb' in typ_lower or 'peak' in title_lower:
                cat = 'expedition'
            elif 'trek' in title_lower or 'trek' in typ_lower:
                cat = 'trek'
            else:
                cat = typ_lower or 'info'
            tdata['category'] = cat

            tdata['_source'] = os.path.relpath(fpath, BASE).replace('\\', '/')

            # tags / highlights
            tdata['highlights'] = data.get('highlights') or (data.get('metadata') or {}).get('highlights') or []
            tdata['tags'] = data.get('tags') or (data.get('metadata') or {}).get('tags') or []

            # attempt to extract missing scalars from text
            text_blob = ' '.join(filter(None, [tdata['title'], tdata['summary']]))
            if tdata['duration_days'] is None:
                tdata['duration_days'] = find_duration(text_blob)
            if tdata['max_altitude_m'] is None:
                tdata['max_altitude_m'] = find_altitude(text_blob)

            records.append(tdata)
    return records


def dedupe_and_canonicalize(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in records:
        key_norm = normalize_title(r.get('title') or '')
        group_key = f"{key_norm}||{r.get('category','') or 'info'}"
        groups[group_key].append(r)

    merged = []
    for key, group in groups.items():
        # scoring: prefer records with more scalars populated and treks
        def score(rec: dict) -> int:
            s = 0
            s += 3 if rec.get('duration_days') else 0
            s += 2 if rec.get('max_altitude_m') else 0
            s += 2 if rec.get('difficulty') else 0
            s += 1 if rec.get('region') else 0
            s += 1 if rec.get('summary') else 0
            s += 2 if rec.get('itinerary') else 0
            s += 2 if rec.get('category') == 'trek' else 0
            s += len(rec.get('highlights') or [])
            return s

        best = max(group, key=lambda r: (score(r), len((r.get('summary') or ''))))
        merged_rec = dict(best)  # shallow copy

        # fill missing fields from other group members
        for field in ('duration_days', 'max_altitude_m', 'difficulty', 'region', 'summary', 'itinerary', 'highlights', 'tags'):
            if not merged_rec.get(field):
                for other in group:
                    val = other.get(field)
                    if val:
                        merged_rec[field] = val
                        break

        # aliases: other titles + abbreviation
        aliases = [other['title'] for other in group if other.get('title') and other['title'] != merged_rec.get('title')]
        ab = abbrev(merged_rec.get('title', ''))
        if ab:
            aliases.append(ab)
        merged_rec['aliases'] = sorted(list(dict.fromkeys([a for a in aliases if a])))

        # canonical title & slug id
        merged_rec['canonical_title'] = merged_rec.get('title')
        if not merged_rec.get('id'):
            slug = re.sub(r"[^a-z0-9]+", '-', (merged_rec.get('title') or '').lower()).strip('-')
            merged_rec['id'] = slug or merged_rec.get('title')

        # beginner friendly boolean
        diff = merged_rec.get('difficulty') or ''
        merged_rec['beginner_friendly'] = True if diff and diff.lower() in ('easy', 'moderate', 'easy/moderate', 'easy/moderate'.lower()) else False

        # explicit flags
        merged_rec['is_trek'] = merged_rec.get('category') == 'trek'
        merged_rec['is_tour'] = merged_rec.get('category') == 'tour'
        merged_rec['is_expedition'] = merged_rec.get('category') == 'expedition'

        # ensure types
        if isinstance(merged_rec.get('duration_days'), float):
            merged_rec['duration_days'] = int(merged_rec['duration_days'])
        if isinstance(merged_rec.get('max_altitude_m'), float):
            merged_rec['max_altitude_m'] = int(merged_rec['max_altitude_m'])

        merged.append(merged_rec)

    # sort alphabetically for stability
    merged.sort(key=lambda r: (r.get('category',''), r.get('canonical_title','')))
    return merged


def main(apply=False):
    recs = scan_rag_ready()
    print(f"Scanned {len(recs)} raw JSON records.")
    cleaned = dedupe_and_canonicalize(recs)
    print(f"Produced {len(cleaned)} cleaned records (after deduplication).")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"Wrote cleaned metadata to {OUT}")

    if apply:
        if os.path.exists(ORIG):
            bak = ORIG + '.bak'
            os.replace(ORIG, bak)
            print(f"Backed up original metadata_index.json to {bak}")
        os.replace(OUT, ORIG)
        print(f"Replaced metadata_index.json with cleaned index.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    main(args.apply)
