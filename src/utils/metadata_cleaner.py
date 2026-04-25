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

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAG_DIR = os.path.join(BASE, "data", "rag_ready")
OUT = os.path.join(RAG_DIR, "metadata_index.cleaned.json")


def parse_int(x):
    if x is None:
        return None
    if isinstance(x, int):
        return x
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
    if "easy" in s:
        return "Easy"
    if "moderate" in s:
        return "Moderate"
    if "challenging" in s:
        return "Challenging"
    if "strenuous" in s or "hard" in s:
        return "Strenuous"
    return s.capitalize()


def scan():
    records = []
    for root, _, files in os.walk(RAG_DIR):
        for fname in files:
            if not fname.endswith('.json'):
                continue
            if fname in ("metadata_index.json", "metadata_index.cleaned.json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"skip {path}: {e}")
                continue
            if isinstance(data, list):
                data = data[0] if data else {}
            if not isinstance(data, dict):
                continue

            title = data.get('title') or (data.get('metadata') or {}).get('title') or os.path.splitext(fname)[0]
            tdata = {}
            tdata['title'] = title

            # duration
            dur = data.get('duration_days') or data.get('duration') or (data.get('metadata') or {}).get('duration_days') or (data.get('metadata') or {}).get('duration')
            tdata['duration_days'] = parse_int(dur)

            # altitude
            alt = data.get('max_altitude_m') or data.get('max_altitude') or (data.get('metadata') or {}).get('max_altitude_m')
            tdata['max_altitude_m'] = parse_int(alt)

            # difficulty
            diff = data.get('difficulty') or (data.get('metadata') or {}).get('difficulty')
            tdata['difficulty'] = canonical_difficulty(diff)

            # region
            region = data.get('region') or (data.get('metadata') or {}).get('region')
            tdata['region'] = region

            # summary
            tdata['summary'] = data.get('summary') or (data.get('metadata') or {}).get('summary') or ""

            # itinerary
            tdata['itinerary'] = data.get('itinerary') or []

            # category/type detection
            typ = (data.get('category') or (data.get('metadata') or {}).get('category') or "")
            typ_lower = str(typ).lower()
            title_lower = str(title).lower()
            if 'helicopter' in title_lower or 'helicopter' in typ_lower or 'fly' in typ_lower or 'tour' in typ_lower:
                tdata['category'] = 'tour'
            elif 'expedition' in typ_lower or 'peak' in typ_lower or 'climb' in typ_lower:
                tdata['category'] = 'expedition'
            else:
                tdata['category'] = 'trek' if 'trek' in title_lower or 'trek' in typ_lower else (typ or 'info')

            # source path relative to project root
            tdata['_source'] = os.path.relpath(path, BASE).replace('\\', '/')

            records.append(tdata)
    return records


def main(apply=False):
    recs = scan()
    print(f"Found {len(recs)} records. Writing cleaned metadata to {OUT}")
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    if apply:
        orig = os.path.join(RAG_DIR, 'metadata_index.json')
        bak = orig + '.bak'
        if os.path.exists(orig):
            os.replace(orig, bak)
        os.replace(OUT, orig)
        print(f"Replaced metadata_index.json (backup at {bak})")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='Replace original metadata_index.json')
    args = p.parse_args()
    main(args.apply)
