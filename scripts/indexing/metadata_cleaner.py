
import os
import json
import re
import argparse
from collections import defaultdict

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAG_DIR = os.path.join(BASE, "data", "rag_ready")
OUT = os.path.join(RAG_DIR, "metadata_index.cleaned.json")
ORIG = os.path.join(RAG_DIR, "metadata_index.json")

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from understanding.normalization import (
    STOPWORDS_NORM,
    ABBR_EXCLUDE,
    REGION_KEYWORDS,
    parse_int,
    canonical_difficulty,
    normalize_title,
    abbrev,
    find_duration,
    find_altitude,
    infer_region,
)



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

            # category detection (enhanced)
            typ = (data.get('category') or (data.get('metadata') or {}).get('category') or '')
            typ_lower = str(typ).lower()
            title_lower = (tdata['title'] or '').lower()
            summary = data.get('summary') or (data.get('metadata') or {}).get('summary') or ''
            summary_lower = str(summary).lower()

            # keyword sets
            tour_keywords = ('helicopter', 'fly', 'tour', 'rafting', 'raft', 'safari', 'kayak', 'kayaking', 'whitewater', 'city tour', 'sightseeing', 'package', 'activity')
            expedition_keywords = ('expedition', 'peak', 'climb', 'summit', 'mount')
            trek_keywords = ('trek', 'trekking', 'circuit', 'trail', 'valley', 'pass', 'la pass', 'route')

            cat = None
            # explicit category provided
            if typ_lower in ('trek', 'trekking', 'tour', 'expedition', 'info', 'package', 'activity'):
                cat = 'trek' if typ_lower == 'trekking' else typ_lower

            # heuristics on title and summary
            if not cat:
                if any(k in title_lower or k in summary_lower for k in tour_keywords):
                    cat = 'tour'
                elif any(k in title_lower or k in summary_lower for k in expedition_keywords):
                    cat = 'expedition'
                elif any(k in title_lower or k in summary_lower for k in trek_keywords):
                    cat = 'trek'

            # normalize fallback
            if not cat:
                cat = typ_lower or 'info'

            # ensure rafting/kayak words force 'tour'
            if any(k in title_lower or k in summary_lower for k in ('rafting','raft','kayak','safari','whitewater')):
                cat = 'tour'

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

        # canonical category normalization and flags
        cat = (merged_rec.get('category') or '').strip().lower()
        if cat == 'trekking':
            cat = 'trek'
        allowed = {'trek', 'tour', 'expedition', 'info'}
        if cat not in allowed:
            title_s = (merged_rec.get('canonical_title') or merged_rec.get('title') or '').lower()
            summary_s = (merged_rec.get('summary') or '').lower()
            if any(k in title_s or k in summary_s for k in ('rafting','raft','safari','kayak','whitewater','tour','package','city tour','sightseeing')):
                cat = 'tour'
            elif any(k in title_s or k in summary_s for k in ('expedition','peak','climb','summit')):
                cat = 'expedition'
            elif any(k in title_s or k in summary_s for k in ('trek','trekking','circuit','trail','valley','pass','la pass')):
                cat = 'trek'
            else:
                cat = 'info'
        merged_rec['category'] = cat
        merged_rec['canonical_category'] = cat
        merged_rec['is_trek'] = cat == 'trek'
        merged_rec['is_tour'] = cat == 'tour'
        merged_rec['is_expedition'] = cat == 'expedition'

        # beginner friendly boolean
        diff = (merged_rec.get('difficulty') or '').lower()
        merged_rec['beginner_friendly'] = diff in ('easy', 'moderate', 'easy/moderate')

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
