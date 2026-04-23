import os
import json
from typing import Optional

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DOCSTORE_PATH = os.path.join(BASE, "data/vector_db/docstore.json")


def ensure_docstore_loaded():
    if not os.path.exists(DOCSTORE_PATH):
        return []
    with open(DOCSTORE_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_numeric(metadata, key):
    v = metadata.get(key)
    try:
        return int(v)
    except Exception:
        # if contains 'm' strip
        if isinstance(v, str):
            import re
            m = re.search(r"(\d+)", v)
            if m:
                return int(m.group(1))
    return None


def shortest_trek():
    docs = ensure_docstore_loaded()
    best = None
    for d in docs:
        md = d.get('metadata', {})
        dur = _get_numeric(md, 'duration_days') or _get_numeric(md, 'duration')
        if dur is None:
            continue
        if best is None or dur < best[0]:
            best = (dur, d)
    return best[1] if best else None


def longest_trek():
    docs = ensure_docstore_loaded()
    best = None
    for d in docs:
        md = d.get('metadata', {})
        dur = _get_numeric(md, 'duration_days') or _get_numeric(md, 'duration')
        if dur is None:
            continue
        if best is None or dur > best[0]:
            best = (dur, d)
    return best[1] if best else None


def highest_altitude():
    docs = ensure_docstore_loaded()
    best = None
    for d in docs:
        md = d.get('metadata', {})
        alt = _get_numeric(md, 'max_altitude_m') or _get_numeric(md, 'max_altitude')
        if alt is None:
            continue
        if best is None or alt > best[0]:
            best = (alt, d)
    return best[1] if best else None


def easiest_trek():
    # simplistic: prefer difficulty labels containing 'easy' or 'normal'
    docs = ensure_docstore_loaded()
    for d in docs:
        md = d.get('metadata', {})
        diff = str(md.get('difficulty', '')).lower()
        if 'easy' in diff or 'normal' in diff:
            return d
    return None


def find_by_name(name: str):
    docs = ensure_docstore_loaded()
    name = name.lower()
    for d in docs:
        md = d.get('metadata', {})
        title = str(md.get('title') or d.get('text','')).lower()
        if name in title:
            return d
    return None
