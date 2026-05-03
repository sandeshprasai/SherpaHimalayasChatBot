import os
import json
import re
from typing import List, Optional

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
            m = re.search(r"(\d+)", v)
            if m:
                return int(m.group(1))
    return None


def _is_trek(doc: dict) -> bool:
    md = doc.get('metadata', {}) or {}
    typ = str(md.get('type', '')).lower()
    if 'trek' in typ:
        return True
    # fallback: check title/text for the word 'trek'
    title = str(md.get('title') or doc.get('text', ''))
    if 'trek' in title.lower():
        return True
    return False


def shortest_treks(top_k: int = 1) -> List[dict]:
    docs = ensure_docstore_loaded()
    items = []
    for d in docs:
        if not _is_trek(d):
            continue
        md = d.get('metadata', {})
        dur = _get_numeric(md, 'duration_days') or _get_numeric(md, 'duration')
        if dur is None:
            continue
        items.append((dur, d))
    items.sort(key=lambda x: x[0])
    return [x[1] for x in items[:top_k]]


def shortest_trek() -> Optional[dict]:
    lst = shortest_treks(1)
    return lst[0] if lst else None


def longest_treks(top_k: int = 1) -> List[dict]:
    docs = ensure_docstore_loaded()
    items = []
    for d in docs:
        if not _is_trek(d):
            continue
        md = d.get('metadata', {})
        dur = _get_numeric(md, 'duration_days') or _get_numeric(md, 'duration')
        if dur is None:
            continue
        items.append((dur, d))
    items.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in items[:top_k]]


def longest_trek() -> Optional[dict]:
    lst = longest_treks(1)
    return lst[0] if lst else None


def highest_altitude(top_k: int = 1) -> Optional[List[dict]]:
    docs = ensure_docstore_loaded()
    items = []
    for d in docs:
        md = d.get('metadata', {})
        alt = _get_numeric(md, 'max_altitude_m') or _get_numeric(md, 'max_altitude')
        if alt is None:
            continue
        items.append((alt, d))
    items.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in items[:top_k]]


def easiest_trek() -> Optional[dict]:
    # simplistic: prefer difficulty labels containing 'easy' or 'normal'
    docs = ensure_docstore_loaded()
    for d in docs:
        if not _is_trek(d):
            continue
        md = d.get('metadata', {})
        diff = str(md.get('difficulty', '')).lower()
        if 'easy' in diff or 'normal' in diff:
            return d
    return None


def find_by_name(name: str) -> Optional[dict]:
    docs = ensure_docstore_loaded()
    name = name.lower()
    for d in docs:
        md = d.get('metadata', {})
        title = str(md.get('title') or d.get('text','')).lower()
        if name in title:
            return d
    return None
