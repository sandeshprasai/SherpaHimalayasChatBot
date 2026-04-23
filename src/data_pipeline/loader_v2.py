import json
import os
import re
from llama_index.core import Document


def parse_markdown(filepath):
    """Parse a cleaned markdown file to extract structured metadata and text."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.rstrip() for l in f]

    title = None
    for l in lines:
        if l.strip().startswith('#'):
            title = l.strip().lstrip('#').strip()
            break
    if not title:
        # fallback to filename
        title = os.path.splitext(os.path.basename(filepath))[0].replace('-', ' ').title()

    # Find Trip Information block
    metadata = {
        'title': title,
        'region': None,
        'start': None,
        'trailhead': None,
        'difficulty': None,
        'max_altitude_m': None,
        'duration_days': None,
        'highlights': [],
        'includes': [],
        'itinerary': []
    }

    content_lines = []
    in_itinerary = False
    for i, l in enumerate(lines):
        low = l.strip().lower()
        if 'trip information' in low:
            # read next ~12 non-empty lines as metadata candidates
            for nl in lines[i+1:i+15]:
                nl = nl.strip()
                if not nl:
                    continue
                # altitude
                m = re.search(r"(\d{3,5})\s?m", nl.lower())
                if m and not metadata['max_altitude_m']:
                    metadata['max_altitude_m'] = int(m.group(1))
                    continue
                # duration (days)
                if re.fullmatch(r"\d{1,2}", nl):
                    val = int(nl)
                    if not metadata['duration_days']:
                        metadata['duration_days'] = val
                        continue
                # difficulty heuristics
                if any(word in nl for word in ['easy', 'normal', 'moderate', 'difficult', 'strenuous']):
                    metadata['difficulty'] = nl
                    continue
                # region detection (capitalized words)
                if metadata['region'] is None and len(nl.split()) <= 3 and nl[0].isupper():
                    metadata['region'] = nl
                    continue
                # highlights
                if 'poon' in nl.lower() or 'rhododendron' in nl.lower() or 'view' in nl.lower():
                    metadata['highlights'].append(nl)
                    continue
                # includes
                if 'include' in nl.lower() or nl.lower().startswith('included'):
                    # attempt to split by commas
                    parts = re.split(r"[:,]", nl, maxsplit=1)
                    if len(parts) > 1:
                        items = [p.strip() for p in parts[1].split(',') if p.strip()]
                        metadata['includes'].extend(items)
                    else:
                        metadata['includes'].append(nl)
                    continue
        if 'itinerary' in low:
            in_itinerary = True
            continue
        if in_itinerary:
            if re.search(r"day\s*\d", l, re.IGNORECASE):
                # new day entry
                metadata['itinerary'].append(l.strip('- ').strip())
            elif l.strip().startswith('- day') or l.strip().lower().startswith('day'):
                metadata['itinerary'].append(l.strip('- ').strip())
            else:
                # continuation of last day
                if metadata['itinerary']:
                    metadata['itinerary'][-1] += ' ' + l.strip()
                else:
                    # still treat as content
                    content_lines.append(l)
        else:
            content_lines.append(l)

    # Fallbacks: try to find numbers elsewhere
    if not metadata['duration_days']:
        # search for patterns like '10 Days' or '10 days'
        txt = '\n'.join(lines[:60])
        m = re.search(r"(\d{1,2})\s+days", txt.lower())
        if m:
            metadata['duration_days'] = int(m.group(1))

    # Build a consolidated text block used for indexing
    text = f"TITLE: {metadata['title']}\n"
    text += f"TYPE: trek\n"
    text += f"REGION: {metadata.get('region') or 'Unknown'}\n"
    text += f"DURATION (DAYS): {metadata.get('duration_days') or 'Unknown'}\n"
    text += f"DIFFICULTY: {metadata.get('difficulty') or 'Unknown'}\n"
    if metadata.get('max_altitude_m'):
        text += f"MAXIMUM ALTITUDE: {metadata.get('max_altitude_m')} meters\n"
    text += '\n'.join(content_lines[:2000])

    return Document(text=text, metadata=metadata)


def load_docs(base_path):
    documents = []
    for root, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                if file.endswith('.json'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # build text similar to previous implementation
                    text = f"TITLE: {data.get('title')}\n"
                    text += f"TYPE: {data.get('category', data.get('type', 'trek'))}\n"
                    text += f"REGION: {data.get('region', 'Unknown')}\n"
                    text += f"DURATION (DAYS): {data.get('duration_days', 'Unknown')}\n"
                    text += f"DIFFICULTY: {data.get('difficulty', 'Unknown')}\n"
                    if 'max_altitude_m' in data:
                        text += f"MAXIMUM ALTITUDE: {data['max_altitude_m']} meters\n"
                    text += f"SUMMARY: {data.get('summary', '')}\n"
                    if 'itinerary' in data:
                        text += f"FULL ITINERARY: {json.dumps(data['itinerary'])}\n"
                    metadata = {k: data.get(k) for k in ['type','id','region','duration_days','max_altitude_m','difficulty','title']}
                    documents.append(Document(text=text, metadata=metadata))
                elif file.endswith('.md'):
                    doc = parse_markdown(full_path)
                    documents.append(doc)
            except Exception as e:
                print(f"Skipping {full_path}: {e}")
    print(f"Successfully loaded {len(documents)} structured documents.")
    return documents
