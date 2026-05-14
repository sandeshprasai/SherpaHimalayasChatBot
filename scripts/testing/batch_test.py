
import os
import sys
import json
import random
import time
from typing import List

# Ensure src is on path
ROOT = os.path.abspath(os.getcwd())
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from core.pipeline import Pipeline

OUT_PATH = os.path.join(ROOT, 'data', 'test_results.json')
META_PATH = os.path.join(ROOT, 'data', 'rag_ready', 'metadata_index.json')


def load_metadata():
    try:
        with open(META_PATH, encoding='utf-8') as f:
            records = json.load(f)
        titles = [r.get('title') for r in records if r.get('title')]
        regions = sorted(list({(r.get('region') or '').strip() for r in records if r.get('region')}))
        return records, titles, regions
    except Exception as e:
        print('Failed to load metadata:', e)
        return [], [], []


def build_queries(titles: List[str], regions: List[str], n: int = 100) -> List[str]:
    templates = [
        "What is the duration of {title}?",
        "Describe the full itinerary of {title}.",
        "What is included in the {title} package?",
        "How do I book a trip to {title}?",
        "Compare {title} vs {other}.",
        "Suggest a 5-day trek in {region}.",
        "Which trek is the easiest?",
        "Find me 3 beginner-friendly treks",
        "Top 3 shortest treks",
        "What permits are required for {title}?",
        "How do I cancel the booking?",
        "What should I pack for a trek?",
        "Is {title} suitable for beginners?",
        "What is the max altitude for {title}?",
        "Which trek has the highest altitude?",
        "Which trek is the most difficult?",
        "List trek destinations in {region}.",
        "Are there any helicopter tours to Annapurna Base Camp?",
        "What is the booking procedure for trekking?",
        "Do I need a permit for {region}?",
    ]

    queries = []
    if not titles:
        # fallback generic queries
        for i in range(n):
            queries.append(random.choice(templates).format(title='Everest Base Camp Trek', other='Annapurna Base Camp Trek', region='Annapurna'))
        return queries

    for i in range(n):
        tmpl = random.choice(templates)
        if '{title}' in tmpl and '{other}' in tmpl:
            a = random.choice(titles)
            b = random.choice(titles)
            while b == a and len(titles) > 1:
                b = random.choice(titles)
            q = tmpl.format(title=a, other=b, region=random.choice(regions) if regions else 'Annapurna')
        elif '{title}' in tmpl:
            a = random.choice(titles)
            q = tmpl.format(title=a, other=random.choice(titles), region=random.choice(regions) if regions else 'Annapurna')
        elif '{region}' in tmpl:
            region = random.choice(regions) if regions else 'Annapurna'
            q = tmpl.format(region=region)
        else:
            q = tmpl
        queries.append(q)

    return queries


def run_batch(n=100):
    records, titles, regions = load_metadata()
    queries = build_queries(titles, regions, n)

    pipeline = Pipeline(debug=False)
    results = []

    for idx, q in enumerate(queries, start=1):
        print(f"[{idx}/{len(queries)}] Q: {q}")
        start = time.time()
        try:
            resp = pipeline.run(q)
            elapsed = time.time() - start

            res = {
                'query': q,
                'answer': (resp.answer or '')[:400],
                'intent': resp.intent.value if getattr(resp, 'intent', None) else None,
                'strategy': resp.strategy.value if getattr(resp, 'strategy', None) else None,
                'used_llm': bool(getattr(resp, 'used_llm', False)),
                'is_fallback': bool(getattr(resp, 'is_fallback', False)),
                'needs_clarification': bool(getattr(resp, 'needs_clarification', False)),
                'clarification': getattr(resp, 'clarification', None),
                'sources_count': len(resp.sources) if getattr(resp, 'sources', None) else 0,
                'top_source_title': (resp.sources[0].title if getattr(resp, 'sources', None) and len(resp.sources)>0 else None),
                'tokens_used': getattr(resp, 'tokens_used', None) or getattr(resp, 'tokens_used', None) or None,
                'time_s': round(elapsed, 3),
            }

            # If clarification requested, attempt a follow-up by selecting first option
            if res['needs_clarification'] and res['clarification']:
                clar = res['clarification']
                # find first numbered option
                picked = None
                for line in clar.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if line[0].isdigit():
                        parts = line.split('.', 1)
                        if len(parts) > 1:
                            picked = parts[1].strip()
                            break
                if picked:
                    print(f"  -> auto-followup selecting: {picked}")
                    fstart = time.time()
                    fresp = pipeline.run('Tell me about ' + picked)
                    fresp_time = time.time() - fstart
                    res['followup'] = {
                        'followup_query': 'Tell me about ' + picked,
                        'followup_answer': (fresp.answer or '')[:400],
                        'followup_used_llm': bool(getattr(fresp, 'used_llm', False)),
                        'followup_time_s': round(fresp_time, 3),
                    }

        except Exception as e:
            res = {'query': q, 'error': str(e)}

        results.append(res)
        # small pause to avoid tight loop
        time.sleep(0.05)

    # write results
    try:
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print('\nWrote results to', OUT_PATH)
    except Exception as e:
        print('Failed to write results:', e)

    # summary
    total = len(results)
    used_llm = sum(1 for r in results if r.get('used_llm'))
    fallback = sum(1 for r in results if r.get('is_fallback'))
    clar = sum(1 for r in results if r.get('needs_clarification'))
    errors = sum(1 for r in results if 'error' in r)

    print('\nSummary:')
    print(' total queries:', total)
    print(' used_llm count:', used_llm)
    print(' fallback count:', fallback)
    print(' clarifications requested:', clar)
    print(' errors:', errors)


if __name__ == '__main__':
    run_batch(100)
