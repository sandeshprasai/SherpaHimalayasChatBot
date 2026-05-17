
import os
import json
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
IN_PATH = os.path.join(ROOT, 'data', 'test_results.json')
OUT_PATH = os.path.join(ROOT, 'data', 'test_summary.json')

try:
    with open(IN_PATH, encoding='utf-8') as f:
        results = json.load(f)
except Exception as e:
    print('ERROR: failed to load test_results.json:', e)
    raise

total = len(results)
used_llm = sum(1 for r in results if r.get('used_llm'))
fallback = sum(1 for r in results if r.get('is_fallback'))
clar = sum(1 for r in results if r.get('needs_clarification'))
errors = sum(1 for r in results if 'error' in r)

intent_counts = Counter(r.get('intent') for r in results if r.get('intent'))

# Which queries had no sources (sources_count == 0)
no_source = [r for r in results if r.get('sources_count') == 0]
no_source_sample = [r.get('query') for r in no_source[:10]]


def categorize(q: str) -> str:
    ql = (q or '').lower()
    if any(k in ql for k in ('book', 'booking', 'deposit', 'refund', 'cancel', 'cancellation', 'change my mind', 'reschedule')):
        return 'booking'
    if any(k in ql for k in ('permit', 'permits', 'tim', 'tim s', 'tim card', 'trek permit')):
        return 'permit'
    if any(k in ql for k in ('pack', 'packing', 'what to bring', 'what should i pack', 'gear')):
        return 'packing'
    if any(k in ql for k in ('price', 'cost', 'fee', 'charge', 'pricing')):
        return 'price'
    if any(k in ql for k in ('how do i', 'how to', 'procedure', 'process', 'steps', 'what is the booking procedure')):
        return 'procedure'
    return 'other'

fallback_list = [r for r in results if r.get('is_fallback')]
fb_by_cat = Counter(categorize(r.get('query','')) for r in fallback_list)
fb_sample = [r.get('query') for r in fallback_list[:10]]

summary = {
    'total_queries': total,
    'used_llm_count': used_llm,
    'fallback_count': fallback,
    'clarifications_requested': clar,
    'errors': errors,
    'intent_counts': dict(intent_counts),
    'no_source_count': len(no_source),
    'no_source_sample': no_source_sample,
    'fallback_by_category': dict(fb_by_cat),
    'fallback_sample_queries': fb_sample,
}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('Test results summary:')
print(json.dumps(summary, indent=2))
print('\nWrote summary to', OUT_PATH)
