import sys
sys.path.insert(0, 'src')
from core.pipeline import Pipeline

qs = [
    'Compare EBC vs ABC',
    'Compare EBC vs Langtang Valley trek',
    'How can I book a trip to Everest Base Camp',
]

pipeline = Pipeline(debug=True)
for q in qs:
    try:
        resp = pipeline.run(q)
        print('Q:', q)
        print('A:', resp.answer)
        print('intent:', resp.intent.value, 'strategy:', resp.strategy.value, 'used_llm:', resp.used_llm, 'fallback:', resp.is_fallback)
    except Exception as e:
        print('Q:', q, 'ERROR:', e)
    print('-'*60)
