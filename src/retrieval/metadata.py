# --- src/retrieval/metadata.py ---
import os, json
from config.settings import settings
from core.schema import QueryRequest, QueryIntent

class MetadataRetriever:
    def __init__(self):
        meta_path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        with open(meta_path, encoding="utf-8") as f:
            all_data = json.load(f)
        
        # Guard: Only allow travel products
        self._records = [
            r for r in all_data 
            if str(r.get("category", "")).lower() in ["trek", "expedition", "peak", "tour"]
            and str(r.get("type", "")).lower() not in ["team", "company", "blog"]
        ]

    def answer(self, req: QueryRequest) -> str | None:
        if req.intent == QueryIntent.AGGREGATE:
            return self._aggregate(req)
        if req.intent == QueryIntent.COMPARE:
            return self._compare(req)
        return None

    def _aggregate(self, req: QueryRequest) -> str | None:
        field = req.agg_field # e.g., "duration_days"
        op = req.agg_op       # e.g., "min"
        
        # DYNAMIC: Find all records that HAVE the requested field and it's a number
        candidates = [
            r for r in self._records 
            if r.get(field) is not None and isinstance(r[field], (int, float))
        ]
        
        if not candidates:
            return f"I found no records with a numeric '{field}' to compare."

        candidates.sort(key=lambda x: x[field], reverse=(op == "max"))
        limit = req.top_k or 1
        winners = candidates[:limit]

        # Dynamic Formatting
        label = field.replace("_", " ").title()
        if limit == 1:
            return f"The {op} {label} belongs to **{winners[0]['title']}** with {winners[0][field]}."
        
        res = [f"Top {limit} destinations by {label} ({op}):"]
        for i, w in enumerate(winners, 1):
            res.append(f"{i}. {w['title']} ({w[field]})")
        return "\n".join(res)

    # ... (Keep compare logic from previous response) ...