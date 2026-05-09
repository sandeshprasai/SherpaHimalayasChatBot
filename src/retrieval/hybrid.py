import json
# Added SourceNode to the import list below
from core.schema import QueryRequest, QueryIntent, SourceNode 

class HybridRetriever:
    def __init__(self):
        from retrieval.metadata import MetadataRetriever
        from retrieval.vector import VectorRetriever
        self._meta = MetadataRetriever()
        self._vec = VectorRetriever()

    def search(self, req: QueryRequest, top_k: int = 5):
        # 1. Apply logic filters (Region, etc.)
        candidates = self._meta.filter_by_logic(req.filters)
        
        # 2. Apply Superlative Sort if requested
        if req.intent == QueryIntent.AGGREGATE:
            field = req.agg_field or "duration_days"
            reverse = (req.agg_op == "max")
            # Mathematical sort of the filtered list
            candidates = [c for c in candidates if c.get(field)]
            candidates.sort(key=lambda x: x[field], reverse=reverse)

        # 3. If it's a general recommendation, use Vector rerank
        elif req.intent == QueryIntent.FILTER:
            # Re-rank based on semantic similarity
            query_text = req.normalized or req.text
            try:
                vec_results = self._vec.search(query_text, top_k=20)
                # Build score map for quick lookup
                score_map = {v.title.lower().strip(): v.score for v in vec_results if v.title}
                # Rerank candidates by vector score
                candidates.sort(
                    key=lambda x: score_map.get(x.get('title', '').lower().strip(), 0.0), 
                    reverse=True
                )
            except Exception:
                pass # Fallback to default metadata order if vector fails

        # Limit results based on extracted top_k or default
        limit = req.top_k or top_k
        top = candidates[:limit]
        
        # Return as list of SourceNodes for the LLM to process
        return [SourceNode(text=f"DATA: {json.dumps(r)}", title=r['title']) for r in top], None