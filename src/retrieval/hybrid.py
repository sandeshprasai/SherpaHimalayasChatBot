import json
from src.core.schema import QueryRequest, QueryIntent, SourceNode 


class HybridRetriever:
    def __init__(self):
        from src.retrieval.metadata import MetadataRetriever
        from src.retrieval.vector import VectorRetriever
        self._meta = MetadataRetriever()
        self._vec = VectorRetriever()

    def search(self, req: QueryRequest, top_k: int = 5):
        candidates = self._meta.filter_by_logic(req.filters)
        
        if req.intent == QueryIntent.AGGREGATE:
            field = req.agg_field or "duration_days"
            reverse = (req.agg_op == "max")
            candidates = [c for c in candidates if c.get(field)]
            candidates.sort(key=lambda x: x[field], reverse=reverse)

        elif req.intent == QueryIntent.FILTER:
            query_text = req.normalized or req.text
            try:
                vec_results = self._vec.search(query_text, top_k=20)
                score_map = {v.title.lower().strip(): v.score for v in vec_results if v.title}
                candidates.sort(
                    key=lambda x: score_map.get(x.get('title', '').lower().strip(), 0.0), 
                    reverse=True
                )
            except Exception:
                pass 

        limit = req.top_k or top_k
        top = candidates[:limit]
        
        return [
        SourceNode(
        text=f"DATA: {json.dumps(r)}",
        title=r.get("title"),
        metadata=r
        )
        for r in top
        ], None