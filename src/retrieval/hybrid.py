import json
from core.schema import QueryRequest, SourceNode

class HybridRetriever:
    def __init__(self):
        from retrieval.metadata import MetadataRetriever
        from retrieval.vector import VectorRetriever
        self._meta = MetadataRetriever()
        self._vec = VectorRetriever()

    def search(self, req: QueryRequest, top_k: int = 5) -> tuple[list[SourceNode], str | None]:
        # 1. Database Logic Filter
        candidates = self._meta.filter_by_logic(req.filters)
        if not candidates: return [], None

        # 2. Vector Semantic Rerank
        query_text = req.normalized or req.text
        vec_results = self._vec.search(query_text, top_k=20)
        
        score_map = {n.title.lower().strip(): n.score for n in vec_results if n.title}
        
        scored = []
        for c in candidates:
            score = score_map.get(c['title'].lower().strip(), 0.1)
            scored.append((c, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [x[0] for x in scored[:top_k]]
        
        # 3. Return raw nodes so the LLM can explain the choices
        nodes = [
            SourceNode(
                text=f"FACT SHEET: {json.dumps(r)}", 
                title=r['title'],
                source_id=r.get('id')
            ) for r in top
        ]
        return nodes, None