import os
import json
from config.settings import settings
from core.schema import QueryRequest, QueryIntent

class MetadataRetriever:
    def __init__(self):
        meta_path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        with open(meta_path, encoding="utf-8") as f:
            all_data = json.load(f)
            
        # GLOBAL TYPE GUARD: Prevents team members from being trekking results
        self._records = [
            r for r in all_data 
            if str(r.get("category", "")).lower() in ["trek", "expedition", "peak", "tour"]
            and str(r.get("type", "")).lower() not in ["team", "company", "blog"]
        ]

    def answer(self, req: QueryRequest) -> str | None:
        """Main dispatcher for metadata-based queries."""
        if req.intent == QueryIntent.AGGREGATE:
            return self._aggregate(req)
        if req.intent == QueryIntent.COMPARE:
            return self._compare(req)
        return None

    def _aggregate(self, req: QueryRequest) -> str | None:
        """Handles: 'Which trek is the shortest?', 'Highest altitude peak', etc."""
        field = req.agg_field or "duration_days"
        op = req.agg_op or "max"
        
        candidates = []
        for r in self._records:
            val = r.get(field)
            if val is not None and isinstance(val, (int, float)):
                # Sanity Check: ignore altitudes > 9000m or < 100m
                if field == "max_altitude_m" and (val > 9000 or val < 100): continue
                candidates.append(r)

        if not candidates: return None

        winner = max(candidates, key=lambda x: x[field]) if op == "max" else min(candidates, key=lambda x: x[field])
        adj = "highest" if "altitude" in field else "longest" if op == "max" else "shortest"
        unit = "meters" if "altitude" in field else "days"
        
        return (f"The {adj} {field.replace('_', ' ')} belongs to **{winner['title']}**.\n\n"
                f"• **{field.replace('_', ' ').title()}:** {winner[field]} {unit}\n"
                f"• **Region:** {winner.get('region', 'N/A')}\n"
                f"• **Summary:** {winner.get('summary', '')}")

    def _compare(self, req: QueryRequest) -> str | None:
        """Handles: 'Compare X vs Y' dynamically for ANY two entities."""
        if not req.entities or len(req.entities) < 2: return None
        
        a, b = req.entities[0], req.entities[1]
        lines = [f"### Comparison: {a['title']} vs {b['title']}\n"]
        lines.append(f"| Feature | {a['title']} | {b['title']} |")
        lines.append("| :--- | :--- | :--- |")
        
        attrs = [("Duration", "duration_days", " days"), 
                 ("Max Altitude", "max_altitude_m", " m"),
                 ("Difficulty", "difficulty", ""), 
                 ("Region", "region", "")]

        for label, key, unit in attrs:
            val_a = f"{a.get(key, 'N/A')}{unit if a.get(key) else ''}"
            val_b = f"{b.get(key, 'N/A')}{unit if b.get(key) else ''}"
            lines.append(f"| {label} | {val_a} | {val_b} |")

        return "\n".join(lines)

    def filter_by_logic(self, filters: dict) -> list[dict]:
        """Logic used by Hybrid Retriever for 'Suggest me' queries."""
        results = self._records
        if region := filters.get("region"):
            results = [r for r in results if region.lower() in str(r.get("region", "")).lower()]
        if days := filters.get("duration_days"):
            results = [r for r in results if r.get("duration_days") and r["duration_days"] <= (days + 2)]
        return results