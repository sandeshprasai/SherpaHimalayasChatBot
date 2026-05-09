import os
import json
from typing import Any
from config.settings import settings

class MetadataRetriever:
    def __init__(self):
        path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        if not os.path.exists(path):
            self._records = []
            self.schema_keys = []
            return
        with open(path, "r", encoding="utf-8") as f:
            self._records = json.load(f)
        
        keys = set()
        for r in self._records:
            keys.update(r.keys())
        self.schema_keys = sorted(list(keys))

    def execute_plan(self, plan: Any) -> list[dict]:
        results = self._records
        if plan.filters:
            for k, v in plan.filters.items():
                if v and k in self.schema_keys:
                    results = [r for r in results if str(v).lower() in str(r.get(k, "")).lower()]

        if plan.sort_field and plan.sort_field in self.schema_keys:
            rev = (plan.sort_direction.lower() == "desc")
            def sort_fn(x):
                val = x.get(plan.sort_field)
                if isinstance(val, (int, float)): return val
                if isinstance(val, str) and val.replace('.','',1).isdigit(): return float(val)
                return 0
            results = sorted(results, key=sort_fn, reverse=rev)
        return results[:plan.limit]