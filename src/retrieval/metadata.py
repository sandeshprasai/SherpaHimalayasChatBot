import os
import json
from typing import Any
from src.config.settings import settings
from src.services.logger import logger

class MetadataRetriever:
    def __init__(self):
        path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        if not os.path.exists(path):
            self._records = []
            self.schema_keys = []
            logger.warning("metadata_index_missing", extra={"path": path})
            return
        with open(path, "r", encoding="utf-8") as f:
            self._records = json.load(f)

        keys = set()
        for r in self._records:
            keys.update(r.keys())
        self.schema_keys = sorted(list(keys))

        logger.info("metadata_index_loaded", extra={
            "path": path,
            "record_count": len(self._records),
            "schema_key_count": len(self.schema_keys),
        })

    def execute_plan(self, plan: Any) -> list[dict]:
        results = self._records
        if plan.filters:
            for k, v in plan.filters.items():
                if k not in self.schema_keys:
                    logger.warning("metadata_filter_key_unknown", extra={
                        "key": k, "known_keys_sample": self.schema_keys[:10]
                    })
                    continue

                # Structured operator filter: {"operator": ">", "value": 7000}
                if isinstance(v, dict) and "value" in v:
                    op = str(v.get("operator", "==")).strip()
                    target = v.get("value")
                    target_num = None
                    if isinstance(target, (int, float)):
                        target_num = target
                    elif isinstance(target, str):
                        cleaned = target.replace(",", "").strip()
                        try:
                            target_num = float(cleaned)
                        except ValueError:
                            target_num = None

                    def matches(r, op=op, target=target, target_num=target_num, k=k):
                        field_val = r.get(k)
                        if field_val is None:
                            return False
                        field_num = None
                        if isinstance(field_val, (int, float)):
                            field_num = field_val
                        elif isinstance(field_val, str):
                            cleaned = field_val.replace(",", "").strip()
                            try:
                                field_num = float(cleaned)
                            except ValueError:
                                field_num = None

                        if target_num is not None and field_num is not None:
                            if op in (">", "gt"): return field_num > target_num
                            if op in (">=", "gte"): return field_num >= target_num
                            if op in ("<", "lt"): return field_num < target_num
                            if op in ("<=", "lte"): return field_num <= target_num
                            if op in ("==", "=", "eq"): return field_num == target_num
                            if op in ("!=", "ne"): return field_num != target_num
                        return str(target).lower() in str(field_val).lower()

                    before = len(results)
                    results = [r for r in results if matches(r)]
                    logger.info("metadata_operator_filter_applied", extra={
                        "field": k, "operator": op, "value": target,
                        "records_before": before, "records_after": len(results),
                    })

                elif v:
                    before = len(results)
                    results = [r for r in results if str(v).lower() in str(r.get(k, "")).lower()]
                    logger.info("metadata_substring_filter_applied", extra={
                        "field": k, "value": v,
                        "records_before": before, "records_after": len(results),
                    })

        if plan.sort_field and plan.sort_field in self.schema_keys:
            rev = (plan.sort_direction.lower() == "desc")
            sortable = [r for r in results if r.get(plan.sort_field) is not None]

            def sort_key(x):
                val = x.get(plan.sort_field)
                if isinstance(val, (int, float)): return val
                if isinstance(val, str) and val.replace('.','',1).isdigit(): return float(val)
                return 0

            results = sorted(sortable, key=sort_key, reverse=rev)

        final = results[:plan.limit]
        logger.info("metadata_plan_executed", extra={
            "filters": plan.filters,
            "sort_field": plan.sort_field,
            "sort_direction": plan.sort_direction,
            "limit": plan.limit,
            "result_count": len(final),
        })
        return final