import json
import os
from src.config.settings import settings


class EntityResolver:
    def __init__(self):
        meta_path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        self.records = []
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.records = json.load(f)
        
        self.name_map = {}
        for r in self.records:
            title = r.get("title", "").lower()
            if title: self.name_map[title] = r
            for alias in r.get("aliases", []):
                self.name_map[alias.lower()] = r
            short = title.replace(" trek", "").replace(" expedition", "").strip()
            if len(short) > 4: self.name_map[short] = r

    def resolve(self, query: str) -> list[dict]:
        query_clean = query.lower()
        found = []
        # Match longest names first (e.g., "Everest Base Camp" vs "Everest")
        keys = sorted(self.name_map.keys(), key=len, reverse=True)
        for key in keys:
            if key in query_clean:
                if self.name_map[key] not in found:
                    found.append(self.name_map[key])
                query_clean = query_clean.replace(key, "")
        return found

    def find_record(self, name_or_title: str) -> dict | None:
        """Helper to find a specific record by its name."""
        return self.name_map.get(name_or_title.lower())