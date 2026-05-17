import re

class DynamicNormalizer:
    def __init__(self, records: list[dict]):
        self.alias_map = {}
        for r in records:
            # Domain-Agnostic identity discovery
            primary_name = None
            for key in ["canonical_title", "title", "name", "id"]:
                if r.get(key):
                    primary_name = r.get(key)
                    break
            
            if primary_name:
                self.alias_map[str(primary_name).lower()] = str(primary_name)
                # Map generic aliases list from data
                aliases = r.get("aliases", [])
                if isinstance(aliases, list):
                    for a in aliases:
                        self.alias_map[str(a).lower()] = str(primary_name)

    def normalize(self, text: str) -> str:
        if not text: return ""
        res = text
        for key in sorted(self.alias_map.keys(), key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            res = pattern.sub(self.alias_map[key], res)
        return res

def normalize(text: str, records: list[dict]) -> str:
    return DynamicNormalizer(records).normalize(text)