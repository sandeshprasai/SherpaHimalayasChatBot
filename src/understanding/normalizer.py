import re

class DynamicNormalizer:
    def __init__(self, records: list[dict]):
        self.alias_map = {}
        for r in records:
            name = r.get("title") or r.get("name") or r.get("id")
            if name:
                self.alias_map[str(name).lower()] = str(name)
                # Map any aliases provided in data (e.g. EBC -> Everest Base Camp)
                aliases = r.get("aliases", [])
                if isinstance(aliases, list):
                    for a in aliases:
                        self.alias_map[str(a).lower()] = str(name)

    def normalize(self, text: str) -> str:
        if not text: return ""
        res = text
        for key in sorted(self.alias_map.keys(), key=len, reverse=True):
            pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
            res = pattern.sub(self.alias_map[key], res)
        return res

def normalize(text: str, records: list[dict]) -> str:
    return DynamicNormalizer(records).normalize(text)