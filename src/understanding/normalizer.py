import re
import json
import os
from config.settings import settings

class DynamicNormalizer:
    def __init__(self):
        self.alias_map = {}
        self._load_aliases()

    def load_aliases(self):
        """Dynamic loading from metadata index to handle all treks automatically."""
        meta_path = settings.cleaned_metadata_index if os.path.exists(settings.cleaned_metadata_index) else settings.metadata_index
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    records = json.load(f)
                    for r in records:
                        canonical = r.get("title")
                        if not canonical: continue
                        
                        # 1. Map aliases from JSON (e.g., EBC, ABC)
                        for a in r.get("aliases", []):
                            self.alias_map[a.lower()] = canonical
                        
                        # 2. Map short names (e.g., "Ghorepani" -> "Ghorepani Circuit Trek")
                        short = canonical.split(" Trek")[0].split(" Expedition")[0].lower()
                        if len(short) > 3:
                            self.alias_map[short] = canonical
            except Exception as e:
                print(f"Normalizer Warning: Could not load metadata aliases: {e}")

    def normalize(self, text: str) -> str:
        if not text: return ""
        
        # Expansion for common contractions
        text = re.sub(r"(?i)\bwhats\b", "what is", text)
        text = re.sub(r"(?i)\bwhat's\b", "what is", text)
        
        words = text.split()
        normalized_words = []
        
        for w in words:
            # Clean punctuation from word for lookup (e.g., "EBC?" -> "ebc")
            clean_w = re.sub(r'[^\w]', '', w.lower())
            
            # If the word is a known alias, replace it with the full title
            if clean_w in self.alias_map:
                normalized_words.append(self.alias_map[clean_w])
            else:
                normalized_words.append(w)
        
        res = " ".join(normalized_words)
        # Collapse multiple spaces
        res = re.sub(r"\s+", " ", res)
        return res.strip()

    # Alias for internal use
    _load_aliases = load_aliases

# --- CRITICAL FIX: Add these lines at the bottom ---

# Create a singleton instance
_inst = DynamicNormalizer()

# Export the standalone function that Pipeline is looking for
def normalize(text: str) -> str:
    return _inst.normalize(text)