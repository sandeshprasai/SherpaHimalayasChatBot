from typing import Dict, Any


class ResponseNormalizer:
    VALID_SECTION_TYPES = {"cards", "list", "table", "text"}

    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return self._fallback(data)

        data.setdefault("answer", "Here is the information you requested.")
        data.setdefault("sections", [])
        data.setdefault("suggestions", [])

        normalized_sections = []

        for section in data["sections"]:
            if not isinstance(section, dict):
                continue

            # Ensure type is valid
            section_type = section.get("type", "list")
            if section_type not in self.VALID_SECTION_TYPES:
                section_type = "list"

            # --- FIX: HEAL ITEMS LIST ---
            raw_items = section.get("items", [])
            normalized_items = []
            
            if isinstance(raw_items, list):
                for item in raw_items:
                    if isinstance(item, dict):
                        normalized_items.append(item)
                    elif isinstance(item, str):
                        # If LLM sent a string, wrap it in a dict to satisfy Pydantic
                        normalized_items.append({"content": item})
            
            # --- FIX: HANDLE "TEXT" TYPE PROPERLY ---
            # If it's a text section but LLM put text in items, move it to content
            content = section.get("content")
            if section_type == "text" and not content and normalized_items:
                content = normalized_items[0].get("content")

            normalized_sections.append({
                "type": section_type,
                "title": section.get("title", "Details"),
                "items": normalized_items,
                "content": content
            })

        data["sections"] = normalized_sections
        return data

    def _fallback(self, raw):
        return {
            "answer": str(raw),
            "sections": [],
            "suggestions": []
        }