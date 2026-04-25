import json
import os
from llama_index.core import Document

def load_json_docs(base_path):
    documents = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # BUILD A VERY DETAILED TEXT BLOCK
                        text = f"TITLE: {data.get('title')}\n"
                        text += f"TYPE: {data.get('type', 'trek')}\n"
                        text += f"SUMMARY: {data.get('summary', '')}\n"
                        
                        if 'itinerary' in data:
                            text += f"FULL ITINERARY: {json.dumps(data['itinerary'])}\n"
                        
                        # ADD INCLUSIONS AND EXCLUSIONS
                        if 'includes' in data:
                            text += f"WHAT IS INCLUDED (INCLUSIONS): {', '.join(data['includes'])}\n"
                        if 'excludes' in data:
                            text += f"WHAT IS EXCLUDED (EXCLUSIONS): {', '.join(data['excludes'])}\n"
                        
                        if 'highlights' in data:
                            text += f"TRIP HIGHLIGHTS: {', '.join(data['highlights'])}\n"

                        if 'content_sections' in data:
                            for sec in data['content_sections']:
                                text += f"{sec.get('heading', '')}: {sec.get('text', '')}\n"

                        doc = Document(
                            text=text,
                            metadata={"type": data.get("type"), "id": data.get("id")}
                        )
                        documents.append(doc)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
                    
    print(f"✅ Successfully loaded {len(documents)} structured documents.")
    return documents