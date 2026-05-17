import os
import json
import time
from dotenv import load_dotenv
from groq import Groq


base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
INPUT_BASE = os.path.join(project_root, "data/cleaned_separated")
OUTPUT_BASE = os.path.join(project_root, "data/rag_ready")

load_dotenv(os.path.join(project_root, ".env"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. THE GENERALIZED PROMPT
UNIVERSAL_SYSTEM_PROMPT = """
You are a high-precision data extraction engine for Sherpa Himalaya.
Task: Convert the provided text into a structured JSON format.

Detection Logic:
- If the text describes a person (e.g. guide, founder), set type to "team".
- If the text is a policy, terms, or about us, set type to "company".
- If the text is an article or news, set type to "blog".
- If it's a travel guide, visa info, or FAQ, set type to "info".

JSON Schema:
{
  "id": "lowercase_slug_with_underscores",
  "type": "blog|team|company|info",
  "title": "Name of person, article, or page",
  "summary": "1-2 sentence overview",
  "attributes": {
     "key": "value" (e.g. role, experience, date, or category)
  },
  "content_sections": [
    {"heading": "Heading Title", "text": "Cleaned paragraph text"}
  ],
  "tags": ["3-5 keywords"],
  "embedding_chunks": [
    {"type": "main_content", "text": "The most important information for the AI to search."}
  ]
}
Return ONLY JSON. No explanations.
"""

def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(input_path)
    print(f"🧠 Universal Processing: {filename}")
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": UNIVERSAL_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            model="llama-3.1-8b-instant", # Faster for large volume
            response_format={"type": "json_object"}
        )
        
        structured_json = json.loads(chat_completion.choices[0].message.content)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_json, f, indent=2)
        return True
    except Exception as e:
        if "rate_limit" in str(e).lower():
            print("⏳ Rate limit. Sleeping 60s...")
            time.sleep(60)
        else:
            print(f"❌ Error in {filename}: {e}")
        return False

def run_universal_conversion():
    # List of folders to process (excluding destinations which you did already)
    target_folders = ["blogs", "company", "information", "team", "insights"]
    
    for folder in target_folders:
        folder_path = os.path.join(INPUT_BASE, folder)
        if not os.path.exists(folder_path): continue
        
        print(f"\n📁 Processing Folder: {folder}")
        for file in os.listdir(folder_path):
            if file.endswith(".md"):
                in_file = os.path.join(folder_path, file)
                out_file = os.path.join(OUTPUT_BASE, folder, file.replace(".md", ".json"))
                
                if os.path.exists(out_file): continue # Skip already done
                
                success = process_file(in_file, out_file)
                if success: print(f"✅ Saved: {file}")
                time.sleep(2) # Protect API quota

if __name__ == "__main__":
    run_universal_conversion()