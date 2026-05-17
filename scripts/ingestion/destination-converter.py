import os
import json
import time
from dotenv import load_dotenv
from groq import Groq


base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))

INPUT_BASE = os.path.join(project_root, "data/cleaned_separated/destinations")
OUTPUT_BASE = os.path.join(project_root, "data/rag_ready/destinations")

load_dotenv(os.path.join(project_root, ".env"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a data extraction expert. Convert the provided travel text into strict JSON.
JSON Schema:
{
  "id": "lowercase_with_underscores",
  "title": "string",
  "category": "Trek/Expedition/Peak/Tour",
  "region": "string",
  "duration_days": integer,
  "difficulty": "Easy/Moderate/Hard/Challenging",
  "max_altitude_m": integer,
  "summary": "2-3 sentences",
  "highlights": ["list"],
  "itinerary": [{"day": int, "title": "string", "description": "string"}],
  "includes": ["list"],
  "excludes": ["list"],
  "embedding_chunks": [
    {"type": "overview", "text": "summary"},
    {"type": "itinerary_summary", "text": "full schedule"}
  ]
}
Return ONLY JSON.
"""

def process_file(input_file_path, output_file_path):
    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    filename = os.path.basename(input_file_path)
    print(f"🧠 Processing with 8B Model: {filename}")
    
    try:
        # CHANGED MODEL TO 8B-INSTANT FOR HIGHER LIMITS
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        structured_json = json.loads(chat_completion.choices[0].message.content)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_json, f, indent=2)
        return True

    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            print("⏳ Daily limit reached or speed too high. Waiting 60 seconds...")
            time.sleep(60)
        else:
            print(f"❌ Error: {e}")
        return False

def run_recursive_conversion():
    for root, dirs, files in os.walk(INPUT_BASE):
        for file in files:
            if file.endswith(".md"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(OUTPUT_BASE, os.path.relpath(root, INPUT_BASE), file.replace(".md", ".json"))

                if os.path.exists(output_path):
                    continue

                success = process_file(input_path, output_path)
                if success:
                    print(f"✅ Converted: {file}")
                
                # Sleep to stay within TPM (Tokens Per Minute)
                time.sleep(2)

if __name__ == "__main__":
    run_recursive_conversion()