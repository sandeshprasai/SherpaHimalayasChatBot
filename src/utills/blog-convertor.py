import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

# Setup Paths
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
INPUT_DIR = os.path.join(project_root, "data/cleaned_separated/blogs")
OUTPUT_DIR = os.path.join(project_root, "data/rag_ready/blogs")

load_dotenv(os.path.join(project_root, ".env"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BLOG_SYSTEM_PROMPT = """
You are a data extraction expert. Convert the provided trekking blog text into a strict JSON format.
JSON Schema:
{
  "id": "lowercase_with_underscores",
  "type": "blog",
  "title": "string",
  "publish_date": "string or null",
  "category": "Travel Advice / News / Health / History",
  "tags": ["list of tags"],
  "summary": "2 sentence summary of the main point",
  "content_sections": [{"heading": "string", "text": "cleaned text content"}],
  "embedding_chunks": [
    {"type": "topic_summary", "text": "summary for search index"},
    {"type": "detailed_content", "text": "full relevant information for AI context"}
  ]
}
Return ONLY the JSON. No conversational filler.
"""

def process_blog(file_path, filename):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"✍️ Processing Blog: {filename}")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": BLOG_SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        
        structured_json = json.loads(chat_completion.choices[0].message.content)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        with open(os.path.join(OUTPUT_DIR, filename.replace(".md", ".json")), "w") as f:
            json.dump(structured_json, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.endswith(".md"):
            process_blog(os.path.join(INPUT_DIR, file), file)
            time.sleep(2) # Avoid rate limits