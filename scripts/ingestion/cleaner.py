import os
import re
from pathlib import Path

# 1. Setup Absolute Paths
base_path = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(base_path, "../../"))

INPUT_DIR = os.path.join(project_root, "data/separated")
OUTPUT_DIR = os.path.join(project_root, "data/cleaned_separated")

def clean_text(text):
   
    
   
    text = re.sub(r'<[^>]+>', '', text)

    
    text = re.sub(r'^- Home\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^- .*\n', '', text) 
    junk_blocks = [
        r"The Route",
        r"Share Your Experience.*",
        r"Ready to Explore the Himalayas\?.*",
        r"Join our newsletter.*",
        r"We're associated with.*",
        r"Your insights help us maintain.*",
        r"No trips found\.",           
        r"Join us for professionally guided.*"
    ]
    for block in junk_blocks:
        text = re.sub(block, "", text, flags=re.IGNORECASE | re.DOTALL)

    
    text = re.sub(r"× -", "", text)
    text = re.sub(r"×", "", text)
    text = re.sub(r"View remaining \d+ Days", "", text)
    
   
    text = text.replace("- \nDay", "- Day")
    
   
    lines = text.split('\n')
    cleaned_lines = []
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if not clean_line:
            cleaned_lines.append("")
            continue
        if i > 0 and clean_line == lines[i-1].strip():
            continue
        cleaned_lines.append(line)
    
    
    result = "\n".join(cleaned_lines)
    result = re.sub(r'\n{3,}', '\n\n', result) 
    
    return result.strip()

def run_cleaner():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Input directory not found at {INPUT_DIR}")
        return

    print(f"🧹 Starting Deep Clean: {INPUT_DIR} -> {OUTPUT_DIR}")
    
    file_count = 0
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".md"):
                input_file_path = os.path.join(root, file)
                
                
                relative_path = os.path.relpath(root, INPUT_DIR)
                target_dir = os.path.join(OUTPUT_DIR, relative_path)
                output_file_path = os.path.join(target_dir, file)
                
                os.makedirs(target_dir, exist_ok=True)
                
                with open(input_file_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
                
                # Clean the content
                cleaned_content = clean_text(raw_content)
                
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
                
                file_count += 1

    print(f"\n✨ Success! {file_count} files cleaned and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_cleaner()