import os
import sys

# Ensure correct path
base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "../../"))
sys.path.append(project_root)

from src.brain.engine import load_index, get_chat_engine

def main():
    print("Loading index and rebuilding Vector DB if necessary...")
    index = load_index()
    engine = get_chat_engine(index)
    
    questions = [
        "What is the maximum altitude of Ghorepani Circuit Trek?",
        "Describe the full itinerary in short for Ghorepani Circuit Trek.",
        "Compare Ghorepani Circuit Trek vs Everest Base Camp Trek in terms of duration.",
        "Which trek has the highest altitude among all treks?"
    ]
    
    for q in questions:
        print("\n" + "="*50)
        print(f"Q: {q}")
        print("="*50)
        response = engine.chat(q)
        print(f"A: {response}")
        print("\nSources retrieved:")
        for node in response.source_nodes:
            print(f"- {node.metadata.get('id', 'unknown')}")

if __name__ == "__main__":
    main()
