import sys
import traceback
from src.retrieval.vector import VectorRetriever


def run():
    try:
        from src.retrieval.vector import VectorRetriever
        v = VectorRetriever()
        v._ensure_ready()
        print('Vector index loaded or built successfully.')
    except Exception as e:
        print('FAILED to build vector index:', e)
        traceback.print_exc()

if __name__ == '__main__':
    run()
