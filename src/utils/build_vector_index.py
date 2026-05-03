import sys
import traceback
sys.path.insert(0, 'src')

def run():
    try:
        from retrieval.vector import VectorRetriever
        v = VectorRetriever()
        v._ensure_ready()
        print('Vector index loaded or built successfully.')
    except Exception as e:
        print('FAILED to build vector index:', e)
        traceback.print_exc()

if __name__ == '__main__':
    run()
