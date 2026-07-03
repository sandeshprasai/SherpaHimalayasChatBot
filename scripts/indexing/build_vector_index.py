import sys
import traceback
import asyncio

async def async_run():
    from src.retrieval.vector import VectorRetriever
    v = VectorRetriever()
    await v._ensure_ready()
    print('Vector index loaded or built successfully.')

def run():
    try:
        asyncio.run(async_run())
    except Exception as e:
        print('FAILED to build vector index:', e)
        traceback.print_exc()

if __name__ == '__main__':
    run()