import json
import redis.asyncio as redis
from src.config.settings import settings
from src.services.logger import logger

# Initialize async Redis client
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

async def get_history(session_id: str) -> list:
    """Fetch recent chat history for a session."""
    if not session_id:
        return []
    try:
        data = await redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else []
    except Exception as e:
        logger.error(f"Redis GET error: {e}")
        return []

async def save_history(session_id: str, history: list):
    """Save history to Redis with a 24-hour expiration."""
    if not session_id:
        return
    try:
        # Keep only the last 6 messages (3 turns) to save Groq Tokens
        trimmed_history = history[-6:]
        await redis_client.setex(
            f"session:{session_id}",
            600, 
            json.dumps(trimmed_history)
        )
    except Exception as e:
        logger.error(f"Redis SET error: {e}")