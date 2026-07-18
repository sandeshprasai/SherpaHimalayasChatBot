import json
import redis.asyncio as redis
from src.config.settings import settings
from src.services.logger import logger

redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    max_connections=20,           
    socket_timeout=2.0,           
    socket_connect_timeout=2.0,
    retry_on_timeout=True,        
)

async def get_history(session_id: str) -> list:
    if not session_id:
        return []
    try:
        data = await redis_client.get(f"session:{session_id}")
        history = json.loads(data) if data else []
        logger.info("history_loaded", extra={
            "session_id": session_id,
            "turns": len(history),
            "cache_hit": data is not None,
        })
        return history
    except Exception as e:
        logger.error("redis_get_error", extra={
            "session_id": session_id,
            "error": str(e),
        })
        return []

async def save_history(session_id: str, history: list):
    if not session_id:
        return
    try:
        trimmed_history = history[-12:]
        await redis_client.setex(
            f"session:{session_id}",
            3600,
            json.dumps(trimmed_history)
        )
        logger.info("history_saved", extra={
            "session_id": session_id,
            "turns": len(trimmed_history),
        })
    except Exception as e:
        logger.error("redis_set_error", extra={
            "session_id": session_id,
            "error": str(e),
        })