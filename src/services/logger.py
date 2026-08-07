import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Every attribute a LogRecord carries by default. Anything passed via
# `extra={...}` that ISN'T in this set is a deliberate structured field and
# gets merged into the top level of the JSON line.
_RESERVED_RECORD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JSONFormatter(logging.Formatter):
    """
    Emits one compact JSON object per line (NDJSON) so logs are directly
    filterable with jq during inspection, e.g.:

        docker compose logs chatbot | jq -c 'select(.event=="chat_completed")'
        docker compose logs chatbot | jq -c 'select(.session_id=="abc-123")'
        docker compose logs chatbot | jq -c 'select(.level=="ERROR")'

    Call sites should pass a short, stable, snake_case event name as the
    log message, and put all dynamic values in `extra={...}`:

        logger.info("classification_completed", extra={
            "session_id": sid, "duration_s": 3.64, "intent": "lookup"
        })
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in _RESERVED_RECORD_ATTRS or key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).strip()

        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_logger(name: str, level: int, filename: str, max_bytes: int, backups: int) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    log.propagate = False

    if log.handlers:
        return log

    formatter = JSONFormatter()

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # stdout, not stderr -- this is what `docker compose logs` streams
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(console_handler)
    return log


logger = _build_logger("Chatbot", logging.INFO, "app.log", 10 * 1024 * 1024, 5)
fallback_logger = _build_logger("Fallback", logging.WARNING, "fallback.log", 5 * 1024 * 1024, 3)