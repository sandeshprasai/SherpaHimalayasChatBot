# src/services/logger.py
"""
Logging configuration.

Two loggers:
  - logger         → data/logs/app.log       (all application events)
  - fallback_logger → data/logs/fallback.log  (analytics: unanswered queries)

The fallback log is the client's analytics tool — any entry here means
the bot couldn't find an answer. These are knowledge base gaps to fill.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── Formatter ─────────────────────────────────────────────────────────────
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# ── Main application logger ───────────────────────────────────────────────
logger = logging.getLogger("sherpa-chatbot")
logger.setLevel(logging.INFO)

# Rotate at 10MB, keep 5 backups
_app_file = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
logger = logging.getLogger("sherpa-chatbot")
logger.setLevel(logging.INFO)

_app_file = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
_app_file.setFormatter(fmt)
_app_console = logging.StreamHandler()
_app_console.setFormatter(fmt)

logger.addHandler(_app_file)
logger.addHandler(_app_console)
logger.propagate = False

# ── Fallback / analytics logger ───────────────────────────────────────────
fallback_logger = logging.getLogger("sherpa-fallback")
fallback_logger.setLevel(logging.WARNING)

_fb_file = logging.FileHandler(os.path.join(LOG_DIR, "fallback.log"), encoding="utf-8")
_fb_file.setFormatter(fmt)

fallback_logger.addHandler(_fb_file)
fallback_logger.propagate = False