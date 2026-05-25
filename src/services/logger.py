import logging
import os
from logging.handlers import RotatingFileHandler


LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger("sherpa-chatbot")
logger.setLevel(logging.INFO)

app_file = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"), 
    maxBytes=10*1024*1024, 
    backupCount=5, 
    encoding="utf-8"
)
app_file.setFormatter(fmt)


console = logging.StreamHandler()
console.setFormatter(fmt)

logger.addHandler(app_file)
logger.addHandler(console)
logger.propagate = False


fallback_logger = logging.getLogger("sherpa-fallback")
fallback_logger.setLevel(logging.WARNING)

fb_file = RotatingFileHandler(
    os.path.join(LOG_DIR, "fallback.log"),
    maxBytes=5*1024*1024,
    backupCount=3,
    encoding="utf-8"
)
fb_file.setFormatter(fmt)

fallback_logger.addHandler(fb_file)
fallback_logger.propagate = False