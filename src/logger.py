import logging
from src.config import LOG_PATH
import os

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

        file_handler = logging.FileHandler(LOG_PATH)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger