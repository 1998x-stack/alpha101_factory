# -*- coding: utf-8 -*-
from loguru import logger
from pathlib import Path
from ..config import LOG_DIR

def setup_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda m: print(m, end=""))  # 控制台
    logger.add(LOG_DIR / "alpha101.log", rotation="5 MB", encoding="utf-8")
    return logger
