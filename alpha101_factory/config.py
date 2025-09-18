# -*- coding: utf-8 -*-
from pathlib import Path
import os

DATA_ROOT = Path(os.getenv("ALPHA101_DATA_ROOT", "./data")).resolve()

PARQ_DIR_SPOT   = DATA_ROOT / "spot"
PARQ_DIR_KLINES = DATA_ROOT / "klines_daily"
PARQ_DIR_TMP    = DATA_ROOT / "tmp_features"
PARQ_DIR_FACT   = DATA_ROOT / "factors"
LOG_DIR         = DATA_ROOT / "logs"

# images
IMG_DIR         = DATA_ROOT / "images"
IMG_KLINES_DIR  = IMG_DIR / "klines"
IMG_BT_DIR      = IMG_DIR / "backtest"

for p in [PARQ_DIR_SPOT, PARQ_DIR_KLINES, PARQ_DIR_TMP, PARQ_DIR_FACT,
          LOG_DIR, IMG_DIR, IMG_KLINES_DIR, IMG_BT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# AkShare K 线参数
ADJUST = os.getenv("ALPHA101_ADJUST", "qfq")   # "qfq" | "hfq" | ""
START_DATE = os.getenv("ALPHA101_START", "20200101")   # "YYYYMMDD" 或空
END_DATE   = os.getenv("ALPHA101_END", "20250917")     # "YYYYMMDD" 或空

# 并发 / 速率
MAX_WORKERS = int(os.getenv("ALPHA101_MAX_WORKERS", "1"))
REQUEST_PAUSE = float(os.getenv("ALPHA101_PAUSE", "0.6"))

# 体量控制：0=全量，>0 表示只抓 N 支用于调试
LIMIT_STOCKS = int(os.getenv("ALPHA101_LIMIT", "0"))
