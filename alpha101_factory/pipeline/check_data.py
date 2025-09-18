# -*- coding: utf-8 -*-
from ..utils.log import setup_logger
from ..data.loader import check_klines_integrity
from loguru import logger

def main():
    setup_logger()
    rpt = check_klines_integrity()
    if rpt.empty:
        logger.warning("No report generated.")
        return
    # Print summary
    ok = rpt[rpt["exists"] & (rpt["rows"] > 0)]
    miss = rpt[~rpt["exists"] | (rpt["rows"] <= 0)]
    logger.info(f"OK files: {len(ok)} | Missing/Empty: {len(miss)}")
    # You can save it if needed
    from ..config import LOG_DIR
    rpt.to_csv(LOG_DIR / "klines_integrity.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
