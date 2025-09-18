# -*- coding: utf-8 -*-
from loguru import logger
from ..utils.log import setup_logger
from ..data.universe import load_universe
from ..factors.tmp_features import build_tmp_all

def main():
    setup_logger()
    symbols = load_universe().tolist()
    if not symbols:
        logger.error("Universe empty. Run fetch first.")
        return
    build_tmp_all(symbols)

if __name__ == "__main__":
    main()
