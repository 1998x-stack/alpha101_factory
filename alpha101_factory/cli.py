# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from loguru import logger

# 确保项目根目录加入 sys.path，便于模块导入
try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
except Exception as e:
    print(f"[警告] 无法设置 sys.path: {e}")

import argparse
from alpha101_factory.utils.log import setup_logger
from alpha101_factory.data.loader import (fetch_spot, fetch_klines_from_spot,
                          check_klines_integrity, load_or_fetch_symbol)
from alpha101_factory.data.universe import load_universe
from alpha101_factory.factors.tmp_features import build_tmp_all
from alpha101_factory.pipeline.compute_factor import compute_and_save
from alpha101_factory.factors.registry import list_factors
from alpha101_factory.config import ADJUST

def cmd_fetch(args):
    spot = fetch_spot(save=True)
    fetch_klines_from_spot(spot)

def cmd_fetch_one(args):
    sym = args.stock
    k = load_or_fetch_symbol(sym, args.start, args.end, adjust=args.adjust or ADJUST, save_image=True)
    if k is None or k.empty:
        logger.warning(f"No data for {sym}")
    else:
        logger.info(f"Loaded/Fetched {sym}: rows={len(k)}, range={k['datetime'].min()}..{k['datetime'].max()}")

def cmd_tmp(args):
    # if single stock specified, only build for that one
    if args.stock:
        ok = build_tmp_all([args.stock])
        return
    symbols = load_universe().tolist()
    build_tmp_all(symbols)

def cmd_factor(args):
    # compute for given factors; if --stock is given, we will limit panel internally in compute_and_save
    # (compute_and_save will read tmp of available symbols; provide a filter list)
    if args.all:
        names = list_factors()
    else:
        names = args.factors
    symbols = None
    if args.stock:
        symbols = [args.stock]
    for n in names:
        compute_and_save(n, symbols=symbols)

def cmd_check(args):
    rpt = check_klines_integrity()
    if rpt.empty:
        logger.info("No data found. Run `fetch` first.")
    else:
        logger.info(rpt.head(10).to_string())
        logger.info(f"exists={rpt['exists'].sum()} / total={len(rpt)}; empty={(rpt['rows']==0).sum()}")

def main():
    setup_logger()
    ap = argparse.ArgumentParser("alpha101-factory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("fetch", help="fetch spot & klines(qfq/hfq) for universe")
    p1.set_defaults(func=cmd_fetch)

    p1a = sub.add_parser("fetch-one", help="load local or fetch one stock; save kline PNG")
    p1a.add_argument("--stock", required=True, help="6-digit code, e.g., 600000")
    p1a.add_argument("--start", default="", help="YYYYMMDD or empty")
    p1a.add_argument("--end", default="", help="YYYYMMDD or empty")
    p1a.add_argument("--adjust", default="", help="qfq/hfq/'' override")
    p1a.set_defaults(func=cmd_fetch_one)

    p2 = sub.add_parser("tmp", help="build tmp features")
    p2.add_argument("--stock", default="", help="build tmp for single stock if provided")
    p2.set_defaults(func=cmd_tmp)

    p3 = sub.add_parser("factor", help="compute factors")
    p3.add_argument("--all", action="store_true", help="compute all registered")
    p3.add_argument("--factors", nargs="*", default=["Alpha101"])
    p3.add_argument("--stock", default="", help="limit computation to single stock if provided")
    p3.set_defaults(func=cmd_factor)

    p4 = sub.add_parser("check", help="check saved parquet for stock data")
    p4.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
