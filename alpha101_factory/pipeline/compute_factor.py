# -*- coding: utf-8 -*-
import pandas as pd
from loguru import logger
from pathlib import Path
from typing import List, Optional

from ..utils.log import setup_logger
from ..config import PARQ_DIR_KLINES, PARQ_DIR_TMP, PARQ_DIR_FACT
from ..utils.io import read_parquet, write_parquet
from ..factors.registry import get_factor

def _load_join(symbols: Optional[List[str]]) -> pd.DataFrame:
    if symbols is None:
        symbols = sorted({p.stem for p in (PARQ_DIR_TMP).glob("*.parquet")})
    dfs = []
    for sym in symbols:
        k = read_parquet(PARQ_DIR_KLINES / f"{sym}.parquet")
        t = read_parquet(PARQ_DIR_TMP / f"{sym}.parquet")
        if k.empty or t.empty:
            continue
        m = pd.merge(
            k, t,
            on=["symbol","datetime","open","high","low","close","volume","amount"],
            how="outer", sort=True
        )
        dfs.append(m)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["datetime","symbol"])

def compute_and_save(factor_name: str, symbols: Optional[List[str]] = None):
    FactorCls = get_factor(factor_name)  # will ensure discovery
    logger.info(f"Computing factor {factor_name} on {('ALL' if symbols is None else symbols)} …")
    df = _load_join(symbols)
    if df.empty:
        logger.error("No data loaded. Build tmp first.")
        return
    fac = FactorCls()
    s = fac.compute(df)  # MultiIndex: [datetime, symbol]
    out = s.reset_index().rename(columns={0:"value"})
    write_parquet(out, PARQ_DIR_FACT / f"{factor_name}.parquet")
    logger.info(f"Saved factor to {PARQ_DIR_FACT / f'{factor_name}.parquet'} rows={len(out)}")

def main():
    setup_logger()
    # 示例：你可以改为从命令行传参或列表
    symbols = sorted({p.stem for p in (PARQ_DIR_TMP).glob("*.parquet")})
    if not symbols:
        logger.error("No tmp files. Run build_tmp first.")
        return
    for name in ["Alpha001","Alpha003","Alpha004","Alpha005","Alpha006","Alpha009","Alpha010",
                "Alpha011","Alpha012","Alpha013","Alpha014","Alpha016","Alpha018","Alpha019",
                "Alpha020","Alpha021","Alpha022","Alpha023","Alpha024","Alpha025","Alpha026",
                "Alpha030","Alpha031","Alpha032","Alpha033","Alpha034","Alpha035","Alpha036",
                "Alpha037","Alpha038","Alpha039","Alpha040","Alpha041","Alpha042","Alpha043",
                "Alpha044","Alpha045","Alpha046","Alpha047","Alpha049","Alpha050","Alpha051",
                "Alpha052","Alpha053","Alpha054","Alpha055","Alpha060","Alpha061","Alpha064",
                "Alpha065","Alpha071","Alpha083","Alpha084","Alpha085","Alpha086","Alpha094",
                "Alpha095","Alpha096","Alpha098","Alpha099","Alpha101"]:
        compute_and_save(name, symbols)

if __name__ == "__main__":
    main()
