import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from alpha101_factory.data import loader


def test_resolve_kline_path_full_range(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "PARQ_DIR_KLINES", tmp_path)
    path = loader._resolve_kline_path("sh600000", "20200101", "20210101", "hfq")
    assert path == tmp_path / "600000_20200101_20210101_hfq.parquet"


def test_resolve_kline_path_without_range(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "PARQ_DIR_KLINES", tmp_path)
    path = loader._resolve_kline_path("600000", "", None, "qfq")
    assert path == tmp_path / "600000.parquet"
