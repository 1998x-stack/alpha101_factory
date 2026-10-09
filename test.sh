# kline PNG now with proper date ticks
python -m alpha101_factory.cli fetch-one --stock 600000 --start 20200101 --end 20240101 --adjust qfq

# backtest (single name): no crash; IC/RankIC CS likely NaN, TS summary reported; quantile step skipped gracefully
python -m alpha101_factory.backtest.run_bt --alpha Alpha101 --horizon 1 --quantiles 5
