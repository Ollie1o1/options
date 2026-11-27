"""Intraday (1-minute) bar data and session-level price extraction.

`alpaca_bars` downloads Alpaca SIP 1-minute bars to monthly Parquet files under
`data/intraday/<SYMBOL>/<YYYY-MM>.parquet`; `sessions` turns them into per-day
prices at fixed clock times. See docs/PREREG_INTRADAY_LATE_DAY_AND_OPEN_20261127.md.
"""
