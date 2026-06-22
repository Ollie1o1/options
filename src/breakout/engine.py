# src/breakout/engine.py
"""Breakout engine orchestrator + CLI. Loads the cached universe, builds forward
return distributions per ticker per horizon, and emits breakout/breakdown
forecasts. The news overlay is applied live only (forward-tracked)."""
from __future__ import annotations
import argparse, json, warnings
from typing import Dict, List, Optional

from src.breakout.data import Series, HORIZONS, DEFAULT_DB, load_series, update_universe
from src.breakout import features as F
from src.breakout.distribution import baseline_distribution, parametric_distribution, make_distribution
from src.breakout import backtest as B
from src.breakout import report as R

UP, DOWN = 0.10, -0.10
FALLBACK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SPY"]


def load_universe(config_path: str = "config.json") -> List[str]:
    try:
        with open(config_path) as f:
            u = ((json.load(f) or {}).get("breakout") or {}).get("universe")
        return u or FALLBACK_UNIVERSE
    except Exception:
        return FALLBACK_UNIVERSE


def live_forecasts(series_by_ticker: Dict[str, Series], model: str = "parametric",
                   seed: int = 0) -> List[dict]:
    rows: List[dict] = []
    for ticker, s in series_by_ticker.items():
        t = len(s.close) - 1
        for label, h in HORIZONS.items():
            if t < h + 1:
                continue
            d = make_distribution(s, t, h, model, seed)
            rows.append({"ticker": ticker, "horizon": label, "point": d.point(),
                         "band": d.band(0.1, 0.9), "up_prob": d.prob_ge(UP),
                         "down_prob": d.prob_le(DOWN)})
    return rows


def _yf_fetcher(ticker: str, start: Optional[str]):
    import pandas as pd, yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.Ticker(ticker).history(period="max", interval="1d")
    if df is None or df.empty:
        return []
    df = df.dropna(subset=["Close"])
    out = []
    for idx, r in df.iterrows():
        out.append({"date": idx.strftime("%Y-%m-%d"), "close": float(r["Close"]),
                    "high": float(r["High"]), "low": float(r["Low"]),
                    "volume": float(r.get("Volume", 0) or 0)})
    return out


def _load_all(db_path: str, tickers: List[str]) -> Dict[str, Series]:
    out = {}
    for t in tickers:
        s = load_series(db_path, t)
        if s is not None:
            out[t] = s
    return out


def menu(config_path: str = "config.json", db_path: str = DEFAULT_DB) -> None:
    """Interactive breakout sub-menu (reached from the launcher's basic UI)."""
    from src import ui
    from src import formatting as fmt
    universe = load_universe(config_path)
    W = 74

    def _row(key, desc):
        return "  " + fmt.style(f"[{key}]", "accent", bold=True) + "  " + desc

    while True:
        print()
        print(ui.rule(W, "BREAKOUT / BREAKDOWN OUTLOOK"))
        print(_row("1", "Live leaderboards — most likely to break out / break down"))
        print(_row("2", "Backtest — featureful model vs vol baseline"))
        print(_row("3", "Backtest — baseline only (vol-only control)"))
        print(_row("4", "Refresh price data (yfinance, ~slow)"))
        print(_row("B", fmt.style("Back", "muted")))
        print(ui.rule(W))
        try:
            choice = (input("  Choice [1]: ").strip() or "1").upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("B", "Q", "BACK", "QUIT"):
            return
        if choice == "4":
            print("  refreshing price cache…")
            res = update_universe(db_path, universe, _yf_fetcher)
            print(f"  updated {sum(1 for v in res.values() if v)} / {len(universe)} tickers")
            continue
        series = _load_all(db_path, universe)
        if not series:
            print("  No cached data yet — choose [4] to refresh first.")
            continue
        if choice == "2":
            print(R.render_backtest(B.run_backtest(series, model="parametric")))
        elif choice == "3":
            print(R.render_backtest(B.run_backtest(series, model="baseline")))
        else:  # "1" / default → live
            print(R.render_forecasts(live_forecasts(series, model="parametric")))


def main(argv=None):
    p = argparse.ArgumentParser(description="Breakout/breakdown probability engine")
    p.add_argument("--backtest", action="store_true")
    p.add_argument("--model", choices=["parametric", "baseline"], default="parametric")
    p.add_argument("--update-data", action="store_true", help="refresh OHLCV cache")
    p.add_argument("--config", default="config.json")
    p.add_argument("--db", default=DEFAULT_DB)
    args = p.parse_args(argv)
    universe = load_universe(args.config)

    if args.update_data:
        res = update_universe(args.db, universe, _yf_fetcher)
        print(f"  updated {sum(1 for v in res.values() if v)} / {len(universe)} tickers")
        return

    series = _load_all(args.db, universe)
    if not series:
        print("  no cached data — run with --update-data first")
        return

    if args.backtest:
        print(R.render_backtest(B.run_backtest(series, model=args.model)))
        return
    forecasts = live_forecasts(series, model=args.model)
    # TODO(news): wire news_overlay.log_prediction here to forward-track live forecasts
    print(R.render_forecasts(forecasts))


if __name__ == "__main__":
    main()
