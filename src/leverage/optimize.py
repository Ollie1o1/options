"""Pick reversion params honestly: grid-search on a TRAIN split, then report the
chosen params' performance on a held-out TEST split. Selection never sees test,
so the test number is an unbiased read on whether the optimization generalizes."""
from __future__ import annotations
from dataclasses import replace
from typing import List, Optional, Tuple
from .reversion import ReversionParams, generate_reversion_signals
from .backtest import run_backtest, BacktestResult

# Default search grid over the parameters that matter most.
_Z = (1.5, 2.0, 2.5, 3.0)
_STOP = (1.0, 1.5, 2.0, 3.0)
_LOOKBACK = (20, 50)


def default_grid() -> List[ReversionParams]:
    base = ReversionParams()
    out = []
    for lb in _LOOKBACK:
        for z in _Z:
            for st in _STOP:
                out.append(replace(base, lookback=lb, z_entry=z, atr_stop_mult=st))
    return out


def optimize_reversion(df5, df15, funding=None, grid: Optional[List] = None,
                       split: float = 0.70, min_trades: int = 30
                       ) -> Tuple[ReversionParams, BacktestResult, BacktestResult]:
    """Return (best_params, train_result, test_result). Best = highest TRAIN
    profit factor among combos with >= min_trades; ties broken by expectancy."""
    grid = grid or default_grid()
    cut = int(len(df5) * split)
    tr5, te5 = df5.iloc[:cut], df5.iloc[cut:]
    tr5.attrs["symbol"] = te5.attrs["symbol"] = df5.attrs.get("symbol", "BTCUSDT")
    tr15 = df15[df15.index <= tr5.index[-1]] if len(tr5) else df15
    te15 = df15[df15.index > tr5.index[-1]] if len(tr5) else df15

    def score(r: BacktestResult):
        if r.n < min_trades:
            return (-1e9, -1e9)
        from .analysis import analyze
        pf = analyze(r)["profit_factor"]
        return (pf, r.expectancy)

    best_p, best_r, best_key = None, None, (-1e18, -1e18)
    for p in grid:
        r = run_backtest(tr5, tr15, p, funding,
                         signal_fn=generate_reversion_signals)
        key = score(r)
        if key > best_key:
            best_p, best_r, best_key = p, r, key
    if best_p is None:  # nothing cleared min_trades — fall back to the base
        best_p = ReversionParams()
        best_r = run_backtest(tr5, tr15, best_p, funding,
                              signal_fn=generate_reversion_signals)
    test_r = run_backtest(te5, te15, best_p, funding,
                          signal_fn=generate_reversion_signals)
    return best_p, best_r, test_r
