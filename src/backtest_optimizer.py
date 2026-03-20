"""
Weight optimizer using synthetic historical backtest.

Approach
--------
1. For each ticker in a training universe, download up to 5 years of daily prices.
2. Roll forward in 14-day steps, generating synthetic short-put trades at 45 DTE.
3. Pre-compute all 21 component scores at each entry point (Black-Scholes + price-derived signals).
4. Simulate P&L using 50% profit-target / 2? stop-loss / 21-DTE-exit rules.
5. Optimize composite_weights to maximise IC (Pearson r between quality_score and P&L),
   regularised with L2 to prevent overfitting.
6. Cross-validate on a held-out time slice.
7. Optionally write optimised weights back to config.json.

Usage
-----
    # Quick run (local L-BFGS-B, ~1 min)
    python -m src.backtest_optimizer

    # More rigorous (global search, ~5-10 min)
    python -m src.backtest_optimizer --method differential_evolution --trials 400

    # Write results back to config.json
    python -m src.backtest_optimizer --save

    # Custom universe
    python -m src.backtest_optimizer --tickers AAPL MSFT NVDA SPY QQQ
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import differential_evolution, minimize
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# -- Config --------------------------------------------------------------------

WEIGHT_KEYS: List[str] = [
    "pop", "em_realism", "rr", "momentum", "iv_rank", "liquidity", "catalyst",
    "theta", "ev", "trader_pref", "iv_edge", "skew_align", "gamma_theta", "pcr",
    "gex", "oi_change", "sentiment", "option_rvol", "vrp", "gamma_pin", "max_pain",
]

# Current config.json weights (baseline)
CURRENT_WEIGHTS: Dict[str, float] = {
    "pop": 0.20, "em_realism": 0.18, "rr": 0.13, "momentum": 0.10,
    "iv_rank": 0.05, "liquidity": 0.10, "catalyst": 0.02, "theta": 0.07,
    "ev": 0.08, "trader_pref": 0.04, "iv_edge": 0.09, "skew_align": 0.04,
    "gamma_theta": 0.03, "pcr": 0.02, "gex": 0.03, "oi_change": 0.02,
    "sentiment": 0.02, "option_rvol": 0.04, "vrp": 0.05, "gamma_pin": 0.03,
    "max_pain": 0.04,
}

# Research-backed starting point (used as x0 for local search)
RESEARCH_WEIGHTS: Dict[str, float] = {
    "pop": 0.22, "em_realism": 0.04, "rr": 0.11, "momentum": 0.05,
    "iv_rank": 0.18, "liquidity": 0.09, "catalyst": 0.01, "theta": 0.07,
    "ev": 0.05, "trader_pref": 0.02, "iv_edge": 0.08, "skew_align": 0.03,
    "gamma_theta": 0.01, "pcr": 0.01, "gex": 0.01, "oi_change": 0.01,
    "sentiment": 0.01, "option_rvol": 0.02, "vrp": 0.06, "gamma_pin": 0.01,
    "max_pain": 0.01,
}

DEFAULT_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "BAC", "XOM", "CVX", "UNH", "JNJ", "V", "MA",
    "HD", "WMT", "PG", "KO", "DIS", "NFLX", "AMD", "INTC",
    "GS", "WFC",
]

RISK_FREE_RATE  = 0.05   # approximate 2022-2024 average
ENTRY_DTE       = 45     # days to expiration at entry
EXIT_DTE_MIN    = 21     # time-exit trigger
TARGET_DELTA    = 0.30   # ~30-delta short put
PROFIT_TARGET   = 0.50   # close at 50 % of premium received
STOP_LOSS_MULT  = 2.0    # close if option value > 2? premium
ROLL_STEP_DAYS  = 14     # generate one trade per 14 trading days
SLIPPAGE_PCT    = 0.02   # 2 % round-trip slippage


# -- Black-Scholes primitives --------------------------------------------------

def _ndtr(x: float) -> float:
    """Normal CDF (scalar)."""
    from math import erfc, sqrt
    return 0.5 * erfc(-x / sqrt(2))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    import math
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return K * math.exp(-r * T) * _ndtr(-d2) - S * _ndtr(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta (negative number, e.g. -0.30 for a 30-delta put)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    import math
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ndtr(d1) - 1.0


def bs_theta_daily(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put theta per calendar day (positive = premium decays towards seller)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    import math
    from math import exp, log, sqrt
    sq = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    phi_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    theta_yr = (
        -S * phi_d1 * sigma / (2 * sqrt(T))
        + r * K * exp(-r * T) * _ndtr(-d2)
    )
    return theta_yr / 365.0


def strike_for_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    target_delta: float = -0.30,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Binary search: return K such that put delta ? target_delta."""
    lo, hi = S * 0.30, S * 0.9999
    for _ in range(max_iter):
        K = (lo + hi) * 0.5
        d = bs_put_delta(S, K, T, r, sigma)
        if abs(d - target_delta) < tol:
            return K
        # More negative delta -> further OTM -> lower K
        if d < target_delta:
            hi = K
        else:
            lo = K
    return K


# -- Component score computation -----------------------------------------------

_NEUTRAL = 0.5  # score for factors with no historical signal


def compute_component_scores(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    hv_pct_rank: float,    # [0,1] HV percentile over trailing year
    hv_ratio: float,       # current HV / 6-month avg HV
    rsi_score: float,      # [0,1] from RSI-14
    vol_rank: float,       # [0,1] volume percentile rank
) -> np.ndarray:
    """Return a length-21 array of component scores in [0,1]."""
    prem = bs_put_price(S, K, T, r, sigma)
    delta = bs_put_delta(S, K, T, r, sigma)
    theta = bs_theta_daily(S, K, T, r, sigma)

    # -- Computable factors ----------------------------------------------------

    # PoP: 1 - |delta| (put delta is negative)
    pop_score = float(np.clip(1.0 - abs(delta), 0, 1))

    # IV rank proxy (HV percentile as substitute for IV rank)
    iv_rank_score = float(np.clip(hv_pct_rank, 0, 1))

    # Risk/reward: premium / max_loss
    max_loss = max(K - S + prem, 0.01)
    rr_score = float(np.clip(prem / max_loss * 5.0, 0, 1))  # 0.20 credit/risk -> 1.0

    # EV: PoP * credit - (1-PoP) * max_loss, normalised to [0,1]
    ev_val = pop_score * prem - (1 - pop_score) * max_loss
    ev_score = float(np.clip((ev_val + max_loss) / (2 * max_loss), 0, 1))

    # Theta score: daily theta decay as % of premium * DTE (how quickly it decays)
    dte = T * 365.0
    theta_frac = (abs(theta) * dte / max(prem, 1e-6))
    theta_score = float(np.clip(theta_frac * 3.0, 0, 1))

    # IV edge: how elevated is current vol vs recent average (sellers want high)
    iv_edge_score = float(np.clip((hv_ratio - 0.8) / 0.8, 0, 1))

    # VRP proxy: positive when HV is elevated above 6m avg (variance risk premium)
    vrp_score = float(np.clip((hv_ratio - 1.0) / 0.5, 0, 1))

    # Momentum (RSI-14): bearish momentum (low RSI) favours short puts
    momentum_score = float(np.clip(1.0 - rsi_score * 0.4, 0, 1))

    # Skew alignment: bearish conditions favour put selling
    skew_align_score = float(np.clip(1.0 - rsi_score * 0.5, 0, 1))

    # EM realism proxy: longer DTE = more time value to decay
    em_realism_score = float(np.clip(T * 6.0, 0, 1))

    # Liquidity from volume rank
    liquidity_score = float(np.clip(vol_rank, 0, 1))

    # -- Neutral for factors without historical data ----------------------------
    scores = np.array([
        pop_score,          # pop
        em_realism_score,   # em_realism
        rr_score,           # rr
        momentum_score,     # momentum
        iv_rank_score,      # iv_rank
        liquidity_score,    # liquidity
        _NEUTRAL,           # catalyst  (no historical earnings calendar)
        theta_score,        # theta
        ev_score,           # ev
        _NEUTRAL,           # trader_pref
        iv_edge_score,      # iv_edge
        skew_align_score,   # skew_align
        _NEUTRAL,           # gamma_theta
        _NEUTRAL,           # pcr
        _NEUTRAL,           # gex
        _NEUTRAL,           # oi_change
        _NEUTRAL,           # sentiment
        _NEUTRAL,           # option_rvol
        vrp_score,          # vrp
        _NEUTRAL,           # gamma_pin
        _NEUTRAL,           # max_pain
    ], dtype=np.float64)

    return np.clip(scores, 0.0, 1.0)


# -- Trade simulation ----------------------------------------------------------

def simulate_pnl(
    entry_premium: float,
    future_closes: np.ndarray,
    K: float,
    sigma: float,
    r: float,
    entry_dte: int,
    profit_target: float = PROFIT_TARGET,
    stop_mult: float = STOP_LOSS_MULT,
    min_dte: int = EXIT_DTE_MIN,
    slippage: float = SLIPPAGE_PCT,
) -> float:
    """Simulate a short put and return pnl_pct (fraction of premium received)."""
    if entry_premium <= 0:
        return 0.0
    received = entry_premium * (1 - slippage / 2)
    stop_val  = entry_premium * stop_mult
    take_val  = entry_premium * profit_target

    for i, S in enumerate(future_closes):
        dte = entry_dte - i - 1
        T = max(dte / 365.0, 1 / 365.0)
        val = bs_put_price(float(S), K, T, r, sigma)
        exit_now = (val <= take_val) or (val >= stop_val) or (dte <= min_dte)
        if exit_now or i == len(future_closes) - 1:
            exit_cost = val * (1 + slippage / 2)
            return float((received - exit_cost) / max(entry_premium, 1e-8))
    return 0.0


# -- Per-ticker backtest -------------------------------------------------------

def _hv_30(closes: pd.Series, end_idx: int) -> Optional[float]:
    ret = closes.iloc[max(0, end_idx - 30):end_idx].pct_change().dropna()
    if len(ret) < 15:
        return None
    v = float(ret.std() * 252 ** 0.5)
    return v if np.isfinite(v) and v > 0 else None


def backtest_ticker(
    symbol: str,
    period: str = "5y",
    r: float = RISK_FREE_RATE,
    entry_dte: int = ENTRY_DTE,
    roll_step: int = ROLL_STEP_DAYS,
    target_delta: float = TARGET_DELTA,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (component_scores [N?21], pnl_pct [N]) for one ticker, or None.
    """
    try:
        raw = yf.download(symbol, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw["Close"].dropna().squeeze()
        if not isinstance(closes, pd.Series) or len(closes) < 300:
            return None
        vols = raw.get("Volume", pd.Series(0, index=raw.index))
        if isinstance(vols, pd.DataFrame):
            vols = vols.iloc[:, 0]
        vols = vols.reindex(closes.index).fillna(0)

        warmup = 260
        step_indices = range(warmup, len(closes) - entry_dte - 5, roll_step)

        scores_list, pnls = [], []
        for idx in step_indices:
            S0 = float(closes.iloc[idx])
            if S0 <= 0:
                continue

            sigma = _hv_30(closes, idx)
            if sigma is None:
                continue

            # HV percentile over trailing 252 days
            hv_history = [
                _hv_30(closes, i)
                for i in range(max(idx - 252, 30), idx, 5)
            ]
            hv_history = [v for v in hv_history if v is not None]
            hv_pct_rank = float(np.mean(np.array(hv_history) < sigma)) if hv_history else 0.5

            # HV ratio vs 6-month avg
            hv_6m = [_hv_30(closes, i) for i in range(max(idx - 130, 30), idx, 5)]
            hv_6m = [v for v in hv_6m if v is not None]
            hv_ratio = sigma / max(float(np.mean(hv_6m)), 0.01) if hv_6m else 1.0

            # RSI-14
            rsi_window = closes.iloc[max(0, idx - 20):idx].diff().dropna()
            gains = rsi_window.clip(lower=0).rolling(14).mean()
            losses = (-rsi_window).clip(lower=0).rolling(14).mean()
            avg_g = float(gains.iloc[-1]) if len(gains) else 0
            avg_l = float(losses.iloc[-1]) if len(losses) else 1e-8
            rsi = 100 - 100 / (1 + avg_g / max(avg_l, 1e-8))
            rsi_score = float(np.clip(rsi / 100.0, 0, 1))

            # Volume rank
            vol_window = vols.iloc[max(0, idx - 252):idx]
            vol_rank = float(np.mean(vol_window < vols.iloc[idx])) if len(vol_window) else 0.5

            # Find strike
            T = entry_dte / 365.0
            try:
                K = strike_for_delta(S0, T, r, sigma, target_delta=-target_delta)
            except Exception:
                K = S0 * (1 - target_delta * sigma * T ** 0.5)
            if K <= 0 or K >= S0:
                continue

            prem = bs_put_price(S0, K, T, r, sigma)
            if prem <= 0.001:
                continue

            comp = compute_component_scores(
                S0, K, T, r, sigma,
                hv_pct_rank=hv_pct_rank,
                hv_ratio=float(np.clip(hv_ratio, 0.1, 5.0)),
                rsi_score=rsi_score,
                vol_rank=vol_rank,
            )

            future = closes.iloc[idx + 1: idx + entry_dte + 1].values
            if len(future) < entry_dte // 2:
                continue

            pnl = simulate_pnl(prem, future, K, sigma, r, entry_dte)
            if not np.isfinite(pnl):
                continue

            scores_list.append(comp)
            pnls.append(pnl)

        if len(scores_list) < 5:
            return None
        return np.array(scores_list, np.float64), np.array(pnls, np.float64)

    except Exception as exc:
        logger.debug("backtest_ticker(%s) failed: %s", symbol, exc)
        return None


# -- Aggregate backtest --------------------------------------------------------

@dataclass
class BacktestResult:
    component_scores: np.ndarray   # (N, 21)
    pnl_pct: np.ndarray            # (N,)
    symbols: List[str]

    @property
    def n_trades(self) -> int:
        return len(self.pnl_pct)

    def factor_ic(self) -> List[Tuple[str, float]]:
        """Pearson IC for each factor column vs P&L."""
        result = []
        for i, k in enumerate(WEIGHT_KEYS):
            col = self.component_scores[:, i]
            if col.std() < 1e-8:
                result.append((k, 0.0))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic, _ = pearsonr(col, self.pnl_pct)
            result.append((k, float(ic) if np.isfinite(ic) else 0.0))
        return result


def run_backtest(
    tickers: List[str],
    period: str = "5y",
    verbose: bool = True,
) -> BacktestResult:
    all_scores, all_pnls, all_syms = [], [], []

    for sym in tickers:
        if verbose:
            print(f"  {sym:>6}: ", end="", flush=True)
        result = backtest_ticker(sym, period=period)
        if result is None:
            if verbose:
                print("skipped")
            continue
        sc, pnl = result
        # Per-ticker IC with current weights
        cw = np.array([CURRENT_WEIGHTS[k] for k in WEIGHT_KEYS])
        cw = cw / cw.sum()
        composite = sc @ cw
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ic = float(pearsonr(composite, pnl)[0]) if composite.std() > 1e-8 else 0.0
        if verbose:
            wr = float(np.mean(pnl > 0))
            print(f"{len(pnl):3d} trades  win={wr:.0%}  IC(current)={ic:+.3f}")
        all_scores.append(sc)
        all_pnls.append(pnl)
        all_syms.extend([sym] * len(pnl))

    if not all_scores:
        raise RuntimeError("No backtest data ? check tickers / internet connection.")

    return BacktestResult(
        component_scores=np.vstack(all_scores),
        pnl_pct=np.concatenate(all_pnls),
        symbols=all_syms,
    )


# -- Optimization --------------------------------------------------------------

def _objective(
    w: np.ndarray,
    component_scores: np.ndarray,
    pnl_pct: np.ndarray,
    l2_lambda: float,
) -> float:
    w = np.abs(w)
    s = w.sum()
    if s < 1e-10:
        return 1.0
    w = w / s

    composite = component_scores @ w
    if composite.std() < 1e-8 or pnl_pct.std() < 1e-8:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ic, _ = pearsonr(composite, pnl_pct)
    if not np.isfinite(ic):
        return 1.0

    # L2: penalise deviation from equal weight (prevents dominance)
    uniform = np.ones(len(w)) / len(w)
    l2 = l2_lambda * float(np.sum((w - uniform) ** 2))

    # Hard cap: any weight > 0.40 incurs quadratic penalty
    cap_pen = 5.0 * float(np.sum(np.maximum(w - 0.40, 0.0) ** 2))

    return -float(ic) + l2 + cap_pen


def optimize_weights(
    bt: BacktestResult,
    method: str = "minimize",
    n_trials: int = 300,
    l2_lambda: float = 0.10,
    verbose: bool = True,
) -> Dict[str, float]:
    n = len(WEIGHT_KEYS)
    bounds = [(0.0, 0.5)] * n
    obj = partial(_objective, component_scores=bt.component_scores,
                  pnl_pct=bt.pnl_pct, l2_lambda=l2_lambda)

    x0_current  = np.array([CURRENT_WEIGHTS[k]  for k in WEIGHT_KEYS], float)
    x0_research = np.array([RESEARCH_WEIGHTS[k] for k in WEIGHT_KEYS], float)
    x0_current  /= x0_current.sum()
    x0_research /= x0_research.sum()

    if verbose:
        ic_cur = -obj(x0_current)
        ic_res = -obj(x0_research)
        print(f"  Baseline IC ? current weights: {ic_cur:+.4f}  research weights: {ic_res:+.4f}")

    if method == "differential_evolution":
        if verbose:
            print(f"  differential_evolution ({n_trials} iter) ? this may take a few minutes...")
        result = differential_evolution(
            obj, bounds=bounds, maxiter=n_trials,
            seed=42, workers=1, tol=1e-5,
            mutation=(0.5, 1.0), recombination=0.7, popsize=15,
            init="latinhypercube",
        )
        raw = result.x
    else:
        if verbose:
            print("  L-BFGS-B (fast local search, two starting points)...")
        # Try both starting points and take the better solution
        r1 = minimize(obj, x0_current,  method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-9})
        r2 = minimize(obj, x0_research, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-9})
        raw = r1.x if r1.fun <= r2.fun else r2.x

    raw = np.abs(raw)
    raw /= raw.sum()

    opt_ic = -obj(raw)
    if verbose:
        base_ic = -obj(x0_current)
        print(f"  Optimised IC: {opt_ic:+.4f}  (? = {opt_ic - base_ic:+.4f})")

    return {k: round(float(v), 4) for k, v in zip(WEIGHT_KEYS, raw)}


def cross_validate(
    bt: BacktestResult,
    weights: Dict[str, float],
    n_folds: int = 5,
) -> Dict[str, float]:
    """Walk-forward cross-validation."""
    w = np.array([weights[k] for k in WEIGHT_KEYS], float)
    w /= w.sum()
    n = bt.n_trades
    fold = n // n_folds

    train_ics, val_ics = [], []
    for f in range(n_folds - 1):
        te = (f + 4) * fold
        vs, ve = te, te + fold
        if ve > n:
            break
        for arr, ics in [(bt.component_scores[:te] @ w, train_ics),
                         (bt.component_scores[vs:ve] @ w, val_ics)]:
            pnl = bt.pnl_pct[:te] if ics is train_ics else bt.pnl_pct[vs:ve]
            if arr.std() > 1e-8:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ic, _ = pearsonr(arr, pnl)
                ics.append(float(ic) if np.isfinite(ic) else 0.0)

    return {
        "train_ic": float(np.mean(train_ics)) if train_ics else 0.0,
        "val_ic":   float(np.mean(val_ics))   if val_ics   else 0.0,
        "degradation": (float(np.mean(val_ics)) - float(np.mean(train_ics)))
                       if (train_ics and val_ics) else 0.0,
    }


# -- Output helpers ------------------------------------------------------------

def print_comparison(current: Dict[str, float], optimised: Dict[str, float]) -> None:
    sep = "-" * 58
    print(f"\n  {sep}")
    print(f"  {'Factor':<18} {'Current':>9} {'Optimised':>9} {'Change':>9}")
    print(f"  {sep}")
    for k in WEIGHT_KEYS:
        c = current.get(k, 0.0)
        o = optimised.get(k, 0.0)
        d = o - c
        arrow = " ^" if d > 0.015 else (" v" if d < -0.015 else "  ")
        print(f"  {k:<18} {c:>9.3f} {o:>9.3f} {d:>+9.3f}{arrow}")
    print(f"  {sep}")


def save_to_config(weights: Dict[str, float], config_path: Path) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    # Renormalise to ensure sum == 1 before saving
    total = sum(weights.values())
    cfg["composite_weights"] = {k: round(v / total, 4) for k, v in weights.items()}
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved to {config_path}")


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimise composite_weights via synthetic short-put backtest."
    )
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_UNIVERSE)
    ap.add_argument("--period",  default="5y", choices=["2y", "3y", "5y"])
    ap.add_argument("--method",  default="minimize",
                    choices=["minimize", "differential_evolution"])
    ap.add_argument("--trials",  type=int,   default=300)
    ap.add_argument("--l2",      type=float, default=0.10,
                    help="L2 regularisation (0 = none, 0.5 = strong, default 0.10)")
    ap.add_argument("--save",    action="store_true",
                    help="Write optimised weights to config.json")
    ap.add_argument("--no-cv",   action="store_true",
                    help="Skip cross-validation")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    config_path = Path(__file__).parent.parent / "config.json"

    print("\n" + "=" * 56)
    print("  OPTIONS SCREENER -- WEIGHT OPTIMIZER")
    print(f"  Universe: {len(args.tickers)} tickers | {args.period} history | {args.method}")
    print("=" * 56)

    # -- 1. Backtest --
    print("\n[1/4] Generating synthetic short-put backtest...")
    bt = run_backtest(args.tickers, period=args.period, verbose=True)
    pnl = bt.pnl_pct
    print(f"\n  Trades: {bt.n_trades}  Win rate: {np.mean(pnl > 0):.1%}  "
          f"Mean P&L: {np.mean(pnl):.3f}  Sharpe: "
          f"{np.mean(pnl) / max(np.std(pnl), 1e-8):.2f}")

    if bt.n_trades < 30:
        print("\nERROR: Too few trades. Add more tickers or extend period.")
        return

    # -- 2. Optimise --
    print("\n[2/4] Optimising weights (IC maximisation + L2 regularisation)...")
    optimised = optimize_weights(bt, method=args.method,
                                 n_trials=args.trials, l2_lambda=args.l2)

    # -- 3. Cross-validate --
    if not args.no_cv:
        print("\n[3/4] Walk-forward cross-validation...")
        cv = cross_validate(bt, optimised)
        print(f"  Train IC: {cv['train_ic']:+.4f}  Val IC: {cv['val_ic']:+.4f}  "
              f"Degradation: {cv['degradation']:+.4f}", end="")
        if cv["degradation"] < -0.04:
            print("  ! overfitting detected ? try --l2 0.20")
        else:
            print("  ? stable")
    else:
        cv = {}

    # -- 4. Report --
    print("\n[4/4] Results")
    print_comparison(CURRENT_WEIGHTS, optimised)

    # Factor-level IC table
    print("\n  Per-factor IC (each component vs realised P&L):")
    fics = sorted(bt.factor_ic(), key=lambda x: abs(x[1]), reverse=True)
    for k, ic in fics:
        bar_len = int(abs(ic) * 30)
        bar = ("#" * bar_len).ljust(30)
        print(f"    {k:<16} {bar} {ic:+.4f}")

    # Suggested command
    if not args.save:
        print("\n  To apply these weights, re-run with --save:")
        print(f"    python -m src.backtest_optimizer --save --method {args.method} --l2 {args.l2}")
    else:
        save_to_config(optimised, config_path)
        print("\n  Weights saved. Restart the screener to apply them.")

    print()


if __name__ == "__main__":
    main()
