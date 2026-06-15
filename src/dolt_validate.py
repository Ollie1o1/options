"""Layer 2: validate the price/IV/Greek-based slice of the scorer against REAL
forward option returns from DoltHub.

Explicitly does NOT cover news / sentiment / catalyst / EDGAR / regime-news
components — those are not reconstructable historically. Each feature function
returns a 0..1 sub-score (higher = more favorable for a long call) so they can
be combined and IC-tested against real marks.

CLI:
    python -m src.dolt_validate --symbols AAPL,SPY --start 2023-01-01 --end 2023-12-31 --weekly
"""
from __future__ import annotations

import datetime as _dt
import math
from statistics import mean
from typing import Any, Dict, List, Optional


def _atm(chain, spot, opt_type="call"):
    cands = [c for c in chain if c.get("type") == opt_type and c.get("strike")]
    return min(cands, key=lambda c: abs(c["strike"] - spot), default=None)


def _dte(asof, expiration):
    return (_dt.date.fromisoformat(expiration) - _dt.date.fromisoformat(asof)).days


# ── Reconstructable features (price / IV / Greek based) ─────────────────────
def term_structure_score(chain, spot, asof, opt_type="call") -> float:
    """Contango (far IV > near IV) is the normal, healthier state for buying near
    calls. Map slope to 0..1; backwardation (event risk) scores low."""
    by_exp: Dict[str, list] = {}
    for c in chain:
        if c.get("type") == opt_type and c.get("iv") is not None:
            by_exp.setdefault(c["expiration"], []).append(c)
    exps = sorted(by_exp, key=lambda e: _dte(asof, e))
    if len(exps) < 2:
        return 0.5
    near = _atm(by_exp[exps[0]], spot, opt_type)
    far = _atm(by_exp[exps[-1]], spot, opt_type)
    if not near or not far or not near["iv"] or not far["iv"]:
        return 0.5
    slope = (far["iv"] - near["iv"]) / max(near["iv"], 1e-6)   # >0 contango
    return max(0.0, min(1.0, 0.5 + slope))                      # 0 slope → 0.5


def skew_score(chain, spot) -> float:
    """Put-call IV skew. Heavy put skew (downside fear) is unfavorable for long
    calls → lower score. Compares ~25-delta put IV vs ~25-delta call IV."""
    puts = [c for c in chain if c.get("type") == "put" and c.get("delta") is not None]
    calls = [c for c in chain if c.get("type") == "call" and c.get("delta") is not None]
    if not puts or not calls:
        return 0.5
    p = min(puts, key=lambda c: abs(abs(c["delta"]) - 0.25))
    c = min(calls, key=lambda c: abs(c["delta"] - 0.25))
    if not p.get("iv") or not c.get("iv"):
        return 0.5
    skew = (p["iv"] - c["iv"]) / max(c["iv"], 1e-6)            # >0 put-skew
    return max(0.0, min(1.0, 0.5 - skew))                      # more put skew → lower


def em_realism_score(atm_iv, dte, spot, realized_abs_move) -> float:
    """How close the realized move was to the 1σ expected move. 1.0 when realized
    ≈ EM, decaying as it diverges. Pure diagnostic of IV calibration."""
    if not atm_iv or atm_iv <= 0 or spot <= 0:
        return 0.5
    em = spot * atm_iv * math.sqrt(max(dte, 1) / 365.0)
    if em <= 0:
        return 0.5
    ratio = realized_abs_move / em
    return max(0.0, 1.0 - abs(ratio - 1.0))


def moneyness_score(delta, target=0.30) -> float:
    """Prefer contracts near the target delta (0.30). 1.0 at target, linear decay."""
    if delta is None:
        return 0.5
    return max(0.0, 1.0 - abs(abs(delta) - target) / target)


def theta_score(theta, mid) -> float:
    """Lower theta burn relative to premium is better for a long holder. Maps
    theta/premium ratio to 0..1 (less negative → higher)."""
    if theta is None or not mid or mid <= 0:
        return 0.5
    burn = abs(theta) / mid           # daily decay fraction
    return max(0.0, min(1.0, 1.0 - burn * 10))   # 10%/day burn → 0


def iv_level_score(atm_iv, iv_history: Optional[List[float]]) -> float:
    """Percentile rank of current ATM IV within the symbol's own IV history.
    LOW IV (cheap options) scores HIGH for a buyer. 0.5 if no history."""
    if not atm_iv or not iv_history:
        return 0.5
    below = sum(1 for v in iv_history if v < atm_iv)
    pct = below / len(iv_history)
    return 1.0 - pct                  # low percentile (cheap) → high score


_WEIGHTS = {"term_structure": 0.20, "skew": 0.15, "moneyness": 0.20,
            "theta": 0.20, "iv_level": 0.25}


def combine_features(feats: Dict[str, float]) -> float:
    """Weighted blend of available sub-scores → 0..1 composite. EM realism is a
    diagnostic, not a predictor, so it is NOT part of the composite."""
    num = sum(_WEIGHTS[k] * v for k, v in feats.items() if k in _WEIGHTS)
    den = sum(_WEIGHTS[k] for k in feats if k in _WEIGHTS)
    return num / den if den else 0.5


# ── IC harness ──────────────────────────────────────────────────────────────
def compute_ic(samples: List[Dict[str, float]]) -> Dict[str, Any]:
    """Pearson + Spearman IC between 'score' and 'ret' over samples."""
    scores = [s["score"] for s in samples]
    rets = [s["ret"] for s in samples]
    n = len(samples)
    out = {"n": n, "ic_pearson": None, "p_pearson": None,
           "ic_spearman": None, "p_spearman": None}
    if n < 3:
        return out
    try:
        from scipy import stats
        pe = stats.pearsonr(scores, rets)
        sp = stats.spearmanr(scores, rets)
        out.update(ic_pearson=float(pe[0]), p_pearson=float(pe[1]),
                   ic_spearman=float(sp[0]), p_spearman=float(sp[1]))
    except Exception:
        import numpy as np
        out["ic_pearson"] = float(np.corrcoef(scores, rets)[0, 1])
    return out


def _spot_history(symbol: str):
    """Adjusted close history (yfinance) → {iso_date: close}. Free, no key."""
    import warnings

    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h = yf.Ticker(symbol).history(period="6y")["Close"]
    return {d.date().isoformat(): float(v) for d, v in h.items()}


def _quintiles(samples):
    if len(samples) < 10:
        return []
    s = sorted(samples, key=lambda x: x["score"])
    q = max(1, len(s) // 5)
    out = []
    for i in range(0, len(s), q):
        grp = s[i:i + q]
        if not grp:
            continue
        out.append({"n": len(grp),
                    "avg_score": round(mean(g["score"] for g in grp), 3),
                    "avg_ret": round(mean(g["ret"] for g in grp), 3),
                    "win_rate": round(sum(1 for g in grp if g["ret"] > 0) / len(grp), 2)})
    return out


def run_validation(symbols, dates, target_dte=30, exit_dte=21,
                   db_path=None) -> Dict[str, Any]:
    """For each (symbol, entry_date): score the top reconstructable call and record
    its REAL forward return exit_dte trading-rows later. Returns IC + quintiles."""
    from src import dolt_options as _do
    kw = {"db_path": db_path} if db_path else {}
    samples: List[Dict[str, float]] = []
    em_samples: List[float] = []
    spot_cache: Dict[str, Dict[str, float]] = {}

    for symbol in symbols:
        symbol = symbol.upper()
        if symbol not in spot_cache:
            spot_cache[symbol] = _spot_history(symbol)
        spots = spot_cache[symbol]
        sdates = sorted(spots)
        iv_hist: List[float] = []
        for entry_date in dates:
            entry_date = _do._clamp_date(entry_date)
            spot = spots.get(entry_date)
            if spot is None:
                continue
            try:
                entry_date_actual, chain = _do.get_chain_near(symbol, entry_date, **kw)
            except _do.DoltRateLimited:
                # Rate-limited: stop fetching, keep whatever we've gathered so far.
                print(f"  [rate-limited at {symbol} {entry_date}] — returning partial results "
                      f"({len(samples)} samples so far)")
                ic = compute_ic(samples)
                ic["em_realism_mean"] = round(mean(em_samples), 3) if em_samples else None
                ic["quintiles"] = _quintiles(samples)
                ic["partial"] = True
                return ic
            except _do.DoltQueryError:
                continue
            if not chain:
                continue
            atm = _atm(chain, spot, "call")
            if atm and atm.get("iv"):
                iv_hist.append(atm["iv"])
            try:
                ei = sdates.index(entry_date)
            except ValueError:
                continue
            xi = min(ei + exit_dte, len(sdates) - 1)
            exit_date = sdates[xi]
            # The contract must outlive the hold, else it expires before exit and
            # is absent from the exit chain (every sample would drop).
            hold_days = (_dt.date.fromisoformat(exit_date)
                         - _dt.date.fromisoformat(entry_date)).days
            floor_dte = max(7, hold_days + 3)
            target_strike = spot * 1.03   # ~30-delta OTM proxy for the picker
            c = _do.nearest_contract(chain, "call", target_strike, entry_date_actual,
                                     target_dte=max(target_dte, hold_days + 10),
                                     min_dte=floor_dte)
            if not c or not c.get("ask") or c["ask"] <= 0:
                continue
            # Moneyness sanity guard: if the nearest strike is far from spot, the
            # symbol likely had a split (yfinance spot is split-adjusted, DoltHub
            # strikes are not) — skip rather than score a corrupted sample.
            if abs(c["strike"] / spot - 1.0) > 0.4:
                continue
            # Exit on the nearest available date with data (DoltHub has gaps).
            try:
                _, exit_chain = _do.get_chain_near(symbol, exit_date, **kw)
            except _do.DoltRateLimited:
                print(f"  [rate-limited at {symbol} {exit_date}] — returning partial results "
                      f"({len(samples)} samples so far)")
                ic = compute_ic(samples)
                ic["em_realism_mean"] = round(mean(em_samples), 3) if em_samples else None
                ic["quintiles"] = _quintiles(samples)
                ic["partial"] = True
                return ic
            except _do.DoltQueryError:
                continue
            exit_c = next((x for x in exit_chain if x["strike"] == c["strike"]
                           and x["expiration"] == c["expiration"] and x["type"] == "call"), None)
            if not exit_c or exit_c.get("bid") is None:
                continue
            real_ret = (exit_c["bid"] - c["ask"]) / c["ask"]   # long call return
            feats = {
                "term_structure": term_structure_score(chain, spot, entry_date),
                "skew": skew_score(chain, spot),
                "moneyness": moneyness_score(c.get("delta")),
                "theta": theta_score(c.get("theta"), c.get("mid")),
                "iv_level": iv_level_score(atm["iv"] if atm else None, iv_hist[:-1] or None),
            }
            score = combine_features(feats)
            samples.append({"score": score, "ret": real_ret})
            exit_spot = spots.get(exit_date, spot)
            em_samples.append(em_realism_score(atm["iv"] if atm else None,
                              target_dte, spot, abs(exit_spot - spot)))

    ic = compute_ic(samples)
    ic["em_realism_mean"] = round(mean(em_samples), 3) if em_samples else None
    ic["quintiles"] = _quintiles(samples)
    return ic


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(
        description="Validate price-based scorer slice vs real option returns")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    from src import dolt_options as _do
    cfg = {}
    try:
        cfg = (json.load(open("config.json")).get("dolt_options") or {})
    except Exception:
        pass
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or cfg.get("basket", ["AAPL", "SPY"])
    dates = _do._date_range(args.start or cfg.get("validate_start"),
                            args.end or cfg.get("validate_end"),
                            weekly=args.weekly or cfg.get("validate_sampling") == "weekly")
    print(f"Validating {len(syms)} symbols x {len(dates)} dates (real DoltHub marks)...")
    out = run_validation(syms, dates,
                         target_dte=int(cfg.get("target_dte", 30)),
                         exit_dte=int(cfg.get("exit_dte", 21)),
                         db_path=args.db or cfg.get("cache_path"))
    print(json.dumps({k: v for k, v in out.items() if k != "quintiles"}, indent=1))
    print("\nScore quintiles (low→high):")
    for qd in out.get("quintiles", []):
        print(f"  score~{qd['avg_score']:.2f}  n={qd['n']:4}  win {qd['win_rate']:.0%}  avg_ret {qd['avg_ret']:+.1%}")
    print("\nNOTE: validates the price/IV/Greek slice only — news/sentiment/catalyst "
          "components are not reconstructable historically.")


if __name__ == "__main__":
    _cli()
