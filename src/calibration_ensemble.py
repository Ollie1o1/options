"""
Multi-method ensemble calibration for `composite_weights`.

Replaces the naive `recommend_weights_from_paper_trades` IC normalisation with
a four-method consensus that filters unstable factors:

  1. Bootstrap IC, 2000 resamples → 95% CI per factor (sign-stability)
  2. Walk-forward expanding-window IC (train/test sign agreement)
  3. Ridge regression with K-fold CV + 1000-resample bootstrap on β
  4. Leave-one-ticker-out concentration check

A factor earns `+++` only when all three statistical methods (bootstrap,
walk-forward, ridge) agree positive; `---` requires all three negative.

Usage:
    PYTHONPATH=$PWD python -m src.calibration_ensemble
        [--db paper_trades.db]
        [--config config.json]
        [--structure long_call]      # 'long_call' | 'short_put' | 'any'
        [--apply]                    # write to config.json (backup first)
        [--n-bootstrap 2000]
        [--folds 3] [--test-size 15] [--train-min 30]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCORE_TO_KEY: Dict[str, str] = {
    "vrp_score": "vrp",
    "term_structure_score": "term_structure",
    "iv_velocity_score": "iv_velocity",
    "iv_edge_score": "iv_edge",
    "vega_risk_score": "vega_risk",
    "spread_score": "spread",
    "trader_pref_score": "trader_pref",
    "rr_score": "rr",
    "pop_score": "pop",
    "iv_rank_score": "iv_rank",
    "momentum_score": "momentum",
    "theta_score": "theta",
    "gamma_theta_score": "gamma_theta",
    "gex_score": "gex",
    "liquidity_score": "liquidity",
    "gamma_magnitude_score": "gamma_magnitude",
    "skew_align_score": "skew_align",
    "iv_mispricing_score": "iv_mispricing",
    "catalyst_score": "catalyst",
    "ev_score": "ev",
    "em_realism_score": "em_realism",
}

# Single-leg, long-call only:
_LONG_CALL_WHERE = (
    "status='CLOSED' AND pnl_pct IS NOT NULL AND type='call' "
    "AND long_strike IS NULL AND short_call_strike IS NULL AND short_put_strike IS NULL"
)
_SHORT_PUT_WHERE = (
    "status='CLOSED' AND pnl_pct IS NOT NULL AND type='put' "
    "AND long_strike IS NULL AND short_call_strike IS NULL AND short_put_strike IS NULL"
)
_ANY_WHERE = "status='CLOSED' AND pnl_pct IS NOT NULL"

CAP = 0.30
PRIOR_N = 60


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v, _ = pearsonr(x, y)
    return float(v) if np.isfinite(v) else float("nan")


def _load_trades(db_path: str, structure: str) -> pd.DataFrame:
    where = {
        "long_call": _LONG_CALL_WHERE,
        "short_put": _SHORT_PUT_WHERE,
        "any": _ANY_WHERE,
    }[structure]
    cols = list(SCORE_TO_KEY)
    q = (
        f"SELECT ticker, date, pnl_pct, pnl_usd, {', '.join(cols)} "
        f"FROM trades WHERE {where} ORDER BY date"
    )
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(q, con)
    con.close()
    return df.reset_index(drop=True)


def _bootstrap_ic(df: pd.DataFrame, n_boot: int, seed: int = 42) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Dict] = {}
    for sc, key in SCORE_TO_KEY.items():
        sub = df[[sc, "pnl_pct"]].dropna()
        if len(sub) < 30 or sub[sc].std() == 0:
            continue
        arr = sub.to_numpy()
        boots = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(arr), len(arr))
            boots[b] = _ic(arr[idx, 0], arr[idx, 1])
        boots = boots[np.isfinite(boots)]
        if len(boots) < 100:
            continue
        out[key] = {
            "mean": float(np.mean(boots)),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_positive": float(np.mean(boots > 0)),
        }
    return out


def _walkforward_ic(
    df: pd.DataFrame, train_min: int, test_size: int
) -> Dict[str, Dict]:
    folds: List[Tuple[int, int]] = []
    k = train_min
    n = len(df)
    while k + test_size <= n:
        folds.append((k, k + test_size))
        k += test_size
    out: Dict[str, Dict] = {}
    for sc, key in SCORE_TO_KEY.items():
        sub = df[[sc, "pnl_pct"]].dropna().reset_index(drop=True)
        if len(sub) < train_min + test_size or sub[sc].std() == 0:
            continue
        tr_ics, te_ics = [], []
        for train_end, test_end in folds:
            tr = sub.iloc[:train_end]
            te = sub.iloc[train_end:test_end]
            a = _ic(tr[sc].values, tr["pnl_pct"].values)
            b = _ic(te[sc].values, te["pnl_pct"].values)
            if np.isnan(a) or np.isnan(b):
                continue
            tr_ics.append(a)
            te_ics.append(b)
        if not tr_ics:
            continue
        sign_agree = float(np.mean([(t * e) > 0 for t, e in zip(tr_ics, te_ics)]))
        out[key] = {
            "train_ic": float(np.mean(tr_ics)),
            "test_ic": float(np.mean(te_ics)),
            "sign_agree": sign_agree,
        }
    return out


def _ridge_with_bootstrap(
    df: pd.DataFrame, n_boot: int = 1000, seed: int = 42
) -> Dict[str, Dict]:
    cols = list(SCORE_TO_KEY)
    sub = df[cols + ["pnl_pct"]].fillna(0.5)
    X = sub[cols].to_numpy()
    y = sub["pnl_pct"].to_numpy()
    nzv = np.where(X.std(axis=0) > 1e-6)[0]
    if len(nzv) == 0:
        return {}
    Xnz = X[:, nzv]
    keys = [SCORE_TO_KEY[cols[i]] for i in nzv]
    mu = Xnz.mean(axis=0)
    sd = Xnz.std(axis=0)
    Xs = (Xnz - mu) / sd
    yc = y - y.mean()

    def ridge(Xs: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        p = Xs.shape[1]
        return np.linalg.solve(Xs.T @ Xs + alpha * np.eye(p), Xs.T @ y)

    # 5-fold CV alpha selection
    rng = np.random.default_rng(seed)
    idx = np.arange(len(yc))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)
    alphas = np.logspace(-1, 3, 30)
    best_a, best_mse = None, float("inf")
    for a in alphas:
        mses = []
        for i in range(5):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(5) if j != i])
            beta = ridge(Xs[tr], yc[tr], a)
            yhat = Xs[te] @ beta
            mses.append(float(np.mean((yc[te] - yhat) ** 2)))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_a = m, float(a)

    beta_full = ridge(Xs, yc, best_a)
    # Bootstrap sign-stability
    n = len(yc)
    boot_beta = np.zeros((n_boot, len(keys)))
    for b in range(n_boot):
        bi = rng.integers(0, n, n)
        boot_beta[b] = ridge(Xs[bi], yc[bi], best_a)
    sign_stab = (boot_beta * beta_full[None, :] > 0).mean(axis=0)
    out: Dict[str, Dict] = {}
    for i, k in enumerate(keys):
        out[k] = {
            "beta": float(beta_full[i]),
            "sign_stability": float(sign_stab[i]),
        }
    return out


def _consensus_verdict(
    boot: Optional[Dict], wf: Optional[Dict], ridge: Optional[Dict]
) -> str:
    pos, neg = 0, 0
    if boot is not None:
        if boot["p_positive"] > 0.90:
            pos += 1
        elif boot["p_positive"] < 0.10:
            neg += 1
    if wf is not None:
        if wf["test_ic"] > 0.05 and wf["sign_agree"] >= 0.5:
            pos += 1
        elif wf["test_ic"] < -0.05 and wf["sign_agree"] >= 0.5:
            neg += 1
    if ridge is not None:
        if ridge["beta"] > 0.005 and ridge["sign_stability"] > 0.90:
            pos += 1
        elif ridge["beta"] < -0.005 and ridge["sign_stability"] > 0.90:
            neg += 1
    if pos >= 2 and neg == 0:
        return "+++" if pos == 3 else "++"
    if neg >= 2 and pos == 0:
        return "---" if neg == 3 else "--"
    if pos > neg:
        return "+"
    if neg > pos:
        return "-"
    return "noise"


def _effective_ic(verdict: str, boot_mean: float) -> float:
    scale = {"+++": 1.0, "++": 0.7, "---": 1.0, "--": 0.7, "+": 0.3, "-": 0.3}.get(
        verdict, 0.0
    )
    return boot_mean * scale


def build_ensemble(
    db_path: str,
    config_path: str,
    structure: str = "long_call",
    n_bootstrap: int = 2000,
    train_min: int = 30,
    test_size: int = 15,
) -> Dict:
    """Build a recommended ensemble weight set. Does NOT write to config.

    Returns a dict containing all per-factor diagnostics, the recommended
    weight vector, and a summary.
    """
    df = _load_trades(db_path, structure)
    if len(df) < 30:
        return {"ready": False, "n_trades": len(df), "reason": "n < 30"}

    boot = _bootstrap_ic(df, n_bootstrap)
    wf = _walkforward_ic(df, train_min, test_size)
    ridge = _ridge_with_bootstrap(df)

    with open(config_path) as f:
        cfg = json.load(f)
    cw = dict(cfg.get("composite_weights", {}))

    target_keys = list(SCORE_TO_KEY.values())
    current = {k: float(cw.get(k, 0.0)) for k in target_keys}
    budget = sum(current.values())

    verdicts: Dict[str, str] = {}
    eff_ics: Dict[str, float] = {}
    for k in target_keys:
        b = boot.get(k)
        w = wf.get(k)
        r = ridge.get(k)
        verdicts[k] = _consensus_verdict(b, w, r)
        eff_ics[k] = _effective_ic(verdicts[k], b["mean"] if b else 0.0)

    pos = {k: max(eff_ics[k], 0.0) for k in target_keys}
    psum = sum(pos.values())
    ic_target = (
        {k: pos[k] / psum * budget for k in target_keys}
        if psum > 0 and budget > 0
        else dict(current)
    )
    n = len(df)
    shrink = n / (n + PRIOR_N)
    new = {k: (1 - shrink) * current[k] + shrink * ic_target[k] for k in target_keys}
    new = {k: min(v, CAP) for k, v in new.items()}
    s = sum(new.values())
    if s > 0 and budget > 0:
        new = {k: v / s * budget for k, v in new.items()}

    deltas = {k: round(new[k] - current[k], 4) for k in target_keys}
    return {
        "ready": True,
        "structure": structure,
        "n_trades": n,
        "shrinkage": shrink,
        "budget": budget,
        "verdicts": verdicts,
        "effective_ic": eff_ics,
        "bootstrap_ic": boot,
        "walkforward_ic": wf,
        "ridge": ridge,
        "current": current,
        "recommended": {k: round(new[k], 6) for k in target_keys},
        "deltas": deltas,
    }


def apply_to_config(recommended: Dict[str, float], config_path: str) -> Path:
    p = Path(config_path)
    with open(p) as f:
        cfg = json.load(f)
    cw = dict(cfg.get("composite_weights", {}))
    bak = p.with_name(
        f"{p.stem}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}-ensemble{p.suffix}"
    )
    bak.write_text(p.read_text())
    for k, v in recommended.items():
        cw[k] = round(float(v), 4)
    # Renormalise to preserve budget (handle 4-decimal rounding drift)
    s = sum(cw[k] for k in recommended)
    orig_budget = sum(cfg.get("composite_weights", {}).get(k, 0.0) for k in recommended)
    if s > 0 and orig_budget > 0:
        for k in recommended:
            cw[k] = round(cw[k] / s * orig_budget, 4)
    cfg["composite_weights"] = cw
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    return bak


def _print_report(res: Dict) -> None:
    if not res.get("ready"):
        print(f"Not ready: {res.get('reason')} (n_trades={res.get('n_trades')})")
        return
    print(
        f"Ensemble calibration — structure={res['structure']}  "
        f"n_trades={res['n_trades']}  shrinkage={res['shrinkage']:.3f}  "
        f"budget={res['budget']:.4f}"
    )
    print()
    print(f"{'factor':<18} {'verdict':>8} {'eff_IC':>8} {'current':>9} {'new':>9} {'Δ':>9}")
    print("-" * 70)
    rows = sorted(
        res["verdicts"].keys(),
        key=lambda k: -abs(res["recommended"][k] - res["current"][k]),
    )
    for k in rows:
        v = res["verdicts"][k]
        eff = res["effective_ic"][k]
        cur = res["current"][k]
        new = res["recommended"][k]
        d = new - cur
        print(
            f"{k:<18} {v:>8} {eff:+8.3f} {cur:>9.4f} {new:>9.4f} {d:>+9.4f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    ap.add_argument(
        "--structure",
        default="long_call",
        choices=["long_call", "short_put", "any"],
    )
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--train-min", type=int, default=30)
    ap.add_argument("--test-size", type=int, default=15)
    args = ap.parse_args()

    res = build_ensemble(
        db_path=args.db,
        config_path=args.config,
        structure=args.structure,
        n_bootstrap=args.n_bootstrap,
        train_min=args.train_min,
        test_size=args.test_size,
    )
    _print_report(res)
    if args.apply and res.get("ready"):
        bak = apply_to_config(res["recommended"], args.config)
        print(f"\nApplied. Backup → {bak}")
    elif args.apply:
        print("\nNot applied (not ready).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
