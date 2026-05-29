"""Walk-forward out-of-sample IC validation against paper_trades.db."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.backtest_optimizer import (
    BacktestResult, CURRENT_WEIGHTS, WEIGHT_KEYS, optimize_weights,
)

# Map WEIGHT_KEYS names to their actual column names in paper_trades.db.
# Most follow the pattern "<key>_score", but a few deviate.
_WEIGHT_KEY_TO_COL = {
    "pop":              "pop_score",
    "em_realism":       "em_realism_score",
    "iv_mispricing":    "iv_mispricing_score",
    "rr":               "rr_score",
    "momentum":         "momentum_score",
    "iv_rank":          "iv_rank_score",
    "liquidity":        "liquidity_score",
    "catalyst":         "catalyst_score",
    "theta":            "theta_score",
    "ev":               "ev_score",
    "trader_pref":      "trader_pref_score",
    "iv_edge":          "iv_edge_score",
    "skew_align":       "skew_align_score",
    "gamma_theta":      "gamma_theta_score",
    "pcr":              "pcr_score",
    "gex":              "gex_score",
    "oi_change":        "oi_change_score",
    "sentiment":        "sentiment_score_norm",   # non-standard name
    "option_rvol":      "option_rvol_score",
    "vrp":              "vrp_score",
    "gamma_pin":        "gamma_pin_score",
    "max_pain":         "max_pain_score",
    "iv_velocity":      "iv_velocity_score",
    "gamma_magnitude":  "gamma_magnitude_score",
    "vega_risk":        "vega_risk_score",
    "term_structure":   "term_structure_score",
    "spread":           "spread_score",
}

_COMPONENT_COLS = [_WEIGHT_KEY_TO_COL[k] for k in WEIGHT_KEYS]


@dataclass
class Trade:
    rowid: int
    entry_date: str
    pnl_pct: float
    components: np.ndarray


def load_trades(db_path: str, strategy: str = "Long Call") -> List[Trade]:
    cols = ", ".join(_COMPONENT_COLS)
    sql = (
        f"SELECT rowid, date, pnl_pct, {cols} FROM trades "
        "WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
        "AND COALESCE(paper_only, 0) = 0 "
        "AND strategy_name = ? "
        "ORDER BY date ASC, rowid ASC"
    )
    out: List[Trade] = []
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(sql, (strategy,)).fetchall():
            rowid, entry_date, pnl = row[0], row[1], row[2]
            comps = np.array(
                [(v if v is not None else 0.5) for v in row[3:]], dtype=float
            )
            try:
                out.append(
                    Trade(
                        rowid=int(rowid),
                        entry_date=str(entry_date),
                        pnl_pct=float(pnl),
                        components=comps,
                    )
                )
            except (TypeError, ValueError):
                continue
    return out


def build_folds(
    trades: List[Trade],
    train_size: int,
    test_size: int,
    step: int,
) -> Iterator[Tuple[List[int], List[int]]]:
    n = len(trades)
    i = 0
    while i + train_size + test_size <= n:
        train_slice = trades[i : i + train_size]
        test_slice = trades[i + train_size : i + train_size + test_size]
        train_ids = [t.rowid for t in train_slice]
        test_ids = [t.rowid for t in test_slice]
        assert not (set(train_ids) & set(test_ids)), (
            f"LEAK in fold starting at {i}"
        )
        yield train_ids, test_ids
        i += step


def _fit_weights_on_fold(train_trades: List[Trade]) -> np.ndarray:
    scores = np.vstack([t.components for t in train_trades])
    pnls = np.array([t.pnl_pct for t in train_trades])
    bt = BacktestResult(
        component_scores=scores,
        pnl_pct=pnls,
        symbols=["FOLD"] * len(train_trades),
    )
    w_dict = optimize_weights(
        bt,
        method="minimize",
        n_trials=200,
        l2_lambda=0.10,
        verbose=False,
        current_weights=CURRENT_WEIGHTS,
        mask_zero_variance=True,
    )
    return np.array([w_dict[k] for k in WEIGHT_KEYS], dtype=float)


def _score_test_fold(test_trades: List[Trade], weights: np.ndarray) -> np.ndarray:
    scores = np.vstack([t.components for t in test_trades])
    return scores @ weights


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    boots = np.array(
        [rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def run_walk_forward(
    db_path: str,
    strategy: str = "Long Call",
    train_size: int = 44,
    test_size: int = 10,
    step: int = 10,
    output_dir: Optional[str] = None,
) -> dict:
    trades = load_trades(db_path, strategy=strategy)
    n_total = len(trades)
    folds = list(build_folds(trades, train_size, test_size, step))

    per_fold: List[dict] = []
    all_test_scores: List[float] = []
    all_test_pnls: List[float] = []

    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        train_set = set(train_ids)
        test_set = set(test_ids)
        train_trades = [t for t in trades if t.rowid in train_set]
        test_trades = [t for t in trades if t.rowid in test_set]
        weights = _fit_weights_on_fold(train_trades)
        composite_test = _score_test_fold(test_trades, weights)
        pnl_test = np.array([t.pnl_pct for t in test_trades])
        if composite_test.std() < 1e-8 or pnl_test.std() < 1e-8:
            ic_p, ic_p_pval = 0.0, 1.0
            ic_s, ic_s_pval = 0.0, 1.0
        else:
            ic_p, ic_p_pval = (
                float(x) for x in pearsonr(composite_test, pnl_test)
            )
            ic_s, ic_s_pval = (
                float(x) for x in spearmanr(composite_test, pnl_test)
            )
        per_fold.append(
            {
                "fold": fold_idx,
                "n_train": len(train_trades),
                "n_test": len(test_trades),
                "ic_pearson": ic_p,
                "p_pearson": ic_p_pval,
                "ic_spearman": ic_s,
                "p_spearman": ic_s_pval,
            }
        )
        all_test_scores.extend(composite_test.tolist())
        all_test_pnls.extend(pnl_test.tolist())

    pooled_s = np.array(all_test_scores)
    pooled_p = np.array(all_test_pnls)
    if pooled_s.size > 1 and pooled_s.std() > 1e-8:
        pooled_ic, pooled_pval = (
            float(x) for x in pearsonr(pooled_s, pooled_p)
        )
    else:
        pooled_ic, pooled_pval = 0.0, 1.0

    fold_ics = np.array([f["ic_pearson"] for f in per_fold])
    ci_lo, ci_hi = (
        _bootstrap_ci(fold_ics)
        if len(fold_ics) > 0
        else (float("nan"), float("nan"))
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": db_path,
        "strategy": strategy,
        "n_total_trades": n_total,
        "n_folds": len(folds),
        "train_size": train_size,
        "test_size": test_size,
        "step": step,
        "pooled_ic": pooled_ic,
        "pooled_pvalue": pooled_pval,
        "fold_ic_mean": float(fold_ics.mean()) if fold_ics.size else 0.0,
        "fold_ic_ci_95": [ci_lo, ci_hi],
        "folds_ic_positive": int((fold_ics >= 0).sum()),
        "folds": per_fold,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d")
        json_name = (
            f"walk_forward_{strategy.lower().replace(' ', '_')}_{stamp}.json"
        )
        md_name = json_name.replace(".json", ".md")
        (out_path / json_name).write_text(json.dumps(summary, indent=2))
        (out_path / md_name).write_text(_format_markdown(summary))
        summary["json_path"] = json_name
        summary["md_path"] = md_name

    return summary


def _format_markdown(s: dict) -> str:
    lines = [
        f"# Walk-Forward OOS IC — {s['strategy']}",
        "",
        f"- Generated: {s['generated_at']}",
        f"- DB: `{s['db_path']}`",
        f"- Total trades: {s['n_total_trades']}",
        (
            f"- Folds: {s['n_folds']}  "
            f"(train={s['train_size']}, test={s['test_size']}, step={s['step']})"
        ),
        "",
        "## Aggregate",
        f"- **Pooled OOS IC:** {s['pooled_ic']:+.3f}  (p={s['pooled_pvalue']:.3f})",
        f"- Per-fold IC mean: {s['fold_ic_mean']:+.3f}",
        (
            f"- Per-fold IC 95% CI: "
            f"[{s['fold_ic_ci_95'][0]:+.3f}, {s['fold_ic_ci_95'][1]:+.3f}]"
        ),
        f"- Folds with IC >= 0: {s['folds_ic_positive']} / {s['n_folds']}",
        "",
        "## Per-fold",
        "| Fold | n_train | n_test | IC (Pearson) | p | IC (Spearman) | p |",
        "|------|---------|--------|--------------|---|---------------|---|",
    ]
    for f in s["folds"]:
        lines.append(
            f"| {f['fold']} | {f['n_train']} | {f['n_test']} | "
            f"{f['ic_pearson']:+.3f} | {f['p_pearson']:.3f} | "
            f"{f['ic_spearman']:+.3f} | {f['p_spearman']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward OOS IC validation")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--strategy", default="Long Call")
    ap.add_argument("--train", type=int, default=44)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--output", default="reports")
    args = ap.parse_args()
    result = run_walk_forward(
        db_path=args.db,
        strategy=args.strategy,
        train_size=args.train,
        test_size=args.test,
        step=args.step,
        output_dir=args.output,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
