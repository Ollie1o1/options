"""Walk-forward out-of-sample IC validation against paper_trades.db."""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.alloc.validate import expected_max_sharpe
from src.backtest_optimizer import (
    BacktestResult, CURRENT_WEIGHTS, WEIGHT_KEYS, optimize_weights,
)
from src.ledger_filters import exclude_ruled_duplicates

# Weights fitted per fold. Below twice this many observations the fit is
# underdetermined and its OOS IC is noise with a decimal point.
MIN_TRAIN_AFTER_PURGE = 2 * len(WEIGHT_KEYS)

# Fewer surviving folds than this and the run reports nothing rather than an
# average of two numbers.
MIN_FOLDS = 3

# Trials per fold inside `optimize_weights` — the size of the weight search,
# needed to state the bar a result has to clear. `_fit_weights_on_fold` passes
# this straight to `optimize_weights(n_trials=...)`, so this is the ONLY
# place the search size is written down.
TRIALS_PER_FOLD = 200

# Default train_size for both `run_walk_forward` and the CLI. A module-level
# constant so the two cannot drift the way they did before: the CLI's
# argparse default was left at the pre-purge value of 44 while the function
# default moved to 100, silently making every unflagged CLI run refuse.
DEFAULT_TRAIN_SIZE = 100

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
    exit_date: str
    pnl_pct: float
    components: np.ndarray


def load_trades(db_path: str, strategy: str = "Long Call") -> List[Trade]:
    cols = ", ".join(_COMPONENT_COLS)
    sql = (
        f"SELECT rowid, date, exit_date, pnl_pct, {cols} FROM trades "
        "WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
        "AND exit_date IS NOT NULL "
        "AND COALESCE(paper_only, 0) = 0 "
        "AND strategy_name = ? "
    )
    out: List[Trade] = []
    with sqlite3.connect(db_path) as conn:
        # A ruled double-log is one decision recorded twice; counting it twice
        # inflates the OOS IC this function feeds into the evidence banner.
        sql += exclude_ruled_duplicates(conn) + " ORDER BY date ASC, rowid ASC"
        for row in conn.execute(sql, (strategy,)).fetchall():
            rowid, entry_date, exit_date, pnl = row[0], row[1], row[2], row[3]
            comps = np.array(
                [(v if v is not None else 0.5) for v in row[4:]], dtype=float
            )
            try:
                out.append(
                    Trade(
                        rowid=int(rowid),
                        entry_date=str(entry_date),
                        exit_date=str(exit_date),
                        pnl_pct=float(pnl),
                        components=comps,
                    )
                )
            except (TypeError, ValueError):
                continue
    return out


def _as_date(value: str) -> date:
    """Parse a stored date. Ledger dates are ISO, sometimes with a time part."""
    return date.fromisoformat(str(value)[:10])


def purge_overlapping(train: List[Trade], test: List[Trade]) -> List[Trade]:
    """Drop training trades whose position was open during the test window.

    A trade entered before the test block but still open inside it has its
    outcome determined by the same price path the test block is scored on.
    `build_folds`' rowid assertion cannot see this: the rows are distinct, the
    information is not.

    The window is the test block's own [earliest entry, latest exit], and the
    overlap test is INCLUSIVE at both ends — a training trade closing on the
    day the window opens shared that day's price path.

    Purging uses each trade's MEASURED interval rather than an assumed holding
    period, so there is no days-to-index conversion to get wrong. `exit_date`
    is complete on every closed strategy in the ledger.
    """
    if not test:
        return list(train)
    lo = min(_as_date(t.entry_date) for t in test)
    hi = max(_as_date(t.exit_date) for t in test)
    return [t for t in train
            if not (_as_date(t.exit_date) >= lo and _as_date(t.entry_date) <= hi)]


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
        n_trials=TRIALS_PER_FOLD,
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


def _write_artifacts(summary: dict, output_dir: Optional[str]) -> None:
    """Write the JSON/markdown report pair, if an output_dir was given.

    Called from both the success and refusal paths in `run_walk_forward` so
    a refusal still leaves a fresh artifact behind — otherwise the evidence
    banner (which always reads the newest `walk_forward_*.json`) would keep
    quoting a stale, pre-purge run as current.
    """
    if not output_dir:
        return
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    strategy = summary["strategy"]
    json_name = (
        f"walk_forward_{strategy.lower().replace(' ', '_')}_{stamp}.json"
    )
    md_name = json_name.replace(".json", ".md")
    (out_path / json_name).write_text(json.dumps(summary, indent=2))
    (out_path / md_name).write_text(_format_markdown(summary))
    summary["json_path"] = json_name
    summary["md_path"] = md_name


def _refused_summary(db_path: str, strategy: str, n_total: int,
                     train_size: int, test_size: int, step: int,
                     n_attempted: int, n_dropped: int, reason: str,
                     output_dir: Optional[str] = None) -> dict:
    """A run that could not measure anything.

    Every statistic is None, never 0.0: a zero here would render in the
    evidence banner as a measured zero IC rather than an absence.
    """
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db_path": db_path,
        "strategy": strategy,
        "n_total_trades": n_total,
        "n_folds": 0,
        "n_folds_attempted": n_attempted,
        "n_folds_dropped": n_dropped,
        "train_size": train_size,
        "test_size": test_size,
        "step": step,
        "refused": True,
        "refused_reason": reason,
        "pooled_ic": None,
        "pooled_pvalue": None,
        "fold_ic_mean": None,
        "fold_ic_ci_95": None,
        "folds_ic_positive": None,
        "n_trials": 0,
        "search_bar_sharpe": None,
        "folds": [],
    }
    _write_artifacts(summary, output_dir)
    return summary


def run_walk_forward(
    db_path: str,
    strategy: str = "Long Call",
    train_size: int = DEFAULT_TRAIN_SIZE,
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
    n_dropped = 0
    n_attempted = len(folds)

    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        train_set = set(train_ids)
        test_set = set(test_ids)
        train_trades = [t for t in trades if t.rowid in train_set]
        test_trades = [t for t in trades if t.rowid in test_set]
        n_requested = len(train_trades)
        train_trades = purge_overlapping(train_trades, test_trades)
        if len(train_trades) < MIN_TRAIN_AFTER_PURGE:
            n_dropped += 1
            continue
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
                "n_train_purged": n_requested - len(train_trades),
                "n_test": len(test_trades),
                "ic_pearson": ic_p,
                "p_pearson": ic_p_pval,
                "ic_spearman": ic_s,
                "p_spearman": ic_s_pval,
            }
        )
        all_test_scores.extend(composite_test.tolist())
        all_test_pnls.extend(pnl_test.tolist())

    if len(per_fold) < MIN_FOLDS:
        if n_attempted == 0:
            # No fold was ever formed, so purging never ran — naming it as
            # the cause would be wrong. Widening train_size only raises the
            # train+test threshold this run already failed to clear, so the
            # remedy is the opposite of the other branch's advice.
            reason = (
                f"no fold could be formed: {n_total} trades < "
                f"train_size+test_size={train_size + test_size}; reduce "
                f"train_size (floor {MIN_TRAIN_AFTER_PURGE}) or wait for "
                f"more closed trades")
        else:
            reason = (f"only {len(per_fold)} of {n_attempted} folds kept "
                      f"{MIN_TRAIN_AFTER_PURGE}+ training trades after purging "
                      f"(minimum {MIN_FOLDS}); widen train_size or wait for more "
                      f"closed trades")
        return _refused_summary(
            db_path, strategy, n_total, train_size, test_size, step,
            n_attempted, n_dropped,
            reason=reason,
            output_dir=output_dir)

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
        "n_folds": len(per_fold),
        "n_folds_attempted": n_attempted,
        "n_folds_dropped": n_dropped,
        "train_size": train_size,
        "test_size": test_size,
        "step": step,
        "refused": False,
        "refused_reason": None,
        "pooled_ic": pooled_ic,
        "pooled_pvalue": pooled_pval,
        "fold_ic_mean": float(fold_ics.mean()) if fold_ics.size else 0.0,
        "fold_ic_ci_95": [ci_lo, ci_hi],
        "folds_ic_positive": int((fold_ics >= 0).sum()),
        "n_trials": TRIALS_PER_FOLD * len(per_fold),
        "search_bar_sharpe": expected_max_sharpe(
            TRIALS_PER_FOLD * len(per_fold),
            trial_variance=1.0 / max(len(pooled_p), 1)),
        "folds": per_fold,
    }

    _write_artifacts(summary, output_dir)

    return summary


def _format_markdown(s: dict) -> str:
    header = [
        f"# Walk-Forward OOS IC — {s['strategy']}",
        "",
        f"- Generated: {s['generated_at']}",
        f"- DB: `{s['db_path']}`",
        f"- Total trades: {s['n_total_trades']}",
        (
            f"- Folds: {s['n_folds']} kept of {s['n_folds_attempted']} "
            f"attempted ({s['n_folds_dropped']} dropped below the training "
            f"floor)  (train={s['train_size']}, test={s['test_size']}, "
            f"step={s['step']})"
        ),
        "",
    ]
    if s.get("refused"):
        header += [
            "## Refused",
            "",
            f"This run reports no statistics: {s['refused_reason']}",
            "",
        ]
        return "\n".join(header) + "\n"
    lines = header + [
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
    ap.add_argument("--train", type=int, default=DEFAULT_TRAIN_SIZE)
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
