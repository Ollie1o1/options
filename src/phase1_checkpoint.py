"""Phase 1 weekly checkpoint: compute forward-cohort IC and emit a gate decision.

Cohort = trades where strategy_name='Long Call', status='CLOSED', paper_only=0,
and date >= phase1_start_date.

Gate rules:
  n < 50                              -> GATHERING
  n >= 50, IC >= 0.08, p < 0.05       -> READY
  n >= 50, 0.03 <= IC < 0.08          -> EXTEND
  n >= 50, IC < 0.03, weeks >= 6      -> STOP
  otherwise                           -> GATHERING

Never modifies paper_trades.db or config.json. Writes only to reports/.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import math

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr


def _load_cohort(db_path: str, phase1_start: str):
    sql = (
        "SELECT quality_score, pnl_pct FROM trades "
        "WHERE strategy_name='Long Call' AND status='CLOSED' "
        "AND COALESCE(paper_only, 0) = 0 "
        "AND date >= ? "
        "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL"
    )
    scores, returns = [], []
    with sqlite3.connect(db_path) as conn:
        for q, p in conn.execute(sql, (phase1_start,)).fetchall():
            try:
                scores.append(float(q)); returns.append(float(p))
            except (TypeError, ValueError):
                continue
    return np.array(scores), np.array(returns)


def _weeks_between(start: str, end: str) -> int:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return max(0, (e - s).days // 7)


def _bootstrap_ci(s: np.ndarray, r: np.ndarray, n_boot: int = 1000, seed: int = 0):
    if len(s) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(s); boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss, rr = s[idx], r[idx]
        if ss.std() < 1e-8 or rr.std() < 1e-8:
            continue
        boots.append(float(pearsonr(ss, rr)[0]))
    if not boots:
        return (float("nan"), float("nan"))
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def posterior_ic_above(ic: float, n: int, threshold: float = 0.08) -> Optional[float]:
    """P(true IC >= threshold | observed ic, n) under a flat prior on the
    Fisher-z scale: true z ~ Normal(atanh(ic), 1/(n-3)).

    Reporting only — never feeds the gate decision (docs/VALIDATION_POWER.md,
    DECISIONS.md 2026-06-07: no silent gate change). Returns None when n < 4
    or ic is not finite.
    """
    if n < 4 or ic is None or not math.isfinite(ic):
        return None
    z_obs = math.atanh(max(-0.999, min(0.999, float(ic))))
    z_thr = math.atanh(max(-0.999, min(0.999, float(threshold))))
    se = 1.0 / math.sqrt(n - 3)
    return float(1 - norm.cdf((z_thr - z_obs) / se))


def compute_checkpoint(db_path: str, phase1_start: str, today: Optional[str] = None) -> dict:
    today = today or datetime.now().strftime("%Y-%m-%d")
    scores, returns = _load_cohort(db_path, phase1_start)
    n = len(scores)
    weeks = _weeks_between(phase1_start, today)

    if n < 3 or scores.std() < 1e-8 or returns.std() < 1e-8:
        ic_p, p_p, ic_s, p_s = 0.0, 1.0, 0.0, 1.0
    else:
        ic_p, p_p = (float(x) for x in pearsonr(scores, returns))
        ic_s, p_s = (float(x) for x in spearmanr(scores, returns))

    ci_lo, ci_hi = _bootstrap_ci(scores, returns)

    if n < 50:
        decision = "GATHERING"
    elif ic_p >= 0.08 and p_p < 0.05:
        decision = "READY"
    elif 0.03 <= ic_p < 0.08:
        decision = "EXTEND"
    elif ic_p < 0.03 and weeks >= 6:
        decision = "STOP"
    else:
        decision = "GATHERING"

    return {
        "today": today, "phase1_start": phase1_start, "weeks_elapsed": weeks,
        "n_trades": n, "ic_pearson": ic_p, "p_pearson": p_p,
        "ic_spearman": ic_s, "p_spearman": p_s, "ic_95_ci": [ci_lo, ci_hi],
        "decision": decision,
        "posterior_ic_ge_008": posterior_ic_above(ic_p, n, threshold=0.08),
    }


def _format_markdown(r: dict) -> str:
    return "\n".join([
        f"# Phase 1 Checkpoint — {r['today']}", "",
        f"- Phase 1 start: {r['phase1_start']}",
        f"- Weeks elapsed: {r['weeks_elapsed']}",
        f"- Cohort size (Long Call, paper_only=0, post-start, closed): **{r['n_trades']}**", "",
        "## Forward-cohort IC",
        f"- Pearson IC: **{r['ic_pearson']:+.3f}**  (p={r['p_pearson']:.3f})",
        f"- Spearman IC: {r['ic_spearman']:+.3f}  (p={r['p_spearman']:.3f})",
        f"- 95% bootstrap CI: [{r['ic_95_ci'][0]:+.3f}, {r['ic_95_ci'][1]:+.3f}]",
        _posterior_line(r.get("posterior_ic_ge_008")), "",
        f"## Gate decision: **{r['decision']}**", "",
        _decision_explainer(r["decision"]), "",
    ]) + "\n"


def _posterior_line(p: Optional[float]) -> str:
    if p is None:
        return "- Bayesian P(true IC >= 0.08): n/a (n < 4) — reporting only, gate unchanged"
    return (f"- Bayesian P(true IC >= 0.08): **{p:.0%}** — reporting only, gate thresholds "
            f"unchanged (see docs/VALIDATION_POWER.md)")


def _decision_explainer(d: str) -> str:
    return {
        "GATHERING": "Need >=50 trades before the gate can fire. Keep auto-logging.",
        "READY": "**Edge proven.** Phase 3 (execution stack) unlocked.",
        "EXTEND": "Edge is positive but below the bar. Continue gathering for 2 more weeks.",
        "STOP": "**Edge not detected** at week 6. Honor the kill criterion: pause and review.",
    }.get(d, "Unknown decision.")


def write_checkpoint(result: dict, output_dir: str = "reports") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"checkpoint_{result['today']}.md"
    md_path.write_text(_format_markdown(result))

    hist_path = out / "checkpoint_history.tsv"
    if not hist_path.exists():
        hist_path.write_text("date\tweeks\tn\tic\tp\tdecision\n")
    with hist_path.open("a") as f:
        f.write(f"{result['today']}\t{result['weeks_elapsed']}\t{result['n_trades']}\t"
                f"{result['ic_pearson']:.4f}\t{result['p_pearson']:.4f}\t{result['decision']}\n")

    if result["decision"] in ("READY", "STOP"):
        (out / "GATE_STATUS.md").write_text(
            f"GATE: **{result['decision']}** as of {result['today']}  "
            f"(n={result['n_trades']}, IC={result['ic_pearson']:+.3f}, "
            f"p={result['p_pearson']:.3f}, weeks={result['weeks_elapsed']})\n"
            f"See `{md_path.name}` for details.\n"
        )
    return {"md": str(md_path), "history": str(hist_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 weekly checkpoint")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--output", default="reports")
    ap.add_argument("--dry-run", action="store_true", help="Compute and print only; do not write")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    phase1_start = (cfg.get("auto_log") or {}).get("phase1_start_date")
    if not phase1_start:
        raise SystemExit("config.json missing auto_log.phase1_start_date")

    result = compute_checkpoint(db_path=args.db, phase1_start=phase1_start)
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        paths = write_checkpoint(result, output_dir=args.output)
        print(f"\nWrote: {paths}")


if __name__ == "__main__":
    main()
