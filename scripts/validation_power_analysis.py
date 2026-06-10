"""Validation power analysis (Sub-project B).

Answers, honestly: can the wait to a trustworthy real-money verdict be shortened
*without statistical dishonesty*? Produces docs/VALIDATION_POWER.md.

Three lenses:
1. Power analysis — given the observed effect (~IC 0.10), how many trades are
   needed for p<0.05 at 80% power? (Reality-check on the n=50 gate.)
2. Minimum detectable effect — at n=50, what observed IC is even significant?
   (Exposes whether the gate's "IC>=0.08 AND p<0.05" is internally binding.)
3. Bayesian sequential read — P(true IC >= 0.08) given what we've observed so far,
   as an honest alternative to a fixed-n threshold.

All gate-relevant numbers are computed from paper_trades.db; nothing here changes
the gate. The output is a recommendation for a human decision logged in
status/DECISIONS.md.

Run: PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.validation_power_analysis
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from scipy.stats import norm, t as t_dist


# ── Pure statistics ─────────────────────────────────────────────────────────

def required_n_pearson(r: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Sample size to detect a Pearson correlation of |r| at the given two-sided
    alpha and power, via the Fisher z-transform. Returns trades needed."""
    r = abs(r)
    if r <= 0 or r >= 1:
        raise ValueError("r must be in (0, 1)")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    c = math.atanh(r)  # Fisher transform
    n = ((z_alpha + z_power) / c) ** 2 + 3
    return int(math.ceil(n))


def min_detectable_r(n: int, alpha: float = 0.05) -> float:
    """Smallest |r| that is significant at two-sided alpha for sample size n,
    inverting the t-statistic t = r*sqrt((n-2)/(1-r^2))."""
    if n < 3:
        return float("nan")
    df = n - 2
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    return float(t_crit / math.sqrt(df + t_crit ** 2))


def pearson_p_value(r: float, n: int) -> float:
    """Two-sided p-value for a Pearson r at sample size n."""
    if n < 3:
        return float("nan")
    df = n - 2
    if abs(r) >= 1:
        return 0.0
    t_stat = r * math.sqrt(df / (1 - r ** 2))
    return float(2 * (1 - t_dist.cdf(abs(t_stat), df)))


def posterior_prob_ic_above(observed_r: float, n: int, threshold: float = 0.08) -> float:
    """P(true IC >= threshold) under a flat prior, using the Fisher-z normal
    approximation: true z ~ Normal(atanh(observed_r), 1/(n-3)).

    Delegates to the canonical implementation in src.phase1_checkpoint (also
    reported in the weekly checkpoint) so the math lives in one place.
    """
    from src.phase1_checkpoint import posterior_ic_above
    p = posterior_ic_above(observed_r, n, threshold=threshold)
    return float("nan") if p is None else p


# ── Data access ─────────────────────────────────────────────────────────────

def _load_long_calls(db_path: str, since: Optional[str] = None,
                     clean_only: bool = True) -> Tuple[list, list]:
    sql = ("SELECT quality_score, pnl_pct FROM trades "
           "WHERE strategy_name='Long Call' AND status='CLOSED' "
           "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL")
    params: list = []
    if clean_only:
        sql += " AND COALESCE(paper_only,0)=0"
    if since:
        sql += " AND date >= ?"
        params.append(since)
    scores, returns = [], []
    with sqlite3.connect(db_path) as conn:
        for q, p in conn.execute(sql, params).fetchall():
            try:
                scores.append(float(q)); returns.append(float(p))
            except (TypeError, ValueError):
                continue
    return scores, returns


def _ic(scores: list, returns: list) -> Tuple[float, float, int]:
    n = len(scores)
    if n < 3:
        return (0.0, 1.0, n)
    import numpy as np
    from scipy.stats import pearsonr
    s, r = np.array(scores), np.array(returns)
    if s.std() < 1e-8 or r.std() < 1e-8:
        return (0.0, 1.0, n)
    ic, p = (float(x) for x in pearsonr(s, r))
    return (ic, p, n)


# ── Report ──────────────────────────────────────────────────────────────────

def build_report(db_path: str, phase1_start: str, today: Optional[str] = None) -> str:
    today = today or datetime.now().strftime("%Y-%m-%d")

    fwd_s, fwd_r = _load_long_calls(db_path, since=phase1_start, clean_only=True)
    fwd_ic, fwd_p, fwd_n = _ic(fwd_s, fwd_r)
    all_s, all_r = _load_long_calls(db_path, since=None, clean_only=True)
    all_ic, all_p, all_n = _ic(all_s, all_r)

    # The historical OOS read we trust is ~+0.10 (walk-forward). Use that as the
    # planning effect size for power; also report what the raw pools show.
    planning_r = 0.10
    n_needed = required_n_pearson(planning_r)
    mdr_50 = min_detectable_r(50)
    mdr_now = min_detectable_r(max(3, fwd_n)) if fwd_n >= 3 else float("nan")
    post_fwd = posterior_prob_ic_above(fwd_ic, fwd_n) if fwd_n > 3 else float("nan")
    post_all = posterior_prob_ic_above(all_ic, all_n) if all_n > 3 else float("nan")

    lines = [
        f"# Validation Power Analysis — {today}", "",
        "_Generated by `scripts/validation_power_analysis.py`. Does not change the "
        "gate; it informs a human decision (see `status/DECISIONS.md`)._", "",
        "## The honest bottom line", "",
        f"- To detect an effect of **IC≈{planning_r:.2f}** (our best out-of-sample "
        f"estimate) at p<0.05 with 80% power, you need **~{n_needed} closed trades** "
        "— not 50.",
        f"- At **n=50**, the smallest IC that is even significant at p<0.05 is "
        f"**{mdr_50:+.3f}**. So the gate's `IC≥0.08 AND p<0.05` is really one rule: "
        f"the `p<0.05` clause forces an observed IC of ~{mdr_50:.2f} at n=50, far "
        "above 0.08. The `IC≥0.08` floor is non-binding at that sample size.",
        "- **Implication:** n=50 will only ever fire READY if the *true* edge is "
        f"strong (≳{mdr_50:.2f}). A real-but-modest edge (~0.10) cannot clear a "
        "fixed-n=50 frequentist gate — it would need hundreds of trades.", "",
        "## 1. Power analysis (how many trades?)", "",
        "| true IC | trades for p<0.05 @ 80% power |",
        "|---|---|",
        f"| 0.10 | {required_n_pearson(0.10)} |",
        f"| 0.15 | {required_n_pearson(0.15)} |",
        f"| 0.20 | {required_n_pearson(0.20)} |",
        f"| 0.28 | {required_n_pearson(0.28)} |",
        "", "## 2. Minimum detectable IC by sample size", "",
        "| n | min significant IC (p<0.05) |",
        "|---|---|",
        f"| 50 | {min_detectable_r(50):+.3f} |",
        f"| 100 | {min_detectable_r(100):+.3f} |",
        f"| 200 | {min_detectable_r(200):+.3f} |",
        f"| 400 | {min_detectable_r(400):+.3f} |",
        "", "## 3. Where the data stands now", "",
        f"- Forward cohort (clean, post-{phase1_start}): n={fwd_n}, "
        f"IC={fwd_ic:+.3f}, p={fwd_p:.3f}.",
        f"- All clean closed Long Calls (pooled, leakage caveats below): "
        f"n={all_n}, IC={all_ic:+.3f}, p={all_p:.3f}.",
        f"- Bayesian P(true IC ≥ 0.08): forward={_pct(post_fwd)}, "
        f"pooled={_pct(post_all)}.", "",
        "### Pooled-read caveat (why we don't just use the pool)",
        "The pooled number mixes trades scored under different config/weight "
        "versions over time and includes calibration in-sample fitting — it is "
        "**not** a clean out-of-sample estimate. The walk-forward harness "
        "(`src/walk_forward.py`) remains the only leakage-controlled read. Treat the "
        "pool as a sanity check, not evidence.", "",
        "## Recommendation", "",
        _recommendation(fwd_n, fwd_ic, mdr_50, post_fwd), "",
    ]
    return "\n".join(lines) + "\n"


def _pct(x: float) -> str:
    return "n/a" if (x is None or math.isnan(x)) else f"{100*x:.0f}%"


def _recommendation(fwd_n: int, fwd_ic: float, mdr_50: float, post_fwd: float) -> str:
    return (
        "1. **Keep the n≥50 gate as the floor, but read it correctly.** At n=50 it "
        f"is effectively a test for a *strong* edge (observed IC ≳ {mdr_50:.2f}). "
        "That is a legitimate, conservative bar — if the screener is genuinely good "
        "the edge will show; if it's marginal, it won't clear, which is the honest "
        "outcome we want.\n"
        "2. **Adopt the Bayesian posterior as the EXTEND/READY tie-breaker.** Once "
        "n≥50, instead of a brittle p-value cliff, also report P(true IC ≥ 0.08). "
        "Going live needs BOTH n≥50 AND posterior ≥ 0.85 (and the frequentist "
        "p<0.05 as the strong-edge confirmation). This avoids both false-go and "
        "endless waiting.\n"
        "3. **Do NOT chase n≈800.** Detecting a 0.10 edge frequentist-clean would "
        "take ~2–3 years at this pace — not worth it. If the edge is only ~0.10, the "
        "correct decision is a *small* allocation justified by the Bayesian posterior "
        "and strict risk caps, not a large one justified by a p-value we can't reach.\n"
        "4. **Action:** leave thresholds unchanged for now; revisit when the forward "
        "cohort reaches n≥50. This document is the basis for that future call."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation power analysis")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--output", default="docs/VALIDATION_POWER.md")
    args = ap.parse_args()

    import json
    with open(args.config) as f:
        phase1_start = (json.load(f).get("auto_log") or {}).get("phase1_start_date")
    if not phase1_start:
        raise SystemExit("config.json missing auto_log.phase1_start_date")

    report = build_report(args.db, phase1_start)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(f"Wrote {args.output}")
    print(report)


if __name__ == "__main__":
    main()
