"""Phase 1 weekly checkpoint: compute forward-cohort IC and emit a gate decision.

Cohort = trades where strategy_name='Long Call', status='CLOSED', paper_only=0,
and date >= phase1_start_date.

Gate v2 (docs/GATE_REDESIGN_SPEC.md, signed off by the operator 2026-07-31) is
the active rule, selected by ``config.gate.version``:

  n_eff < 50                                   -> GATHERING
  P(true RANK IC >= 0.08) >= 0.85              -> READY   (sign-guard applies)
  P(true RANK IC >= 0.08) <= 0.15              -> STOP
  rank IC < 0.03                               -> STOP
  EXTEND granted twice already                 -> STOP
  otherwise                                    -> EXTEND  (bounded, 2 x 2 weeks)

Three deliberate differences from v1. The statistic is the RANK IC, because
long-call returns are floored at -100% and driven above by a few take-profits,
which is precisely where Pearson misleads. The sample size is EFFECTIVE n —
batch auto-logging clusters entries by day and same-day trades share that day's
move. And EXTEND is bounded: under v1 an IC drifting between 0.03 and 0.08
extended forever, which made "keep gathering" a permanent answer rather than a
decision.

v1 is preserved verbatim in ``decide_v1`` and its verdict is reported beside
v2 on every run. A superseded rule that vanishes cannot be audited, and the
two agreeing (or not) is itself evidence.

Never modifies paper_trades.db or config.json. Writes only to reports/.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import math

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr


def _load_cohort(db_path: str, phase1_start: str, max_capital_at_risk: Optional[float] = None):
    """Cohort scores and returns.

    With ``max_capital_at_risk`` set, restricts to trades the account could
    actually have opened. Rows with NULL capital_at_risk are excluded from that
    subset rather than assumed affordable — unbounded risk is not small risk.
    """
    sql = (
        "SELECT quality_score, pnl_pct, date FROM trades "
        "WHERE strategy_name='Long Call' AND status='CLOSED' "
        "AND COALESCE(paper_only, 0) = 0 "
        "AND date >= ? "
        "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL"
    )
    params: tuple = (phase1_start,)
    if max_capital_at_risk is not None:
        sql += " AND capital_at_risk IS NOT NULL AND capital_at_risk <= ?"
        params = (phase1_start, float(max_capital_at_risk))
    scores, returns, dates = [], [], []
    with sqlite3.connect(db_path) as conn:
        for q, p, d in conn.execute(sql, params).fetchall():
            try:
                scores.append(float(q)); returns.append(float(p))
            except (TypeError, ValueError):
                continue
            dates.append(str(d)[:10] if d else "")
    return np.array(scores), np.array(returns), dates


def design_effect(returns: np.ndarray, dates) -> tuple:
    """(icc, design_effect, n_eff) for returns clustered by entry date.

    Batch auto-logging enters several trades on the same day, and same-day
    trades share that day's market move — so they are not independent
    observations and nominal n overstates the evidence. One-way ANOVA over
    entry-day clusters gives the intraclass correlation; the design effect is
    ``1 + (mean cluster size - 1) * ICC`` and n_eff = n / DE.

    ICC is floored at 0: a negative estimate means "no positive clustering
    detected", not "these observations are better than independent".
    """
    n = len(returns)
    if n < 3 or not dates or len(dates) != n:
        return 0.0, 1.0, float(n)

    groups: dict = {}
    for value, day in zip(returns, dates):
        groups.setdefault(day, []).append(float(value))
    k = len(groups)
    if k < 2 or k == n:
        # Every trade on its own day (or all on one day): no cluster structure
        # to estimate, so nominal n stands.
        return 0.0, 1.0, float(n)

    grand = float(np.mean(returns))
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    ss_within = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in groups.values())
    df_between, df_within = k - 1, n - k
    if df_within <= 0:
        return 0.0, 1.0, float(n)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    sizes = [len(v) for v in groups.values()]
    # Unequal cluster sizes: the standard m0 correction, not the plain mean.
    m0 = (n - sum(s * s for s in sizes) / n) / df_between
    if m0 <= 0 or ms_between + (m0 - 1) * ms_within == 0:
        return 0.0, 1.0, float(n)

    icc = (ms_between - ms_within) / (ms_between + (m0 - 1) * ms_within)
    icc = max(0.0, float(icc))
    mean_size = n / k
    de = 1.0 + (mean_size - 1.0) * icc
    de = max(1.0, float(de))
    return icc, de, float(n) / de


# Gate v2 bands (docs/GATE_REDESIGN_SPEC.md, signed 2026-07-31).
GATE_V2_READY_POSTERIOR = 0.85
GATE_V2_STOP_POSTERIOR = 0.15
GATE_V2_IC_FLOOR = 0.03
GATE_V2_MIN_N_EFF = 50
GATE_V2_MAX_EXTENSIONS = 2


def decide_v1(n: int, ic_p: float, p_p: float, weeks: int) -> str:
    """The original gate. Preserved verbatim so its verdict stays reportable
    beside v2 — a decision rule that silently disappears cannot be audited."""
    if n < 50:
        return "GATHERING"
    if ic_p >= 0.08 and p_p < 0.05:
        return "READY"
    if 0.03 <= ic_p < 0.08:
        return "EXTEND"
    if ic_p < 0.03 and weeks >= 6:
        return "STOP"
    return "GATHERING"


def decide_v2(n_eff: float, ic_rank: float, ic_pearson: float,
              extensions_used: int = 0) -> tuple:
    """The signed redesign. Returns (decision, reason).

    Reads RANK IC, because long-call returns are floored at -100% and
    outlier-driven above, which is the distribution Pearson handles worst.
    Sample size is effective n, not nominal. EXTEND is bounded: it can be
    granted at most twice, after which the decision must resolve.
    """
    if n_eff < GATE_V2_MIN_N_EFF:
        return "GATHERING", f"effective n {n_eff:.1f} < {GATE_V2_MIN_N_EFF}"

    post = posterior_ic_above(ic_rank, int(round(n_eff)), threshold=0.08)
    if post is None:
        return "GATHERING", "posterior not computable"

    disagree = (ic_rank > 0) != (ic_pearson > 0)

    if post >= GATE_V2_READY_POSTERIOR:
        if disagree:
            # Sign-agreement guard: never authorise on a statistic its
            # counterpart contradicts. Documented, not silent.
            return "EXTEND", (f"posterior {post:.0%} clears the bar but rank and "
                              "Pearson IC disagree in sign — guard holds it back")
        return "READY", f"P(true rank IC >= 0.08) = {post:.0%}"

    if post <= GATE_V2_STOP_POSTERIOR:
        return "STOP", f"P(true rank IC >= 0.08) = {post:.0%}, at or below the floor"

    if ic_rank < GATE_V2_IC_FLOOR:
        return "STOP", f"rank IC {ic_rank:+.3f} below the {GATE_V2_IC_FLOOR} floor"

    if extensions_used >= GATE_V2_MAX_EXTENSIONS:
        return "STOP", (f"EXTEND allowance exhausted ({extensions_used} of "
                        f"{GATE_V2_MAX_EXTENSIONS}) and posterior {post:.0%} "
                        "never reached the bar")

    return "EXTEND", (f"posterior {post:.0%} between the bands; extension "
                      f"{extensions_used + 1} of {GATE_V2_MAX_EXTENSIONS}")


def _dual_ic(scores: np.ndarray, returns: np.ndarray):
    """(pearson, p_pearson, spearman, p_spearman).

    Both statistics, always, so no surface has to report one without the other:
    Pearson on floored/unbounded option returns is dominated by the few +100%
    take-profits, and the rank statistic is the one that survives that skew.
    Degenerate samples fall back to (0.0, 1.0, 0.0, 1.0), matching ``_ic``.
    """
    if len(scores) < 3 or scores.std() < 1e-8 or returns.std() < 1e-8:
        return 0.0, 1.0, 0.0, 1.0
    ic_p, p_p = (float(x) for x in pearsonr(scores, returns))
    ic_s, p_s = (float(x) for x in spearmanr(scores, returns))
    return ic_p, p_p, ic_s, p_s


# The credit family real money would trade first. Strings match the
# strategy_name values actually present in paper_trades.db.
SHORT_PREMIUM_STRATEGIES = ("Bull Put", "Bear Call", "Short Put")


def _load_short_premium_cohort(db_path: str, phase1_start: str,
                               max_capital_at_risk: Optional[float] = None) -> list:
    """(quality_score, pnl_usd, capital_at_risk) for closed short-premium trades.

    Same window and closed-only logic as the Long Call gate cohort, restricted
    to positions that fit the budget. Two deliberate differences:

    * ``paper_only`` is NOT filtered. The whole short-premium family is logged
      ``paper_only=1`` today (config ``auto_log.paper_only_strategies``), so the
      gate cohort's ``paper_only=0`` filter would leave this block permanently
      empty and it would report nothing about the strategy it exists to watch.
    * Return is measured in dollars over capital at risk, not ``pnl_pct``:
      percentage return on a short is meaningless without its denominator.

    Never raises — a DB without the v16 columns yields an empty cohort rather
    than taking the checkpoint down.
    """
    placeholders = ",".join("?" for _ in SHORT_PREMIUM_STRATEGIES)
    sql = (
        f"SELECT quality_score, pnl_usd, capital_at_risk FROM trades "
        f"WHERE strategy_name IN ({placeholders}) AND status='CLOSED' "
        "AND date >= ? "
        "AND quality_score IS NOT NULL AND pnl_usd IS NOT NULL "
        "AND capital_at_risk IS NOT NULL AND capital_at_risk > 0"
    )
    params: list = [*SHORT_PREMIUM_STRATEGIES, phase1_start]
    if max_capital_at_risk is not None:
        sql += " AND capital_at_risk <= ?"
        params.append(float(max_capital_at_risk))
    rows: list = []
    try:
        with sqlite3.connect(db_path) as conn:
            fetched = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    for q, pnl, car in fetched:
        try:
            car_f = float(car)
            if car_f <= 0:
                continue
            rows.append((float(q), float(pnl), car_f))
        except (TypeError, ValueError):
            continue
    return rows


def short_premium_report(db_path: str, phase1_start: str,
                         max_capital_at_risk: Optional[float] = None) -> dict:
    """Reporting-only read of the short-premium cohort. Never feeds the gate.

    Rank IC is measured against per-trade return-on-risk (pnl / capital at
    risk), and the headline return is capital-weighted (sum pnl / sum capital
    at risk) with the median per-trade figure beside it, because one large
    contract must not be able to carry the line on its own.
    """
    rows = _load_short_premium_cohort(db_path, phase1_start, max_capital_at_risk)
    out: dict = {
        "strategies": list(SHORT_PREMIUM_STRATEGIES),
        "max_capital_at_risk": max_capital_at_risk,
        "n": len(rows),
        "ic_pearson": None, "p_pearson": None,
        "ic_spearman": None, "p_spearman": None,
        "sum_pnl": 0.0, "sum_capital_at_risk": 0.0,
        "ror_sum": None, "ror_median": None,
    }
    if not rows:
        return out

    scores = np.array([r[0] for r in rows], dtype=float)
    pnl = np.array([r[1] for r in rows], dtype=float)
    car = np.array([r[2] for r in rows], dtype=float)
    ror = pnl / car

    out["sum_pnl"] = float(pnl.sum())
    out["sum_capital_at_risk"] = float(car.sum())
    if car.sum() > 0:
        out["ror_sum"] = float(pnl.sum() / car.sum())
    out["ror_median"] = float(np.median(ror))

    if len(rows) >= 3:
        ic_p, p_p, ic_s, p_s = _dual_ic(scores, ror)
        out.update(ic_pearson=ic_p, p_pearson=p_p,
                   ic_spearman=ic_s, p_spearman=p_s)
    return out


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

    Under gate v1 this was reporting only. Under v2 (docs/GATE_REDESIGN_SPEC.md,
    signed 2026-07-31) it IS the decision statistic, applied to the rank IC at
    effective n. Returns None when n < 4 or ic is not finite.
    """
    if n < 4 or ic is None or not math.isfinite(ic):
        return None
    z_obs = math.atanh(max(-0.999, min(0.999, float(ic))))
    z_thr = math.atanh(max(-0.999, min(0.999, float(threshold))))
    se = 1.0 / math.sqrt(n - 3)
    return float(1 - norm.cdf((z_thr - z_obs) / se))


def compute_checkpoint(db_path: str, phase1_start: str, today: Optional[str] = None,
                       max_capital_at_risk: Optional[float] = None,
                       gate_version: int = 2,
                       extensions_used: int = 0) -> dict:
    """Cohort IC and the gate decision.

    ``gate_version`` selects which rule governs. 2 is the signed redesign
    (docs/GATE_REDESIGN_SPEC.md, 2026-07-31): rank IC, posterior bands at
    effective n, bounded EXTEND. 1 is the original Pearson/p-value rule, kept
    so its verdict can still be computed and shown. BOTH are always returned
    under ``decision_v1`` / ``decision_v2`` regardless of which one governs —
    a decision rule that quietly vanishes cannot be audited.

    ``max_capital_at_risk`` adds a parallel read of the subset the account could
    actually have traded — roughly half the cohort ties up more than the whole
    budget. That subset, and the short-premium cohort block, remain reporting
    only: neither is read by the gate.
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    scores, returns, dates = _load_cohort(db_path, phase1_start)
    n = len(scores)
    weeks = _weeks_between(phase1_start, today)

    n_affordable: Optional[int] = None
    ic_affordable: Optional[float] = None
    p_affordable: Optional[float] = None
    ic_s_affordable: Optional[float] = None
    p_s_affordable: Optional[float] = None
    if max_capital_at_risk is not None:
        aff_scores, aff_returns, _aff_dates = _load_cohort(db_path, phase1_start, max_capital_at_risk)
        n_affordable = len(aff_scores)
        (ic_affordable, p_affordable,
         ic_s_affordable, p_s_affordable) = _dual_ic(aff_scores, aff_returns)

    short_premium = short_premium_report(db_path, phase1_start, max_capital_at_risk)

    ic_p, p_p, ic_s, p_s = _dual_ic(scores, returns)

    ci_lo, ci_hi = _bootstrap_ci(scores, returns)

    icc, de, n_eff = design_effect(returns, dates)

    decision_v1 = decide_v1(n, ic_p, p_p, weeks)
    decision_v2, reason_v2 = decide_v2(n_eff, ic_s, ic_p, extensions_used)

    # Which one governs. v2 is the signed redesign; v1 stays computed and
    # reported so the change is never invisible and can always be audited.
    decision = decision_v2 if gate_version >= 2 else decision_v1

    return {
        "today": today, "phase1_start": phase1_start, "weeks_elapsed": weeks,
        "n_trades": n, "ic_pearson": ic_p, "p_pearson": p_p,
        "ic_spearman": ic_s, "p_spearman": p_s, "ic_95_ci": [ci_lo, ci_hi],
        "decision": decision,
        "gate_version": gate_version,
        "decision_v1": decision_v1,
        "decision_v2": decision_v2,
        "decision_v2_reason": reason_v2,
        "icc": icc,
        "design_effect": de,
        "n_effective": n_eff,
        "extensions_used": extensions_used,
        # v2 reads RANK IC, so its posterior is the rank one. The Pearson
        # posterior stays below under its original key for continuity.
        "posterior_rank_ic_ge_008": posterior_ic_above(
            ic_s, int(round(n_eff)), threshold=0.08),
        "posterior_ic_ge_008": posterior_ic_above(ic_p, n, threshold=0.08),
        "max_capital_at_risk": max_capital_at_risk,
        "n_affordable": n_affordable,
        "ic_pearson_affordable": ic_affordable,
        "p_pearson_affordable": p_affordable,
        "ic_spearman_affordable": ic_s_affordable,
        "p_spearman_affordable": p_s_affordable,
        "short_premium": short_premium,
    }


def _format_markdown(r: dict) -> str:
    return "\n".join([
        f"# Phase 1 Checkpoint — {r['today']}", "",
        f"- Phase 1 start: {r['phase1_start']}",
        f"- Weeks elapsed: {r['weeks_elapsed']}",
        f"- Cohort size (Long Call, paper_only=0, post-start, closed): **{r['n_trades']}**", "",
        "## Forward-cohort IC",
        f"- Pearson IC (gate statistic): **{r['ic_pearson']:+.3f}**  (p={r['p_pearson']:.3f})",
        f"- Spearman rank IC: **{r['ic_spearman']:+.3f}**  (p={r['p_spearman']:.3f})",
        _sign_disagreement_line(r.get("ic_pearson"), r.get("ic_spearman")),
        f"- 95% bootstrap CI (Pearson): [{r['ic_95_ci'][0]:+.3f}, {r['ic_95_ci'][1]:+.3f}]",
        _posterior_line(r.get("posterior_ic_ge_008")),
        *_affordable_lines(r),
        "",
        f"## Gate decision: **{r['decision']}**", "",
        _decision_explainer(r["decision"]), "",
        *_gate_version_lines(r),
        *_short_premium_lines(r.get("short_premium")),
    ]) + "\n"


def _gate_version_lines(r: dict) -> list:
    """Which rule decided, why, and what the other one said.

    Both verdicts appear on every run. When they agree the line is short and
    reassuring; when they disagree that disagreement is the single most
    important thing on the page, and it must not be discoverable only by
    reading the source.
    """
    v1, v2 = r.get("decision_v1"), r.get("decision_v2")
    if v1 is None or v2 is None:
        return []
    active = int(r.get("gate_version", 2))
    n_eff = r.get("n_effective")
    lines = ["### Rule versions", ""]
    lines.append(f"- Active rule: **v{active}** "
                 f"({'rank IC, posterior bands, effective n' if active >= 2 else 'Pearson IC, p-value'})")
    lines.append(f"- v2 (signed 2026-07-31) says **{v2}** — {r.get('decision_v2_reason', '')}")
    lines.append(f"- v1 (superseded) says **{v1}**")
    if n_eff is not None:
        lines.append(
            f"- Effective n: **{n_eff:.1f}** of {r['n_trades']} nominal "
            f"(ICC {r.get('icc', 0.0):.3f}, design effect {r.get('design_effect', 1.0):.2f}) "
            "— same-day entries share that day's move, so they are not "
            "independent observations")
    post = r.get("posterior_rank_ic_ge_008")
    if post is not None:
        lines.append(f"- P(true rank IC >= 0.08) = **{post:.0%}** "
                     f"(READY at >= {GATE_V2_READY_POSTERIOR:.0%}, "
                     f"STOP at <= {GATE_V2_STOP_POSTERIOR:.0%})")
    if v1 != v2:
        lines.append("")
        lines.append(f"> **The two rules disagree** (v1 {v1}, v2 {v2}). v{active} "
                     "governs; the other is shown so the choice stays auditable.")
    lines.append("")
    return lines


def _sign_disagreement_line(ic_p: Optional[float], ic_s: Optional[float]) -> str:
    """Flag the case the two statistics tell opposite stories.

    Not a rule — the gate still reads Pearson. It is here so the disagreement
    cannot be read past without noticing it (docs/GATE_REDESIGN_SPEC.md).

    A cohort too small or too degenerate to correlate reports both statistics
    as 0.000; that is the absence of a measurement, not agreement, and must not
    be printed as if the two had been computed and matched."""
    if ic_p is None or ic_s is None:
        return "- Statistic agreement: n/a"
    if abs(ic_p) <= 1e-9 and abs(ic_s) <= 1e-9:
        return "- Statistic agreement: n/a (cohort too small or too degenerate to correlate)"
    if (ic_p > 0) != (ic_s > 0) and abs(ic_p) > 1e-9 and abs(ic_s) > 1e-9:
        return ("- **Statistics disagree in sign** — the rank IC does not confirm "
                "the Pearson reading. Treat the gate statistic with suspicion.")
    return "- Statistics agree in sign."


def _short_premium_lines(sp: Optional[dict]) -> list:
    """The credit family's cohort, reported every week and gating nothing.

    Real money would trade short premium first, but the gate cohort is Long
    Call. Until a short-premium gate is signed off (docs/GATE_REDESIGN_SPEC.md
    §2.4) this block exists so the family is at least measured.
    """
    if not sp:
        return []
    fam = " + ".join(sp.get("strategies") or [])
    cap = sp.get("max_capital_at_risk")
    scope = (f"capital at risk <= ${cap:,.0f}" if cap else "no capital cap applied")
    lines = [
        "",
        "## Short-premium cohort — REPORTING ONLY, not a gate",
        "",
        f"{fam}, closed, entered on/after the Phase 1 start ({scope}). "
        "Includes paper_only rows: the whole family is paper-only today.",
        f"- Cohort size: **{sp['n']}**",
    ]
    if sp["n"] == 0:
        lines.append("- No closed short-premium trades with a recorded capital at risk yet.")
    else:
        if sp.get("ic_spearman") is None:
            lines.append("- Rank IC (quality_score vs return-on-risk): n/a (n < 3)")
        else:
            lines.append(
                f"- Rank IC (quality_score vs return-on-risk): **{sp['ic_spearman']:+.3f}**  "
                f"(p={sp['p_spearman']:.3f})")
            lines.append(
                f"- Pearson IC (same pair): {sp['ic_pearson']:+.3f}  "
                f"(p={sp['p_pearson']:.3f})")
        if sp.get("ror_sum") is not None:
            lines.append(
                f"- Return on risk (sum P&L / sum capital at risk): **{sp['ror_sum']:+.1%}**  "
                f"(${sp['sum_pnl']:,.0f} / ${sp['sum_capital_at_risk']:,.0f})")
        if sp.get("ror_median") is not None:
            lines.append(f"- Median per-trade return on risk: {sp['ror_median']:+.1%}")
    lines.append("- **REPORTING ONLY — not a gate.** The decision above reads the "
                 "Long Call cohort and nothing here can move it.")
    return lines


def _affordable_lines(r: dict) -> list:
    """The same IC over trades that fit the budget — reporting only.

    Half the cohort has historically tied up more than the whole account, so
    the nominal IC describes positions that could not have been opened. Both
    numbers are shown; the gate still reads the nominal one.
    """
    n_aff = r.get("n_affordable")
    if n_aff is None:
        return []
    cap = r.get("max_capital_at_risk") or 0
    ic = r.get("ic_pearson_affordable")
    p = r.get("p_pearson_affordable")
    ic_s = r.get("ic_spearman_affordable")
    p_s = r.get("p_spearman_affordable")
    return [
        "",
        f"### Affordable subset (capital at risk <= ${cap:,.0f})",
        f"- Cohort size: **{n_aff} of {r['n_trades']}**",
        f"- Pearson IC: **{ic:+.3f}**  (p={p:.3f})" if ic is not None else
        "- Pearson IC: n/a",
        f"- Spearman rank IC: **{ic_s:+.3f}**  (p={p_s:.3f})" if ic_s is not None else
        "- Spearman rank IC: n/a",
        "- Reporting only — the gate decision above reads the nominal cohort.",
    ]


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


HISTORY_HEADER = "date\tweeks\tn\tic\tp\tdecision\tspearman\tp_spearman\n"


def _ensure_history_header(hist_path: Path) -> None:
    """Header names the rank-IC columns; historical rows are left exactly as
    they were written.

    The file therefore holds rows of two widths — 6-field rows from before the
    rank IC was recorded, 8-field rows after — and every reader must tolerate
    both (``src/evidence.py`` does). Nothing is ever rewritten but the header
    line, because the history is a record, not a derived artifact. That rewrite
    goes through a temp file and ``os.replace`` — the history cannot be
    regenerated from anything, so it must never be left truncated by an
    interrupted write.
    """
    if not hist_path.exists():
        _atomic_write(hist_path, HISTORY_HEADER)
        return
    text = hist_path.read_text()
    if not text.strip():
        _atomic_write(hist_path, HISTORY_HEADER)
        return
    lines = text.splitlines(keepends=True)
    if lines[0].startswith("date\t") and "spearman" not in lines[0]:
        lines[0] = HISTORY_HEADER
        text = "".join(lines)
        _atomic_write(hist_path, text)
    if not text.endswith("\n"):
        # A truncated last row would otherwise be fused with the new one.
        with hist_path.open("a") as f:
            f.write("\n")


def _atomic_write(path: Path, text: str) -> None:
    """Replace ``path`` in one step: write a sibling temp file, fsync, rename.

    A same-directory rename is atomic, so a reader or a crash sees either the
    old file or the new one, never a half-truncated one.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _last_history_row(hist_path: Path) -> Optional[list]:
    """Fields of the last data row, or None if there is no data row yet.

    Fields are returned positionally and of whatever width the row happens to
    be: the history holds both 6-field legacy rows and 8-field current ones, so
    callers index (date=0, n=2) rather than zipping against a header.
    """
    if not hist_path.exists():
        return None
    try:
        text = hist_path.read_text()
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        if not line.strip() or line.startswith("date\t"):
            continue
        return line.split("\t")
    return None


def _history_append_is_redundant(hist_path: Path, today: str, n_trades: int) -> bool:
    """True when the last row already records this day at this cohort size.

    The checkpoint is re-run whenever startup maintenance decides it is due, and
    it used to append unconditionally — so a day with two launches got two rows,
    and anything reading the history as a time series (the evidence banner, any
    n-over-time read) saw a step that never happened.

    Same day at a *different* n is a real new observation — the cohort grew
    between runs — and still appends. Only the exact repeat is dropped.
    """
    fields = _last_history_row(hist_path)
    if not fields or len(fields) < 3:
        return False
    if fields[0].strip() != str(today):
        return False
    try:
        return int(float(fields[2])) == int(n_trades)
    except (TypeError, ValueError):
        return False


def write_checkpoint(result: dict, output_dir: str = "reports") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"checkpoint_{result['today']}.md"
    md_path.write_text(_format_markdown(result))

    hist_path = out / "checkpoint_history.tsv"
    _ensure_history_header(hist_path)
    appended = not _history_append_is_redundant(hist_path, result["today"], result["n_trades"])
    if appended:
        with hist_path.open("a") as f:
            f.write(f"{result['today']}\t{result['weeks_elapsed']}\t{result['n_trades']}\t"
                    f"{result['ic_pearson']:.4f}\t{result['p_pearson']:.4f}\t{result['decision']}\t"
                    f"{result['ic_spearman']:.4f}\t{result['p_spearman']:.4f}\n")

    if result["decision"] in ("READY", "STOP"):
        _v = int(result.get("gate_version", 2))
        # Which statistic decided has to be stated, not inferred. Under v2 the
        # rank IC is the gate statistic; under v1 it was reporting only.
        _rank_label = ("Rank IC (THE gate statistic under v2)" if _v >= 2
                       else "Rank IC (reporting only, not the gate statistic)")
        _neff = result.get("n_effective")
        _neff_txt = f", n_eff={_neff:.1f}" if _neff is not None else ""
        (out / "GATE_STATUS.md").write_text(
            f"GATE: **{result['decision']}** as of {result['today']}  "
            f"(rule v{_v}, n={result['n_trades']}{_neff_txt}, "
            f"IC={result['ic_pearson']:+.3f}, "
            f"p={result['p_pearson']:.3f}, weeks={result['weeks_elapsed']})\n"
            f"{_rank_label}: "
            f"{result['ic_spearman']:+.3f} (p={result['p_spearman']:.3f})\n"
            f"v1 says {result.get('decision_v1')}, v2 says {result.get('decision_v2')}"
            f" — {result.get('decision_v2_reason', '')}\n"
            f"See `{md_path.name}` for details.\n"
        )
    return {"md": str(md_path), "history": str(hist_path), "history_appended": appended}


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

    cap = (cfg.get("auto_log") or {}).get("max_capital_at_risk")
    gate_cfg = cfg.get("gate") or {}
    result = compute_checkpoint(
        db_path=args.db, phase1_start=phase1_start,
        max_capital_at_risk=float(cap) if cap else None,
        gate_version=int(gate_cfg.get("version", 2)),
        extensions_used=int(gate_cfg.get("extensions_used", 0)),
    )
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        paths = write_checkpoint(result, output_dir=args.output)
        print(f"\nWrote: {paths}")


if __name__ == "__main__":
    main()
