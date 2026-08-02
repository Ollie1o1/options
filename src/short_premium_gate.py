"""The short-premium gate — §2.4 of docs/GATE_REDESIGN_SPEC.md (signed 2026-07-31).

The Phase-1 gate validated Long Calls and resolved STOP. The family the book
actually earns on is short premium, and it had never been validated at all:
validating one strategy to authorise going live, then trading a different one,
is a mismatch between the experiment and the decision it licenses.

This is NOT the Long Call gate pointed at different rows. Three things about
short-premium returns force a different design:

**The tail is on the other side.** Long-call returns are floored at -100% and
dragged up by a few take-profits. Short premium is the mirror: many small wins
and occasional large losses (measured on the live cohort: 66% win rate, skew
-0.25, a left tail reaching -100%). A mean is the wrong summary of that, so
Arm A asks about the MEDIAN — one blown-up condor must not be able to sink a
line that is otherwise working, and one lucky one must not carry it.

**"Is the edge positive" is not a correlation question.** The Long Call gate
asks whether `quality_score` RANKS trades, which is an IC and has a tidy
Fisher-z posterior. The first question about short premium is simpler and more
fundamental: does the family make money at all, net of what it costs to trade?
That is a question about a location parameter, and it gets a bootstrap, not a
correlation posterior.

**Entries cluster hard.** 188 trades on 27 entry days, up to 33 on a single
day. Same-day trades share that day's market move, so resampling individual
trades would treat 188 correlated observations as 188 independent ones and
report far more certainty than the data supports. The bootstrap therefore
resamples ENTRY DAYS, not trades.

Two arms, because "does this family make money" and "does the scorer pick which
ones" are separate questions with separate consequences:

* **Arm A — family viability.** P(true median net return-on-risk > 0), by
  clustered bootstrap. This is the arm that would authorise capital.
* **Arm B — selection skill.** Rank IC of `quality_score` against return-on-risk,
  through the same posterior bands the Long Call gate uses. Arm B failing does
  NOT veto Arm A: it decides whether live trading would use the scorer to pick
  contracts, or trade the family on structural rules and ignore the ranking.

Returns are net of MEASURED per-structure costs, not the flat constant the
ledger charged — that constant is undercharged 3.2x on Bull Put and
overcharged 2x on Bear Call, which is exactly the asymmetry that would decide
this question wrongly.

Read-only. Never writes to the ledger.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.phase1_checkpoint import (GATE_V2_MAX_EXTENSIONS,
                                   GATE_V2_MIN_N_EFF,
                                   GATE_V2_READY_POSTERIOR,
                                   GATE_V2_STOP_POSTERIOR,
                                   decide_v2, design_effect)

SHORT_PREMIUM_STRATEGIES: Tuple[str, ...] = ("Bull Put", "Bear Call", "Short Put")

# Legs per structure, for costing the round trip. A cash-secured put is one
# leg; the verticals are two.
LEG_COUNTS: Dict[str, int] = {"Bull Put": 2, "Bear Call": 2, "Short Put": 1}

# Arm A asks whether the median is above zero — "does this family make money",
# not "does it beat some invented hurdle". A non-zero threshold here would be a
# second decision smuggled in beside the first.
ARM_A_THRESHOLD = 0.0

BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 12345

# A credit trade is only tradeable if the market you must cross is small next
# to the credit you collect. Measured on the live cohort, 31 of 188 logged
# trades carry round-trip friction EXCEEDING the entire credit received — a
# $26.50-max-loss Bull Put against $64.80 of spread. Those cannot be profitable
# at any win rate, and including them measures a population no rational trader
# would open. This is the mirror of the affordability filter: that one excluded
# positions too large for the account, this one excludes positions too small to
# survive their own spread.
#
# 0.50 is deliberately generous — at half the credit gone to friction you still
# need to be right roughly 2:1 to break even. Tightening it strengthens the
# result rather than weakening it (at 0.33 the median rises to +30.5%), so the
# loose setting is the conservative choice for a gate.
TRADEABILITY_MAX_FRICTION_RATIO = 0.50


def load_cohort(db_path: str, phase1_start: str,
                max_capital_at_risk: Optional[float] = None) -> List[Dict[str, Any]]:
    """Closed short-premium trades in the window, with what costing needs.

    ``paper_only`` is deliberately NOT filtered — see `cohort_caveats`, which
    reports what that means rather than hiding it.
    """
    placeholders = ",".join("?" for _ in SHORT_PREMIUM_STRATEGIES)
    sql = (
        f"SELECT strategy_name, quality_score, pnl_usd, capital_at_risk, date, "
        f"       net_credit, entry_price, paper_only "
        f"FROM trades WHERE strategy_name IN ({placeholders}) AND status='CLOSED' "
        "AND date >= ? AND pnl_usd IS NOT NULL "
        "AND capital_at_risk IS NOT NULL AND capital_at_risk > 0"
    )
    params: list = [*SHORT_PREMIUM_STRATEGIES, phase1_start]
    if max_capital_at_risk is not None:
        sql += " AND capital_at_risk <= ?"
        params.append(float(max_capital_at_risk))
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params)]
    except sqlite3.Error:
        return []


def _entry_credit(row: Dict[str, Any]) -> float:
    """Credit taken at entry, per share."""
    for key in ("net_credit", "entry_price"):
        try:
            v = float(row.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return 0.0


def net_return_on_risk(row: Dict[str, Any], old_model=None, new_model=None) -> Optional[float]:
    """Per-trade return on capital at risk, net of MEASURED costs.

    ``pnl_usd`` is already net of the flat friction the ledger charged at close.
    That constant is wrong per structure and in opposite directions, so the old
    friction is added back and the measured one taken off. With no cost models
    supplied the stored figure is used unchanged.
    """
    try:
        pnl = float(row["pnl_usd"])
        car = float(row["capital_at_risk"])
    except (TypeError, ValueError, KeyError):
        return None
    if car <= 0:
        return None

    if old_model is not None and new_model is not None:
        strategy = row.get("strategy_name") or ""
        n_legs = LEG_COUNTS.get(strategy, 2)
        credit = _entry_credit(row)
        try:
            old_f = old_model.friction(strategy, n_legs, credit)
            new_f = new_model.friction(strategy, n_legs, credit)
            # friction is per share; a contract is 100 shares.
            pnl -= (new_f - old_f) * 100.0
        except Exception:
            pass
    return pnl / car


def friction_to_credit(row: Dict[str, Any], model) -> Optional[float]:
    """Round-trip friction as a fraction of the credit received.

    Above 1.0 the trade cannot profit at any win rate: crossing the market
    costs more than the whole credit. None when there is no credit recorded —
    a missing credit is not a free trade, it is a row to skip.
    """
    if model is None:
        return None
    credit = _entry_credit(row)
    if credit <= 0:
        return None
    strategy = row.get("strategy_name") or ""
    n_legs = LEG_COUNTS.get(strategy, 2)
    try:
        return float(model.friction(strategy, n_legs, credit)) / credit
    except Exception:
        return None


def cluster_bootstrap_medians(values: Sequence[float], clusters: Sequence[Any],
                              n_boot: int = BOOTSTRAP_DRAWS,
                              seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Bootstrap distribution of the median, resampling CLUSTERS not trades.

    Entry days are the clusters. Drawing whole days preserves the within-day
    correlation that makes 188 trades worth far fewer independent observations;
    drawing individual trades would destroy it and manufacture confidence.
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return np.array([])

    groups: Dict[Any, List[float]] = {}
    for v, c in zip(vals, clusters):
        groups.setdefault(c, []).append(float(v))
    keys = list(groups.keys())
    if len(keys) < 2:
        return np.array([])

    rng = np.random.default_rng(seed)
    k = len(keys)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, k, size=k)
        sample: List[float] = []
        for idx in picked:
            sample.extend(groups[keys[idx]])
        out[i] = np.median(sample) if sample else np.nan
    return out[~np.isnan(out)]


def posterior_median_above(values: Sequence[float], clusters: Sequence[Any],
                           threshold: float = ARM_A_THRESHOLD,
                           n_boot: int = BOOTSTRAP_DRAWS,
                           seed: int = BOOTSTRAP_SEED) -> Optional[float]:
    """P(true median > threshold), read off the clustered bootstrap.

    The bootstrap distribution of a statistic is used directly as the posterior
    under a flat prior — the same interpretation the Long Call gate's Fisher-z
    posterior carries, arrived at without assuming a distribution this data
    does not have.
    """
    boots = cluster_bootstrap_medians(values, clusters, n_boot, seed)
    if boots.size == 0:
        return None
    return float((boots > threshold).mean())


def decide_arm_a(n_eff: float, posterior: Optional[float],
                 extensions_used: int = 0,
                 median_ror: Optional[float] = None,
                 capital_weighted_ror: Optional[float] = None) -> Tuple[str, str]:
    """Family viability. Same bands and bounded EXTEND as the Long Call gate,
    plus a coherence guard.

    The guard exists because the median alone can say yes while the money says
    no. Short premium is the classic shape for it: many small wins and a few
    large losses, so a positive median is entirely compatible with losing money
    overall. The median answers "does a typical trade win"; the capital-weighted
    return answers "did the book make money". Authorising capital needs both,
    for the same reason the Long Call gate refuses to fire when its two
    statistics disagree in sign.
    """
    if n_eff < GATE_V2_MIN_N_EFF:
        return "GATHERING", f"effective n {n_eff:.1f} < {GATE_V2_MIN_N_EFF}"
    if posterior is None:
        return "GATHERING", "posterior not computable (needs >= 2 entry days)"
    if posterior >= GATE_V2_READY_POSTERIOR:
        if (median_ror is not None and capital_weighted_ror is not None
                and (median_ror > 0) != (capital_weighted_ror > 0)):
            return "EXTEND", (
                f"posterior {posterior:.0%} clears the bar, but the median "
                f"({median_ror:+.1%}) and the capital-weighted return "
                f"({capital_weighted_ror:+.1%}) disagree in sign — the typical "
                "trade and the book are telling different stories, so the guard "
                "holds it back")
        return "READY", f"P(true median net RoR > 0) = {posterior:.0%}"
    if posterior <= GATE_V2_STOP_POSTERIOR:
        return "STOP", (f"P(true median net RoR > 0) = {posterior:.0%}, "
                        "at or below the floor")
    if extensions_used >= GATE_V2_MAX_EXTENSIONS:
        return "STOP", (f"EXTEND allowance exhausted ({extensions_used} of "
                        f"{GATE_V2_MAX_EXTENSIONS}); posterior {posterior:.0%} "
                        "never reached the bar")
    return "EXTEND", (f"posterior {posterior:.0%} between the bands; extension "
                      f"{extensions_used + 1} of {GATE_V2_MAX_EXTENSIONS}")


def cohort_caveats(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """What this cohort is NOT, stated plainly beside whatever it says.

    A gate that would authorise real money has to declare the ways its evidence
    falls short, or the number it prints is worth less than it looks.
    """
    out: List[str] = []
    if not rows:
        return ["Cohort is empty."]

    n_paper = sum(1 for r in rows if int(r.get("paper_only") or 0) == 1)
    if n_paper == len(rows):
        out.append(
            "**Every row is `paper_only=1`.** The family is quarantined by "
            "`auto_log.paper_only_strategies` — a strategy-selection decision, "
            "not a judgement about data quality. These rows were not intended "
            "as a validation cohort, and promoting them to one is an operator "
            "decision that this gate does not make for you.")
    elif n_paper:
        out.append(
            f"{n_paper} of {len(rows)} rows are still `paper_only=1` — a mixed "
            "cohort, which is worth resolving one way or the other.")
    else:
        out.append(
            "Cohort promoted out of `paper_only` on 2026-08-01 by operator "
            "decision, so these rows now count as validation evidence. That "
            "removed a CLASSIFICATION, not a limitation: the caveats below are "
            "unchanged by it, and the tail one still binds.")

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.get("strategy_name") or "?"] = counts.get(r.get("strategy_name") or "?", 0) + 1
    thin = [f"{k} ({v})" for k, v in sorted(counts.items()) if v < 30]
    if thin:
        out.append(
            "Thin by structure: " + ", ".join(thin) + ". The family verdict is "
            "carried by the better-represented lines; it is not evidence about "
            "these individually.")

    days = {str(r.get("date"))[:10] for r in rows}
    if days and len(rows) / max(1, len(days)) >= 3:
        out.append(
            f"{len(rows)} trades on {len(days)} entry days "
            f"({len(rows) / len(days):.1f} per day). The bootstrap resamples "
            "days for this reason; nominal n materially overstates the evidence.")

    out.append(
        "Exit fidelity: stops in this window were checked at manual cadence "
        "(the scheduler has been dead since 2026-06-15), and 94% of stopped "
        "trades ran past their stop. Losses here are overstated relative to the "
        "rules that were meant to govern them — see the stop-overshoot note in "
        "reports/TRACK_RECORD.md. This biases the result DOWN, so it is the one "
        "caveat that does not threaten a positive verdict.")

    days_sorted = sorted(d for d in days if d)
    if days_sorted:
        span = f"{days_sorted[0]} to {days_sorted[-1]}"
        out.append(
            f"**The tail is unobserved.** The window is {span} — under two "
            "months, containing no volatility shock. Short premium's "
            "characteristic failure is not a slow bleed but a single event that "
            "returns months of credit at once, and a sample this short cannot "
            "price that risk. Every large loss in the wider ledger is a "
            "cash-secured put at $32k-$83k of collateral, all of which the "
            "affordability cap excludes, so the cohort measured here has never "
            "actually taken a big loss. A high posterior on this data means "
            "'no evidence against', not 'the tail has been survived'.")
    return out


def evaluate(db_path: str, phase1_start: str,
             max_capital_at_risk: Optional[float] = None,
             extensions_used: int = 0,
             use_measured_costs: bool = True) -> Dict[str, Any]:
    """Both arms over the short-premium cohort. Pure read."""
    rows = load_cohort(db_path, phase1_start, max_capital_at_risk)
    result: Dict[str, Any] = {
        "strategies": list(SHORT_PREMIUM_STRATEGIES),
        "max_capital_at_risk": max_capital_at_risk,
        "n": len(rows),
        "n_effective": 0.0, "icc": 0.0, "design_effect": 1.0,
        "median_net_ror": None, "mean_net_ror": None, "ror_sum": None,
        "capital_weighted_net_ror": None,
        "n_tradeable": 0, "n_untradeable": 0, "n_friction_over_credit": 0,
        "posterior_median_gt_0": None,
        "arm_a": "GATHERING", "arm_a_reason": "empty cohort",
        "arm_b": "GATHERING", "arm_b_reason": "empty cohort",
        "ic_spearman": None, "ic_pearson": None,
        "caveats": cohort_caveats(rows),
        "costs_measured": False,
    }
    if not rows:
        return result

    old_model = new_model = None
    if use_measured_costs:
        try:
            from src.execution_costs import CostModel, load_measured_model
            new_model = load_measured_model(commission_per_contract=0.0, fx_rate=0.0)
            old_model = CostModel(table={}, default_half_spread=0.05,
                                  commission_per_contract=0.0, fx_rate=0.0)
            result["costs_measured"] = bool(getattr(new_model, "table", None))
        except Exception:
            old_model = new_model = None

    rors, clusters, scores, risks = [], [], [], []
    n_untradeable = 0
    n_impossible = 0
    for r in rows:
        v = net_return_on_risk(r, old_model, new_model)
        if v is None:
            continue
        ratio = friction_to_credit(r, new_model)
        if ratio is not None:
            if ratio > 1.0:
                n_impossible += 1
            if ratio > TRADEABILITY_MAX_FRICTION_RATIO:
                n_untradeable += 1
                continue  # a trade nobody would open is not evidence
        rors.append(v)
        clusters.append(str(r.get("date"))[:10])
        scores.append(r.get("quality_score"))
        risks.append(float(r["capital_at_risk"]))

    result["n_untradeable"] = n_untradeable
    result["n_friction_over_credit"] = n_impossible
    result["n_tradeable"] = len(rors)
    if not rors:
        result["arm_a_reason"] = "no tradeable trades in the cohort"
        return result

    arr = np.asarray(rors, dtype=float)
    risk_arr = np.asarray(risks, dtype=float)
    icc, de, n_eff = design_effect(arr, clusters)
    cap_wtd = float((arr * risk_arr).sum() / risk_arr.sum()) if risk_arr.sum() > 0 else None
    result.update(n_effective=n_eff, icc=icc, design_effect=de,
                  median_net_ror=float(np.median(arr)),
                  mean_net_ror=float(arr.mean()),
                  capital_weighted_net_ror=cap_wtd)

    total_risk = sum(float(r["capital_at_risk"]) for r in rows)
    if total_risk > 0:
        result["ror_sum"] = float(sum(float(r["pnl_usd"]) for r in rows) / total_risk)

    post = posterior_median_above(arr, clusters)
    result["posterior_median_gt_0"] = post
    result["arm_a"], result["arm_a_reason"] = decide_arm_a(
        n_eff, post, extensions_used,
        median_ror=result["median_net_ror"], capital_weighted_ror=cap_wtd)

    # Arm B: does the scorer rank within the family? Same machinery as the
    # Long Call gate, deliberately — this half IS a correlation question.
    scored = [(s, v) for s, v in zip(scores, arr) if s is not None]
    if len(scored) >= 3:
        from src.phase1_checkpoint import _dual_ic
        s_arr = np.array([float(s) for s, _ in scored])
        v_arr = np.array([float(v) for _, v in scored])
        ic_p, _pp, ic_s, _ps = _dual_ic(s_arr, v_arr)
        result.update(ic_pearson=ic_p, ic_spearman=ic_s)
        result["arm_b"], result["arm_b_reason"] = decide_v2(
            n_eff, ic_rank=ic_s, ic_pearson=ic_p, extensions_used=extensions_used)
    else:
        result["arm_b_reason"] = "fewer than 3 scored trades"
    return result


def format_report(r: Dict[str, Any]) -> str:
    """Markdown for the checkpoint. Both arms, their disagreement, the caveats."""
    L: List[str] = []
    L.append("## Short-premium gate")
    L.append("")
    L.append(f"Cohort: {' + '.join(r['strategies'])}, closed, post-window"
             + (f", capital at risk <= ${r['max_capital_at_risk']:,.0f}"
                if r.get("max_capital_at_risk") else "") + ".")
    L.append("")
    L.append(f"- Trades: **{r['n']}** "
             f"(effective **{r['n_effective']:.1f}**, ICC {r['icc']:.3f}, "
             f"design effect {r['design_effect']:.2f})")
    if r.get("median_net_ror") is not None:
        L.append(f"- Median net return on risk: **{r['median_net_ror']:+.1%}** "
                 f"per trade (mean {r['mean_net_ror']:+.1%})")
    if r.get("capital_weighted_net_ror") is not None:
        L.append(f"- Capital-weighted net return on risk: **{r['capital_weighted_net_ror']:+.1%}**")
    if r.get("ror_sum") is not None:
        L.append(f"- Same figure before re-costing: {r['ror_sum']:+.1%} "
                 "(what the ledger's flat $0.05 friction implied)")
    if r.get("n_untradeable"):
        L.append(f"- Excluded as untradeable: **{r['n_untradeable']} of {r['n']}** "
                 f"— round-trip friction above {TRADEABILITY_MAX_FRICTION_RATIO:.0%} "
                 f"of the credit received, of which **{r['n_friction_over_credit']}** "
                 "cost MORE than the entire credit and could not profit at any win rate")
    L.append(f"- Costs: {'measured per structure' if r.get('costs_measured') else 'flat fallback (archive unavailable)'}")
    L.append("")
    post = r.get("posterior_median_gt_0")
    L.append("### Arm A — does the family make money (authorises capital)")
    L.append("")
    if post is not None:
        L.append(f"- P(true median net RoR > 0) = **{post:.0%}** "
                 f"(clustered bootstrap over entry days; READY >= "
                 f"{GATE_V2_READY_POSTERIOR:.0%}, STOP <= {GATE_V2_STOP_POSTERIOR:.0%})")
    L.append(f"- **{r['arm_a']}** — {r['arm_a_reason']}")
    L.append("")
    L.append("### Arm B — does the scorer rank within the family")
    L.append("")
    if r.get("ic_spearman") is not None:
        L.append(f"- Rank IC: **{r['ic_spearman']:+.3f}** "
                 f"(Pearson {r['ic_pearson']:+.3f})")
    L.append(f"- **{r['arm_b']}** — {r['arm_b_reason']}")
    L.append("- Arm B does not veto Arm A. It decides whether live trading "
             "would use the scorer to pick contracts, or trade the family on "
             "structural rules and ignore the ranking.")
    L.append("")
    if r.get("caveats"):
        L.append("### What this cohort is not")
        L.append("")
        for c in r["caveats"]:
            L.append(f"- {c}")
        L.append("")
    return "\n".join(L)


def main() -> None:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description="Short-premium gate")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = _json.load(f)
    start = (cfg.get("auto_log") or {}).get("phase1_start_date")
    cap = (cfg.get("auto_log") or {}).get("max_capital_at_risk")
    gate_cfg = cfg.get("gate") or {}
    r = evaluate(args.db, start, float(cap) if cap else None,
                 extensions_used=int(gate_cfg.get("short_premium_extensions_used", 0)))
    print(format_report(r))


if __name__ == "__main__":
    main()
