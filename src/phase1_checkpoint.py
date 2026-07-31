"""Phase 1 weekly checkpoint: compute forward-cohort IC and emit a gate decision.

Cohort = trades where strategy_name='Long Call', status='CLOSED', paper_only=0,
and date >= phase1_start_date.

Gate rules:
  n < 50                              -> GATHERING
  n >= 50, IC >= 0.08, p < 0.05       -> READY
  n >= 50, 0.03 <= IC < 0.08          -> EXTEND
  n >= 50, IC < 0.03, weeks >= 6      -> STOP
  otherwise                           -> GATHERING

IC in those rules means the Pearson IC and nothing else. Everything else the
checkpoint reports — the Spearman rank IC, the affordable subset, the
short-premium cohort — sits beside the decision and is never read by it
(DECISIONS.md 2026-06-07 forbids silent gate changes; a redesign is specified
in docs/GATE_REDESIGN_SPEC.md and is not active until signed off).

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
        "SELECT quality_score, pnl_pct FROM trades "
        "WHERE strategy_name='Long Call' AND status='CLOSED' "
        "AND COALESCE(paper_only, 0) = 0 "
        "AND date >= ? "
        "AND quality_score IS NOT NULL AND pnl_pct IS NOT NULL"
    )
    params: tuple = (phase1_start,)
    if max_capital_at_risk is not None:
        sql += " AND capital_at_risk IS NOT NULL AND capital_at_risk <= ?"
        params = (phase1_start, float(max_capital_at_risk))
    scores, returns = [], []
    with sqlite3.connect(db_path) as conn:
        for q, p in conn.execute(sql, params).fetchall():
            try:
                scores.append(float(q)); returns.append(float(p))
            except (TypeError, ValueError):
                continue
    return np.array(scores), np.array(returns)


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


def compute_checkpoint(db_path: str, phase1_start: str, today: Optional[str] = None,
                       max_capital_at_risk: Optional[float] = None) -> dict:
    """Cohort IC and the gate decision.

    ``max_capital_at_risk`` adds a parallel read of the subset the account could
    actually have traded — roughly half the cohort ties up more than the whole
    budget. It is reporting only: the decision below still reads the nominal
    cohort, because changing the gate is a decision to be made deliberately
    (DECISIONS.md 2026-06-07), not a side effect of adding a column.

    The same applies to the two other numbers this returns: the Spearman rank
    IC and the short-premium cohort block. Both are reported beside the gate
    and neither is read by it.
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    scores, returns = _load_cohort(db_path, phase1_start)
    n = len(scores)
    weeks = _weeks_between(phase1_start, today)

    n_affordable: Optional[int] = None
    ic_affordable: Optional[float] = None
    p_affordable: Optional[float] = None
    ic_s_affordable: Optional[float] = None
    p_s_affordable: Optional[float] = None
    if max_capital_at_risk is not None:
        aff_scores, aff_returns = _load_cohort(db_path, phase1_start, max_capital_at_risk)
        n_affordable = len(aff_scores)
        (ic_affordable, p_affordable,
         ic_s_affordable, p_s_affordable) = _dual_ic(aff_scores, aff_returns)

    short_premium = short_premium_report(db_path, phase1_start, max_capital_at_risk)

    ic_p, p_p, ic_s, p_s = _dual_ic(scores, returns)

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
        *_short_premium_lines(r.get("short_premium")),
    ]) + "\n"


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
        (out / "GATE_STATUS.md").write_text(
            f"GATE: **{result['decision']}** as of {result['today']}  "
            f"(n={result['n_trades']}, IC={result['ic_pearson']:+.3f}, "
            f"p={result['p_pearson']:.3f}, weeks={result['weeks_elapsed']})\n"
            f"Rank IC (reporting only, not the gate statistic): "
            f"{result['ic_spearman']:+.3f} (p={result['p_spearman']:.3f})\n"
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
    result = compute_checkpoint(db_path=args.db, phase1_start=phase1_start,
                                max_capital_at_risk=float(cap) if cap else None)
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        paths = write_checkpoint(result, output_dir=args.output)
        print(f"\nWrote: {paths}")


if __name__ == "__main__":
    main()
