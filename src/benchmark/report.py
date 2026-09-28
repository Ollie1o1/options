"""Run manifest (with corpus fingerprint) and markdown emission.

Follows `src.policy_lab.report`'s pattern: a run without a manifest is not a
result, because without the git SHA and a snapshot of what the corpus looked
like at read time, a number on a page cannot be checked by anyone — including
its author a month later. `paper_trades.db` is live (closed trades went
954 -> 968 during prior, unrelated work on this repo), so the fingerprint
records `max(exit_date)`, the closed-trade count offered by the ledger, and
the count that actually survived the loader's invariants, all three always
present.

**This module emits no verdict.** There is no promote/reject/pass/fail
anywhere below — see `excess.py`'s module docstring for why a significance
test is not meaningful on this data, and `_LIMITS` for the paragraph every
report carries stating that plainly.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.benchmark.excess import (
    AggregateResult, CumulativeComparison, DailyAggregateResult, LoadReport,
    MatchStats,
)
from src.benchmark.prices import StitchValidation, _ro


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def corpus_fingerprint(ledger_db_path: str, loaded: int) -> Dict[str, Any]:
    """A read-only snapshot of what `paper_trades.db` offered at read time.

    `max_exit_date` and `terminal_count` describe the corpus itself
    (independent of what the loader kept); `loaded` is what
    `excess.load_closed_trades` actually returned — so both the offered and
    the used counts are visible on the same manifest, and two runs are
    comparable only when all three agree.
    """
    con = _ro(ledger_db_path)
    try:
        max_exit_date = con.execute(
            "SELECT MAX(exit_date) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()[0]
        terminal_count = con.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()[0]
    finally:
        con.close()
    return {
        "max_exit_date": max_exit_date,
        "terminal_count": int(terminal_count),
        "loaded": int(loaded),
    }


def run_manifest(ledger_db_path: str, load_report: LoadReport,
                  match_stats: MatchStats, stitch: StitchValidation,
                  seed: int) -> Dict[str, Any]:
    """Everything needed to reproduce or challenge this run."""
    return {
        "git_sha": _git_sha(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "cluster_unit": "(symbol, entry_date)",
        "corpus_fingerprint": corpus_fingerprint(ledger_db_path,
                                                  load_report.loaded),
        "row_counts": {
            "loaded": load_report.loaded,
            "dropped": dict(load_report.dropped),
        },
        "match_stats": {
            "exact": match_stats.exact,
            "tolerant": match_stats.tolerant,
            "missing": match_stats.missing,
            "tolerance_days": match_stats.tolerance_days,
        },
        "stitch_validation": {
            "n_common_dates": stitch.n_common,
            "median_diff": stitch.median_diff,
            "mean_diff": stitch.mean_diff,
            "max_abs_diff": stitch.max_abs_diff,
            "median_abs_diff": stitch.median_abs_diff,
            "threshold": stitch.threshold,
            "ok": stitch.ok,
        },
    }


def manifests_comparable(a: Dict[str, Any], b: Dict[str, Any]) -> "tuple[bool, str]":
    """Whether two manifests' numbers may be compared, and why not if not.

    Mirrors `policy_lab.report.manifests_comparable`: a differing git SHA
    disqualifies on its own, and beyond that the corpus fingerprint's
    `max_exit_date`/`terminal_count` must agree — `paper_trades.db` is a
    live ledger, so two runs are comparable only when they read the same
    snapshot of it.
    """
    sha_a, sha_b = a.get("git_sha"), b.get("git_sha")
    if sha_a != sha_b:
        return False, f"different git sha: {sha_a} vs {sha_b}"
    fp_a = a.get("corpus_fingerprint") or {}
    fp_b = b.get("corpus_fingerprint") or {}
    ts_a, ts_b = fp_a.get("max_exit_date"), fp_b.get("max_exit_date")
    if ts_a != ts_b:
        return False, f"corpus drifted: max_exit_date {ts_a} -> {ts_b}"
    tc_a, tc_b = fp_a.get("terminal_count"), fp_b.get("terminal_count")
    if tc_a != tc_b:
        grown = tc_a is not None and tc_b is not None and tc_b > tc_a
        verb = "grew" if grown else "changed"
        return False, f"corpus {verb}: terminal_count {tc_a} -> {tc_b}"
    return True, "same git sha, same corpus fingerprint"


_LIMITS = """## What this engine cannot tell you

This engine **cannot establish statistically that the book under- or
out-performs a passive alternative.** The per-trade excess confidence
interval contains zero and would need on the order of 100,000+ trades to
resolve at the observed effect size and noise; the daily series would need
on the order of thousands of trading days (years). What it provides instead
is an honest, reproducible point comparison and a permanent fixture for
accumulating forward observations — not a hypothesis test with a pass/fail
bar.

Why pairing does not rescue this the way it does in `policy_lab`: there,
both arms replay the SAME position on the SAME price path, so the
underlying's move cancels exactly in the paired difference. Here the book's
trade and SPY are different instruments on different paths — there is no
shared path to cancel, and the book's per-trade return noise is many times
larger than SPY's return over the same window (see the `variance ratio`
figures below, which stay near 1.0 rather than falling the way a working
pairing would).

The cumulative dollar comparison is different in kind: it is **one
realisation of one path and carries no confidence interval.** It says what
actually happened, not what would happen on average — a description, not
an inference.

Null 3 (a mechanical SPY put spread) is **out of scope for this report.**
The options chain that would price it (`chain_archive.db`) only covers 52
snapshot dates, a small fraction of the closed book's date range, and
building it as though it covered the whole book is the failure this project
exists to avoid.
"""


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:+.3%}"


def _fmt_ci(lo: Optional[float], hi: Optional[float]) -> str:
    if lo is None or hi is None:
        return "not computed"
    return f"[{lo:+.3%}, {hi:+.3%}]"


def _fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def render_markdown(manifest: Dict[str, Any],
                     cumulative: CumulativeComparison,
                     per_trade: Dict[str, AggregateResult],
                     daily: Optional[DailyAggregateResult]) -> str:
    """The BENCHMARK_RESULT_*.md body. Descriptive throughout — no verdict."""
    stitch = manifest["stitch_validation"]
    match = manifest["match_stats"]
    fp = manifest["corpus_fingerprint"]
    rows = manifest["row_counts"]

    lines = [
        "# Benchmark Result",
        "",
        f"- git: `{manifest['git_sha']}`",
        f"- generated: {manifest['generated_at']}",
        f"- seed: {manifest['seed']}",
        f"- clustering unit: {manifest['cluster_unit']}",
        f"- corpus fingerprint: max_exit_date={fp['max_exit_date']}, "
        f"terminal_count={fp['terminal_count']}, loaded={fp['loaded']}",
        f"- rows dropped: {rows['dropped'] or '{}'}",
        f"- SPY date match: {match['exact']} exact, {match['tolerant']} "
        f"within {match['tolerance_days']}d tolerance, {match['missing']} "
        f"missing (of {match['exact'] + match['tolerant'] + match['missing']} "
        "matched trades)",
        f"- SPY stitch validation: {stitch['n_common_dates']} common dates, "
        f"median diff {_fmt_pct(stitch['median_diff'])}, "
        f"mean diff {_fmt_pct(stitch['mean_diff'])}, "
        f"max abs diff {_fmt_pct(stitch['max_abs_diff'])}, "
        f"median abs diff {_fmt_pct(stitch['median_abs_diff'])} "
        f"(threshold {_fmt_pct(stitch['threshold'])}, "
        f"{'OK' if stitch['ok'] else 'FAILED'})",
        "",
        "## Headline: cumulative dollar comparison",
        "",
        "**One realisation of one path. No confidence interval attaches to "
        "this section.**",
        "",
        f"- matched trades: {cumulative.n_trades}",
        f"- book total realised P&L: {_fmt_usd(cumulative.book_pnl_usd)}",
        f"- total capital-at-risk (primary convention, sum over trades): "
        f"{_fmt_usd(cumulative.total_capital_at_risk_usd)}",
        f"- SPY-equivalent P&L, primary convention (same dollars, same "
        f"entry->exit windows, per trade): "
        f"{_fmt_usd(cumulative.spy_equiv_primary_usd)}",
        f"- mean concurrent capital deployed (secondary convention): "
        f"{_fmt_usd(cumulative.mean_concurrent_usd)}",
        f"- peak concurrent capital deployed: "
        f"{_fmt_usd(cumulative.peak_concurrent_usd)}",
        f"- SPY return over the full book span: "
        f"{_fmt_pct(cumulative.spy_span_return)}",
        f"- SPY-equivalent P&L, secondary convention (mean concurrent "
        f"deployment held in SPY for the whole span): "
        f"{_fmt_usd(cumulative.spy_equiv_secondary_usd)}",
        "",
    ]

    lines += ["## Per-trade excess (descriptive; CI shown)", ""]
    lines += ["| vs. null | n trades | n clusters | mean book | mean null | "
              "mean excess | 95% CI | CI contains zero | variance ratio | "
              "corr(book, null) | n needed (80% power) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in per_trade.items():
        mean_null = _fmt_pct(r.mean_null)
        mean_book = _fmt_pct(r.mean_book)
        ci = _fmt_ci(r.ci_lo, r.ci_hi)
        contains_zero = "n/a" if r.ci_contains_zero is None else (
            "YES" if r.ci_contains_zero else "no")
        corr = "n/a" if r.corr_book_null is None else f"{r.corr_book_null:+.4f}"
        n_needed = ("inf" if r.n_needed_80pct_power == float("inf")
                    else f"{r.n_needed_80pct_power:,.0f}")
        lines.append(
            f"| `{name}` | {r.n_rows} | {r.n_clusters} | {mean_book} | "
            f"{mean_null} | {_fmt_pct(r.mean)} | {ci} | {contains_zero} | "
            f"{r.variance_ratio:.3f} | {corr} | {n_needed} |")
    lines.append("")
    lines.append(
        "`mean excess` = mean(book_ret - null_ret) per trade, "
        "`(symbol, entry_date)`-clustered bootstrap CI "
        "(`policy_lab.stats.cluster_bootstrap_mean_ci`, n_boot=10000). "
        "`variance ratio` near 1.0 means pairing against this null removed "
        "essentially no variance (contrast with `policy_lab`, where a "
        "working pairing drives this well below 1.0).")
    lines.append("")

    lines += ["## Daily portfolio series (descriptive; CI shown)", ""]
    if daily is None or daily.n_days == 0:
        lines.append("Not computed (no aligned trading days).")
    else:
        contains_zero = "n/a" if daily.ci_contains_zero is None else (
            "YES" if daily.ci_contains_zero else "no")
        t_str = "n/a" if daily.t_stat is None else f"{daily.t_stat:+.3f}"
        n_needed = ("inf" if daily.n_needed_80pct_power == float("inf")
                    else f"{daily.n_needed_80pct_power:,.0f}")
        lines += [
            f"- aligned daily observations: {daily.n_days}",
            f"- mean book daily return (on deployed capital): "
            f"{_fmt_pct(daily.mean_book)} (sd {_fmt_pct(daily.sd_book)})",
            f"- mean SPY daily return over the same days: "
            f"{_fmt_pct(daily.mean_spy)} (sd {_fmt_pct(daily.sd_spy)})",
            f"- mean daily excess: {_fmt_pct(daily.mean_excess)} "
            f"(sd {_fmt_pct(daily.sd_excess)})",
            f"- t-statistic: {t_str}",
            f"- 95% CI: {_fmt_ci(daily.ci_lo, daily.ci_hi)} "
            f"(contains zero: {contains_zero})",
            f"- variance ratio (sd(excess)/sd(book)): {daily.variance_ratio:.3f}",
            f"- n needed to resolve at 80% power: {n_needed} trading days",
        ]
    lines.append("")

    lines.append(_LIMITS)
    return "\n".join(lines)
