"""Rendering, and the refusal that makes the rest of it worth reading."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import src.formatting as fmt
from src.catalyst.backtest.prereg import HYPOTHESES
from src.catalyst.backtest.study import Result

WIDTH = 100
_EXPLORATORY = {h.key for h in HYPOTHESES if h.exploratory}


def render(results: Sequence[Result], horizon_counts: Dict[int, int],
           dropped_delisted: int, prereg_ok: bool,
           universe_pinned_at: Optional[str] = None,
           universe_n: Optional[int] = None,
           run_failed: Optional[str] = None,
           failed_fetches: int = 0) -> str:
    """The study report. Refuses outright if the prereg hash does not match."""
    lines: List[str] = [fmt.style("CATALYST BACKTEST", "heading"),
                        fmt.draw_separator(WIDTH)]
    if not prereg_ok:
        lines.append("  REFUSED: reports/CATALYST_PREREG.md is missing or does "
                     "not match the declared hypotheses.")
        lines.append("  No results are shown. Run `--write-prereg`, commit it, "
                     "then re-run.")
        return "\n".join(lines)
    if run_failed:
        # A failed run must never be rendered as a measurement. On 2026-08-28
        # a rate-limited benchmark fetch nulled every XBI-relative outcome and
        # this printed "n = 0 vs 0 ... UNDERPOWERED" for all three hypotheses,
        # which reads as a finding about the data. UNDERPOWERED means "we
        # measured and the arm was small"; this means nothing was measured.
        lines.append(f"  RUN FAILED — {run_failed}")
        lines.append("  No results are shown. This is NOT an UNDERPOWERED "
                     "result and NOT evidence of absence: nothing was "
                     "measured. Re-run once the data source recovers.")
        return "\n".join(lines)

    for r in results:
        tag = "  [EXPLORATORY]" if r.key in _EXPLORATORY else ""
        lines.append(fmt.style(f"{r.key}  {r.label}{tag}", "emph"))
        if r.verdict != "NOT COMPUTABLE":
            # The cluster count is the honest sample size. A k of 0 means it
            # was not measured, so nothing is printed rather than "0 tickers".
            clusters = (f"   ({r.k_true} vs {r.k_false} tickers)"
                        if r.k_true or r.k_false else "")
            lines.append(f"    n = {r.n_true} vs {r.n_false} rows{clusters}")
            lines.append(f"    mean {r.mean_true:+.3f} vs {r.mean_false:+.3f}"
                         f"   diff {r.diff:+.3f}")
        if r.verdict == "NOT COMPUTABLE":
            lines.append(f"    {fmt.style('NOT COMPUTABLE', 'warn')} "
                         f"— declared, but the data to test it does not exist")
        elif r.verdict == "UNDERPOWERED":
            lines.append(f"    {fmt.style('UNDERPOWERED', 'warn')} "
                         f"— an arm is too small to say anything")
        else:
            lines.append(f"    95% CI [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]"
                         f"   {fmt.style(r.verdict, 'warn')}")
        lines.append("")

    lines.append(fmt.draw_separator(WIDTH))
    # Observations, not vintages. There are 12 vintages; the 2026-08-27 run
    # printed "3mo: 2103 vintages", which was this count wearing the wrong
    # noun. Horizons differ because a 12-month window has not elapsed for the
    # late vintages, so the arms are genuinely different sizes per horizon.
    counts = ", ".join(f"{m}mo: {n} observations"
                       for m, n in sorted(horizon_counts.items()))
    if counts:
        lines.append(f"  elapsed observations per horizon — {counts}")
    if failed_fetches:
        # "Delisted" and "we could not look" are different claims. A ticker
        # lost to a source failure takes its whole cluster out of the study
        # and moves every interval, so it is never folded into the
        # survivorship line.
        lines.append(f"  {failed_fetches} tickers dropped after FAILED price "
                     f"fetches — a source outage, NOT a delisting; the sample "
                     f"is smaller than the pinned universe")
    lines.append(f"  {dropped_delisted} names dropped as delisted and "
                 f"unpriceable; measured 6.4% of resolvable 2024 sponsors, "
                 f"skewed toward ACQUISITIONS, so the residual bias is likely "
                 f"conservative")
    lines.append("  market cap is TODAY'S, not the vintage's — no historical "
                 "cap source; the band is a universe definition, not a feature "
                 "under test")
    lines.append("  overlapping quarterly vintages are NOT independent "
                 "observations")
    lines.append("  CIs are cluster-robust, resampling TICKERS — the forward "
                 "return is a property of the ticker, so trials sharing one "
                 "ticker and vintage are one observation, not several")
    # WHICH universe ran. Two runs on different populations are not
    # comparable, and before this was printed there was no way to tell them
    # apart from the output — the sweep moved H3's arms between two runs a day
    # apart. A fresh sweep is deliberately NOT called "pinned".
    if universe_pinned_at:
        lines.append(f"  universe PINNED {universe_pinned_at}"
                     + (f" — {universe_n} trials" if universe_n is not None
                        else "")
                     + "; --refresh-universe to re-sweep")
    elif universe_n is not None:
        lines.append(f"  universe swept fresh — {universe_n} trials")
    lines.append("  a CI containing zero is NO EVIDENCE, not a weak finding")
    return "\n".join(lines)
