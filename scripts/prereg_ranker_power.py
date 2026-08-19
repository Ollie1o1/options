"""Compute n* and write the frozen pre-registration. Run ONCE.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.prereg_ranker_power \
        --deadline 2026-11-19

Uses only cluster STRUCTURE, never outcomes, so it can be run before any
position closes — which is the only time a pre-registration is worth writing.

See docs/PREREG_RANKER_TEST_SPEC.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

from src import prereg_ranker as pk

DEFAULT_REGISTRATION = "docs/PREREG_RANKER_TEST.md"

# The book's own measured ICC range from batch entries is 0.08-0.11. The upper
# end is assumed, because under-powering is the failure that wastes a quarter.
DEFAULT_ASSUMED_ICC = 0.11


def parse_field(text: str, name: str) -> Optional[str]:
    """Read a `name: value` line out of the registration's parameter block."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name}:"):
            return stripped.split(":", 1)[1].strip()
    return None


def build_registration(cohort: pd.DataFrame, *, target_ic: float, power: float,
                       alpha: float, deadline: str, n_boot: int, seed: int,
                       assumed_icc: float) -> str:
    """The registration document, with n* computed from observed clustering."""
    need_effective = pk.required_effective_n(target_ic, power, alpha)

    rows = len(cohort)
    clusters = int(cohort["contract_key"].nunique()) if rows else 0
    mean_cluster = (rows / clusters) if clusters else 1.0
    design_effect = 1.0 + (mean_cluster - 1.0) * assumed_icc
    n_star = need_effective * design_effect

    stamped = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""# Pre-registered ranker test — REGISTRATION

**Frozen {stamped}. Immutable.**

Written before any outcome existed. Every term below was fixed in advance;
that is the only thing that makes the eventual result mean anything.

## Hypothesis

**H1.** Among gate survivors, `ev_net` predicts `pnl_pct`.

Statistic: rank IC on ranks demeaned within `(entry_date, strategy)` cells.
Interval: percentile cluster bootstrap resampling `contract_key`.

## Parameters

```
feature: ev_net
outcome: pnl_pct
cells: entry_date, strategy
cluster: contract_key
target_ic: {target_ic}
power: {power}
alpha: {alpha}
n_boot: {n_boot}
seed: {seed}
min_cell_rows: {pk.MIN_CELL_ROWS}
assumed_icc: {assumed_icc}
mean_cluster_size_at_registration: {mean_cluster:.4f}
design_effect: {design_effect:.4f}
n_star_effective: {need_effective:.1f}
n_star_nominal: {n_star:.0f}
deadline: {deadline}
```

## Decision rule

One look, at `n_star_nominal` closed survivor positions or `deadline`,
whichever comes first.

| outcome | condition | consequence |
|---|---|---|
| PASS | 95% CI lower bound > 0 | ranking may be *proposed* again, as its own change |
| FAIL | CI contains 0 | refuse-don't-rank stands; entries stay random |
| INVERTED | CI entirely below 0 | treated as FAIL; motivates a NEW registration |
| UNDERPOWERED | n < n_star_nominal at deadline | treated as FAIL |

There is no EXTEND state — that is the trap that let the LC gate run forever.
PASS does **not** authorise real money.

`n_star` is powered FOR `target_ic`, so "CI lower bound > 0" and "the effect is
meaningful" cannot disagree. That is exactly where the LC gate failed:
`IC >= 0.08 AND p < 0.05` at a trigger of n >= 50 made the 0.08 decorative,
because detecting 0.08 needs n around 1224.

## Secondary, no decision authority

`quality_score`, `carry`, `delta` — reported beside the primary. A secondary
result may motivate a new registration; it can never move this one.

## Guards

- Negative control: outcome shuffled within cells must return null.
- Sign consistency across halves, split at the median `entry_date`. Reported.
- Effective n reported, never nominal.

## Result

*Not yet run.* `scripts/prereg_ranker_test.py` writes here, once.
"""


def write_registration(text: str, path: str) -> None:
    """Write the registration, refusing to overwrite an existing one.

    Immutability is the whole mechanism. Rewriting after outcomes exist is
    precisely the abuse pre-registration prevents, so it is blocked in code
    rather than left to discipline.
    """
    if os.path.exists(path):
        raise FileExistsError(
            f"{path} already exists — a registration is immutable. "
            "Delete it deliberately if you truly mean to re-register.")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/candidates.db")
    ap.add_argument("--out", default=DEFAULT_REGISTRATION)
    ap.add_argument("--target-ic", type=float, default=0.08)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--deadline", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--assumed-icc", type=float, default=DEFAULT_ASSUMED_ICC)
    args = ap.parse_args(argv)

    # Structure ONLY. Deliberately not `load_cohort`, which reads outcomes:
    # this must be runnable before any position closes, and reading outcomes
    # here would be the first peek.
    import sqlite3
    try:
        with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
            cohort = pd.read_sql(
                "SELECT contract_key FROM candidate_positions", conn)
    except Exception:
        cohort = pd.DataFrame(columns=["contract_key"])

    text = build_registration(
        cohort, target_ic=args.target_ic, power=args.power, alpha=args.alpha,
        deadline=args.deadline, n_boot=args.n_boot, seed=args.seed,
        assumed_icc=args.assumed_icc)
    write_registration(text, args.out)
    print(f"Registration written to {args.out}")
    print(f"  positions seen:  {len(cohort)}")
    print(f"  n_star_nominal:  {parse_field(text, 'n_star_nominal')}")
    print(f"  deadline:        {parse_field(text, 'deadline')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
