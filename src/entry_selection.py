"""Which survivors get the top-N entry slots.

They are drawn at random, and that is a decision rather than the absence of
one.

Three findings force it. Ranking was tested and failed: no ordering of the
survivors beat any other at the #1 slot out of sample — 23 of 48 paired
(day, board) cells, Wilcoxon p=0.89 (`scripts/validate_gates.py`, 2026-08-09).
Removal, by contrast, held in five folds out of five, which is why the gates
stay and only the ordering goes.

The ordering the entry path was ACTUALLY using was not the one it documented.
`rank_single_legs_by_verdict` ranks EV-descending, then `gate_and_report`
re-sorts the survivors by carry (`pick_ranking._carry_key`), and `.head(N)`
consumed that. On a live board the two orderings were disjoint — 0 of 10
overlap, 1 of 5 after the per-symbol dedup — and the carry order entered a set
whose median EV was 43.94 against 59.40 for the survivor pool it was drawn
from. Carry is documented as "an ordering, not a ranking ... deliberately not
a quality signal" (`pick_ranking.py:101`), and its apparent whole-book
correlation with return is a strategy-mix artifact: +0.104 overall but -0.282
within Iron Condor, -0.070 within Long Put.

And a random draw is the only one that makes the recorded data clean. Every
selection rule imprints itself on the ledger, so a cohort selected by rule X
cannot be used to test rule X — the reason the existing book cannot settle
whether its own ranker works. Drawing at random makes the entered set an
unbiased sample of the survivor pool, which is what `data/candidates.db` needs
if it is ever to answer the question.

This module does NOT decide what is eligible. The gates still refuse, and the
allowlist and budget filters still apply. Randomness operates only among what
survived all of them.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

ENTRY_DISCLOSURE = (
    "Entry slots drawn at RANDOM among gate survivors — not a ranking. "
    "No ordering of survivors beat another out of sample (p=0.89), and a "
    "random draw keeps the logged cohort an unbiased sample of the pool."
)


def entry_seed(scan_id: str) -> int:
    """A reproducible seed for one scan's draw.

    Derived with SHA-256 rather than the builtin `hash`, which is salted per
    process: a seed from `hash()` would change every run and the draw could
    not be reproduced from the `scan_id` recorded in `data/candidates.db`.
    Reproducibility is the whole reason the seed is derived from the scan_id
    rather than taken from entropy — an audit has to be able to replay it.
    """
    digest = hashlib.sha256(str(scan_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def dedup_by_symbol(df: Any, strategy_of_row: Any, *, alloc: Any = None) -> Any:
    """One row per symbol: the highest-weighted eligible structure wins.

    Without `alloc` this is exactly `drop_duplicates(subset=["symbol"],
    keep="first")` on whatever order the frame already carries — plain
    first-wins, and what still runs when no strategy allocation is
    configured (the name allowlist rules instead, and every candidate here
    is already the same structure it would have been under that list).

    With `alloc`, rows are stable-sorted by `alloc.share(strategy)` before
    the same drop_duplicates, so ties within one structure still resolve by
    the caller's own draw order (e.g. `draw_entry_queue`), but which
    STRUCTURE wins a symbol's one shot follows the allocation's weight
    rather than which structure happened to have more raw candidate rows
    for that symbol.

    That distinction is not academic. `strategy_allocation.allocate` targets
    a SHARE per structure (Bull Put ~88%, Long Call/Long Put/Short Put ~4%
    each) and gates each candidate independently against it — but a
    structure that is constructible on many strikes of a few names produces
    far more raw rows per symbol than one constructible on only a handful of
    strikes across many names. Deduping by plain shuffle order rewards row
    COUNT, not weight, and a symbol's one shot went to the low-weight
    structure about as often as the high-weight one. Measured 2026-09-03:
    entered mix 40% Bull Put / 40% Long Call+Put against an 88%/8% target.
    Sorting by weight first fixes the ORDER of operations; it does not
    remove the randomness — the winning row still has to clear its own
    structure's admission probability afterward, same as always.
    """
    if df is None or len(df) == 0 or "symbol" not in getattr(df, "columns", ()):
        return df
    weights = getattr(alloc, "weights", None)
    if alloc is not None and weights:
        weight = df.apply(lambda r: alloc.share(strategy_of_row(r)), axis=1)
        df = (df.assign(_alloc_weight=weight)
                .sort_values("_alloc_weight", ascending=False, kind="mergesort")
                .drop(columns="_alloc_weight"))
    return df.drop_duplicates(subset=["symbol"], keep="first")


def draw_entry_queue(df: Any, *, scan_id: str) -> Any:
    """Shuffle survivors into the order the top-N cut will consume.

    Returns a permutation — every input row comes back. Dropping candidates
    here would starve the forward cohort; the allowlist and budget filters
    downstream do the dropping.

    Failure-safe like the rest of the scan path: if anything raises, the frame
    is returned as it came in. A broken draw must degrade to "unshuffled",
    never to "empty".
    """
    if df is None or len(df) == 0:
        return df
    try:
        import numpy as np

        rng = np.random.default_rng(entry_seed(scan_id))
        return df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    except Exception:
        log.warning("entry draw failed; leaving the queue in scan order",
                    exc_info=True)
        return df
