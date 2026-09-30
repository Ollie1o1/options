"""Benchmark engine: what the trading book returned against what doing
nothing (Cash, SPY buy-and-hold) would have returned, over identical
holding periods.

Deliberately separate from `src.policy_lab`: that package compares two
policies on the SAME path (the underlying's move cancels in the paired
difference); this package compares the book's actual trades against paths
it never took (a different instrument, SPY, over the same calendar window).
There is no shared path to cancel here, which is why this engine's results
carry much wider uncertainty than `policy_lab`'s — see `excess.py` and
`report.py` for the honest accounting of that.

Phase 1 (this package, as built): nulls 1 (Cash) and 2 (SPY buy-and-hold)
only. Null 3 (a mechanical SPY put spread) is out of scope — the options
chain that would price it only covers 52 archive dates, a small fraction of
the book's 968 closed trades — and is deferred to a later, separately
scoped piece of work.
"""
