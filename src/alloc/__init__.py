"""Allocation backtester: which structure, on which underlyings, at what size.

Built because contract ranking is a settled dead end here — `quality_score` shows
no within-family predictive power across 829 closed trades, and selectivity made
results monotonically worse. The only reliable signal in the ledger is which
structure family gets deployed, so that is what this measures.
"""
