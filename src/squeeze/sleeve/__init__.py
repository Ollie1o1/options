"""The squeeze-call sleeve's measurement layer.

Builds the evidence that decides whether long calls on high-short-interest
names are viable. Nothing here trades, sizes, or writes to the ledger: the
sleeve itself is deliberately not built until the gate returns GO.

See docs/superpowers/specs/2026-08-02-squeeze-call-sleeve-design.md.
"""
