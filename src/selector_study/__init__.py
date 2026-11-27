"""Paper study: the generic bull-put selector against a matched liquidity-first baseline.

Rules: docs/PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md. Research data only. Nothing
here reads or writes the trading ledger, a promotion gate, or a scoring weight.

capture  -> one observation per (session, symbol) from the chain archive, both arms built
marks    -> later archived quotes for every frozen entry, gaps kept as rows
outcomes -> the book's spread exit rules, crossed (primary) and mid (sensitivity)
readout  -> coverage, matching, outcome counts, paired difference clustered by ISO week
"""
