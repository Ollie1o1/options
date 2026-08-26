# Catalyst Backtest — Pre-Registration

Written BEFORE any result was computed. The runner refuses to emit a report
unless this file's SHA-256 matches the hypotheses declared in
`src/catalyst/backtest/prereg.py`.

Vintages: quarter-starts 2023-01-01 through 2025-10-01 (12).
Benchmark: absolute and XBI-relative.
Universe: catalyst-calendar rows, $50M-$10B, sponsor resolving to a live ticker.

A confidence interval containing zero is reported as NO EVIDENCE, and is never
re-sliced until it does not.

## H1 (PRIMARY, 6-month horizon)

Rows flagged FUNDED THROUGH outperform rows flagged RAISE BEFORE over the forward window, XBI-relative.

## H2 (EXPLORATORY, 6-month horizon)

Trials whose primary endpoint was amended underperform trials with no endpoint amendment.

## H3 (EXPLORATORY, 6-month horizon)

Phase 3 rows outperform Phase 2 rows.

## H4 (EXPLORATORY, 3-month horizon)

The options-implied move is biased relative to the realised move over the event window.
