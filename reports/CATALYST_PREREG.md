# Catalyst Backtest — Pre-Registration v2

The runner refuses to emit a report unless this file's SHA-256 matches the
hypotheses declared in `src/catalyst/backtest/prereg.py`.

Vintages: quarter-starts 2023-01-01 through 2025-10-01 (12).
Benchmark: absolute and XBI-relative.
Universe: catalyst-calendar rows, $50M-$10B, sponsor resolving to a live ticker.

A confidence interval containing zero is reported as NO EVIDENCE, and is never
re-sliced until it does not.

## v2 supersedes v1 — the ESTIMATOR changed, the hypotheses did not

**v1 (frozen 2026-08-25, run 2026-08-26) used a bootstrap that resampled ROWS.
That was wrong, and this version corrects it.** The hypotheses below are
unchanged, word for word; only the interval around them is.

`outcomes.outcomes_for(ticker, vintage, today, prices, bench)` never receives
an `nct_id`. The forward return is therefore a property of the TICKER and the
VINTAGE alone, so every trial on one ticker at one vintage appended a
BYTE-IDENTICAL value to its arm. Measured 2026-08-27 from the point-in-time
cache: 832 trials resolve to 270 distinct tickers, mean 3.08 trials each,
VNDA alone 17. Those copies are not independent evidence, and resampling rows
counted them as if they were.

**Estimator, v2:** percentile bootstrap resampling TICKERS with replacement,
2,000 iterations, seeded. A ticker appearing in both arms is drawn once and
contributes to both, preserving the within-ticker correlation. UNDERPOWERED is
decided on the CLUSTER count, not the row count.

**This is a RE-ANALYSIS of the same observations, not an independent
replication.** It cannot confirm v1 and must never be reported as a second
study that agreed. Widening an interval that already contained zero leaves it
containing zero, so v1's NO EVIDENCE verdicts are expected to stand; what
changes is any claim that rests on the WIDTH of those intervals — above all
the claim that large effects were ruled out.

v1 remains in git history. Nothing about it is deleted or amended in place.

## H1 (PRIMARY, 6-month horizon)

Rows flagged FUNDED THROUGH outperform rows flagged RAISE BEFORE over the forward window, XBI-relative.

## H2 (EXPLORATORY, 6-month horizon)

Trials whose primary endpoint was amended underperform trials with no endpoint amendment.

## H3 (EXPLORATORY, 6-month horizon)

Phase 3 rows outperform Phase 2 rows.

## H4 (EXPLORATORY, 3-month horizon)

The options-implied move is biased relative to the realised move over the event window.
