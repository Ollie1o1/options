# Discovery collection audit — 2026-11-24

Collection quality only. No predictive outcomes were evaluated, no hypothesis
was added, and no strategy, gate, calibration weight, exit or real-money setting
was changed. The eight previously registered channel findings remain dated to
their original August price sample, including negative and insufficient results.

## Starting evidence

The checkout and origin were clean at `4f3c196` (PR #150). The existing collectors
and archives were inspected before choosing the repair:

| Archive / collector | Measured coverage before this change | Decision |
|---|---|---|
| `data/equity_ohlcv.db`, `src/breakout/data.py`, `src/breakout/engine.py` | 310,723 daily bars, 32 symbols, every symbol ending 2026-08-14 | Repair this shared archive |
| `data/squeeze_prices.db`, `src/squeeze/backtest/prices.py` | 20,502,167 closes, 21,244 symbols, 2017-10-02–2026-08-07 | Different, broad raw-price substrate; do not splice it into the adjusted archive |
| `data/news_archive.db`, `src/news_archive.py` | 122,638 observations, 230 symbols, 112 first-seen dates, through 2026-11-24 | Preserve first archival timestamps; keep unknown measurements NULL going forward |
| `data/estimates_archive.db`, `src/estimates_archive.py` | 61,690 estimate fields, 105 symbols, six snapshot dates through 2026-11-20; 49,526 analyst actions on four first-seen dates through 2026-11-17 | Reuse existing after-close collection; historical grade dates do not establish availability |
| `data/chain_archive.db`, `src/chain_archive.py` | Existing option-chain/spot source | Intraday spot is not a replacement for an adjusted daily-close series |
| `src/data_fetching.py` | Live scan history requests and caches | Continue using these for scans; do not add a second durable equity-history collector |

The prior breakout refresh ran on demand, always requested `period=max`, ignored
its start argument, threw away overlapping dates, inserted with `OR IGNORE`, and
represented exceptions as zero inserts. Nothing scheduled an equity-history
refresh. A dividend or split revision could therefore leave older and newer
prices on different adjustment bases without an observable failure.

## Repair and measured result

The existing `ohlcv` table is still the shared price store. The repaired path:

- Explicitly requests daily Yahoo history with `auto_adjust=False`, actions on,
  repair off, and an exclusive end date. It stores `Adj Close` and scales high
  and low by `Adj Close / Close`. Yahoo `Close` is already split-adjusted;
  calling it an original as-traded price would be incorrect. Provider close,
  dividends and split ratios are preserved alongside each accepted bar.
- Uses a 35-calendar-day overlap for incremental requests. Price/volume overlap
  disagreement beyond numerical tolerance triggers a full refetch before any
  prices change. Legacy/unidentified histories and histories not fully checked
  for 30 days also require a full refresh. Every previous date must survive a
  replacement. Every overwritten value is versioned, including tiny revisions.
- Retains compressed normalized provider observations and SHA-256 fingerprints,
  requested windows, actual observation timestamps, acceptance and errors in
  `price_fetches`. `ohlcv_evidence.first_seen` is immutable on reobservation;
  old values are retained in `ohlcv_revisions`. Prices, evidence and collector
  state commit atomically per symbol. Original archive values are also retained
  in the local pre-refresh SQLite backup.
- Rejects empty, duplicate-date, weekend, out-of-window, nonfinite and invalid
  OHLCV responses. Retries are bounded; five consecutive failing symbols stop
  the feed. Failure is distinct from a successful no-change observation.
- Waits until 18:00 Eastern to include today's daily bar, including on half days.
  It uses the maintained US holiday calendar and returns unknown for unsupported
  years. The existing installed maintenance schedule has 18:30 and 20:30 weekday
  retries; this change adds no scheduler and changes no entry/exit schedule.
- Marks maintenance complete only when the covered universe is current and
  verified. Partial/stale/gapped runs remain unstamped and retryable.

All 32 existing symbols were preserved. Collection advanced to **2026-11-23**,
adding **2,240 bars: 70 sessions per symbol**, for **312,963 total bars**. November
24 was deliberately excluded because collection occurred before 18:00 Eastern.
From 2017-10-02 (or each later first observation), all 32 symbols had complete
matching SPY dates, positive finite prices, source evidence for every bar, and
no daily move exceeding 45% in magnitude. Maintained-year expected sessions also
check for a date absent from SPY itself. Before-listing/history gaps stay unknown.

The follow-up overlap collection added zero new bars and completed successfully.
Provider rounding revisions are retained rather than silently discarded. Two
full-history large moves remain visible, without inferred split repairs: AAPL
2000-09-29 (approximately −51.9%) and AMD 2016-04-22 (approximately +52.3%). These
precede the research window and are observations to review, not a feed outage.

Numerical spot checks preserved the provider's explicit split ratios for NVDA
2024-06-10 (10), AAPL 2020-08-31 (4), AMZN 2022-06-06 (20), and GOOGL 2022-07-18
(20). Their adjusted daily returns were approximately +0.75%, +3.39%, +1.99%,
and −2.46%, respectively. No ratio inferred from a price jump was used. SQLite
integrity, per-symbol row counts, all fetch payload hashes and evidence coverage
were checked. [The machine-readable audit](discovery/coverage_20261124.json)
retains each accepted/rejected fetch fingerprint, per-symbol price fingerprints,
before/after counts and these split observations.

The existing SPY source-consistency check also passed against archived option
chain spots on 80 exact common dates: median absolute difference 0.234%, below
its unchanged 0.50% threshold (maximum 1.462%). This is a limited intraday-spot
cross-check, not independent verification of every daily close or adjustment.

An initial full-history validation rejected one floating-point unit of
high/low scaling error at an exactly equal close/high or close/low. The measured
maximum discrepancy was below 2e-16 relative. Validation now tolerates 1e-10
relative for this range comparison, with a regression test. The rejected fetches
remain in the audit. This tolerance does not repair source prices.

## Available workflow

```bash
python -m src.discovery_research --coverage --no-open
python -m src.discovery_research --update-prices --no-open
```

The first command is read-only against the archives. The second collects prices
and produces coverage only. Both write `reports/discovery/coverage.html` and
`coverage.json`, plus timestamped coverage copies, and return nonzero for
incomplete price coverage. The report distinguishes stale/missing/unverified
symbols, exact-date gaps, large-move review, failed requests and missing event
availability. News and analyst observation counts are not claims that their
collectors successfully fetched every symbol on every day. An absent observation
stays unknown. Existing news with historical missing-score defaults cannot be
retrospectively distinguished from measured neutrality; only future missing
measurements are fixed.

The saved Discovery Lab links to this audit. Opening saved findings refreshes
coverage locally, without evaluating hypotheses or fetching anything. The dated
study's original JSON and preregistration remain unchanged.

## Limits and next priorities

This validates same-provider consistency and supplied corporate actions. It is
not an independent second-vendor verification, a historical point-in-time price
vintage, or evidence of forecasting skill. Changes outside the overlap may wait
until the monthly full refresh. The US holiday calendar is currently maintained
for 2026 only and must be extended before 2027 collection can claim freshness.
An earlier listing date cannot be inferred just because a current symbol has
prices now. SQLite payload retention grows with collection; retention must not
silently discard vintages used by a research run.

Analyst actions retain immutable `first_seen`. The separate daily estimate-field
table currently keeps its latest same-day snapshot, with that version's
`archived_at`; it must not be treated as an earlier intraday vintage. Neither
publication dates nor backfilled analyst grade dates can repair missed first-seen
history. Per-symbol fetch-attempt ledgers for news/estimates remain useful next
collection work, especially to distinguish a measured empty fetch from no fetch.

With prices available, next investigate dated customer/supplier disclosures,
guidance vintages and sector operating drivers using the existing SEC/catalyst
collectors where applicable. Preserve source documents and availability before
registering genuinely new economic hypotheses. Do not recycle the closed simple
sentiment, VIX-direction or peer-leadership tests as new ideas. New tests need
own momentum, market/sector and volatility controls, chronological purged
holdouts, realistic entry timing, clustered uncertainty and family correction.
Stock direction, movement magnitude and option profitability remain separate
questions. Candidate/contract scan-change comparisons follow this collection
and evidence work; this patch does not claim to implement that later step.
