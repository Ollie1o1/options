# Open Items After the 2026-08-08 Pass

Written so a later session does not have to re-derive any of this. Companion to
`docs/TAIL_OBSERVED_20260808.md` (data + tail) and
`docs/ATTRIBUTION_20260808.md` (what predicts what).

---

## In flight when this was written

**2025-2026 daily backfill.** 32,867 symbol-days across 121 symbols, ~2.7h at
`--workers 6`, adding ~740 MB. Resumable and interrupt-safe — if it died, just
re-run the same command and it picks up:

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options --backfill \
    --symbols "$(...all cached symbols...)" --workers 6 \
    --start 2025-01-01 --end 2026-06-12
# check what is left first, offline:
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options --gaps \
    --start 2025-01-01 --end 2026-06-12
```

It does **not** change any conclusion in the two companion documents — those
are measured on 2020-2024, and the MEGA cohort already had 2025 daily coverage
for 8 of its 14 names. What it unlocks is `--all-names` work on the recent
window.

## Ranked, with reasons

### 1. Measure the effective spread. This is the only thing that moves a verdict.

Every negative result in this repo assumes an entry crosses the full quoted
spread. That assumption has never been measured, and §2 of
`ALLOCATION_BACKTEST_FINDINGS.md` puts the toll at **half the credit**. If real
fills land near mid, the arithmetic that produces "-$100 EV" changes sign.

Cheapest path: **Databento's $125 free signup credits** (6-month validity,
priced per GB). One surgical OPRA pull — SPY trades + NBBO on ~40 known entry
dates — answers it. This is the highest-value action available at any price.

### 2. optionsDX, free, four more tails

Registration only, no billing info. SPY/SPX/QQQ/VIX/AAPL/NVDA/TSLA EOD,
**2010-2023**, monthly CSVs, **all strikes and all expirations**, with
`C_VOLUME` and `C_SIZE` (no OI). Covers 2011, 2015, 2018 Volmageddon and 2020.

This is the only free way past the two hard walls of the Dolt data: **DTE 10-67
and 2.6 expirations per symbol-day**. It makes calendars, diagonals, LEAPS and
a real liquidity filter testable for the first time. Needs a new loader.

### 3. ThetaData free tier, for the window optionsDX misses

EOD from 2023-06 (quotes from 2023-12), 30 req/min, whole US universe. Its EOD
endpoint returns `volume, count, bid, ask, bid_size, ask_size`. Tiles
2023-12 → 2026 where optionsDX stops. No OI at the free tier.

### 4. Re-derive §4c's friction comparison at proper n

The mega-vs-all comparison cannot currently be reproduced: `max_concurrent = 3`
caps open positions **portfolio-wide**, so a 117-symbol universe still yields
~36 trades/year and n=105 rather than §4c's 10,363. The cap is correct for an
account-level return figure and wrong for measuring a per-trade effect. Needs a
run with concurrency lifted, reported as per-trade statistics only.

### 5. Re-run §4d's signal table

It is flagged in-file as unconfirmed. Every number in it was computed with
splits unhandled, and its headline IV-rank result does not reproduce as a rank
IC on 2020-2024. Either re-run it with splits wired or delete the claim.

### 6. Re-run the news sentiment test when the archive has power

`python -m src.news_signal --horizons 1,3,5,10` — it prints its own power
verdict, so it tells you whether its own null means anything. A large effect
(IC 0.10) is already excluded. IC 0.05 becomes answerable around **mid-October
2026**, IC 0.03 around **late November 2026**, purely by letting the archive
accrue. See `docs/NEWS_SENTIMENT_20260808.md`.

Higher-leverage than waiting: **improve the scorer, not the weight.** All
36,400 archived rows keep their raw headline, so a finance-tuned sentiment
model can be applied retroactively to the entire history and would make the
eventual test a test of sentiment rather than a test of TextBlob. Also
deduplicate by `dedup_key` before aggregating — one ruling currently counts
four times.

## Do NOT do these

- **Do not add an IV-rank or VRP entry filter.** Both measure flat
  (|IC| < 0.05) across all three structures on 2020-2024 with splits wired.
- **Do not select on `credit_pct_width` or `friction_pct_credit`.** They are
  accounting identities for a held-to-expiry structure, they are non-monotone
  by quartile, and selecting on credit just means selling closer to the money.
- **Do not try to fill the 2022-2024 alternate-day gaps.** That cadence is
  upstream's own — verified, `SPY 2024-03-05/07/12` return 0 rows from the API.
- **Do not try to clone the Dolt repo.** Disk sits at 97% (~6 GiB free) and the
  repo covers 2,098 symbols. Range queries (`date BETWEEN`) hit the API's 30s
  deadline, so it is one call per symbol-day regardless.
- **Do not tune `DELTA_TOLERANCE` against returns.** It is set at 0.10 as the
  conventional reading of "a 40-delta option". Tuning it would convert a
  correctness guard into an overfit parameter.
- **Do not give `sentiment` a nonzero weight.** Measured IC ~0 at every
  horizon, and a large effect is already powered out. Adding it would inject a
  term scored by a general-English lexicon into a score whose hand-set
  constants already carry IC -0.096.
- **Do not download FNSPID.** 29.6 GB against ~5 GiB free, and CC BY-NC-4.0
  forbids commercial use — disqualifying for a real-money system.

## Watch the disk

It sat at **98% / ~5 GiB free** at the end of this pass, and the 2025-2026
backfill was still writing. `data/` is 3+ GB of research data that
`project_trust_and_polish_pass` says explicitly not to "clean". Free space
elsewhere before the next large fetch — this is now the binding constraint on
every data option in this file, and it is what ruled out cloning the Dolt repo
and downloading FNSPID.

## Standing hazards this pass discovered

- **A guard that is never called is not a guard.** `splits=` existed, was
  documented, was tested in isolation, and was never passed by the one
  production call site — so `split_closed` was structurally always 0 and every
  published result was computed without it. Nothing in 3,554 tests caught this.
  When a safety branch exists, assert on the call site, not just the function.
- **"Nearest" without a tolerance is substitution.** `_nearest_delta` returned
  whatever existed. Same failure class as the §4c DTE-window bug: the
  instrument tested was not the instrument specified.
- **Comparing rows across a data gap fabricates events.** Split detection read
  a 21-month drift as a corporate action. Any consecutive-row comparison in
  this cache needs a calendar-distance guard.
- **A big t-statistic on an accounting identity is still arithmetic.** Check
  bucket monotonicity before believing any IC.
