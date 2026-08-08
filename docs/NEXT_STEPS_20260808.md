# Open Items After the 2026-08-08 Pass

Written so a later session does not have to re-derive any of this. Companion to
`docs/TAIL_OBSERVED_20260808.md` (data + tail) and
`docs/ATTRIBUTION_20260808.md` (what predicts what).

---

## RESUMING A BACKFILL — read this first if you have an hour

**Nothing is ever lost by stopping one.** Every completed symbol-day is
committed as it lands, and a pair whose fetch FAILED is deliberately left
unmarked so a later run retries it. Kill it, reboot, come back next week — just
re-run the same command and it picks up where it stopped. Jobs do survive a
closed terminal (`nohup`) and a sleeping laptop, but not a reboot.

**Always check the gap first. It is offline, instant, and free:**

```bash
cd ~/Projects/options
export PYTHONPATH=$PWD OPTIONS_MAINTENANCE_CHILD=1
PY=~/.venvs/options/bin/python

# all symbols currently in the cache, as a comma list
SYMS=$($PY -c "
import sqlite3
c=sqlite3.connect('data/dolt_options.db', timeout=300)
print(','.join(sorted(r[0] for r in c.execute('SELECT DISTINCT symbol FROM dolt_chain'))))")

$PY -m src.dolt_options --gaps --symbols "$SYMS" --start 2020-01-27 --end 2026-06-12
```

`--gaps` prints the unfetched grid by year plus a runtime estimate. If it says
0 unfetched, that window is done.

**Then run whichever window you want.** ~3.3 pairs/s at `--workers 6`; the
pacing lock is the ceiling, so more workers buy nothing:

```bash
# (A) the COVID window for symbols that lack it — the highest-value one
$PY -m src.dolt_options --backfill --symbols "$SYMS" --workers 6 \
    --start 2020-01-27 --end 2021-12-31

# (B) 2025-2026 daily
$PY -m src.dolt_options --backfill --symbols "$SYMS" --workers 6 \
    --start 2025-01-01 --end 2026-06-12

# (C) just the 14 tight-spread names, any window — ~35 min, good for a quick pass
$PY -m src.dolt_options --backfill --mega --workers 6 \
    --start 2020-01-27 --end 2021-12-31
```

**Run only ONE at a time.** Two API-heavy jobs compete and get 403'd — observed
again on 2026-08-08 when a probe run during a backfill returned empty.

### Why (A) is the one that matters

The tail result in `TAIL_OBSERVED_20260808.md` — every spread open into the
COVID crash lost 100% of its capital at risk — rests on **3 trades**, because
only the 14 tight-spread names had 2020-21 data. 107 symbols still lack it.
Filling them should turn 3 crash trades into 30-50, which is the difference
between witnessing the tail and being able to characterise it. ~54,000
symbol-days, ~5h, ~630 MB.

### Status at the end of the 2026-08-08 pass

| window | state |
|---|---|
| 2020-01-27..2021-12-31, MEGA 14 | DONE (6,909 symbol-days) |
| 2020-01-27..2021-12-31, other 107 | **the open one — see (A)** |
| 2022-2024 | complete, and upstream's own cadence is every-other-day |
| 2025-01..2026-06, 121 symbols | was running at 32,867 pairs; re-check with `--gaps` |

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

## Disk — RESOLVED 2026-08-08, was 99%, now 84%

It hit **99% / 3.3 GiB free** mid-pass and was the binding constraint on every
data option here. Reclaimed **~28 GiB** without touching `data/` (3.4 GB of
research data that `project_trust_and_polish_pass` says explicitly not to
"clean") and without touching the operator's wallpapers:

| freed | item |
|---:|---|
| 8.8 GB | UTM `Ubuntu.utm` VM, untouched since 2025-09-29 |
| 6.4 GB | npm cache (`npm cache clean --force`) |
| 5.6 GB | Parallels Windows 11 install ISO |
| 3.5 GB | 19 stale VS Code extension dirs across 13 extensions |
| 245 MB | `brew cleanup --prune=all` |

**Now at 31 GiB free**, which unblocks **optionsDX** (item 2 above).

Still blocked, and NOT on size: **FNSPID remains disqualified by its
CC BY-NC-4.0 licence** — commercial use is explicitly prohibited, which rules
it out for a system whose stated destination is real money. Do not re-litigate
that one on the grounds that the disk now has room.

Left on the table if more is ever needed: Fotor 4.4 GB, Downloads 4.1 GB,
Docker 3.2 GB (`docker system prune -a`), CoreSimulator 2.0 GB (needs full
Xcode for `simctl`), codex/puppeteer caches ~3.4 GB.

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
