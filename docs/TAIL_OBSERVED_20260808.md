# The Tail, Observed — 2026-08-08

Every short-premium result in this repo has carried the same caveat: *the tail
is unobserved*. `docs/ALLOCATION_BACKTEST_FINDINGS.md` §4b put it plainly —
"an 84.9% win rate with skew -2.80 is not a good strategy; it is a strategy
whose losses have not arrived." The short-premium gate carries the same words.

The reason was data, not analysis. The local Dolt cache started **2022-01-03**,
and 2022-2026 contains no vol crash.

It does now.

---

## 1. Two years of free data were sitting unclaimed

The upstream DoltHub dataset (`post-no-preference/options`) begins
**2020-01-27** — three weeks before the COVID top. Probed directly:

```
SPY  2020-02-21 ->  66 rows      local: 0
SPY  2020-03-16 ->  66 rows      local: 0     <- VIX 82
SPY  2020-03-20 ->  78 rows      local: 0
LMT  2020-03-16 ->  66   NVDA 60   AAPL 60   XLV 62    <- full universe present
```

Local coverage before this pass: **2020 = 12 symbol-days, 2021 = 8**. The
2020-2021 window was essentially unfetched.

Two other things were confirmed while mapping the gap, both of which
constrain what can ever be asked of this source:

- **The every-other-day cadence in 2022-2024 is upstream's own**, not a
  backfill shortfall (`SPY 2024-03-05/07/12` return 0 rows from the API too).
  Not fixable.
- **2025-2026 upstream is daily**, and 113 of 121 symbols hold only ~65 of
  ~250 days locally. That gap *is* fixable and is still open.

## 2. What was fetched

`--mega` cohort (the 14 tight-spread names — the only stratum that ever
produced a non-negative result), 2020-01-27 to 2021-12-31:

```
6,909 symbol-days, 495,164 rows, 0 failed, 34.8 min
2020: 8 symbol-days -> 1,849        2021: 6 -> 2,111
```

Serial backfill would have taken 6.4 hours. See §5 for why it didn't.

## 3. The tail

Bull put spread, 25-delta, `dte [25,60]`, held to expiry, over 2020-01-27 to
2021-12-31. The trades opened into the crash:

| symbol | entry | exit | width | capital at risk | P&L |
|---|---|---|---:|---:|---:|
| MSFT | 2020-02-21 | 2020-03-20 | $5 | $430 | **-$430** |
| NVDA | 2020-02-21 | 2020-03-20 | $10 | $790 | **-$790** |
| AAPL | 2020-02-24 | 2020-03-20 | $5 | $435 | **-$435** |
| MSFT | 2020-03-20 | 2020-04-17 | $10 | $910 | +$90 |
| NVDA | 2020-03-20 | 2020-04-17 | $10 | $845 | +$155 |
| SPY | 2020-03-23 | 2020-04-20 | $10 | $805 | +$195 |

**Every position open into the crash lost 100% of its capital at risk.** P&L
equals risk exactly in all three cases — not a large loss, the *maximum*
loss, on every one. The 50% win rate in this window exists only because three
more were opened at the bottom.

That is the shape the skew of -2.8 was pointing at, now measured rather than
inferred.

### UPDATE, same day: n=3 becomes n=527

The 107 symbols that lacked 2020-21 data were backfilled. The crash window is
now complete — 121 symbols, 64 trading days, 253,135 rows — and the anecdote
above is confirmed as representative, not a fluke.

**Everything opened into the crash (2020-02-01 .. 03-20), 120 symbols:**

```
n = 527
mean RoC      -68.97%
median RoC   -100.00%      <- the MEDIAN trade lost everything
win rate        22.6%
TOTAL LOSSES  346 of 527 = 65.7%   (lost >=99% of capital at risk)
best           +33.33%
```

Across the whole crash-and-recovery window (2020-01-27 .. 06-30):
`n=1,772, win 65.3%, RoC -23.08%, t -19.66, clustered t -2.75, DSR 0.000`.

A 65% win rate and a median outcome of total loss, in the same sample. That is
the short-premium signature stated as plainly as this data can state it.

### Does a high IV-rank filter protect against this? NO — it selects INTO it

The IVR>=70 arm carries DSR 0.797 in-sample on 2022-2024 (`ATTRIBUTION` §4e),
and 2020-21 is a genuine holdout no configuration search has touched. So this
is the test that matters, and the answer has two halves.

**It does not avoid the crash.** Median IV rank of crash-window entries was
**93**, and **220 of 280 (79%)** carried IV rank >= 70. The filter would have
selected four out of five of these trades.

**But within the crash it was much less bad:**

| cohort | n | mean RoC | win | total-loss rate |
|---|---:|---:|---:|---:|
| IV rank <= 30 | 24 | -91.80% | 4.2% | 91.7% |
| IV rank 30-70 | 51 | -99.90% | 0.0% | 98.0% |
| **IV rank >= 70** | 205 | **-46.94%** | **42.9%** | **46.8%** |

Mechanically sensible: entering when IV is already elevated puts a 25-delta
strike much further from spot in dollar terms and collects far more credit.

**And the confound is unresolved.** High-IVR entries came a median of **9 days
later** (2020-03-02 vs 02-21), i.e. after ~8% of the drawdown had already
happened, so part of that gap is simply less remaining downside. Two attempts
to separate it both failed:

* Restricting to pre-crash entries (2020-02-01..02-19) yields **zero** trades
  with an IV rank — the signal needs 10 prior snapshots and the data begins
  2020-01-27, so February is excluded by construction.
* A within-day comparison (same entry date, so timing is fixed) finds only
  **3 days** with >=3 trades in both cohorts: mean advantage +15.68%, high IVR
  won 2 of 3, **paired t = 1.56**. Suggestive, and nowhere near conclusive.

**The reading that survives all of this:** IVR>=70 is a *return* filter with a
fat tail, **not a risk filter**. "Better" here means losing 47% instead of 92%.
Anything that sizes positions on the assumption that high IV rank is protective
would be sized on a misreading.

### The window as a whole

```
bull_put              n=75  win=89.3%  RoC=  4.48%  t= 1.22  tc= 1.74  skew=-2.81  DSR=0.226  [reject]
bear_call             n=75  win=60.0%  RoC=-21.74%  t=-3.76  tc=-3.71  skew=-0.78  DSR=0.000  [reject]
iron_condor           n=74  win=74.3%  RoC= -9.24%  t=-1.84  tc=-1.78  skew=-1.33  DSR=0.000  [reject]
long_call [CONTROL]   n=69  win=36.2%  RoC=  8.25%  t= 0.43  tc=-0.29  skew= 1.54  DSR=0.040  [reject]
```

Three things worth keeping:

1. **Bull put survives the crash in aggregate (+4.48%) and is still rejected**
   (DSR 0.226 against the 0.5 line). Surviving one tail is not evidence of an
   edge; n=75.
2. **Selling calls into 2020-2021 is catastrophic** (-21.7%, t=-3.76). The
   iron condor's loss is the call side. This is a regime effect, not a
   discovery.
3. **The known-negative control makes money here** (+8.25%, skew **+1.54** —
   the only positive skew in the table). 2020-2021 is the regime where long
   premium works, which is consistent with the standing
   "long-premium negative-EV *outside* low-VIX" result. It is a caution about
   reading any of these numbers as regime-free.

### Caveats specific to this window

- **The 2020 chains are the thinnest in the dataset**: ~66 rows/day against
  ~140 later, 3 expirations, ~11 strikes each. Of 78 entries attempted,
  61 were `skipped_missing` and 27 `skipped_no_legs`.
- **Strike sparsity widens the spreads.** A $5 width was requested; the median
  obtained was **$6**, mean **$7.35**, max **$17.50**. NVDA's -$790 is a
  correct max loss on a $10-wide spread, not an engine fault. Risk per trade
  in this window is systematically larger than the spec asks.
- **n=3 on the way down — SUPERSEDED, see the UPDATE above.** The 107-symbol
  backfill took this to n=527 the same day, and the anecdote held: median
  outcome -100%, 65.7% total losses.

## 4. Two live defects found on the way

### 4a. Splits were never passed to the engine

`src/alloc/splits.py` exists to stop the backtest reading a 20:1 split as a
95% crash. `replay()` accepts `splits=`. **Nothing ever passed it.** The only
production call site (`src/alloc/__main__.py`) passed `terminal=` and
`stratum_of=` and nothing else, so `splits` defaulted to `{}` and the guard
branch was dead code — `split_closed` was structurally always 0.

Every result in `ALLOCATION_BACKTEST_FINDINGS.md` was computed with splits
unhandled, against that file's own warning that doing so "corrupts two things
badly". Now wired; the COVID run reports `split_closed: 2`.

### 4b. Split detection read drift across data gaps as a corporate action

`detect_splits` compared consecutive *observed* rows regardless of the
calendar distance between them. With the cache jumping from 2020-03-20 to
2022-01-03, SPY's mean strike went 228.9 -> 465.7 and was reported as a split
— a 21-month doubling read as an overnight event. Eight of the fourteen
tight-spread names tripped it at that boundary.

Detection is now limited to observations within `MAX_GAP_DAYS = 7`. Validated
against ground truth:

| | before | after |
|---|---:|---:|
| symbols flagged | 20 | 15 |
| events | 25 | 17 |
| false positives at the 2022 boundary | 8 | **0** |
| false positives inside the crash window | — | **0** |

Every surviving event is a verifiable real split: AAPL 4:1, NVDA 4:1 and 10:1,
AMZN 20:1, GOOG 20:1, TSLA 3:1. **AAPL's date corrected from `2020-10-28` to
`2020-08-31`** — the true split date — because the backfill densified the
series enough to pin it. The fix and the data improve each other.

The tradeoff is explicit: a real split hidden inside a gap longer than 7 days
is now missed. Densifying the cache is what shrinks that risk.

## 5. The backfill is 11x faster

A DoltHub round trip measured **~3.3s** against a 0.30s throttle: 90% of the
wall clock was the socket, not politeness. `backfill_parallel()` runs N
concurrent fetches behind a *fleet-wide* pacing lock, so concurrency hides
latency without raising the offered request rate.

Measured: **0.31 pairs/s serial -> 3.31 pairs/s at `--workers 6`**, 6,909
consecutive fetches, **0 failures**. The run is now paced-bound rather than
latency-bound, so more workers buy nothing.

Two properties it holds, both tested:

- **A failed fetch is never cached as an empty day.** Caching a network error
  as "0 rows" would mark real data permanently absent and every later run
  would skip it. Failures are left unmarked and retried.
- **Rate limiting stops the run** rather than grinding the queue into
  failures. It resumes on the next invocation.

```bash
PY="PYTHONPATH=$PWD ~/.venvs/options/bin/python"
$PY -m src.dolt_options --gaps --mega --start 2020-01-27 --end 2021-12-31
$PY -m src.dolt_options --backfill --mega --workers 6 --start 2020-01-27 --end 2021-12-31
```

`--gaps` reports the unfetched grid by year and never touches the network.

## 6. What is still missing

Ranked by what it would change, with the free options first.

1. **2025-2026 daily fill** — 113 symbols x ~185 days, free, already wired,
   ~2.5h at `--workers 6`. Upstream is daily there; the cache is not.
2. **optionsDX** (free, registration, no billing) — SPY/SPX/QQQ/VIX/AAPL/NVDA/
   TSLA EOD, **2010-2023**, *all* strikes and *all* expirations, with
   `C_VOLUME` and `C_SIZE`. Covers 2011, 2015, 2018 Volmageddon and 2020 —
   four tails instead of one — and removes the ladder wall below.
3. **ThetaData free tier** — EOD `volume, count, bid, ask, bid_size, ask_size`
   from 2023-06 (quotes 2023-12), whole US universe. Tiles the 2024-2026
   window that optionsDX stops short of.
4. **Databento $125 free credits** — one surgical OPRA pull of SPY trades +
   NBBO to measure *effective* spread against quoted. The entire negative
   verdict in this repo rests on the assumption that entries cross the full
   spread. That assumption has never been measured.

### Walls this dataset cannot pass, at any effort

Measured on the cache, and unchanged by any backfill:

| property | value |
|---|---|
| expirations per symbol-day | mean **2.6** |
| strikes per (symbol, date, expiry) | mean **18** |
| **DTE range, entire dataset** | **10 to 67 days** |
| open interest | **absent — not a column upstream** |
| volume | **absent** (`vol` is implied vol) |
| bid/ask size | absent |
| snapshots per day | 1 (EOD) |

No 0DTE, no weeklies under 10 days, no LEAPS, no PMCC. Calendars and diagonals
are structurally untestable. There is no liquidity or capacity data of any
kind. Sources 2 and 3 above are the only free way past this.
