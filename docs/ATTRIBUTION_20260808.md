# What Made the Profitable Trades Profitable — 2026-08-08

The question: across the real-marks backtest, which entry-time features
separated the winners from the losers, and can anything be changed on the back
of it?

**The answer is a negative result, and it is the important kind.** No feature
knowable at entry predicts outcome once accounting identities are removed. The
strongest correlations in the table are arithmetic. Nothing predicts the
disasters at all. One genuine specification defect was found and fixed, and it
moves no verdict.

Reproduce with:

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.alloc --attribute \
    --start 2020-01-27 --end 2024-12-31 --trials 14
```

---

## 1. The harness

`src/alloc/attribution.py`. Every trade now carries `features` — a snapshot of
what was knowable on its entry date (`engine._entry_features`), covering market
state from `SignalHistory` and the geometry of the legs **actually obtained**,
which is not the geometry requested.

Two features are new. `rv` is annualised realized vol of the observed spot
path, and `iv_minus_rv` is the variance risk premium as it could be seen on the
day. That second one is the whole theoretical case for buying or selling
premium, and it had never been measured in this engine.

`rv` had to handle two properties of this cache, both tested:

- **Spacing.** The data is every-other-trading-day before 2025 and daily after.
  A raw stdev of consecutive returns would read the denser era as lower vol
  purely because each step spans less time — the backfill itself would look
  like a regime change. Each return is scaled by the root of its own elapsed
  time.
- **Holes.** Consecutive rows can straddle a 21-month gap. Annualising that as
  one return fabricates a vol explosion, so steps over 10 days are dropped.

## 2. The measurement

MEGA cohort, 2020-01-27 to 2024-12-31, 14 symbols x 790 dates, held to expiry.
14 features examined — which is a **14-way search**, and `rank_features` carries
that count so every row can be deflated by it.

### long_call (n=180)

| feature | IC | t | t_clustered | p |
|---|---:|---:|---:|---:|
| long_delta | 0.2349 | 3.22 | **3.08** | 0.0015 |
| moneyness | -0.1434 | -1.93 | -2.01 | 0.0548 |
| capital_at_risk | 0.0852 | 1.14 | 0.79 | 0.2552 |
| rv | 0.0717 | 0.95 | -0.10 | 0.3442 |
| **iv_minus_rv** | **0.0214** | 0.28 | 0.86 | 0.7784 |
| **iv_rank** | **-0.0100** | -0.13 | -0.06 | 0.8956 |
| dte | -0.0051 | -0.07 | 0.19 | 0.9461 |

### bull_put (n=187)

| feature | IC | t_clustered | p |
|---|---:|---:|---:|
| credit_pct_width | 0.5572 | 7.38 | 0.0000 |
| friction_pct_credit | -0.3088 | -4.39 | 0.0000 |
| atm_iv | 0.2702 | 3.98 | 0.0002 |
| rv | 0.2531 | 4.08 | 0.0006 |
| moneyness | -0.2373 | -3.58 | 0.0011 |
| **iv_minus_rv** | **-0.0432** | -1.28 | 0.5627 |
| **trend** | **0.0388** | 0.52 | 0.6031 |
| **iv_rank** | **-0.0288** | -0.91 | 0.6991 |

## 3. Why the big numbers are not findings

**`credit_pct_width` is an accounting identity.** For a credit spread held to
expiry, `RoC = (credit - loss) / (width - credit)`. Among the 82% that expire
worthless, RoC is a deterministic increasing function of the credit received.
An IC of 0.56 there is arithmetic. It is also not actionable: you cannot choose
to receive more credit without selling closer to the money — which is exactly
why `moneyness` shows -0.24 in the same table. The two are the same fact.

`friction_pct_credit` is the same story with the sign flipped: friction is
subtracted from P&L, so of course it correlates.

**And they fail this repo's own shape test.** From
`ALLOCATION_BACKTEST_FINDINGS.md` §4d: "Monotonicity across a range is much
harder to produce by chance than one good cell." Neither is monotone:

```
bull_put, credit_pct_width by quartile:
  Q1 meanRoC -0.0113 | Q2 +0.0189 | Q3 +0.0900 | Q4 -0.0679   <- up then down
bull_put, friction_pct_credit by quartile:
  Q1 -0.0318 | Q2 +0.0861 | Q3 -0.0243 | Q4 +0.0012           <- no order at all
```

The `friction_pct_credit` top quartile also runs to **24.75 and 34.50** — i.e.
friction of 2,475% of the credit. Those are degenerate near-zero-credit trades,
not a signal.

## 4. What genuinely has no signal

This is the part worth keeping. Across all three structures, every feature that
is *not* an accounting identity is flat:

| feature | long_call | bull_put | bear_call |
|---|---:|---:|---:|
| **iv_minus_rv** (the VRP) | +0.021 | -0.043 | -0.090 |
| **iv_rank** | -0.010 | -0.029 | +0.006 |
| **trend** | +0.049 | +0.039 | -0.075 |
| **ret_4w** | -0.071 | +0.112 | -0.087 |
| **dte** | -0.005 | -0.010 | -0.062 |

**The variance risk premium at entry does not predict the outcome.** That is
the single most theory-motivated feature available — the reason anyone buys or
sells options — and it is indistinguishable from zero in all three structures.

**IV rank is flat here, which contradicts §4d.** That section reported a
monotone IV-rank effect (-0.97% at IVR<=30 rising to +3.00% at IVR>=70) and
called it one of two things worth carrying forward. It was measured on a
different window (2022-2026), **with splits unhandled**, and as a conditional
mean rather than an IC. It does not reproduce as a rank correlation on
2020-2024 with splits wired. Treat §4d's IV-rank result as unconfirmed.

## 5. Nothing predicts the disaster

For a short-premium book this is the question that matters, and it is not the
same question as "does this shift the mean". The losses are rare and total, so:
does anything knowable at entry flag the trades that lose more than half their
capital?

Mann-Whitney AUC, disaster vs the rest. 0.5 is a coin flip.

| bull_put (25 disasters / 187) | AUC | p |
|---|---:|---:|
| credit_pct_width | 0.636 | 0.0287 |
| rv | 0.604 | 0.0954 |
| atm_iv | 0.580 | 0.1977 |
| iv_minus_rv | 0.428 | 0.2457 |
| iv_rank | 0.538 | 0.5423 |
| trend | 0.460 | 0.5264 |

| bear_call (45 disasters / 188) | AUC | p |
|---|---:|---:|
| rv | 0.572 | 0.1485 |
| atm_iv | 0.566 | 0.1809 |
| iv_rank | 0.513 | 0.7887 |
| trend | 0.499 | 0.9883 |

**Nothing separates them.** The best cell across an 11-way search is AUC 0.636
at p=0.0287, which does not survive even a Bonferroni correction (0.05/11 =
0.0045) — and its sign is the wrong way round for a warning: the trades that
blew up had *higher* credit for their width (0.1619 vs 0.1404). You were paid
more precisely because the risk was real.

The magnitude is the headline. bull_put's 25 disasters are **13.4% of trades
and 1,577% of total absolute RoC**. The entire P&L is the tail, and the tail is
invisible at entry.

## 6. The one real defect: substituted strikes

`select_legs` promises in its own docstring that "a missing wing is a skipped
trade, never a substituted strike". The delta target had **no tolerance**:
`_nearest_delta` returned the closest listed contract however far away it was.
Asked for a 40-delta call against a chain whose nearest listed call was
2-delta, the engine bought the 2-delta lottery ticket and recorded it as a
40-delta trade.

Fixed: `DELTA_TOLERANCE = 0.10`, the conventional reading of "a 40-delta
option" (0.30-0.50), applied to every structure's delta target and deliberately
**not tuned against returns**. `strike_selection: random` still bypasses it,
because constraining a deliberate control arm to a band would make it not
random.

### How big was it? Much smaller than it first looked.

The `long_delta` quartile table showed Q1 spanning delta **0.0000** to 0.3740
with a mean RoC of **-0.5674**, which reads like substitution wrecking the arm.
A controlled A/B — identical window, tolerance the only difference — says
otherwise:

| arm | n | RoC | off-spec (>0.10 from target) | median &#124;delta-target&#124; |
|---|---:|---:|---:|---:|
| long_call, tolerance OFF | 180 | 2.62% | **2.8%** | 0.026 |
| long_call, tolerance ON | 181 | 4.23% | **0.0%** | 0.026 |
| bull_put OFF / ON | 187 / 187 | 0.78% / 0.78% | 0.0% | 0.016 |
| bear_call OFF / ON | 188 / 188 | -13.35% / -13.35% | 0.0% | 0.017 |
| iron_condor OFF / ON | 190 / 190 | -8.07% / -8.07% | 0.0% | 0.015 |

**The credit structures are completely unaffected — 0% of their short legs were
ever off-spec.** Only long_call had any, at 2.8%, and removing them changes
n by one and moves no verdict (both arms reject).

So: the guard is correct and worth having — a delta-0.000 "40-delta call" is
not the strategy, and nothing should record it as one — but it is a
correctness fix, **not** the explanation for the long-call quartile pattern.
That pattern is just the mechanical fact that higher-delta calls finish in the
money more often (win rate rose monotonically 20.0% -> 26.7% -> 33.3% ->
42.2%).

## 7. Changes made

| change | why | effect |
|---|---|---|
| `DELTA_TOLERANCE = 0.10` in `select_legs` | instrument integrity | 2.8% of long calls refused; credit structures unchanged |
| `rv` + `iv_minus_rv` in `SignalHistory` | the VRP had never been measured | both flat — a negative result |
| `Trade.features` + `attribution.py` + `--attribute` | attribution was not possible before | permanent capability |
| `SqliteChainSource` holds one connection | a window is ~10,000 chain reads, each opening a database | faster, and no longer locks out of a concurrent backfill |
| `READ_TIMEOUT_S = 120` on every cache reader | the standing workaround was "snapshot the DB first" | readers wait instead of erroring |

## 7b. §4c's friction result, re-derived with splits wired — it half holds

§4c's mega-vs-all comparison could not be reproduced at first: `max_concurrent`
caps open positions **portfolio-wide**, so across 117 names the engine opens
~36 trades/year regardless of universe width and the sample came out at n=105
rather than §4c's 10,363. That cap is right for an account-level return figure
and wrong for measuring a per-trade effect, so it is now a CLI argument
(`--max-concurrent`); `report.summarise` still applies its own capacity cap
independently, so the account-level line stays honest either way.

Re-run uncapped, 2022-01-07 to 2024-12-31, splits wired, deflating by 37:

| universe | structure | n | win | RoC/trade | t | t clustered | skew |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL 117 | bull_put | 6,917 | 76.5% | **-6.40%** | -13.86 | -4.55 | -1.67 |
| **MEGA 14** | **bull_put** | 3,808 | 79.8% | **-1.15%** | -1.69 | **-1.57** | -1.68 |
| ALL 117 | bear_call | 6,911 | 76.5% | -7.92% | -16.98 | -5.42 | -1.66 |
| **MEGA 14** | **bear_call** | 3,746 | 70.4% | **-12.54%** | -15.94 | -8.25 | -1.10 |
| ALL 117 | iron_condor | 6,751 | 73.0% | -8.37% | -18.04 | -7.17 | -1.53 |
| MEGA 14 | iron_condor | 3,404 | 70.4% | -7.87% | -10.09 | -6.13 | -1.21 |
| ALL 117 | long_call | 6,953 | 28.1% | -2.92% | -1.34 | -0.37 | 2.63 |
| MEGA 14 | long_call | 5,736 | 28.0% | -0.30% | -0.12 | 0.14 | 2.61 |

**The ALL-names bull put row replicates §4c almost exactly** (-6.40% vs -6.76%,
76.5% vs 76.8% win, skew -1.67 vs -1.66), which is a good sign the engine did
not drift under the splits fix.

**And the friction claim holds for bull puts:** restricting to tight-spread
names moves it from decisively negative (-6.40%, t=-13.86) to indistinguishable
from zero (-1.15%, clustered t=-1.57). Same conclusion §4c reached.

### But it does not generalise, and §4c implied that it did

§4c reported the bear call and condor as staying "clearly negative even there",
which reads as friction helping everywhere and merely not being enough. That is
not what a controlled comparison shows:

**The bear call is markedly WORSE on the tight-spread universe** — -12.54%
against -7.92% on the broad one. Tighter spreads cannot make a strategy worse,
so something else dominates: the MEGA cohort is SPY plus mega-cap tech over
2022-2024, and selling calls into that rally is a directional loss that swamps
any friction saving. The condor, which contains a short call, sits between the
two as expected.

So the honest statement is narrower than §4c's: **friction is the driver for
the put side; for the call side, the beta of the tight-spread universe is.**
Restricting to mega-caps is not a general improvement — it is a bullish tilt
that happens to help puts and hurt calls over this window.

Everything still rejects.

## 8. What would actually change the answer

Nothing in this dataset. The features that could plausibly carry signal are the
ones it does not have — open interest, volume, quote sizes, intraday path — and
the one measurement that would move every verdict is whether real fills land
near mid or at the touch. See `docs/TAIL_OBSERVED_20260808.md` §6.

Two things NOT to do on the back of this file:

1. **Do not add an IV-rank or VRP entry filter.** Both are flat here, and §4d's
   IV-rank result does not reproduce with splits wired.
2. **Do not chase `credit_pct_width`.** It is arithmetic, non-monotone, and
   selecting on it just means selling closer to the money.
