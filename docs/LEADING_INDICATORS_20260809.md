# Leading indicators for verdicts — where the remaining edge actually is

2026-08-09. A research pass over what this repo has already measured about
entry-time signal, what its own evidence says to change *today*, and which
untested indicators are worth the next measurement.

Companion to `ATTRIBUTION_20260808.md` (what predicts what),
`CONDOR_COMPOSITE_20260807.md` (component ICs) and `NEXT_STEPS_20260808.md`.

**Summary.** Three scoring changes are already justified by evidence in hand and
need no new research — the live config currently contradicts this repo's own
holdout. Beyond that, the search for *absolute level* signals is exhausted:
every one has been tested and killed. The four untested candidates that remain
are all **shape and rate-of-change** features, they all have a mechanism, and
all four are computable from `data/dolt_options.db` without a single API call.

---

## 1. Act now: the config still weights a feature your holdout killed

`ATTRIBUTION_20260808.md` §4f withdrew IV rank "as a scoring input" on
2026-08-08. It is still an input, in three places:

| where | value | what §4f measured |
|---|---|---|
| `config.credit_spread_weights.iv_rank` | **0.15** | q_spread +7.88% in-sample → **−12.50% on the holdout** — sign flips |
| `config.iron_condor_weights.iv_rank` | **0.12** | same |
| `config.filters.min_iv_percentile` | **25** | a hard entry filter on the same quantity |
| `config.vix_regime_multipliers.*.iv_rank` | 0.6 / 0.5 | rescales it by regime |

And from `TAIL_OBSERVED_20260808.md`: **79% of the trades that lost 100% of
capital at risk carried IVR ≥ 70.** High IV rank selects *into* a crash. The
in-sample DSR of 0.797 was a 44-configuration artifact.

So the credit-spread composite currently puts 15% of its weight on a feature
whose only out-of-sample measurement is negative, and which concentrates the
tail that wipes the book. Zeroing it in both weight tables is the single
best-evidenced change available, and it is a *deletion*, not a fit — no new
parameter, nothing to overfit.

`min_iv_percentile: 25` is a separate decision. It is a floor, not a preference
for high IVR, and dropping it widens the candidate set rather than tilting it.
Worth re-deriving, but it is not the same defect.

### 1b. The condor `pop` weight, still unactioned

`CONDOR_COMPOSITE_20260807.md` called this "the best-evidenced defect found in
this entire scoring review" and it is still shipped as-is:

- `pop` carries **0.30** — the largest single weight — at rank IC **−0.3115** (p 0.001)
- mechanism identified: `spearman(pop_score, net_credit) = −0.7197`. High PoP *is* a tiny credit.
- top-PoP quartile: $1.25 credit against a $4 width, **−37.8% median, 26.7% win**
- stable in **4 of 4** walk-forward windows, no sign flips
- **worsens** to −0.3437 when restated at the crossed credit — it survives the cost correction
- clears Bonferroni at 7 comparisons

0.60 of the condor weight sits on negative-IC components; 0.13 on the two that
work. That is not noise, it is a weighting close to reversed.

### 1c. The vertical composite has the same shape, milder

| component | weight | rank IC |
|---|---:|---:|
| pop | 0.25 | −0.0594 |
| iv_rank | 0.15 | −0.1187 |
| liquidity | 0.10 | −0.1395 |
| **sum on negative-IC** | **0.50** | |
| credit_to_width | 0.20 | +0.2315 |
| return_on_risk | 0.10 | +0.2316 |
| theta | 0.08 | +0.1270 |
| **sum on positive-IC** | **0.38** | |

Caveat that must travel with this table: `credit_to_width` is an accounting
identity for a held-to-expiry structure (§3 of `ATTRIBUTION_20260808.md`), so
its +0.23 is not a licence to select on it. The actionable half is the negative
0.50, not the positive 0.38.

### 1d. `width` is validated and wired to nothing

The one non-identity feature that survived the holdout, and its evidence got
*stronger* out of sample:

| | in-sample (2022-24) | holdout (2020-21) |
|---|---:|---:|
| q_spread | +20.75% | **+13.88%** |
| clustered t | −0.70 (p 0.58) | **+5.41 (p 0.0002)** |

Mechanism: credit scales with width, the two crossings do not. It is
independently the best-supported mechanism in the whole repo.

§4f is right that it does not belong in `quality_score` — width is a property of
a constructed spread, not of a contract in a chain. Its home is the structure
builder and `config.json`. Right now it has no home at all. That is the gap
between "measured and survived a holdout" and "changes what the system does."

---

## 2. The screen itself is the bottleneck, not the feature list

This is the most transferable lesson in the repo and it deserves to be
promoted from a footnote to the default method.

`width` reads **IC 0.0089, p = 0.58** — indistinguishable from nothing — while
separating its extreme quintiles by **20.75 points of RoC**. The screen that was
supposed to find signal was hiding a 20-point effect.

The reason is structural: short-premium returns have skew −1.7 to −2.0. A
feature whose value is **tail avoidance** changes the mean a long way and the
median rank ordering barely at all. Rank IC is close to blind to exactly the
kind of edge a short-premium book needs.

**So any future indicator search should report four things, not one:**

1. rank IC (keeps comparability with everything already measured)
2. **quantile spread** — mean RoC, bottom vs top quintile
3. **monotonicity across the graded parameter** — the shape that is hard to fake
4. **tail AUC** — Mann-Whitney, disaster vs the rest (§5's method, made first-class)

and then, before anything is believed:

5. **a pre-registered holdout.** IV rank is the cautionary tale: DSR 0.797
   in-sample, sign inversion out. No feature should reach `config.json` on
   in-sample evidence again.
6. **an honest trial count.** §4e declares 44 and admits that undercounts the
   true search. DSR falls as the count grows, so a single registry of every
   configuration tried across this effort would make every future DSR mean
   something. Nothing has one today.

Point 5 is about to get much cheaper: the running COVID backfill takes the
2020-21 holdout from 14 symbols to 121.

### 2b. Everything tested so far has been a marginal

Every feature in every table here is one-dimensional. No interaction has ever
been screened, and the two most obvious ones are motivated by results already in
hand:

- **IVR × ΔIV.** IV rank is a *level*, and §4f showed the level selects into
  crashes. "High IV rank and falling" is the textbook short-premium entry;
  "high IV rank and rising" is a crash in progress. The same level, opposite
  trades. This single interaction is the most plausible reconciliation of §4e
  (monotone and positive in-sample) with §4f (sign flips on a window containing
  a crash) — and it would mean IV rank was never wrong, just under-specified.
- **width × friction**, the surviving mechanism against the thing it is a proxy for.

---

## 3. Four untested indicators, with mechanism and measured feasibility

The features tested so far — `iv_rank`, `atm_iv`, `rv`, `iv_minus_rv`, `trend`,
`ret_4w`, `dte`, plus leg geometry — share a property: they are all **levels of
one name at one moment**. The four below are shapes, rates of change, or
cross-sectional, which is why they are not redundant with a search that came
back empty.

`dolt_chain` carries `symbol, date, expiration, strike, type, bid, ask, mid, iv,
delta, gamma, theta, vega, rho` over 11.5M rows. No OI, no volume — that wall is
real and unchanged. But it is enough for all four.

### 3.1 Term-structure slope — the best candidate

Near-dated ATM IV minus far-dated ATM IV. Backwardation means the market is
pricing near-term stress.

**Why it is different from everything that failed.** IV rank is a level against
a name's own history and is *coincident* with stress. Term structure is a
shape, and it inverts as stress arrives. It is the best-documented vol-timing
signal in the public literature, and it is the natural answer to §5's "nothing
predicts the disaster" — a finding measured entirely on levels.

**Feasibility, measured today:** SPY has **1,147 symbol-days, median 3
expirations per day, 100% with ≥ 2.** Fully computable.

**Real limit:** the cache is DTE 10-67, so this is a short-dated slope only —
roughly 10d against 60d. That happens to be the segment that inverts first, so
the constraint is survivable, but it is not the 1M/3M slope the literature
usually means, and the writeup must say so.

### 3.2 Skew / risk reversal

25Δ put IV minus 25Δ call IV, and separately the short strike's own IV against
ATM.

**Why it matters specifically for the short side.** Selling a bull put *is*
selling the put wing. How rich that wing is relative to ATM is literally the
price of the thing being sold, and no feature tested so far captures it —
`atm_iv` is measured at the money, `iv_rank` is a time-series rank of that same
ATM number. Skew is the missing cross-strike dimension. Skew *steepening* is
also a stress leading indicator in its own right.

**Feasibility, measured today:** across the whole universe there are **325,088
symbol-day-expiries, of which 85.4% (277,670) carry both a ~25Δ put and a ~25Δ
call.** That is a very large sample by this repo's standards.

### 3.3 IV velocity and vol-of-vol

Δ`atm_iv` over 5-10 days, and the trailing stdev of `atm_iv`.

The level failed. The derivative has never been looked at, and §4f's failure
mode — high IVR selecting into the crash — is precisely what a direction term
would separate. Cheapest of the four to compute; `SignalHistory` already keeps
the window that `iv_rank` is built from, so this is a handful of lines in
`src/alloc/signals.py` beside the existing `rv` / `iv_minus_rv`.

### 3.4 Correlation / dispersion regime — the answer to §5

§5 asked whether anything at entry flags the trades that lose more than half
their capital, tested eleven **per-name** features, and found nothing (best cell
AUC 0.636, failing Bonferroni, with its sign the wrong way round).

That is the expected result. A crash is a **systemic** event — every short put
in the book loses together — and no property of one ticker can flag it. The
matching feature has to be a property of the market:

- median pairwise correlation of the universe over a trailing window
- share of the 121 names with rising ATM IV
- cross-sectional dispersion of `iv_minus_rv`

This is the highest-value one *if* it works, because the tail is where the
entire P&L lives: bull_put's 25 disasters were 13.4% of trades and **1,577% of
total absolute RoC**. It is also the hardest to power — systemic events are
rare, so n is small in the only dimension that counts, and 2020-21 is one
event, not a sample of them.

### 3.5 A transform worth trying on all of the above

The one place in this repo where something worked was the outlook engine, whose
real edge is **relative, IC +0.05-0.08**. Every feature here is currently
absolute. Ranking a feature **cross-sectionally across names on the same date**
removes the market-wide component that makes same-day positions co-move — which
is also exactly what the clustered-t correction penalises, and clustered t is
what killed IVR ≥ 70 (2.26 against the 3.0 hurdle). A feature that is flat in
absolute terms can rank cross-sectionally. It costs one groupby.

---

## 4. The call side has never been holdout-tested at all

The user's question covered calls as well as shorts, and the asymmetry in the
evidence is worth stating plainly.

For `long_call`, `long_delta` reads **IC +0.2349, clustered t 3.08** — one of
very few things in this repo to clear Harvey's 3.0 hurdle — with win rate rising
monotonically across quartiles, **20.0% → 26.7% → 33.3% → 42.2%**. The mechanism
is mechanical and unmysterious: higher-delta calls finish in the money more
often.

But: that is **in-sample on 2022-2024**, a window that is a near-uninterrupted
mega-cap rally. Delta is beta, so "higher delta did better" and "the market went
up" are the same sentence over that window. It is not a signal until it is
measured somewhere the market did not go up.

**2020-21 is that window, and it contains both a crash and a melt-up.** The
running backfill is what makes it testable, and this is the call-side test that
has never been run.

---

## 5. Do not bother with

- **Re-fitting `quality_score` weights to the ledger.** The ledger's own headline
  verdict is "NO SIGNIFICANT EDGE (IC = −0.03, p = 0.433)" and its top score
  quintile is the worst cell in it. Fitting weights on a sample where the
  composite does not rank is fitting to noise. §1's changes are *deletions* of
  components measured negative, which is a different act.
- **Anything selected on `credit_pct_width` or `friction_pct_credit`.** Accounting
  identities, non-monotone by quartile.
- **A news-sentiment weight.** IC ~0 at every horizon, large effect powered out.
- **More features before the friction measurement.** Every negative result here
  assumes an entry crosses the full quoted spread, and §2 of
  `ALLOCATION_BACKTEST_FINDINGS.md` puts that toll at half the credit. A signal
  search under a wrong cost model can invert. Databento's $125 signup credits
  and one surgical OPRA pull still answers it, and it is still the highest-value
  action available at any price.

---

## 6. Ranked

| # | action | cost | evidence behind it |
|---|---|---|---|
| 1 | Zero `iv_rank` in both weight tables | minutes | holdout sign flip + 79% of total-loss trades |
| 2 | Cut condor `pop` from 0.30 | minutes | IC −0.31, mechanism −0.72, 4/4 windows, survives cost restatement |
| 3 | Give `width` a home in the structure builder | small | only non-identity feature to survive the holdout, t improves to +5.41 |
| 4 | Make the 4-part screen + holdout the default in `--attribute` | medium | width: IC 0.0089 p=0.58 vs q_spread +20.75% |
| 5 | Add term-structure slope and skew to `SignalHistory` | medium | untested, mechanism, 277,670 usable slices measured |
| 6 | IV velocity, and the IVR × ΔIV interaction | small | reconciles §4e with §4f |
| 7 | Re-run the call side on the 2020-21 holdout | small once backfill lands | long_delta t 3.08 is in-sample in a rally |
| 8 | Correlation-regime feature vs the disaster set | medium | §5 tested only per-name features |
| 9 | Measure the effective spread (Databento) | $0, one pull | conditions every result in this repo |

1-3 need no new measurement. 4 makes every later measurement worth more. 5-8 are
the actual research queue, and none of them needs a byte of new data.
