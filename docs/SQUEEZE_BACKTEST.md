# Squeeze grader — point-in-time backtest

Does an `assess_squeeze` **SETUP** grade actually raise the odds of a right-tail
up-move, or does it just select volatile, heavily-shorted junk?

`src/squeeze/detector.py` has always been display-only — it never touches
`quality_score` and never suppresses a pick. This harness is the evidence layer
for it. It grades history with the *production* detector (imported, never
copied) and measures what happened next.

```bash
python -m src.squeeze.backtest.finra  --discover --backfill   # FINRA short interest
python -m src.squeeze.backtest.prices --load-csv --all-symbols --start 2017-10-01
python -m src.squeeze.backtest.shares --backfill              # EDGAR shares outstanding
python -m src.squeeze.backtest --run                          # the study
```

## Data

| Input | Source | Coverage |
|---|---|---|
| shares short, days-to-cover, prior-month | FINRA consolidated short interest (free, no auth) | 2018-01 → present, ~205 settlement dates, 16–20k names each |
| daily close / volume | DoltHub `post-no-preference/stocks` CSV export | 2011 → present, ~12k names/day |
| shares outstanding | SEC EDGAR XBRL `dei:EntityCommonStockSharesOutstanding` | per-filing history, includes delisted filers |

Both the universe and the prices are **survivorship-free**. The universe is
whatever FINRA reported on each settlement date, not a present-day screen, and
the DoltHub table still prices names that later delisted — BBBY, MULN, NKLA are
all there. Using today's Finviz high-short-float screen instead would be badly
look-ahead biased: it can only contain companies that survived.

## Method

**Entry is delayed 8 trading days.** FINRA disseminates a settlement date's short
interest roughly eight business days later. Entering on the settlement date would
trade information nobody could see yet, and that gap is exactly when a squeeze
can begin — so the look-ahead would flatter the result precisely where it
matters. Every price feature is measured at the entry bar, not the settlement bar.

**The outcome is a path maximum, σ-normalised.** A squeeze that spikes 40% and
gives it all back is a winning trade for a call holder and invisible to an
endpoint return, so the measure is the highest close reached over the horizon.
It is divided by the name's own trailing 60-day realised vol scaled to the
horizon, because a 20% move in a 100%-vol microcap is unremarkable and the same
move in a 30%-vol name is not. Without that, the test would mostly rediscover
that shorted stocks are volatile.

**Asymmetry is the headline, not the raw up-tail.** σ-normalisation fixes scale
but not shape. Junkier names reach ±2σ more often in *both* directions, so a raw
up-tail lift can be pure kurtosis. Only an excess of upside over downside is
evidence that trapped shorts push the right side of the distribution out
further. Both are reported; the asymmetry is the one that answers the question.

**Inference is a moving-block bootstrap over settlement dates.** Two dependencies
would otherwise fake significance: every name on a date shares market beta, and
neighbouring dates observe overlapping futures (dates are ~11 trading days apart,
horizons run to 42). Draws therefore take contiguous blocks of dates, never
individual rows. Treating ~10⁵ overlapping observations as independent would
shrink the standard error by more than an order of magnitude.

**Robustness cuts.** The sample contains the January 2021 meme episode, which
could be the entire effect on its own. Every headline is re-reported excluding
that window, and split 2018–2022 / 2023–2026.

## Results (2026-07-28, 205 settlement dates, 480,744 graded observations)

**There is real signal, and it is in the short-interest level — not in the
grader's scoring.**

SETUP vs NONE, 21 trading days, float assumed at 80% of shares outstanding:

| | P(+2σ) | P(−2σ) | asymmetry | P(+20%) | median end |
|---|---|---|---|---|---|
| SETUP | 7.9% | 4.2% | +3.7pp | 24.2% | −0.8% |
| WATCH | 7.3% | 3.7% | +3.6pp | 23.1% | −0.5% |
| NONE | 5.8% | 4.6% | +1.2pp | 11.0% | +0.4% |

The asymmetry lift is +2.57pp (95% CI [+1.38, +3.48]) — the up-tail rises
*without* the down-tail rising, so this is not fat-tailedness. It survives every
cut: excluding the 2021 meme window (+2.15pp), 2018–2022 train (+1.88pp) and
2023–2026 holdout (+2.36pp), and every float assumption from 1.0 to 1.5.

Note the shape: SETUP names have a **fatter right tail and a worse median**
(−0.8% vs +0.4%). That is the short-interest anomaly and the squeeze thesis
coexisting, and it is exactly the payoff a long call wants and a share position
does not.

**The scoring adds nothing over its own gate.** Ranking by short interest alone,
taking the same number of names, matches or beats the full grader:

| cohort | P(+2σ) | asymmetry | P(+20%) |
|---|---|---|---|
| grader SETUP | 7.93% | +3.74pp | 24.2% |
| short interest level only | 7.96% | **+4.36pp** | **26.6%** |

The two cohorts overlap 77%. Spearman between evidence points and the normalised
move is +0.001 (CI [−0.047, +0.039]) — indistinguishable from zero. Short
interest deciles, by contrast, are monotone: P(+20%) climbs 9.0% → 23.1% from
D1 to D10.

**Two scored factors are backwards.** Within the top 5% by short interest,
bootstrapped effect on asymmetry:

| factor | grader | effect | 95% CI |
|---|---|---|---|
| days-to-cover ≥ 5 | **+2 pts** | **−2.38pp** | [−4.82, −0.75] |
| 5-day return ≥ +10% | *not scored* | **+3.31pp** | [+1.31, +5.77] |
| SI rising MoM | +1 pt | +1.30pp | [−0.01, +2.70] |
| RVOL > 1.5 | +1 pt | −1.39pp | [−5.46, +1.77] |
| 5-day return ≤ −10% | +1 pt | −1.96pp | [−8.53, +2.30] |

Days-to-cover is the grader's largest single bonus and it significantly *hurts*.
Upward momentum is significantly helpful and is not scored at all, while the
rule that rewards a sharp drop points the wrong way — squeezes follow strength,
not weakness. Only "SI rising" is directionally right, and it is borderline.

Because the `ret_5d` rule is dead (see below), the mismatch costs almost nothing
in aggregate: re-running with it live moves the lift by +0.06pp. The bug is
accidentally protective.

**What the sharp end looks like.** Forward 42 trading days, by short-interest
rank (SI as a share of assumed float):

| bucket | SI threshold | names/date | P(+20%) | P(+30%) | P(+50%) | median max | median end |
|---|---|---|---|---|---|---|---|
| top 1% | ≥32% | 24 | 41.7% | 29.9% | 17.1% | +15.0% | −3.2% |
| top 5% | ≥19% | 119 | 39.0% | 26.4% | 13.8% | +13.9% | −1.8% |
| top 10% | | 237 | 36.7% | 24.0% | 11.8% | +13.2% | −1.2% |
| all graded | | 2,373 | 22.5% | 12.2% | 4.8% | +9.0% | +0.5% |
| **top 5% SI and 5d return ≥ +10%** | | **16** | **50.5%** | **37.6%** | **21.8%** | **+20.5%** | −1.9% |

The best simple rule found — heavy short interest *plus upward momentum* — hits
+20% within 42 days **half the time**, against a 22.5% base rate, and +50% about
one time in five against 4.8%. It fires on ~16 names per settlement date. Note
its median endpoint return is still negative: this is a right-tail trade, not a
directional one.

## Known limitations

- **`iv_skew` cannot be reconstructed.** None of the high-short-interest names
  have option history in the DoltHub chains (checked: NBIS, SMCI, LCID, CVNA,
  UPST, IONQ, RKLB, SOUN, GME, AMC, BBBY — all empty; the dataset is mega-cap
  only). That is one of the grader's points, and it is passed as None rather
  than faked.
- **The option trade itself is untestable.** For the same reason, there are no
  historical bid/ask marks for these names, so this measures the *underlying's*
  move. A fat tail is necessary for a long-call squeeze trade to work, not
  sufficient — it still has to beat the IV you pay, and high-SI names carry
  expensive vol.
- **Shares outstanding is not float.** It includes insider and restricted stock,
  so the raw ratio understates short interest as a share of float. `--si-scale`
  states the assumed float fraction (1.0 = float is all shares out, 1.25 = 80%,
  1.5 = 67%) and is swept, so no conclusion rests on one threshold.
- **EDGAR coverage is partial.** `company_tickers.json` is a present-day snapshot,
  so ETFs (correctly, not squeeze candidates) and some delisted operating
  companies have no CIK. Rather than assume that is harmless, the report includes
  a coverage check: dropped rows still have prices, so their forward outcomes are
  compared against kept rows directly. A small gap means the join is roughly
  random with respect to outcome.
- **σ-normalisation is mildly conservative for SETUP.** Points are awarded for
  hot RVOL and (as intended) a sharp 5-day drop, both of which raise trailing
  vol and therefore raise the 2σ bar. The asymmetry measure is unaffected, since
  σ scales both tails equally.

## Bug found while building this

`src/squeeze/detector.py` compares `ret_5d` to `LATE_SHORT_RET5D = -10.0` on a
percent scale, but the pipeline stores `ret_5d` as a **fraction**
(`data_fetching.py:1475` → `close[-1]/close[-6] - 1.0`). A stock down 15% in a
week arrives as `-0.15`, never `-15.0`, so the "late shorts pressing" point can
never fire in the live grader — it would need a -1000% week. The study reports
both `--ret5d as_written` (reproducing live behaviour) and `--ret5d as_intended`,
which quantifies what the mismatch costs.
