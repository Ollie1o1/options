# Sector / Asset Outlook Engine — Findings

**Date:** 2026-06-08
**Code:** `src/outlook/` (factors, engine, backtest), `tests/outlook/` (21 tests)
**Run:** `python -m src.outlook.backtest` (live) · `--backtest` (validation)

A cross-sectional, regime-aware ranking of sector/asset ETFs (most bullish →
most bearish) over a ~1–3 month horizon, with explainable drivers and an honest,
measured track record. Free data (yfinance). Information tool only — no trading.

## Factors (all with academic support at 1–3 months)
12-1 momentum (Jegadeesh-Titman), trend vs 200d (time-series momentum),
1-month reversal (contrarian), relative strength vs SPY. Each z-scored
cross-sectionally and blended (weights in `config.json → outlook`). A **regime
gate** suppresses absolute BEARISH calls when the broad market is in an uptrend.

## The central finding — directional accuracy is asymmetric

Backtest: 16 instruments, monthly rebalance, ~8 years aligned history.

| horizon | overall | **bullish** | **bearish** | rel-hit | IC | base-rate up |
|---|---|---|---|---|---|---|
| 2mo (42d) | 60.0% | 66.5% | 30.1% | 51.4% | +0.053 | 65.1% |
| 3mo (63d) | 64.6% | **71.6%** | 29.9% | 49.2% | **+0.076** | 68.3% |

**What this means, honestly:**
- **Bullish/long calls are reliable: ~66–72% right.** Useful.
- **Bearish/absolute-down calls are NOT — ~30%, and no tuning fixed it.** Equity
  markets drift up and recover fast from below-200d levels, so "down over the
  next 2–3 months" is a sub-coin-flip bet outside sustained bear markets. The
  regime gate makes the tool *say it rarely* (and only in downtrends) rather than
  pretend skill it doesn't have.
- **The genuine, both-directions edge is RELATIVE:** IC +0.05…+0.08 — modest but
  legitimate (publishable-grade). Use it to **overweight the top / underweight the
  bottom**, not to time absolute market direction.
- Headline "hit rate" is inflated by market drift; an absolute-trend overlay
  raised it to ~60% but *hurt* the real relative skill (rel-hit 53%→47%), so it's
  off by default.

## Product framing
Output uses **FAVOR / NEUTRAL / AVOID** (overweight / neutral / underweight).
"AVOID" is explicitly relative weakness, **not** a high-confidence short. In an
uptrend the tool favors leaders and shorts nothing; in a confirmed downtrend the
gate opens and genuine bearish calls can appear.

## Limits / honest caveats
- Skill is modest (IC ~0.06) — a tilt, not a crystal ball.
- News/analyst-revision factors (free, available) are not yet in the backtest
  (no historical sentiment); they can enhance the *live* signal but are unproven
  here.
- ~8y aligned history (XLC/XLRE start 2018) is mostly a bull regime, so the
  bearish side is under-tested by construction — another reason to treat shorts
  with skepticism.

## reversal_1m weight — tested 2026-07-28, no change warranted

The 1-month reversal factor is negated by construction, so a sharp selloff
contributes *positively* to a bullish score. On 2026-07-28 that meant SMH read
+0.130 on the factor while down 13% over the month, which prompted the question
of whether the 0.15 weight is miscalibrated.

Four weight settings were compared on identical aligned price columns (one
fetch, so the comparison is like-for-like), with a paired bootstrap over
per-rebalance ICs — 5,000 resamples of the per-date IC difference.

| horizon | variant | ΔIC vs current | 95% CI | verdict |
|---|---|---:|---|---|
| 42d | removed (renormalised) | +0.0088 | [−0.0100, +0.0276] | indistinguishable |
| 42d | sign flipped (= 1m momentum) | +0.0166 | [−0.0142, +0.0487] | indistinguishable |
| 42d | half weight | +0.0029 | [−0.0081, +0.0139] | indistinguishable |
| 63d | removed (renormalised) | −0.0034 | [−0.0222, +0.0163] | indistinguishable |
| 63d | sign flipped | −0.0061 | [−0.0367, +0.0242] | indistinguishable |
| 63d | half weight | −0.0012 | [−0.0126, +0.0101] | indistinguishable |

Every interval spans zero, and the two horizons disagree on sign — the 42d
point estimates favour removing or flipping it, the 63d estimates favour
keeping it. With n≈83 rebalances the study cannot resolve IC differences of
~0.01, which is the size of every effect here.

**Verdict: leave the weight at 0.15.** There is no evidence it is wrong, and
changing it on these numbers would be fitting noise. The factor's odd live
reading is a property of short-horizon reversal, not a demonstrated defect.
Revisit only with materially more rebalances, not with a different opinion.
