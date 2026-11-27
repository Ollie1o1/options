# Result — Intraday I1 (late-day fade) and S1–S7 (open / premarket), index ETFs, shares

**Run:** 2026-11-27 · **Registration:** `docs/PREREG_INTRADAY_LATE_DAY_AND_OPEN_20261127.md`
(frozen `2467952`, amendment A1 `b29fec0`, both before any return was computed).
**Estimator:** `python -m scripts.intraday_tests` · raw output
`reports/intraday/intraday_tests_20261127T132330.json` (local, gitignored; every number is reproduced below).
**Data:** Alpaca SIP 1-minute bars, SPY/QQQ/IWM/DIA, 2016-01-04 .. 2026-11-25, **2,720 full
sessions** (22 early closes removed via the exchange calendar). 1¢/share each way after 10:00,
2¢ before; moving-block bootstrap (10-day blocks, 10,000 draws, seed 20261127).

## Verdict

**I1 FAIL — and the sign is the opposite of the hypothesis.** Fading the 10:30→15:30 move from
15:30 to the close **lost** money on 2016–2023: **−2.60 bps/day, 95% CI [−4.00, −1.27]**,
n = 1,997 days, hit 43.7%, negative in **8 of 8 years** and in both halves (−2.56 / −2.64).
Gross −1.67 bps/day; cost ≈ 0.93 bps/day. In 2016–2023 the midday move **continued** into the
close.

On the SEEN window (2024+, where the idea came from) the fade is positive: +1.33 bps/day
[−0.09, +2.82], positive in 2024, 2025 and 2026. The Alpaca-minute estimate reproduces the
independent Yahoo-hourly probe (+1.3 vs +1.6..+2.1), so the flip is in the data, not the code.

## Secondaries (binding 2016–2023, Holm across 10, no authority)

| id | rule | mean bps/day | 95% CI | p_holm | 2024+ |
|---|---|---:|---|---:|---:|
| S1 SPY | fade, SPY only | −2.24 | [−3.97, −0.63] | 0.062 | +1.29 |
| S1 QQQ | fade, QQQ only | −2.97 | [−4.76, −1.32] | **0.009** | +1.50 |
| S1 IWM | fade, IWM only | −2.20 | [−3.87, −0.45] | 0.076 | +1.25 |
| S1 DIA | fade, DIA only | −2.92 | [−4.75, −1.43] | **0.010** | +1.25 |
| S2 | fade, big-move days | −7.94 | [−14.79, −0.86] | 0.106 | +7.96 |
| S3 | fade at 15:45 | −2.44 | [−3.55, −1.39] | **0.001** | +0.81 |
| S4 | Gao intraday momentum (prev close→10:00 predicts last 30 min) | −1.56 | [−2.80, −0.17] | 0.106 | −1.05 |
| S5 | gap > 0.5% continuation, 09:35→close | +2.82 | [−3.30, +9.51] | 0.763 | −7.90 |
| S6 | premarket direction into open, 09:31→10:00 | −2.20 | [−4.56, −0.01] | 0.174 | −3.15 |
| S7 | opening-range (first 30 min) continuation | +0.27 | [−3.10, +3.70] | 0.874 | +1.96 |

The three Holm survivors are **the same negative fade result** (per ticker, and at 15:45), not
new edges. Per the registration, a negative result is not turned into the opposite rule.

- **Premarket / gaps:** nothing. Gap continuation (S5) is noise and flips sign in 2024+.
  Premarket direction tends to *reverse* in the first 30 minutes (S6 negative in both halves
  and 2024+), but at ~2 bps against ~2 bps of opening-spread cost the fade nets ≈ 0.
- **Classic intraday momentum (S4)** is absent in 2016–2023 as well as 2024+.

## Exploratory follow-up (no authority)

`python -m scripts.intraday_regime_explore`:

- Gross fade by half-year: strongly negative in **2018 H1/H2 (−4.2/−6.0), 2020 H1 (−4.3),
  2022 H1 (−7.4)** — late-day momentum is concentrated in stressed markets — and positive in
  4 of 6 half-years since 2024.
- **An adaptive rule** (fade if the trailing-N-day gross fade is positive, else follow, decided on
  prior days only): N=60 +0.56 [−0.51, +1.69], N=120 −0.16 [−1.22, +0.97], N=250 +0.63
  [−0.57, +1.85] bps/day net. Unstable in N and every CI spans zero. **Not tradable in shares.**
- Scale: the 15:30→16:00 move averages 19 bps (median 12); the predictable part is ≈ 1.5–2 bps.

## What this means

1. **Last-30-minute and premarket timing rules on index ETFs: CLOSED for shares.** The late-day
   pattern is real and large in t-stat terms, but it **changed sign around 2024**, so a rule
   fitted on either period lost in the other, and the adaptive version does not clear ≈ 0.8 bps of
   cost.
2. **For 0DTE options it is not yet answerable.** The live question would be a regime-conditional
   one (e.g. late-day momentum in high-volatility regimes, cf. dealer-hedging accounts of
   intraday momentum). That is a new hypothesis, chosen after seeing this data, and it needs
   (a) its own registration with a forward window and (b) intraday option quotes (ThetaData /
   Databento) to price the option leg. Nothing here supports trading 0DTE now.
3. The downloader stays useful: `python -m src.intraday.alpaca_bars --symbols … --start YYYY-MM`
   extends any universe; months already on disk are skipped.

## Not done (registered sensitivity)

The closing-auction exit price: the downloader keeps bars starting before 16:00 only, so the
16:00 auction print was not stored. The primary uses the 15:59 bar close as registered.
