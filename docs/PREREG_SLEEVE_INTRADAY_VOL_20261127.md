# Preregistration — V1: does intraday data forecast the sleeve's volatility better than RV20?

**Registered:** 2026-11-27, **before** any 5-minute bar for the sleeve universe was downloaded
(`data/intraday_5m/` did not exist at registration).
**Implements:** `scripts/sleeve_vol_forecast_test.py` on `src/intraday/realized.py`
(tests: `tests/intraday/test_realized.py`).
**Concerns:** the VALIDATED core sleeve `VM_OUTLOOK_TOP3`
(`docs/PREREG_VOL_MANAGED_OUTLOOK_20261124.md`, live in `src/outlook/vm_rotation.py`).

## Why

The sleeve scales its top-3 basket by `clip(0.14 / σ̂, 0.25, 1.00)` once a month, with
σ̂ = annualised 20-day close-to-close realised vol of the basket (RV20, 20 daily returns).
That is the sleeve's only estimated input that higher-frequency data can improve: intraday
realised variance uses ~78 returns a day instead of 1, and HAR forecasts (Corsi 2009) built on
it are the standard improvement in the volatility-forecasting literature. Nothing here
changes the ranking, the universe or the rebalance calendar.

## Population (frozen)

- **Rebalance grid:** exactly the extended dynamic run of
  `scripts/backtest_vol_managed_outlook.py` (`_align_matrix(.., strict_common=False,
  min_date="2005-01-01")`, `STEP = 21`, same eligibility rules). The grid is not re-chosen.
- **Basket at t:** Arm A of that run (top 3 with positive composite, 1/3 each), weights
  normalised to 1 for vol estimation. Months with an empty basket are skipped.
- **Holding period:** sessions `exec_in + 1 .. exec_out` of that run.
- **Window:** every grid rebalance t on or after the first date on which E1 below is defined
  (≈ 2016-02) whose holding period ends by the last date in `data/equity_ohlcv.db`.
- **MIN_N = 100 rebalance months** (per comparison).

## Estimators of the basket's annualised volatility at t (data through t's close only)

| id | estimator | role |
|---|---|---|
| **E0** | RV20 close-to-close, `_trailing_basket_rv` unchanged (adjusted daily closes) | incumbent |
| **E1** | **Intraday RV20:** basket 5-minute realised variance over the 20 sessions ending t | claimant |
| **E2** | **HAR-RV forecast** on basket intraday RV (pooled coefficients, expanding window) | claimant |
| E3 | EWMA (λ = 0.94) of basket daily close-to-close returns, 60-day warm-up | diagnostic only |

**E1 detail.** Source: Alpaca SIP **5-minute** bars, `adjustment=all` (split + dividend
adjusted, so overnight returns carry no ex-dividend drop), regular session only. For each
full session (Alpaca calendar close = 16:00) take the closes of the 78 bars starting
09:30 … 15:55. A ticker-session is valid iff ≥ 75 of the 78 bars exist (missing closes
forward-filled within the day) and the previous full session's 15:55-bar close exists.
Interval returns: the first is previous 15:55-bar close → today's 09:30-bar close (it
absorbs the overnight gap), then 77 consecutive 5-minute returns. Basket interval return =
Σ wᵢ rᵢ; basket daily RV = Σ (basket interval return)². E1 = √(252 · mean daily RV) over the
sessions among the last 20 full sessions ending t on which all basket tickers are valid;
undefined if fewer than 18.

**E2 detail.** Per-ETF daily RV series (same construction, single ticker). HAR features at
session s: `RV_d(s)`, `RV_w(s)` = mean over s−4..s, `RV_m(s)` = mean over s−21..s; target
`y(s)` = mean RV over s+1..s+21. At each t, OLS of y on [1, RV_d, RV_w, RV_m] pooled over all
16 ETFs, using only pairs with s + 21 ≤ t (no leakage); undefined until ≥ 1,000 pairs. Forecast
for the basket = β applied to the **basket's** own RV_d / RV_w / RV_m at t (basket RV series
built with the month's weights), floored at 0.25 × basket RV_m; E2 = √(252 · forecast).

## Outcome and loss (frozen)

- **Realised proxy** for the holding period: `σ²_real = (252/n) Σ r_d²` over the basket's
  daily close-to-close returns (adjusted closes, same weights, n = holding sessions). This
  proxy uses neither estimator's data source and is unbiased for variance.
- **Loss:** QLIKE on variance, `L = σ²_real/σ̂² − ln(σ²_real/σ̂²) − 1` (Patton 2011: robust to
  noise in the proxy).
- **Statistic per claimant k ∈ {E1, E2}:** paired monthly difference `D = L(E0) − L(Ek)`
  (positive = Ek better), over months where both are defined.
- **Inference:** circular moving-block bootstrap over months, block 3, 10,000 draws, seed
  20261127; **Bonferroni over 2 claimants → 97.5% two-sided CI**.
- **PASS (k) iff:** CI lower bound on mean(D) > 0 **and** mean(D) > 0 in both 2016–2020 and
  2021–2026 **and** n ≥ MIN_N.

## Adoption rule (frozen) — what a PASS changes

A passing claimant replaces RV20 in the live sleeve **only if**, re-simulating the sleeve's
Arm B on the same window with that estimator's σ̂ in place of RV20 (3 bps turnover cost,
otherwise identical): its Sharpe ≥ E0-sleeve Sharpe − 0.05 **and** its max drawdown is no
worse than E0-sleeve max drawdown − 2 pp. If both claimants pass and qualify, the one with
the larger mean(D) is adopted. Reported for every estimator (no further authority): CAGR,
vol, Sharpe, max DD, turnover, and the realised-vol error versus the 14% target on months
where the scale was below 1.

## What is NOT changed by any outcome

Ranking, universe, top-k, the 14% target, the [0.25, 1.00] clip, the monthly calendar.

## Amendments

Only as dated notes, before results, in the stricter direction.
