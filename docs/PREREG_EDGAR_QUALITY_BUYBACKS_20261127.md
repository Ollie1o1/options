# Preregistration — Study 2D: Point-in-Time SEC EDGAR Quality + Net Buybacks TFSA Sleeve

**Registered:** 2026-11-27 (frozen before downloading full 286-stock historical SEC XBRL filings or running `scripts/backtest_edgar_quality_buybacks.py`).  
**Target Account:** Wealthsimple USD TFSA (`$0` commission shares, long-only quarterly rebalance).

## Why

Our TFSA currently has one validated core sleeve (`VM_OUTLOOK_TOP3`, monthly ETF rotation) and no validated structural single-stock fundamental sleeve. Academic evidence (Novy-Marx 2013 *Gross Profitability*, Boudoukh et al. 2007 *Net Payout/Buyback Yield*, FCF profitability) suggests low-turnover quality + net share shrinkage delivers persistent cross-sectional alpha at stock-share friction (`~10 bps/side`). Unlike stale 13Fs or paid vendor datasets, SEC EDGAR XBRL `companyfacts` is 100% free and carries exact `filed` dates back to ~2010.

## Honest Survivorship Bias Control (Read First)

* The local `data/equity_ohlcv.db` universe (286 non-ETF US stocks with `2016-01 .. 2026-10` price coverage and active SEC CIKs) was curated in 2026 and **excludes delisted/bankrupt companies** (survivorship bias).
* Consequently, raw CAGR versus `SPY` would overstate absolute returns.
* **Primary Benchmark (`EW_UNIV`):** Every arm is judged on **quarterly excess return over the Equal-Weight Portfolio of the exact same eligible stocks on each rebalance date** (same survivorship pool!). This neutralizes index-level survivorship inflation and isolates pure cross-sectional selection power.
* **Promotion Ceiling:** Even if an arm clears all statistical hurdles below, it may be promoted at most to **`LEAD` (survivorship-caveated)** — NEVER `VALIDATED` — until confirmed either on a survivorship-free CRSP/Compustat feed or forward OOS (`opt discover` forward ledger).

## Strict Point-in-Time (Zero Look-Ahead & Split-Safe) Factor Definitions

At each quarterly signal date `t` (last trading day of Mar, Jun, Sep, Dec from `2016-06-30` through `2026-09-30`, `~41` quarters; trades executed at close of `t + 1` trading day), **only SEC 10-K/10-Q facts with `filed_date <= t` are visible**:

1. **Split-Safe Same-Filing Net Diluted Buyback Yield (`NBY`):** Within the latest `10-K`/`10-Q` accession (`accn`) filed on or before `t` that contains `WeightedAverageNumberOfDilutedSharesOutstanding` (fallback `WeightedAverageNumberOfSharesOutstandingBasic`), compare the most recent reporting period `(start, end)` against the exact comparative prior-year period inside the **same `accn`** (`end_cur - end_prev` in `[330, 400]` days and `|dur_cur - dur_prev| <= 15` days). Because both periods are reported inside the same filing, stock splits are automatically restated on the same basis and stock-based compensation (SBC) dilution is included. `NBY = clip(1 - S_cur / S_prev, -0.25, +0.25)`. (Require `end_cur` within `400` days of `t` so stale/defunct filers are excluded.)
2. **Novy-Marx Gross Profitability (`GP/A`):** Latest `10-K` (annual duration `330..400` days, `filed_date <= t`, `end` within `500` days of `t`) `GrossProfit` (fallback `Revenues - CostOfRevenue` from standard US-GAAP tags) divided by latest `Assets` (`filed_date <= t`). Winsorized cross-sectionally at `[1%, 99%]`.
3. **Free Cash Flow / Assets (`FCF/A`):** Latest `10-K` (annual duration `330..400` days, `filed_date <= t`) `(NetCashProvidedByUsedInOperatingActivities - |PaymentsToAcquirePropertyPlantAndEquipment|)` divided by latest `Assets` (`filed_date <= t`). Winsorized at `[1%, 99%]`.
4. **Composite Quality + Buyback Score (`Q_NBY`):** Equal-weight mean of available cross-sectional percentile ranks (`0..1`) across `{NBY, GP/A, FCF/A}`, requiring at least 2 valid factors (so financials/insurers without `GrossProfit` are still ranked cleanly on `NBY` + `FCF/A`).

## Frozen Claimants (3 Quarterly Long-Only Arms, Zero Tuning)

| Arm | Selection Rule (Quarterly Rebalance at `t+1` Close) | Friction |
|---|---|---|
| **Arm Q1 (`TOP20_NBY`)** | Equal-weight top 20 eligible stocks by `NBY`. | `10 bps` one-way |
| **Arm Q2 (`TOP20_GPA`)** | Equal-weight top 20 eligible stocks by `GP/A` (among stocks with valid `GP/A`). | `10 bps` one-way |
| **Arm Q3 (`TOP20_COMPOSITE`)** | Equal-weight top 20 eligible stocks by composite `Q_NBY`. | `10 bps` one-way |

* **Eligible universe at date `t`:** Non-ETF US stocks in `data/equity_ohlcv.db` with valid OHLCV on `t`, `t+1`, and `t_next+1`, `>= 252` trailing trading days, and valid factor value as of `filed_date <= t`.
* **Stress Check:** `20 bps` one-way turnover friction.
* **Diagnostics (no adoption authority):**
  * Q5 (top 20%) minus Q1 (bottom 20%) long-only spread on `Q_NBY`.
  * Momentum-orthogonal check: double-sort / excess return after neutralizing trailing `12-1m` price momentum quintile (`MOM_12_1`).
  * `TOP20_COMPOSITE` with `Close > SMA200` trend filter (uninvested slots in 0% cash).
  * Quarterly correlation between `TOP20_COMPOSITE` excess/total returns and `VM_OUTLOOK_TOP3` (TFSA diversification audit).

## Inference & Adoption Rule (Frozen)

* **Primary Metric:** Annualized quarterly arithmetic excess return vs `EW_UNIV` (exact same eligible pool), net of `10 bps` one-way turnover cost (`EW_UNIV` also charged quarterly rebalance friction).
* **Bootstrap:** Circular block bootstrap over quarterly excess returns (`block = 2` quarters, `10,000` draws, seed `20261127`), **Bonferroni across 3 arms (`alpha = 0.05 / 3`, `98.33%` two-sided CI)**.
* **Sub-period Halves:** `2016–2021 H1` (`t <= 2021-06-30`) vs `2021 H2–2026` (`t > 2021-06-30`).
* **PASS Bar:** An arm earns **`PASS`** (eligible for **`LEAD — survivorship-caveated`** status in `opt discover --lens edgar_quality`) ONLY IF:
  1. `98.33%` CI lower bound of excess return vs `EW_UNIV` (at 10 bps) is strictly **`> 0`**; AND
  2. Pointwise excess return vs `EW_UNIV` is positive in **both halves**; AND
  3. Pointwise excess return remains **`> 0`** under `20 bps` one-way friction stress; AND
  4. Top quintile beats bottom quintile (`Q5 - Q1 > 0`).
* Otherwise verdict = **`FAIL`** (`CLOSED` or kept `WATCHLIST` only if descriptive context; never promoted).
