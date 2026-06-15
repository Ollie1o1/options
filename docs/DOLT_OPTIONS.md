# DoltHub Real-Options Layer

Free, real EOD option marks (bid/ask/IV/Greeks) from the public DoltHub dataset
[`post-no-preference/options`](https://www.dolthub.com/repositories/post-no-preference/options),
used to make the backtester's P&L real and to validate the price-based slice of
the scorer against real forward returns.

## Why this exists

Everything else in the system is either Black-Scholes-repriced (the walk-forward
backtester) or forward paper trades (the n≈4 cohort). This layer gives **real
historical option prices** so the numbers are backed by what actually traded.

## Source & coverage

- **Endpoint:** `https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master?q=<SQL>`
- **Auth:** none required for reads. (A DoltHub *CLI credential* is NOT an HTTP
  API token; sending an `authorization` header returns HTTP 400.)
- **Table `option_chain`:** `date, act_symbol, expiration, strike, call_put,
  bid, ask, vol (IV), delta, gamma, theta, vega, rho`.
- **Coverage:** 2019-02-09 → 2026-06-12, EOD, trading days only.
- **Access pattern:** queries MUST be scoped by `date` (leading PK column) or
  they hit the API's ~30s deadline. A (symbol,date) chain is ~200 rows in <1s.

## Rate-limit etiquette (built in)

- **Cache-first:** every (symbol,date) is fetched once then served from
  `data/dolt_options.db` forever. Empty (non-trading) days are cached as misses.
- **Throttle:** ~0.3s between live calls.
- **Resumable backfill:** skips pairs already fetched.
- Don't run two live backfills/validations at once — you'll race the rate limit.

## CLIs

### 1. Fetch / backfill chains
```bash
# Probe one chain
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options --probe

# Backfill a basket over a range (weekly samples)
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options \
    --backfill --symbols AAPL,SPY,NVDA --start 2023-01-01 --end 2023-12-31 --weekly

# Cache stats
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.dolt_options --stats
```

### 2. Real-marks backtest
```bash
# Real DoltHub fills instead of Black-Scholes (entry=bid, exit=ask; short premium)
PYTHONPATH=$PWD OPTIONS_MAINTENANCE_CHILD=1 ~/.venvs/options/bin/python \
    -m src.backtester AAPL SPY --price-source dolt
```
The report header labels the run `[REAL DOLTHUB MARKS]` vs `[THEORETICAL PRICES]`.
First run is slow (live fetch); subsequent runs read the cache. Note: real fills
are materially worse than the BS spread assumption — e.g. AAPL short-premium
≈ −66% (real) vs −25% (BS).

### 3. Historical scorer-slice validation
```bash
PYTHONPATH=$PWD OPTIONS_MAINTENANCE_CHILD=1 ~/.venvs/options/bin/python \
    -m src.dolt_validate --symbols AAPL,SPY,NVDA --start 2023-01-01 --end 2023-12-31 --weekly
```
Reconstructs the **price/IV/Greek-based** features (term structure, skew,
moneyness, theta, IV level), picks the top call, looks up its **real** forward
return `exit_dte` rows later, and reports IC + quintiles.

## Honest limitation

`dolt_validate` covers only the **price/IV/Greek slice** of the scorer.
News, sentiment, catalyst, EDGAR insider, and regime-news components are NOT
reconstructable historically, so they are excluded. A positive IC here validates
the price-based signals; it does not validate the full production `quality_score`.

## Findings on real marks (2026-06-15, all reproduced from cache)

Every number below was produced by the CLIs above on real DoltHub marks, not estimated.

- **The flat 7%/side spread was wrong both ways.** Real `(ask-bid)/mid`: ATM/ITM
  ~1.0–1.5%/side, but deep-OTM 3.4–5.9%/side (25% at p75). See `dolt_slippage --table`.
- **Black-Scholes flattered short premium by ~40 points.** AAPL short-premium walk-forward:
  −24.5% (BS) vs −65.6% (real fills).
- **The price/IV/Greek slice of the scorer has no edge.** AAPL+SPY, 70 samples, 2022–2024:
  IC +0.05 (p=0.66). Consistent with the live paper IC (0.03) and the BS backtest.
- **The actual long-call cohort is marginally positive.** AAPL+SPY, n=185, 2022–2024,
  using the canonical exit rules (TP +100% / deep-ITM Δ≥0.80 / time-exit / SL −50%):
  win 40.5%, **avg +5.3% all-in** (net of real spread + $0.65/contract commission),
  median −9.5%, PF 1.17. Tail-driven. (Gross, pre-commission: +5.7%.)
  Caveat: 2 symbols; broaden the basket before trusting it.
- **Earnings IV crush is real but a modest cohort drag.** AAPL ATM IV drops a median
  ~24% post-earnings (13 events), yet holding through earnings cost ~1 pt of avg return
  (+4.8% vs +5.8% clean) — only ~11% of trades touch earnings.

## Config (`config.json → dolt_options`)

```json
{
  "cache_path": "data/dolt_options.db",
  "basket": ["AAPL", "SPY", "NVDA", "QQQ", "TSLA", "MSFT", "AMD", "META", "GOOG", "AMZN", "NFLX", "IWM"],
  "validate_start": "2022-01-01",
  "validate_end": "2025-12-31",
  "validate_sampling": "weekly",
  "target_dte": 30,
  "exit_dte": 21
}
```

The cache DB lives under `data/` which is gitignored (binary, regenerable).
