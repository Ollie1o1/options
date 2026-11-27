# Audit plan — how wrong is the inferred underlying price in the DoltHub studies?

**Written:** 2026-11-27, before any Alpaca daily close was downloaded for this purpose.
**Implements:** `scripts/audit_dolt_spot.py`. An audit, not a hypothesis test: it measures an
error and re-runs one study; it cannot create a strategy.

## Why

DoltHub chains carry no underlying price. `src/selector_study/dolt_history.py` infers it:
- **entry spot** = put-call-parity intercept on the nearest expiration (this is the forward,
  not spot, and feeds leg selection via `capture.build_arms(.., spot=..)`);
- **settlement close** = adjusted close × (parity spot / adjusted close) from the nearest
  fetched day (adjusted closes must be un-adjusted for dividends and splits).
`src/dolt_stocks.py` (used by the earnings and pre-earnings studies) instead un-adjusts
yfinance closes for splits only. Alpaca `adjustment=raw` daily bars give the actual traded
close on the strikes' own basis.

## Truth source and its check

Alpaca SIP daily bars, `adjustment=raw`, 2019-01 .. 2026-06, for the 10 symbols in the local
Dolt cache (AAPL, AMZN, DIA, GOOGL, JPM, META, MSFT, NVDA, SPY, XLE). **Validated first**:
compare Alpaca's raw daily close for SPY and DIA (no splits) against yfinance's unadjusted
`Close` (`auto_adjust=False`); the audit proceeds only if the median absolute difference is
≤ 0.02% (otherwise the truth source is reported as unfit and the audit stops).

## Measurements (all reported)

1. **Entry spot error:** `parity_spot(chain, day) / true_close(day) − 1` on every cached
   chain (Saturday-stamped weeklies before 2020-01-06 compared with the prior Friday's close).
   Median, 95th percentile and max of |error|, per symbol and pooled.
2. **Settlement error:** the study's settlement close vs the true raw close at each settled
   expiration in `data/selector_study_dolt/` — median / p95 / max |error|, and how many entries
   change from "short strike in the money" to "out of the money" or back.
3. **dolt_stocks method:** yfinance split-adjusted close un-adjusted with yfinance's split
   history (the same arithmetic as `src/dolt_stocks.py`) vs true close — median / p95 / max.
4. **Study re-run:** the selector-vs-liquidity history (`dolt_history.run`) re-run into a
   separate directory with **true closes** for both entry spot and settlement; primary readout
   (crossed paired mean and CI) compared with the recorded interim (+0.03% [−0.17, +0.22]).

## How the result is used

- If measurement 1 or 2 has p95 |error| ≤ 0.25%, the inferred prices are recorded as
  adequate and no study is re-opened.
- If the re-run changes the interim's conclusion (its CI excludes zero, or the point estimate
  moves by more than half the original CI half-width), the interim result doc gets a dated
  correction and the study's spot source is switched to Alpaca.
- Either way, an `alpaca` raw-close provider is added so future Dolt work can use true closes.
