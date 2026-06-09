# Intel Briefing — Backtest Findings

Date: 2026-06-09
Harness: `src/intel/reliability.py` · basket of 12 liquid names
(SPY QQQ IWM SMH NVDA AMD AAPL MSFT AMZN META GOOGL TSLA) · ~4y daily ·
pooled, every-3rd-day subsample.

## What we measured

For each price-derived signal we reconstructed its value at every historical
day (using only data available at that point) and paired it with the realized
forward N-day return. We report the **information coefficient** (Spearman rank
correlation of signal vs forward return) and the directional **hit-rate**. A
signal earns weight in the verdict only if its IC is **positive and clears a
null margin** (IC > 0.02) with adequate sample; otherwise it is shown but given
**zero weight** (tagged "context only").

## Results (10-day forward horizon, the default)

| signal | IC | hit-rate | n | weight | verdict |
|---|---|---|---|---|---|
| momentum | **+0.027** | 0.544 | 3852 | 0.27 | `ok` — earns weight |
| trend | −0.005 | 0.558 | 3852 | 0 | context only |
| support | −0.028 | 0.578 | 3852 | 0 | context only |
| bounce | −0.017 | 0.512 | 3852 | 0 | context only |
| rsi | −0.032 | 0.419 | 3852 | 0 | context only |
| news | n/a | n/a | 0 | 0 | not backtestable from price |
| analyst | n/a | n/a | 0 | 0 | not backtestable from price |

## Trend across horizons (why we do NOT bullish-weight trend)

| horizon | trend IC | momentum IC |
|---|---|---|
| 10d | −0.005 | +0.027 |
| 20d | −0.031 | +0.031 |
| 60d | **−0.156** | −0.018 |
| 120d | **−0.283** | −0.078 |

Trend's IC is negative and **worsens** with horizon. In this basket and period
(a regime where mega-cap tech ran up and then mean-reverted), being above the
200-day / in a golden cross **preceded lower** forward returns. Trend-following
at these horizons was counterproductive. Hard-coding "uptrend = bullish" into
the verdict would therefore have been **misleading** — so trend is shown for
context and given zero weight.

## Conclusions / design implications

1. **Short-horizon equity direction is close to random.** Only momentum shows a
   whisper of positive edge (IC ≈ 0.027, ~1.7 standard errors). The verdict is
   built to reflect this: it is deliberately **humble**, and confidence is
   usually low/medium. This is the anti-misleading design the spec required.
2. **The actionable value does not depend on signal IC.** It comes from
   (a) consolidating every data point in one view, (b) the **empirical
   bounce base-rate** (a conditional historical frequency, not a prediction),
   and (c) the deterministic **playbook** "what to do" line, which reads all
   signals regardless of their verdict weight.
3. **News / analyst** are not reconstructable from a price series, so they stay
   context-only (zero weight) until wired to a historical sentiment source.
4. Weights are cached in `data/intel_reliability.json` and recomputed weekly
   (or via `load_or_compute_reliability(force=True)`).

## How to reproduce

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -c \
  "from src.intel.reliability import load_or_compute_reliability as f; \
   import json; print(json.dumps(f(force=True), indent=2))"
```
