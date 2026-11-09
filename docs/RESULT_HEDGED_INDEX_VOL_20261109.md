# Delta-hedged index premium — RESULT

Run 2026-11-09 against `docs/PREREG_HEDGED_INDEX_VOL_20261109.md` (frozen in
`e80298a` before running). Script: `scripts/hedged_index_vol_odx.py`.
SPY optionsDX 2010-01-04 .. 2023-12-29, 165 monthly expiries, one
non-overlapping trade each.

## Verdict: 0 of 3 pass. No §3 candidate.

| test | n | mean r (per trade, on CAR) | PF | 98.33% lower bounds | verdict |
|---|---|---|---|---|---|
| T1 hedged 25/10Δ condor | 165 | +1.23% [−0.84, +3.28] | 1.289 [0.848, 2.037] | −1.27% / 0.778 | fail |
| T2 T1 when VRP > prior median | 61 | −0.72% [−4.19, +2.59] | 0.863 [0.437, 1.789] | −4.99% / 0.372 | fail |
| T3 hedged ATM fly, 10Δ wings | 165 | +1.08% [−1.23, +3.38] | 1.213 [0.808, 1.869] | −1.69% / 0.742 | fail |

## Readings (no authority beyond the verdict)

- **The point estimate is positive, but 14 years can't confirm it.** T1 earns
  +1.2% of capital at risk per month (annualised Sharpe ≈ 0.32, monthly sd
  13.3%). Confirming that mean at 2 standard errors needs ≈ 470 monthly trades,
  about 39 years. This data has 165. The test is underpowered for any premium
  of plausible size, so "fail" here means "not shown", not "shown absent".
- **Hedging does what it should to risk, not to return.** Unhedged mean +1.15%
  vs hedged +1.23%. The sd drops 33% → 13%, and the worst trade drops −100% →
  −46%. Daily hedging gave back $984 of $3,551 option P&L (the gamma cost) plus
  $304 in share costs over 165 trades.
- **T2 points the wrong way.** High-VRP months did worse than low-VRP months
  (above − below −3.05% [−7.32, +1.17]). The CI spans zero, so this is no
  evidence for timing either way, and the naive "sell when IV is rich vs RV"
  idea is not supported.
- **Mid fills** lift T1 to +1.66% [−0.40, +3.71]. Cost is still a lever, but
  not enough to pass.
- **By year:** 2022 was the worst year for both T1 and T3 (about −1.0 of sum r). 2020 was the best
  (+1.26). Losses come in grinding trend years (2013-14, 2018, 2022), not just
  crashes.

## What this closes, what it leaves

Closed: hedged and unhedged index short-vol as a §3 candidate on optionsDX
SPY 2010-23. Don't re-run variants on this data; the sensitivities were fixed
in advance and none clears the bar.

Open, and it needs more data rather than more variants: a longer index history
(e.g. SPX 2004+, which includes 2008), or more independent index underlyings, could
give the power this one lacks. Per the decision rule, the end state stands:
hold the index; the screener stays a research tool.
