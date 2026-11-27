# Result (interim) — selector vs liquidity-first baseline, DoltHub hold-to-expiry history

**Run:** 2026-11-26, `opt lab study history`, rules frozen in
[PREREG_SELECTOR_BASELINE_DOLT_HISTORY_20261126.md](PREREG_SELECTOR_BASELINE_DOLT_HISTORY_20261126.md).
**Interim:** 10 of the 17 registered symbols. The download was stopped by the system for low
memory before XLF, XLI, XLK, XLP, XLU, XLV and XLY were fetched (XLE is partial, to 2022-06).
Resume with `opt lab study history --fetch`. The cache skips everything already fetched.

**Verdict (primary, crossed): No difference established.**

## Coverage

| | |
|---|---|
| Chains converted | 3,349 (AAPL, AMZN, DIA, GOOGL, JPM, MSFT, NVDA, SPY 2019-02 → 2026-06; META from 2022-07; XLE to 2022-06) |
| Weekly entry days | 399 sessions (Saturdays before 2020, as DoltHub stamps them; Wed / Thu / Tue after) |
| Observations | 3,348 captured · 523 empty upstream days (Dolt holes, retried on the fallback day) · 1 missing spot |
| Selector picks | 2,244. The other 1,104 captured chains failed the selector's own rules |
| Baseline | **2,004 identical legs (89%)**, 240 different, 0 no-match |
| Outcomes | 2,206 pairs settled · 38 missing (no settlement price within 7 days) · 0 not filled |
| Settlement basis | 4,412 settlements; 28 used the entry-day ratio because a split fell inside the hold |

Spot is implied by put-call parity. Its ratio to the local adjusted close moves 0.03–0.15% a week
(median by symbol). Jumps above 5% match the known splits (AAPL 2020; AMZN, GOOGL 2022; NVDA 2021,
2024) plus isolated NVDA outlier days.

## Paired difference, selector − baseline, return on capital at risk

| Case | Pairs | Weeks | Mean of weekly means | 95% CI (block 8 wk) | Selector | Baseline |
|---|---:|---:|---:|---|---:|---:|
| **Crossed entry (primary)** | 2,206 | 370 | **+0.03%** | **[−0.17%, +0.22%]** | +1.96% | +1.94% |
| Midpoint entry (sensitivity) | 2,206 | 370 | +0.06% | [−0.14%, +0.26%] | +3.09% | +3.05% |

Diagnostic, not a registered test: the 239 settled pairs where the legs differed (165 weeks) give
crossed −0.42% [−2.29%, +1.30%] and mid −0.13% [−2.00%, +1.62%]. The baseline's spreads were
tighter relative to credit (median 9.1% vs 10.8%), but it took wider strikes at slightly lower
delta and collected less credit per dollar of width (13.8% vs 15.3%). That offsets the saving.

## Reading

- On DoltHub's ~27-strike grid, the liquidity-first rule almost always lands on the selector's
  own spread. Where it does not, its cost saving is small against outcome noise. **No evidence
  that switching the generic selector to liquidity-first would change net results** at one crossing.
- The registration said a null here does not rule out an advantage under the book's exits,
  which cross twice, or on live chains with 1-point strikes. The forward study (read ≥ 2027-03-01) tests that.
- The arms' absolute returns (+1.96% crossed) are **not** evidence of an edge. They are not
  delta-adjusted, and the 2026-10-15 finding that put spreads trail their own delta still stands.
