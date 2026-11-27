# Audit result — inferred underlying prices in the DoltHub studies

**Run:** 2026-11-27 · **Plan:** `docs/AUDIT_DOLT_SPOT_PLAN_20261127.md` (with amendment A1,
made after the truth-source check and before any measurement) · **Script:**
`python -m scripts.audit_dolt_spot --rerun` · **Provider added:** `src/raw_closes.py`.

## Bottom line

1. The put-call-parity spot used by the selector history study is **adequate for comparisons
   between arms but biased for absolute returns**: it is a forward, +0.09% high on average,
   p95 |error| 0.37% (plan threshold 0.25% — **not met**).
2. Re-running that study on official raw closes **does not change its verdict** (+0.00%
   [−0.18, +0.20] vs +0.03% [−0.17, +0.22]) but cuts both arms' absolute crossed return on capital
   at risk from **+1.96% to +1.29%** per trade. Correction appended to the interim result.
3. Every other DoltHub module (`dolt_spread`, `dolt_short`, `dolt_vol`, `dolt_cohort`,
   `dolt_earnings*`) already uses `dolt_stocks.close_history` — split-unadjusted official closes,
   the same construction validated here — so no other study is reopened.

## M0 — truth source

Official close = yfinance `Close` (split-adjusted only) × every later split. Cross-checked
against Alpaca SIP raw daily closes, 2019-01 .. 2026-06, 1,882 days per symbol:

| | agree within 0.05% (all / 2021+) | splits bracketed |
|---|---|---|
| AAPL | 99.4% / 99.9% | 2020-08-31 4:1 ✓ |
| AMZN | 99.8% / 100% | 2022-06-06 20:1 ✓ |
| DIA | 97.7% / 99.9% | — |
| GOOGL | 99.9% / 99.9% | 2022-07-18 20:1 ✓ |
| JPM | 99.9% / 100% | — |
| META | 99.9% / 100% | — |
| MSFT | 99.6% / 100% | — |
| NVDA | 99.8% / 100% | 2021-07-20 4:1 ✓, 2024-06-10 10:1 ✓ |
| SPY | 97.1% / 99.6% | — |
| XLE | 96.8% / 99.1% | 2025-12-05 2:1 ✓ |

The disagreements are Alpaca's, mostly 2019–2020, where its daily close carries post-market
prints (SPY 2019-01-08: Alpaca 256.62, official 256.77, 15:59 one-minute close 256.69). That is
why Alpaca daily closes are a cross-check and not the truth (amendment A1).

## M1 — entry spot (parity) vs official close, 3,348 chains

Pooled |error|: median **0.13%**, p95 **0.37%**, max 6.3%; 543 chains (16%) above 0.25%; mean
signed **+0.087%** (forward above spot). Per symbol p95: AAPL 0.27%, AMZN 0.29%, DIA 0.30%, GOOGL
0.36%, JPM 0.53%, META 0.39%, MSFT 0.35%, NVDA 0.32%, SPY 0.35%, XLE 1.03%. The five worst are
all **2025-03-26** (NVDA +6.3%, GOOGL +3.4%, META +2.7%, AMZN +2.4%) — the DoltHub chain for that
date is inconsistent with that day's close across names; the study read the NVDA one as a split.

## M2 — settlement close vs official close (contract basis), 4,412 settlements

Median |error| 0.15%, p95 0.34%, max 6.3% (the NVDA 2025-03-26 entry). 27 entries flip between
short put in / out of the money; mean |P&L change| 0.58% of capital at risk, p99 8.3%, 513 entries
change by more than 1%.

Note on method: 24 entries spanning a split looked 300–1,900% wrong on a naive comparison. They
are right — a split adjusts the listed option (strike ÷ r, deliverable × r), so the study's
pre-split-basis settlement is economically correct. The audit's truth uses the same contract basis
(`raw_closes.settlement_close`).

## M3 — `dolt_stocks` arithmetic

It is the truth construction (split-unadjusted official closes), so M0 is its validation —
provided DoltHub's split table matches yfinance's split history.

## M4 — study re-run

| run | pairs | weeks | paired mean (crossed) | 95% CI | selector RoR | baseline RoR |
|---|---:|---:|---:|---|---:|---:|
| recorded interim (2026-11-26) | 2,206 | 370 | +0.03% | [−0.17%, +0.22%] | +1.96% | +1.94% |
| control: original method, fresh run | 2,206 | 370 | +0.03% | [−0.17%, +0.22%] | +1.96% | +1.94% |
| **official raw closes** (entry spot + settlement) | 2,239 | 373 | **+0.00%** | **[−0.18%, +0.20%]** | **+1.29%** | **+1.30%** |

The point estimate moved 0.03 pp against a CI half-width of 0.19 pp — well under the plan's
"half the half-width" trigger — so the study's spot source is **not** switched; the correction is
recorded instead. 33 more pairs resolve because true closes exist where the parity spot did not.

## For future DoltHub work

Use `src/raw_closes.official_raw_closes` (or `dolt_stocks.close_history`) for any absolute claim.
When the remaining 7 selector-history symbols are fetched, re-running with true closes is
cheap: `scripts/audit_dolt_spot.py::rerun` shows the substitution.
