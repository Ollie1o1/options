# Pre-registration — does structure profitability differ by SECTOR?

Written **2026-08-13, before any sector-conditioned outcome was computed.**
Sample sizes were inspected to set cell minimums (a power calculation, not a
peek); no return, IC, PF or win rate was looked at per sector beforehand.

## 1. Why this, and why it is not a fishing trip

Everything in this repo that has survived a holdout is a **conditioner** or a
**cross-sectional** comparison. Everything that has died is an **absolute
per-contract ranker**.

| survived | kind |
|---|---|
| VIX regime (low-VIX long calls beat high-VIX, across time *and* symbol holdouts) | conditioner |
| Segment split: index short put PF 2.28, semi 1.17, **tech 0.95 TRAP** | conditioner |
| Sector outlook relative IC +0.05–0.08 (absolute calls ~30% reliable) | cross-sectional |
| `long_delta` survives residualisation both windows — only feature that did | conditional on side |
| Short-interest **level** — real, OOS-robust | cross-sectional |

| died | kind |
|---|---|
| `quality_score` ranking (p=0.89) | absolute ranker |
| `iv_rank`, `term_slope`, condor `pop`, ~40 features | absolute rankers |

The segment result above is the **largest structural effect measured anywhere
in this system — a 2.4× spread in profit factor** — and it was measured on
`{"index": ["SPY"], "tech": [5 names], "semi": [2 names]}`. Eight symbols. The
code's own comment names the reason: *"the per-segment sample — the binding
constraint everywhere in this research."*

The Dolt cache holds **121 symbols with >200 days of real chains**. The sector
hypothesis has been tested on **7% of the available universe** and found a large
effect. That is an under-powered search, not an exhausted one.

## 2. Hypothesis

**H-SECTOR.** Structure-family profitability differs systematically by sector,
and pooling across sectors masks offsetting effects.

**H-TECH (the pre-existing claim, now testable).** Short premium on technology
names underperforms short premium on the index — "TECH = TRAP" — and this
replicates on ~20 tech names rather than 5, in a held-out window.

## 3. Data

`data/dolt_options.db`, real EOD bid/ask/greeks, 121 symbols, 2020-01-27 to
2026-06-12. Entry at `short_bid - long_ask` / exit at `short_ask - long_bid`
(crossed both ways). Returns on **max risk** for defined-risk structures.

**Known limitation, stated up front:** on SPY put spreads only 8% of positions
were ever quoted twice, so exit rules mostly never fire and positions ride to
expiry. See §7 for the data-quality gate this forces.

## 4. Sector mapping — LOCKED HERE

`src/data_fetching.SECTOR_MAP` covers 43/121. It is extended to the remaining
78 Dolt names below, to SPDR sector buckets. **This mapping is fixed by this
document and must not be adjusted after seeing results.**

```
XLK  Technology       ADSK AKAM CRUS EPAM GLW JKHY LFUS MANH MSI NTAP OSIS SYNA TTEC WEX
XLV  Health Care      MED MOH THC VTRS
XLY  Cons. Disc.      ANF CZR DHI HBI HRB LEN LOPE MTH NCLH PENN RL VC WGO YETI
XLP  Cons. Staples    CLX EL FIZZ HAIN MKC SJM
XLI  Industrials      AAWW ALLE AXON CPRT GNRC GTLS HA LNN OSK OTIS SNA SWK UNF
XLE  Energy           KMI NOV
XLF  Financials       AMP FHN IBKR MKTX PBCT RGA WLTW
XLB  Materials        FMC IOSP MLM
XLRE Real Estate      CBRE ESS MAC PCH REG SBAC SRC
XLC  Comm. Services   LYV OMC
BENCH (not a sector)  SPY XHE XME XPH        # index + industry ETFs, held out as benchmark
CDAY -> XLK (payroll software; Ceridian)
FB   -> XLC (pre-rename Meta; see project_sq_rename — data ends 2022-06)
```

## 5. Design

* **Structures:** bull put spread, bear call spread (`src/dolt_spread.py`),
  short put (`src/dolt_short.py`), long call (`src/dolt_cohort.py`).
* **Sampling:** weekly entries, the only mode every Dolt result on record used.
* **Split, LOCKED:** train `2020-01-27 .. 2023-12-31`; holdout
  `2024-01-01 .. 2026-06-12`. The holdout is never fitted on and is read once.
* **Outcome:** return on max risk, all-in of modelled crossing cost.
* **Benchmark:** SPY, so every sector claim is stated *relative to the index*
  rather than in absolute terms — absolute short-premium P&L is dominated by
  the 2020-2026 equity drift, which is beta, not a finding.

## 6. Cell minimums — LOCKED

A cell is **sector × structure**. A cell is eligible only if:

* **≥ 4 distinct symbols** (kills the 2-name "semi" conclusion this replaces);
* **≥ 100 closed trades in train** and **≥ 60 in holdout**;
* **≥ 20% of trades "managed"** (quoted at least twice, so an exit rule could
  fire). Below that the cell measures the archive's strike coverage, not the
  strategy — the trap identified on SPY spreads.

Ineligible cells are reported as **INSUFFICIENT** and carry no verdict. They
may not be rescued by merging sectors after the fact.

## 7. Promotion rule — LOCKED, all four required

A sector effect is **REAL** only if:

1. **Sign holds out of sample** — the sector's return-on-risk *relative to
   SPY* has the same sign in train and holdout.
2. **Survives multiplicity** — Benjamini-Hochberg **q < 0.10** across the
   entire sector × structure grid, computed over every eligible cell tested,
   not per cell.
3. **Not one name** — dropping the single largest-contributing symbol leaves
   the sign unchanged.
4. **Not the vol level restated** — the effect survives residualising the
   outcome on entry credit-to-width (the `IC|ctl` control that exposed four
   fake ICs on 2026-08-11).

Anything failing any of the four is **recorded and dropped**, not
down-weighted. No config or weight changes ship from this run.

## 8. Honest prior

Stated before the run, as `PREREG_OPTIONSDX` did:

The 2.4× segment spread is the strongest structural signal on record here, so
**something** is probably there. But the same document that measured it also
found the effect leaned on 2024 and on tiny samples, and this repo's history is
that broadening a basket *flips* verdicts (`AAPL+SPY` PF 1.17 → 4-name PF 1.02;
3-name tech "STAND DOWN" → 5-name "SHORT puts PF 1.14"). A 10-sector × 4-structure
grid is 40 tests; at q<0.10 roughly four false positives are expected if
nothing is real, which is exactly why §7.2 exists.

**Most likely outcome: 1-2 cells survive all four criteria, and the honest
result is a refusal rule ("do not sell premium on sector X") rather than a
selection rule.** A refusal rule is still worth having — this system is already
built to refuse.

## 9. What ships if it works

Nothing automatic. A surviving sector effect becomes a **display-only** verdict
line first, as `dolt_verdict` already does per segment, and would need a second
independent window before it could gate anything.

---

## ADDENDUM — deviations, recorded when they happened

Two harness defects surfaced after the run started and before any
sector-conditioned **outcome** (return, PF, IC, win rate) was read. Cell
counts and managed-fractions had been seen; nothing else. Both are recorded
here rather than silently corrected.

**D1. `run_cohort_backtest` returned no per-trade rows.** Every `long_call`
cell came back `n=0`. The data existed; the runner reported summary statistics
only, unlike its two siblings. Fixed in the repo (PR #40) rather than worked
around, because the same gap blocks attribution of the Phase 1 gate cohort.
The grid is re-run from scratch against the fixed runner.

**D2. The §6 managed-fraction gate applies to SPREADS ONLY.** `marks_seen` is
recorded by `dolt_spread` and by neither `dolt_short` nor `dolt_cohort`, so the
harness was reading a missing key as 0% and would have marked every
single-leg cell INSUFFICIENT for a reason that was an artefact of my own code.

The gate therefore stands as written for `bull_put` and `bear_call`, and is
**not evaluable** for `short_put` and `long_call`. Those two are reported with
`managed = n/a`, and their exit-observability is **unverified** — a real
weakness in the evidence for them, not a clean pass. It is not a licence to
treat them as if they had passed the gate.

No hypothesis, split, mapping, cell minimum or promotion criterion in §§2-7 is
changed by either deviation.
