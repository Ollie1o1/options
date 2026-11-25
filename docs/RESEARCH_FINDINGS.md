# Research findings — dated register

What we've learned about making money with this system, **newest first**. Each
entry has the date it was found, its status, and where the detail lives.

Status key: **LEAD** = promising, being re-checked · **PENDING** = forward test
scheduled · **CLOSED** = answered, don't re-run · **BASELINE** = the default.

---

## 2026-11-24 (newest) — Phase 1 equity-first playbooks (low stock friction, 3–50 bps)

Five preregistered stock/ETF studies run to test whether edges that die under options bid-ask spreads survive at Wealthsimple USD share friction (`$0` commission, `3–10 bps` spreads):

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | **Vol-Managed Sector/Asset ETF Rotation (`VM_OUTLOOK_TOP3`) PASSED on both 2019–26 and 2006–26.** Monthly top-3 `src/outlook` ETFs scaled by `clip(0.14 / sigma_20, 0.25, 1.00)` at 3 bps/side and 0% cash yield delivered **Sharpe `0.983` / MaxDD `-17.2%`** vs SPY `0.778` / `-32.8%` (2019–26) and **Sharpe `0.819` / MaxDD `-17.3%`** vs SPY `0.685` / `-55.0%` (2006–26). Outperformed `VM_SPY` control (`0.570` / `0.716`), survived 10 bps stress and 2022 bear (`+3.4%` vs SPY `-8.9%`). Caveat: Sharpe difference CIs still span zero, while drawdown compression (`+38pp` with VIX cap, CI `[+1.5pp, +37.4pp]`) is massive and robust. | **VALIDATED** (core TFSA ETF sleeve) | `RESULT_VOL_MANAGED_OUTLOOK_20261124.md` |
| 2 | **Uptrend Pullback Mean Reversion (`equity_pullback`) PASSED standalone holdout (H-A) at 10 bps, FAILED SPY-excess (H-B).** `Close > SMA200`, `12-1m > 0`, `RSI(2) < 10`, `3d <= -3%`, next-day fill: broad 13,890-symbol holdout (`n=25,629`) `+0.39%` `[+0.04%, +0.73%]`, PF `1.23` `[1.02, 1.50]`; liquid 62-symbol holdout (`n=971`) `+0.55%` `[+0.11%, +1.00%]`, PF `1.43` `[1.08, 1.94]`. Excess vs SPY flat (`+0.01%` / `+0.18%`), lost in 2018 & 2022 bears, and fails at 20 bps friction (confirming shares-only execution). | **LEAD** (shares only, bear-regime flagged) | `RESULT_EQUITY_PULLBACKS_20261124.md` |
| 3 | **Path-managed Squeeze Momentum (`squeeze_equity`) FAILED CI across all 3 arms (`S1/S2/S3`) and `HOLD42`.** Stops materially improved compounding (`HOLD42` 4-slot CAGR `-1.2%` → `S1 +18.6%` / `S2 +13.7%`), but 61% loss rate and right-tail concentration (`drop top 3` halves mean) left moving-block 98.33% CIs spanning zero (`S2 +2.15% [-1.30%, +6.26%]`). | **CLOSED** | `RESULT_SQUEEZE_PATH_EQUITY_20261124.md` |
| 4 | **Large-cap PEAD (`pead_breakout`, 20d/40d) FAILED (`+0.04%` excess vs SPY, `n=614`).** Two useful diagnostics: (a) negative earnings breakdowns (`-5% / 2x vol`) lagged SPY by **`-1.16%` `[-2.19%, -0.19%]`** across both halves (retained as Radar **`POST_EARN_CRASH`** avoid badge); (b) non-earnings `+5% / 2x vol` control earned `+3.00%` `[+0.71%, +5.21%]` (PF `1.62`, unpreregistered future lead). Arm B (Estimate Revisions) preregistered & date-gated until `2027-01-15` (`2/4` snapshots collected). | **CLOSED** (PEAD) · **PENDING** (Revisions ≥ 2027-01-15) | `RESULT_PEAD_EQUITY_20261124.md` |
| 5 | **Funded Biotech Pre-Catalyst Run-Up (`biotech_runup`, `T-45 → T-3`) FAILED (`n=215`, 94 tickers).** Mean `+0.81% [-3.70%, +5.63%]`, excess vs XBI `-0.32% [-4.38%, +3.89%]`. ClinicalTrials.gov `PCD` dates are too noisy compared to exact PDUFA dates. | **CLOSED** | `RESULT_BIOTECH_RUNUP_20261124.md` |
| 6 | **Test O1 (`60–120 DTE`, `5%`-wide SPY Bull Put, 2010–23, `n=708`, med friction `4.9%`):** Primary full-touch crossed `O1-M` (TP50/no-stop/21-DTE) missed the 97.5% CI by 27 bps (`RoC +4.30%`, `PF 2.24 [1.28, 4.95]`, `x_entry +1.12% [-0.27%, +2.25%]` → **FAIL crossed**), **BUT PASSED at realistic `k=0.35` limit fills** (`x_entry +1.59% [+0.26%, +2.67%]`, `PF 2.41 [1.37, 5.42]`, `x_dyn +0.70%`, positive in 13/14 yrs including 2022). First defined-risk options rule in the repo to clear delta-adjusted excess at `k=0.35` limit fills; strictly requires measured limit execution (`fill_experiment.py` / IBKR). | **LEAD** (limit-only `k <= 0.35`) | `RESULT_ODX_WIDE_60_120_20261124.md` |
| 7 | **Test O2 (Pre-Earnings Anticipation `T-10 → T-1`, closed strictly before report, `2020–26`):** All 3 claimants FAILED: `O2-EQ` shares `+0.70%` (`-0.01%` excess vs SPY `[-0.34%, +0.31%]`, `n=2,278`), `O2-LC` crossed `-7.55%` (`+8.98%` at mid — 16.5% crossing wall), `O2-ST` crossed `-11.37%` (`+1.69%` at mid, lost in 7/7 years). | **CLOSED** | `RESULT_PRE_EARNINGS_RUNUP_20261124.md` |

## 2026-11-10 — audit of the findings below

Each finding was re-read against the script that produced it, except where a row says otherwise. The Bear Call
and gate-separation numbers were re-run and reproduce to the digit.

| Finding | As recorded | Could it flip? | What the check showed |
|---|---|---|---|
| #1 Bear Calls made money (PF 1.89) | LEAD | **Yes — it does** | The paper P&L exits at the **mid** and concedes 35% of the half-spread at entry. Crossing both ends: PF 1.19 [0.68, 2.24], excess −1.4% [−6.3, +2.2]. Index ETFs: PF 1.41 [0.69, 3.14], excess −0.1%. Single names: excess −4.8% [−6.9, −2.8]. **Not profitable at the fills the earnings and SPY tests were judged on.** Still a lead only in the sense that mid-fill profit exists. |
| #6 Friction-refused Bear Calls trail their delta | supports the gate | No | −3.1% as recorded; −20% once crossed. Stronger, same sign. |
| #5 The gate separates credit spreads | CLOSED (replicated) | No | Reproduces (+0.19 / +0.17). Holds with the 08-29/30 weekend exits removed (original window +0.24 / +0.23) and at mid-to-mid prices (forward +0.10 / +0.07, both CIs above zero). The negative-EV arm alone separates at mid (+0.19 / +0.13), so it is not only the spread being charged to the wide contracts. One exception: forward Bull Put, passed vs friction-refused at mid, +0.04 [−0.01, +0.09]. |
| #2 Earnings premium real at mid, dead at the touch | CLOSED | No | Entry/exit days are correct for before-open and after-close reports; no look-ahead. The touch result (−20% to −33%) is far from zero. "Real at mid" rests on mids of quotes up to 50% wide and is the weaker half. |
| #3 Earnings held to expiry is worse | CLOSED | **Not audited** | `scripts/earnings_to_expiry_dolt.py` was not read. |
| #4 Hedged SPY short vol | CLOSED | No | Hedge uses the prior close's delta; the VRP split uses only prior trades. Null stands. |
| #8 Nine Long Puts lagged their delta | descriptive | No | No duplicates or weekend exits in the ledger; the +$241 / +$786 figures were not recomputed. |
| 10-15 live Bull Put rule is costly beta | CLOSED | No | Crosses both ends, month clusters, absolute deltas. Negative in 12 of 13 years. |
| 10-15 SPY put spread held to expiry | CLOSED | **Not audited** | `scripts/index_hold_to_expiry_odx.py` was not read. |
| 10-10..14 direction signals | CLOSED | No (not re-run) | All null or below the cost hurdle with tight CIs; a bug would have to create an effect 5× larger. |
| 09-28 book vs SPY, exit sweep | CLOSED | No (not re-run) | Read from the real ledger on capital at risk. |

**Bugs found in the code behind the shadow corpus** (fixed, each with a failing
test; apply from the merge on, nothing back-filled):

- A quarter of simulated positions (206,221) carried an entry date 1 or 3 days
  after the scan that priced them. Their minimum hold ran from the later day.
- 31,559 simulated time exits (6%) were booked on a quote from before the
  minimum hold, because an exit could be decided on a mark of any age.
- The "top quintile" gate had never fired on a discovery board: 0 refusals in
  840,691 candidates; eight MRNA contracts above the cutoff were shown as
  cleared on 09-22 and 09-25. The auto-log board was unaffected.
- `delta_matched_benchmark` would price exits from a stale price cache without
  saying so. It now stops.

None of these moves a recorded verdict: gate separation is a within-day rank
comparison and both arms shared the defects.

## 2026-11-09

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | **Gate-passed Bear Calls made money on paper — at a mid exit only (see 11-10: PF 1.19 [0.68, 2.24] once the spread is crossed).** 4,456 trades, Aug-Oct 2026: profit factor 1.89 [1.09, 3.67]. About half of it came from the underlying falling, though, and the excess over delta (+2.8% [−2.0, +6.3] of capital at risk) isn't proven. **Index/sector ETFs carry it** (PF 2.2); single names are flat. | **LEAD**, re-check on fresh trades **≥ 2027-01-20** (`python -m scripts.bear_call_absolute --cache DIR --forward`) | `RESULT_EARNINGS_AND_BEAR_CALL_20261109.md` |
| 2 | **The earnings premium is real, but trading costs eat it.** Over 2,466 earnings events (2020-26), selling a defined-risk condor or fly the day before and buying back the day after earns about **+4% of capital at risk per event at mid**. Crossing the bid-ask on all 4 legs, both ends, turns that into −20% to −33%. Break-even needs paying ≤ 12-15% of the half-spread. Buying straddles into earnings loses even at mid. | CLOSED as a next-day trade. Real-fill question open. | same |
| 3 | **Holding the earnings trade to expiry makes it worse.** One crossing instead of two, but the result goes to −10% (condor) / −15% (fly), and even at mid it is about zero or negative. The premium is the overnight IV crush only; after that the move belongs to the buyer. | CLOSED | `RESULT_EARNINGS_TO_EXPIRY_20261109.md` |
| 4 | **Delta-hedged SPY short vol** (2010-23, monthly): +1.2% of capital at risk per trade [−0.8, +3.3], Sharpe ~0.3. Not significant; it would take ~39 years of data to confirm. Hedging cuts risk (worst trade −100% → −46%) but doesn't add return. "Sell when IV is rich vs realised" pointed the **wrong** way. | CLOSED on this data; needs more data (SPX 2004+, more index underlyings) | `RESULT_HEDGED_INDEX_VOL_20261109.md` |
| 5 | **The gate still picks better credit spreads** on fresh data (Oct 13 - Nov 6): Bear Call +0.19, Bull Put +0.17, 73-76 symbols. That is relative (passed vs refused), not profit. It does nothing for Long Call/Long Put. | CLOSED (replicated) | `reports/gate_separation/gate_separation_20261109T215401Z.json` |
| 7 | **The earnings filter had been mostly blind since 08-20.** Its calendar cache was never refreshed, and a report on the entry day after the close counted as "already priced in". The XYZ Long Put bought on report day (11-03) lost 99.6% overnight. **Fixed:** weekly calendar refresh, same-day after-close reports count as held through, and long options are now gated too (buying into earnings lost −16% on average, E1-L). | FIXED (live) | `config.json` → `auto_log.refuse_long_through_earnings` |
| 8 | **The 9 trades while you were away (10-16 .. 11-09):** all Long Puts, +$241. Holding the same market exposure would have made +$786 (SPY −2.5%), so the picks lagged their own direction. Two winners (INTC, SLV) carried it. The gate doesn't separate Long Puts, so these were effectively random draws. | CLOSED (descriptive) | `paper_trades.db` ids 1116-1124 |
| 6 | **Friction-refused Bear Calls are reliably worse than their delta** (−3.1%). Cost control is the part of the system that clearly works. | supports the gate | B1 result |

## 2026-10-15

| Finding | Status |
|---|---|
| **The live Bull Put rule is "costly beta"**: on SPY 2010-23 it trails holding its delta by 2.1% per trade [−2.7, −1.6]. The deficit is the bid-ask crossing on a ~3-day hold. Vetoed for real money; auto-logging stopped. | CLOSED |
| **A SPY put spread held to expiry** (one crossing) also fails to beat its delta (−0.8%). Every put-spread measurement lands at or below its delta: on this data a put spread is just a way to hold the market. | CLOSED |
| **Decision rule agreed**: the only path to real money is a pre-registered rule that passes history, then ≥ 8 forward weeks, then a $1,500 pilot to measure real fills. | **BASELINE** (`DECISION_RULE_20261015.md`) |
| The EV gate's skew treatment puts index puts at a disadvantage (index pass rate 5.2% vs single names 13.2%). | CLOSED |

## 2026-10-10 .. 10-14 — directional signals

All null or below the cost of trading options: earnings-surprise drift (+0.14% at 2 sessions, real but ~5× too small), news event types, analyst up/downgrades, post-shock drift (MRNA was the right tail, which can't be picked in advance), long options at 1-3 sessions (lose ~8% of premium, mostly the spread). **Predicting direction from public data is exhausted here.** CLOSED.

## 2026-09-28 .. 09-30

- The paper book trailed SPY by about $7,700 over 5.4 months; no edge was measurable (PF 1.04 [0.87, 1.24]). CLOSED.
- 750 exit-rule combinations: none promoted. Exits are second-order. CLOSED.
- Bull Put scarcity is caused by the EV gate, not the universe. CLOSED.

## 2026-08 and earlier

- Friction (bid-ask cost) filter: −6.76% → −0.64% return on capital. **The largest effect measured anywhere.** BASELINE.
- Single-name short vol: PF 0.30, no edge. CLOSED.
- optionsDX SPY bull put 2010-23: +2.30% per trade before delta adjustment. Superseded by the 10-15 delta-adjusted results.

---

## The one-paragraph summary (as of 2026-11-09)

Nothing has yet passed the decision rule, so real money stays off and the
default is to hold an index fund. Every edge found is either **real but eaten
by trading costs** (earnings premium, index short vol at mid, and — as of the
11-10 audit — ETF Bear Calls, whose paper profit is a mid-exit figure). The system's clearest value is **cost control**. The
next verdicts are scheduled: Bear Call forward ≥ 2027-01-20, earnings forward
on our own archive ≥ 2027-03-15.

## Pending checks

| Date | Check | Command / doc |
|---|---|---|
| ≥ 2027-01-15 | Bull Put §2 (for the record) | `python -m scripts.delta_matched_benchmark --cache NEW_EMPTY_DIR --exit-through 2027-01-15` |
| mid-Jan 2027 | Analyst estimate revisions | Registered (`PREREG_PEAD_AND_REVISIONS_20261124.md`), date-gated estimator ready: `python -m scripts.estimate_revisions_test` (`--coverage-only` any time; 2/4 non-overlapping snapshots as of 2026-11-24). |
| ≥ 2027-01-20 | **B1-F Bear Call forward** | `python -m scripts.bear_call_absolute --cache DIR --forward` (prints the crossed-fill line, Amendment 3) |
| ≥ 2027-03-15 | E2 earnings forward (own archive) | `python -m scripts.earnings_premium_dolt --forward`; `--coverage` any time. **At risk: 7 events already lost to chain-archive gaps.** |
