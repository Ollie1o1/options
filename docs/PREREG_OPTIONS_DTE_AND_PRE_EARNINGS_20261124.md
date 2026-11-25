# Focused Options Tests O1 & O2: 60–120 DTE Wider Index Spreads & Pre-Earnings Run-Up — REGISTRATION

**Frozen 2026-11-24, before either estimator has been run.**
Part of Phase 2C of the execution plan. Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9 and `docs/DECISION_RULE_20261015.md` §3.

---

## Provenance & Explicit Non-Duplication

Every single IV-surface feature (`atm_iv`, `iv_rank`, `vol_of_vol`, `skew_25d`, `term_slope_1m3m`, `entry_depth`, single-name VRP) is **CLOSED** per `docs/OPTIONSDX_RESULTS_20260811.md` and `docs/RESEARCH_PROCESS.md` Step 6. Neither test below introduces a new IV-surface feature or re-sweeps exit grids.

Instead, both tests isolate **untested structural/calendar mechanics** directly motivated by prior measurements:
1. **Why O1 is genuinely new (vs `OPTIONSDX_RESULTS_20260811.md` H3 and `RESULT_LIVE_BULL_PUT_ODX_STRESS_20261015.md`):**
   - H3 (`2026-08-11`) measured `$5`-wide held-to-expiry SPY bull puts across 6 DTE bands and found **`60–120 DTE`** had the **lowest friction of any tenor** (`4.9%` mean / `4.2%` median, `0.3%` over the 25% gate; vs `58.9%` past 250 DTE), with `meanRoC = +6.25%` (`n=691, effN=70`), **but H3 never delta-adjusted it** (concluding raw RoC was confounded by SPY's bull drift).
   - Task 4 (`2026-10-15`) ran a **delta-matched benchmark** on SPY 2010–2023, but ONLY on the short-dated live rule (`14–31 DTE`, `1.5%`-of-spot narrow width, 3-day hold), where two crossings killed it (`-2.10%` excess vs delta).
   - **Never tested:** Delta-adjusted excess return of **wider (`5%`-of-spot, ~`$10–$25`) spreads at the `60–120 DTE` friction sweet spot** (combining the two best-evidenced friction levers: `width` `t = +5.41` OOS + `60–120 DTE` lowest friction!).
2. **Why O2 is genuinely new (vs `RESULT_EARNINGS_AND_BEAR_CALL_20261109.md` E1):**
   - E1 tested holding **THROUGH** earnings (`D - 1 → D + 1`): long straddles lost `-16.1%` at touch, short condors/flies earned `+4%` at mid but `-20%` to `-33%` crossed.
   - **Never tested:** The **pre-earnings anticipation window (`T - 10` trading days → `T - 1` close, strictly closed BEFORE the binary report)** — zero overnight earnings-gap exposure, testing whether systematic retail anticipation flow produces tradable pre-event drift or IV expansion net of crossed spreads!

---

## Test O1 — 60–120 DTE Wider SPY Credit Spreads, Delta-Adjusted (`optionsdx.db`, 2010–2023)

### Population & Data
- **Data:** `data/optionsdx.db` (`odx_chain`), `SPY`, `2010-01-04` .. `2023-12-29` (`18,913,800` rows).
- **Entry cadence:** **Weekly (every Friday, or prior trading day if holiday)**, `max_concurrent = 200` (non-binding for per-trade inference; capacity-capped book reported as secondary diagnostic).
- **Expiry selection:** Nearest listed expiry with `DTE in [60, 120]` (preferring closest to `90 DTE`).
- **Strike geometry (Frozen):**
  - Short put nearest **`-0.25` delta** (within `±0.08` tolerance; drop day if missing).
  - Long put nearest **`5.0%` of implied spot below the short strike** (e.g., `~$15` wide when SPY=$300, `~$25` wide when SPY=$500 — scales cleanly over 14 years without nominal-dollar drift!).
  - Quote filter: both legs `bid > 0`, `ask >= bid`, round-trip crossing cost `<= 25%` of credit (`max_friction_to_credit = 0.25`).
- **Fills:** **Crossed at the touch on BOTH ends** (`short at bid, long at ask` on entry; reverse on early exit; intrinsic settlement with zero exit crossing if held to expiry).

### Frozen Claimants (Exactly 2 Exit Arms — Bonferroni `97.5%` two-sided CI, `alpha = 0.025`)
1. **O1-H (Hold to Expiry — Single Crossing):** No TP, no stop, no time exit — pays crossing friction **once** (entry only).
2. **O1-M (Managed TP 50%, No Stop, 21-DTE Time Exit):** Take profit at `+50%` of credit (crossed), **no stop-loss** (motivated by Policy Lab's `slnone` observation in `POLICY_LAB_FINDINGS_20260928.md`), time exit at `DTE <= 21`.

### Outcomes, Delta Benchmark & Inference (Matches `DECISION_RULE_20261015.md` §3 Exactly!)
- Per trade `i`:
  - `c_i = capital_at_risk = 100 * (width - crossed_credit)`
  - `RoC_i = pnl_i / c_i`
  - **Dynamic / Entry Delta-Equivalent `H_i`:** Primary uses the exact registered Task-4 definition: `H_i = Δ_net_entry * 100 * (S_exit - S_entry)` (for direct apples-to-apples comparison against `DECISION_RULE_20261015.md` §3!), AND reports a **daily re-hedged delta-equivalent sensitivity** `H_dyn` (since 90-DTE holds span longer moves where gamma shifts delta!).
  - **Excess:** `x_i = (pnl_i - H_i) / c_i`
  - **PF on capital at risk:** `sum(max(RoC, 0)) / |sum(min(RoC, 0))|`
- **Clustering:** Because 60–120 DTE holds overlap ~2–3 months, use **moving-block bootstrap over entry calendar months with block length = 3** (NOT length 1, preventing overlap inflation!), `10,000` draws, seed `20261124`. Also report `effective_n` via `src/alloc/validate.py::effective_n`.
- **Minimum Sample Floor:** `effective_n >= 40` independent holding periods (14 years / ~75d hold ≈ `68` independent periods — clears floor!).
- **Binding Pass Rule (Either Arm, Decision Rule §3 Step 2 Bar):**
  - `mean(x)` 97.5% block-bootstrap CI lower `> 0` **AND** `PF(RoC)` 97.5% CI lower `> 1.0` **AND** daily-rehedged excess `mean(x_dyn)` point estimate `> 0` (confirms excess isn't an entry-delta convexity artifact!).
- **Sensitivities (No Authority):**
  - `limit` fill policy (`k = 0.35` half-spread conceded) and `mid` fill upper bound.
  - Same `60–120 DTE, 5%-wide` geometry on **Bear Calls** (`+0.25` delta short call) as a symmetric directional check.
  - Per-year breakdown explicitly highlighting **2011, 2015, 2018, 2020, 2022**.

---

## Test O2 — Pre-Earnings Anticipation Window (`T - 10 → T - 1`, Closed BEFORE Report)

### Population & Data
- **Data:** `data/dolt_options.db` (`earnings_cal`: 5,067 events across 196 symbols, `2020-01-22` .. `2026-06-12`; `dolt_chain` + `stocks_close`, cross-checked against `data/squeeze_prices.db`).
- **Event Timing (Strict Pre-Announcement Cutoff — ZERO Binary Exposure):**
  - Determine announcement reaction session `D` identically to `PREREG_EARNINGS_AND_BEAR_CALL_20261109.md`:
    - `before-open`: `D` = event date → **last safe pre-report close is `D - 1`**.
    - `after-close`: `D` = next session after event date → **last safe pre-report close is event date (`D - 1`)**.
  - **Entry Snapshot `T_in`:** Last Dolt snapshot on or before `D - 11` calendar days (within `[D - 16, D - 10]` calendar days — roughly `T - 10` to `T - 8` trading days before the report).
  - **Exit Snapshot `T_out`:** Last Dolt snapshot strictly **on or before `D - 1`** (within `[D - 3, D - 1]` calendar days). **Any event lacking a snapshot strictly prior to the announcement is dropped and counted — NEVER held across the report!**
- **Expiry Selection (for option arms):** Nearest listed expiration in `T_in` strictly **after `D`** (spanning the upcoming earnings date so it captures pre-earnings IV demand) with `DTE_at_entry <= 45`.
- **Quote Filter (each leg at entry):** `bid > 0`, `ask >= bid`, `(ask - bid) / mid <= 0.25`. Missing exit strike handled via the exact convex linear-interpolation rule from `PREREG_EARNINGS_AND_BEAR_CALL_20261109.md` Amendment 1 (and exact-match sensitivity reported).

### Frozen Claimants (3 Instruments, Bonferroni `98.33%` CI, `alpha = 0.0167`)
1. **O2-EQ (Shares Only — Equity Anticipation Drift):** Buy shares at `T_in` close, sell at `T_out` close (`10 bps` round-trip friction). Benchmark: matched-window `SPY` return `b_i`.
2. **O2-LC (Long ATM Call — Directional + Pre-Earnings IV Run-Up):** Buy call nearest `0.50` delta at `T_in` **ask**, sell at `T_out` **bid** (full round-trip crossing).
3. **O2-ST (Long ATM Straddle — Pure Pre-Earnings IV Run-Up, Delta-Neutral):** Buy call + put at strike nearest `0.50` call delta at `T_in` **ask**, sell both at `T_out` **bid**.

### Clustering & Binding Pass Rules (Test O2)
- **Cluster unit:** **ISO week of `D`** (earnings seasons cluster tightly). `10,000` cluster-bootstrap draws, seed `20261124`.
- **Minimum Sample Floor:** `MIN_EVENTS = 200` AND `MIN_WEEKS = 50`.
- **Pass (Per Claimant):**
  - `O2-EQ`: Excess return `mean(r - b_spy)` 98.33% CI lower `> 0` AND `PF(r)` 98.33% CI lower `> 1.0`.
  - `O2-LC` / `O2-ST`: Mean return on debit `mean(r)` 98.33% CI lower `> 0` AND `PF(r)` 98.33% CI lower `> 1.0` **at crossed fills**!
- **Sensitivities (No Authority):**
  - Mid-fill upper bound (`O2-LC` and `O2-ST` — measures whether pre-earnings IV run-up exists gross of spread, just like E1 measured for post-earnings crush!).
  - Exact-quoted exit matches only (no strike interpolation).
  - Mega-cap tight-spread subset (`SPY` component mega-caps) vs rest of Dolt universe.
  - Per-year breakdown (`2020` .. `2026`).

---

## Authority & Prior

- **Authority:**
  - If **O1** passes, it satisfies `DECISION_RULE_20261015.md` §3 Step 2 (historical optionsDX pass) and authorizes **Step 3 (≥ 8 forward paper weeks)** — NOT live money yet!
  - If **O2-EQ** passes, it joins the Equity Playbook Radar (`configs/playbooks.json`). If **O2-LC / O2-ST** pass crossed, they enter forward paper validation; if positive only at mid, they reinforce the execution/IBKR priority.
- **Prior:**
  - **O1:** Cautiously skeptical on delta-adjusted excess at crossed fills — wider `5%` spreads at `60–120 DTE` cut friction to `~4%` of credit (the lowest anywhere in the repo), so if any defined-risk SPY short-put rule can beat its own delta net of crossing, **this is the best shot**. If O1 fails delta-adjusted, index put credit spreads as a standalone strategy are definitively closed!
  - **O2:** Moderate prior on `O2-EQ` (pre-earnings drift in shares is documented in Frazzini & Lamont / Barber et al. at low stock friction); low prior that `O2-ST` clears crossed spreads (theta burn over 10 days fights IV expansion), though mid-fill sensitivity will show cleanly whether IV expansion exceeds theta.

**Estimators:**
- O1: `scripts/backtest_odx_wide_60_120.py` → `docs/RESULT_ODX_WIDE_60_120_20261124.md`
- O2: `scripts/backtest_pre_earnings_runup.py` → `docs/RESULT_PRE_EARNINGS_RUNUP_20261124.md`
