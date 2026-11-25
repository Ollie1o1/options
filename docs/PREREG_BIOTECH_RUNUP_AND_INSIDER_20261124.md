# Event-Driven Equity Playbooks 5 & 6: Funded Pre-Catalyst Biotech Run-Up & Insider Cluster Buys — REGISTRATION

**Frozen 2026-11-24, before any outcome has been computed for Playbook 6 (Biotech Run-Up) or the equity-execution extension of Playbook 5 (Insider Clusters).**
Part of the stock-first daily profit reset (Phase 1 of the execution plan).
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **Playbook 6 (Biotech Pre-Catalyst Run-Up):**
  - `reports/CATALYST_PREREG.md` (v3, 2026-08-27) tested 6-month **holding THROUGH** Phase 2/3 primary completion dates (`FUNDED THROUGH` vs `RAISE BEFORE`, endpoint amendments, Phase 3 vs Phase 2) across 12 quarterly vintages (`2023-01-01` .. `2025-10-01`, `270` distinct tickers, `data/catalyst_pit.db`), returning `NO EVIDENCE` on holding through the binary window.
  - **Untested mechanism:** Exiting **BEFORE** the binary catalyst (`T - 45` trading days → `T - 3` trading days) — the well-known biotech "run-up" effect, where speculative positioning bids funded late-stage biotechs higher into a dated catalyst while avoiding the asymmetric `-60%` binary trial-failure gap and avoiding pre-catalyst equity dilution via the `FUNDED THROUGH` cash-runway filter!
  - Zero run-up window returns (`T - 45 → T - 3`) have ever been computed on `data/catalyst_pit.db`.
- **Playbook 5 (Insider Cluster Buys):**
  - `docs/superpowers/specs/2026-09-21-insider-cluster-backtest-design.md` (approved 2026-09-21) already specifies Stage 1 (underlying-only EDGAR bulk Form 4 backfill `2024-01-01..present` → `data/insider_backtest.db`, filed-date entry, moving-block bootstrap over filed dates, primary test `CLUSTER BUY vs NONE` asymmetry at `k=2, 42d`).
  - This document does **not** alter or duplicate that approved spec — it registers the exact **executable equity P&L companion metric** so that once `data/insider_backtest.db` lands, Playbook 5 can be promoted cleanly into `configs/playbooks.json` under the same standard as Playbooks 1–4.

---

## Playbook 6 — Funded Pre-Catalyst Biotech Run-Up (Shares, `T - 45 → T - 3`)

### Population & Point-in-Time Discipline
- **Data:** Existing `data/catalyst_pit.db` (`pit_universe`, `pit_study`, `pit_versions`, `pit_facts`, `pit_prices`, `pit_caps`) + `data/squeeze_prices.db` (fallback daily closes/volumes if a window extends beyond `pit_prices`).
- **Strict Point-in-Time Rule:** Every feature (trial status, primary completion date `PCD`, `FUNDED THROUGH` cash runway from `pit_facts`, market cap `$50M–$10B`) is reconstructed **strictly `as_of <= entry_date`** using `src/catalyst/pit.py` (never live ClinicalTrials.gov endpoints!).
- **Eligibility (Evaluated at `T_entry = PCD - 45 trading days`):**
  1. Trial is **Phase 2 or Phase 3**, industry-sponsored, US-listed ticker `$50M <= cap <= $10B`.
  2. Date precision guard: Include only trials whose `PCD` recorded at `T_entry` has **day-level OR month-level** precision; if month-level (`~YYYY-MM`), anchor `PCD` conservatively at the **1st trading day of that month** (so `T_exit = PCD - 3` always exits *before* the month begins, preventing accidental binary exposure!).
  3. **Runway Gate (Primary Arm `B_FUNDED`):** Cash runway `as_of <= T_entry` is `FUNDED THROUGH` (`cash / quarterly_burn` covers `PCD + 6 months` per `src/catalyst/runway.py`).
  4. **Tradability Gate:** `close[T_entry] >= $2.00` AND `median_20d(close * volume) >= $1,000,000`.
  5. **One position per `(ticker, PCD_month)`:** Multiple trials on the same ticker expiring in the same month collapse into **one trade** (fixes the VNDA×17 row-duplication bug caught in `CATALYST_PREREG.md` v2!).

### Execution Timing, Stops & Friction
- **Entry:** Close of `T_entry + 1` (`PCD - 44` trading days).
- **Exit:** First of:
  1. **Pre-Event Mandatory Exit:** Close of `PCD - 3` trading days (strictly BEFORE primary completion / readout window!).
  2. **Early Trial-Termination / Gap Stop:** Hard stop `-20%` on daily close (executed at next close `k + 1`, protecting against surprise early interim halt / surprise secondary offering).
- **Friction:** **`25 bps` per side = `50 bps` (`0.0050`) round-trip** (small/mid-cap biotech spread realism).

### Observation Unit, Clustering & Hypotheses (Playbook 6)
- **Cluster unit:** **Ticker** (resampling distinct tickers with replacement, matching `CATALYST_PREREG.md` v2!) **AND** secondary check clustering by **entry quarter** (biotech beta via `XBI` correlates across dates). Both CIs reported; **the wider CI governs**.
- **Bootstrap:** `10,000` draws, seed `20261124`, 95% percentile CI.
- **Benchmark:** Matched-window **`XBI`** return `b_xbi` AND matched-window **`SPY`** return `b_spy` over exact `[entry_date, exit_date]`.
- **Minimum Sample Floor:** `MIN_TICKERS = 40` distinct funded tickers in `data/catalyst_pit.db`. Below 40 → `INSUFFICIENT`.
- **H6-A (Primary — Absolute Net Run-Up Return):** `mean(r)` 95% CI lower `> 0` AND `PF(r)` 95% CI lower `> 1.0`.
- **H6-B (Secondary — XBI-Relative Alpha):** `mean(r - b_xbi)` 95% CI lower `> 0`.
- **Control Arm (No Authority, Mechanism Check):** Same window on `RAISE BEFORE` (underfunded) biotechs — tests whether the `FUNDED THROUGH` dilution filter actually separates run-up performance.
- **Sensitivities (No Authority):** Day-precision dates only vs month-precision; Phase 3 only vs Phase 2; `T - 60 → T - 5` window; `100 bps` slippage stress.

---

## Playbook 5 — Insider Cluster Buys Equity Companion Spec (Tied to `2026-09-21` Approved Spec)

Per `docs/superpowers/specs/2026-09-21-insider-cluster-backtest-design.md`, Stage 1 tests underlying asymmetry (`P(+2σ) - P(-2σ)` at 42d) on `CLUSTER BUY` vs `NONE` using EDGAR daily-index Form 4 filings (`2024-01-01..present`, entry at **filed date**, moving-block bootstrap over filed dates).

### Companion Executable Equity Metric (Frozen Now Before Backfill Runs)
Once `src/insider/backtest/` executes Stage 1:
- **Entry:** Close of `filed_date + 1` trading day on tickers passing `cluster_score().label == "CLUSTER BUY"` (≥ 2 distinct open-market code `P` buyers within 90d), `close >= $5.00`, `median_20d(close * volume) >= $2,000,000`.
- **Exit:** Fixed **`42` trading days** (`filed_date + 43`), with `-15%` hard stop (executed next close).
- **Friction:** **`15 bps` per side = `30 bps` round-trip**.
- **Pass Bar:**
  1. Stage 1 primary asymmetry cell (`CLUSTER BUY vs NONE, k=2, 42d`) clears its approved moving-block 95% CI `> 0` on the `2025+` holdout; **AND**
  2. Net equity return `mean(r)` AND matched-`IWM` excess return `mean(r - b_iwm)` both have moving-block 95% CI lower `> 0` (minimum `30` distinct filed-date clusters in holdout).

---

## Authority & Prior

- **Authority:** Passing promotes `biotech_runup` and/or `insider_cluster` from `[RESEARCH]` to `[VALIDATED]` in `configs/playbooks.json` and elevates them on `src/radar/`. No live automated execution.
- **Prior:** Biotech run-up is historically strongest on exact-dated **PDUFA** dates (`src/catalyst/pdufa.py`) and noisier on approximate ClinicalTrials.gov primary completion dates (because `PCD` lags topline by 1–3 months, as noted in `README.md`), so `B_FUNDED` on `catalyst_pit.db` may be underpowered or noisy — running this pre-registered test settles whether the existing `catalyst_pit.db` dataset has tradable pre-event alpha or stays purely descriptive context. Insider cluster buys have strong multi-decade academic support in shares (especially mid/small caps) once the 30-hour EDGAR backfill completes.

**Estimators:**
- Playbook 6: `scripts/backtest_biotech_runup.py` → `docs/RESULT_BIOTECH_RUNUP_20261124.md`
- Playbook 5: `src/insider/backtest/` (per `2026-09-21` spec) + equity P&L report → `docs/RESULT_INSIDER_CLUSTER_20261124.md`
