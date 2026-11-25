# Equity Playbook 4: Multi-Week PEAD & Analyst Estimate Revisions (Shares) — REGISTRATION

**Frozen 2026-11-24, before any outcome has been computed on either Arm A or Arm B.**
Also formally fulfills the open registration requirement in `docs/RESEARCH_FINDINGS.md` ("mid-Jan 2027: Analyst estimate revisions — write the hypothesis down before opening the archive").
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **What has already been seen:**
  - `docs/RESEARCH_FINDINGS.md` (2026-10-10..14): Earnings-surprise drift over **2 trading sessions** measured `+0.14% ± 0.10%` (`n = 19,867`) — statistically real, ~5× too small for a `~0.70%` options spread hurdle.
  - `docs/DATA_COVERAGE_AUDIT_20261124.md`: `data/estimates_archive.db` holds `61,690` estimate fields across `105` symbols (`6` snapshot dates, `2026-10-14` .. `2026-11-20`) and `49,526` analyst actions across `105` symbols (`4` first-seen dates, `2026-10-14` .. `2026-11-17`). No forward returns were ever joined or inspected against `estimates_archive.db`.
  - `data/dolt_options.db::earnings_cal` holds `5,067` timed earnings events across `196` symbols (`2020-01-22` .. `2026-12-17`).
- **What has NOT been seen:**
  - Zero multi-week (`20d` / `40d`) post-earnings reaction drift tests in **shares**.
  - Zero returns or correlations computed from `data/estimates_archive.db`.

---

## Two-Arm Design

Because historical estimate-revision vintages do not exist before `2026-10-14` (and `DATA_COVERAGE_AUDIT_20261124.md` warns never to backdate historical grade dates), this study splits cleanly into:
- **Arm A (Historical, Runnable Now):** Post-Earnings Price-Reaction Drift (**PEAD-Price**) in shares over `20` and `40` trading days (`2020-01` .. `2026-08`).
- **Arm B (Forward, Locked Until `>= 2027-01-15`):** Point-in-time **Analyst EPS Estimate Revisions** from `data/estimates_archive.db` using immutable snapshot / `first_seen` dates only.

---

## Arm A — Multi-Week Post-Earnings Reaction Drift (Shares, 2020–2026)

### Economic Mechanism
When a liquid stock reports earnings and gaps sharply higher on heavy volume (a market-validated positive fundamental surprise, avoiding dependence on noisy third-party EPS surprise scraping), institutional portfolios rebalance gradually over the following 4–8 weeks (Post-Earnings Announcement Drift). Entering **1 session after** the announcement in **shares** (5 bps friction) avoids the binary earnings gap and captures the multi-week drift.

### Population & Timing (Zero Look-Ahead)
- **Events:** All rows in `data/dolt_options.db::earnings_cal` (`whn` in `before-open` or `after-close`, `2020-01-22` .. `2026-06-30`) matched to `data/squeeze_prices.db` (`px`, daily close & volume; cross-checked against `data/equity_ohlcv.db` where present).
- **Announcement Reaction Session `D`:**
  - `before-open`: `D` = event date (first trading session on or after event date).
  - `after-close`: `D` = first trading session strictly after event date.
  - Let `D - 1` be the last trading session strictly before `D`.
- **Entry:** Signal evaluated at close of session `D`. **Enter at close of session `D + 1`** (1 full trading day after the earnings reaction, guaranteeing realistic retail entry after initial chaos subsides!).
- **Exit:** Fixed horizon close of **`D + 21`** (primary: ~1 month, 20 sessions held) and **`D + 41`** (secondary: ~2 months, 40 sessions held).
- **Signal Definition (Frozen — Positive Earnings Breakout):**
  1. **Earnings Reaction Jump:** `reaction_ret = close[D] / close[D - 1] - 1.0 >= +0.05` (`>= +5%` reaction)
  2. **Volume Confirmation:** `volume[D] >= 2.0 * median_20d(volume[D-21..D-1])` (institutional participation)
  3. **Follow-Through / No Immediate Fade:** `close[D + 1] >= close[D] * 0.98` (holding at least 98% of the reaction close by entry bar `D + 1`; prevents buying instant blow-off reversals).
  4. **Liquidity Gate:** `close[D - 1] >= $10.00` and `median_20d(close * volume) >= $20M`.
- **Friction:** **`5 bps` per side = `10 bps` round-trip**.

### Benchmark, Controls & Inference (Arm A)
- **Matched Benchmark:** `SPY` return over exact `[D + 1, D + 21]` (and `[D + 1, D + 41]`). Excess return `x_i = r_i - b_i`.
- **Negative Control (Residualization per `RESEARCH_PROCESS.md` Step 6):** Compare against **non-earnings `+5%` / `2x volume` jump days** on the same universe over 2020–2026 to test whether earnings catalysts drift differently from generic momentum shocks!
- **Split:** `2020-01` .. `2023-06` (first half) vs `2023-07` .. `2026-06` (second half).
- **Cluster unit:** **Earnings Quarter-Month (`(year, month)` of `D`)** — earnings seasons cluster tightly inside Jan/Apr/Jul/Oct. `10,000` cluster-bootstrap draws, seed `20261124`.
- **Minimum Sample Floor:** `MIN_EVENTS = 100` AND `MIN_MONTHS = 24`.
- **Binding Pass Rule (Arm A, Primary Horizon = 20 trading days `D+1 → D+21`):**
  - Excess return `mean(x)` 95% month-clustered CI lower `> 0` **AND** absolute `PF(r)` 95% CI lower `> 1.0` **AND** same positive sign in both halves (`2020–2023H1` and `2023H2–2026`).
  - *(40d horizon and negative-earnings short/avoid mirror `reaction_ret <= -5%` are reported as secondary diagnostics with no promotion authority on their own.)*

---

## Arm B — Point-in-Time Analyst Estimate Revisions (Locked Readout `>= 2027-01-15`)

### Strict Anti-Leakage Contract
- Uses **ONLY** `data/estimates_archive.db::estimate_snapshots` where `snap_date >= '2026-10-14'`.
- **NEVER** uses `analyst_actions.grade_date` as an availability timestamp; any analyst-action diagnostic uses strictly `substr(first_seen, 1, 10)` and excludes the initial backfill dump (`first_seen` on `2026-10-14`).
- The estimator script (`scripts/estimate_revisions_test.py`) **refuses to run before `2027-01-15`** unless passed `--coverage-only` (which prints snapshot/symbol counts and zero returns).

### Feature Definition (Frozen Before Looking)
For each archived symbol `s` on each snapshot date `t` (`snap_date` in `2026-10-14` .. `2026-12-31`):
1. **Primary Revision Breadth (`rev_breadth_30d`):** From `dataset = 'eps_revisions'`, summed across current quarter (`0q`) and next fiscal year (`+1y`):
   ```
   up   = upLast30days(0q)   + upLast30days(+1y)
   down = downLast30days(0q) + downLast30days(+1y)
   rev_breadth_30d = (up - down) / (up + down)     if (up + down) >= 2 else NaN
   ```
2. **Secondary Consensus Magnitude (`eps_delta_0y`):** Percentage change in `earnings_estimate | +1y | avg` between consecutive archive snapshots `t_prev` and `t` (at least 14 calendar days apart): `(avg[t] - avg[t_prev]) / abs(avg[t_prev])`, clipped to `[-0.5, +0.5]` to guard against near-zero EPS denominators.

### Outcome & Estimator (Arm B)
- **Entry:** Close of first trading session strictly **after** `snap_date` (`t + 1`; handles `archived_at` after-close timing cleanly).
- **Horizon:** `21` trading days (`t + 1` → `t + 22`), stock excess return over `SPY`: `y = ret_stock_21d - ret_spy_21d`.
- **Primary Test (Cross-Sectional Rank IC):** Cross-sectional Spearman rank correlation (`src/prereg_ranker.py::rank_ic`) between `rev_breadth_30d` and `y` within each `snap_date`, averaged across non-overlapping `snap_date` clusters (minimum 15 trading days apart).
- **Controls (Step 6):** Residualize `rev_breadth_30d` on trailing 21d own stock return (controlling for short-term momentum) and sector ETF 21d return.
- **Minimum Sample Floor:** `MIN_SNAPSHOTS = 4` non-overlapping snapshot dates AND `MIN_OBS_PER_SNAP = 40` symbols with valid `rev_breadth_30d`. (If `< 4` non-overlapping 21d windows by Jan 15, extend readout automatically to `2027-03-01` — no early peeking!)
- **Binding Pass Rule (Arm B):** Mean residualized cross-sectional rank IC `> +0.04` with cluster-bootstrap 95% CI lower `> 0` (or, if `n_snapshots < 10`, record as `INSUFFICIENT_POWER` and report sign + top-quintile vs bottom-quintile spread honestly to inform whether to keep archiving through Q2 2027).

---

## Authority & Prior

- **Authority:** Arm A pass promotes `pead_breakout` to `[VALIDATED]` on the daily Equity Radar (paper/`own_picks` forward validation). Arm B pass integrates `rev_breadth_30d` as a **validated badge & filter** on the Radar and Ticker Dossier (combining directly with Arm A!). Neither authorizes live automated capital.
- **Prior:** Arm A has a solid historical literature prior (`+0.5%` to `+1.5%` 20d excess drift net of 10 bps stock friction), though mega-caps in `dolt_options` may be more efficient than mid-caps. Arm B by Jan 15 will have ~3 months of archive (~3–4 non-overlapping 21d windows), so it will likely read `INSUFFICIENT_POWER` for strict significance unless the effect is large — registering now locks the exact estimator so we don't p-hack in January.

**Estimators:**
- Arm A: `scripts/backtest_pead_equity.py` → `docs/RESULT_PEAD_EQUITY_20261124.md`
- Arm B: `scripts/estimate_revisions_test.py` (date-gated `>= 2027-01-15`) → `docs/RESULT_ESTIMATE_REVISIONS_20270115.md`
