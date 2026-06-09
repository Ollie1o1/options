# Trust & Data-Integrity Roadmap — Status

Status of the four-phase effort to make this screener a *trusted* source of options
data: provenance on every number, facts separated from forecasts, and engineering
that signals seriousness. All four phases are implemented and committed on the
`trust-data-integrity` branch.

---

## Phase 1 — Data provenance threading ✅

yfinance serves delayed/stale quotes with no per-contract record of when a quote
was struck. Now every contract carries provenance that threads from fetch →
scoring → display → API → CSV.

- **`src/data_quality.py`** (new): `classify_quote_freshness(age, market_open)` and
  `check_market_hours()`. The latter is now the single source of truth —
  `options_screener._check_market_hours` delegates to it.
- **`data_fetching.fetch_options_yfinance`** stamps each contract after the chain
  concat; `_process_option_chain` sets the base source.
- **`enrich_and_score`** re-tags synthesized rows and computes `quote_freshness`
  early so it survives all filtering.
- **`calculate_scores`** applies a −0.05 quality penalty to stale quotes
  (`-stale_quote(-0.05)` in `score_drivers`) — downgraded, never dropped.
- Surfaced: per-pick freshness flag, aggregate `Data quality: …` line, four new
  columns in API serialization and CSV export, documented in `schemas.py`.

## Phase 2 — Internal IV cross-validation ✅

Yahoo's `impliedVolatility` is unreliable on illiquid strikes. We verify it with a
Black-Scholes inversion of each contract's own mid price.

- **`data_quality.implied_vol_from_price`** — Brent solver, bracket σ ∈ [0.005, 5.0],
  returns None for invalid inputs or when no root exists (below intrinsic / above
  model max).
- **`data_quality.cross_validate_iv`** — adds `iv_solved`, `iv_residual_pct`,
  `iv_verified`. Pure; does not mutate `impliedVolatility`.
- **`enrich_and_score`** runs it right after the IV smile filter. When Yahoo's IV is
  unverified (or missing) and an IV can be solved, the solved IV is adopted for all
  downstream Greeks/PoP/EV; corrections are logged at INFO. Verified rows pass through
  unchanged (confirmed: no spurious over-filtering).
- Surfaced: per-pick `IV verified ✓` / `IV corrected (yahoo X% → solved Y%)`,
  aggregate `IV cross-check: …` line, three new columns in API + CSV.

## Phase 3 — Trust infrastructure ✅

- **`.github/workflows/ci.yml`** — push/PR CI. Test job on Python 3.11 + 3.12,
  installs `requirements-lock.txt` + `requirements-dev.txt`, runs
  `pytest tests/ -x -q --ignore=tests/_phase3_stress.py -m "not network"`. Separate
  non-blocking mypy job.
- **`pyproject.toml`** — registers the `network` pytest marker + testpaths.
- **README** — CI badge; install leads with `requirements-lock.txt` (with the
  "yfinance breaks between versions" rationale); **Module Maturity** table
  (Stable / Beta / Experimental).

## Phase 4 — Honest uncertainty labeling + public track record ✅

- **`src/evidence.py`** (new): `load_model_evidence()` parses the latest
  `reports/walk_forward_*.json` and `reports/checkpoint_history.tsv` into
  `{pooled_ic, p_value, n_oos, cohort_n, gate_decision, as_of}` with safe defaults;
  `format_evidence_banner()` renders the one-line EXPERIMENTAL label.
- **`ranking.py` + `cli_display.py`** show the banner under the AI panel and in the
  executive summary — always read from artifacts, never hardcoded, so it updates as
  evidence accumulates.
- **`scripts/publish_track_record.py`** renders `reports/TRACK_RECORD.md` from
  `paper_trades.db` (win rate, returns, per-strategy breakdown, gate status, full
  closed-trade table) with the paper/delayed-data/friction caveat stated plainly.
  Wired into the weekly startup-maintenance throttle.
- **README** — "What You Can Trust Today" (facts vs forecasts) + TRACK_RECORD link.

---

## New data-quality columns (what they mean)

| Column | Type | Meaning |
|--------|------|---------|
| `quote_source` | str | `yfinance`, `yfinance+synthetic_spread` (bid/ask reconstructed around lastPrice when the chain is zero-quoted), or `yahooquery` |
| `quote_as_of` | str | UTC ISO timestamp of the contract's last print (Yahoo `lastTradeDate`); NA if missing |
| `quote_age_min` | float | Minutes between `quote_as_of` and the fetch; NaN if unknown |
| `quote_freshness` | str | `fresh` (≤20m, market open) / `delayed` (≤120m) / `stale` (>120m, or any age while market closed) / `unknown` (no timestamp) |
| `iv_solved` | float | IV solved from the contract's mid via Black-Scholes inversion; NaN when unsolvable |
| `iv_residual_pct` | float | `(yahoo_iv − iv_solved) / iv_solved`; NaN if either is missing |
| `iv_verified` | bool/None | True if `|residual| ≤ 15%`, False if not, None when no IV could be solved |

Internal helper columns also added: `iv_yahoo` (original IV before any correction)
and `iv_corrected` (bool, drives the display).

---

## What remains manual / deferred

- **Polygon cross-source verification (deliberately deferred).** A `src/polygon_client.py`
  exists and `POLYGON_API_KEY` already enriches news/VWAP, but we do **not** yet
  cross-check Yahoo prices/IV against a second source (Polygon). That is the next
  obvious trust step: compare the two providers per contract and widen
  `quote_freshness` / IV verification into a true cross-source agreement check.
  It needs a paid key and live network, so it was left out of this scope.
- **First CI run triage.** CI could not be executed locally (see deviations below).
  The first GitHub Actions run is the real verification of the full `pytest tests/`
  subset, including the experimental `tests/crypto`, `tests/leverage`, etc. subdirs;
  expect to triage any module-specific or dependency gaps there.
- **`network` marker is registered but unused.** No current test hits live network
  (the only HTTP test fully mocks `requests`). Mark any future live-network test
  with `@pytest.mark.network` and CI will deselect it automatically.

---

## Testing note

New tests are written as `unittest.TestCase` (not bare pytest functions) so they run
both locally via `python -m unittest` and under pytest in CI. They are pure and
network-free: `tests/test_data_quality.py`, `tests/test_iv_solver.py`,
`tests/test_evidence.py`.
