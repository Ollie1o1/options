# Delta-matched benchmark for the Bull Put book — REGISTRATION

**Frozen 2026-10-15**, before any delta-equivalent P&L was computed. Task 3 of
the 10-15 strategic reset.

The book's own P&L was already known (+$878 on $30.5k at risk, 59 closed
sized-era trades), so this is a benchmark definition, not a blind test. It
fixes what "beats" means before the hedge leg is computed.

## Question

A Bull Put is long delta. Over 2026-08-20 → 10-15 the market mostly rose. Does
the book earn anything **beyond** the P&L of holding its own delta in the
underlying over the same window? If not, it is an expensive way to be long.

## Population

`paper_trades.db` (repo root), table `trades`:
- `strategy_name = 'Bull Put'`
- `status = 'CLOSED'`
- `date >= '2026-08-20'` (the sized era)
- `pnl_usd IS NOT NULL`

That gives 59 trades on 28 symbols. Open trades are excluded.

## Definitions

Per trade *i*, with q = `quantity` and 100 shares per contract:

1. **Entry spot S₀:**
   - Solve S₀ from the short leg so that BS_put(S₀, K_short, `entry_iv`, T) =
     short mid (`short_bid_entry`/`short_ask_entry`).
   - Use r = 4%, q_div = `dividend_yield` (or 0 if NULL), T = calendar days to
     `expiration` 16:00 ET from entry-date 12:00 ET.
   - Sanity: |S₀ / daily close(entry date) − 1| ≤ 3%. Otherwise fall back to
     that close and count it.
2. **Position delta per share:** Δ = |Δ_short| − |Δ_long|, where
   - Δ_short = BS delta at (S₀, K_short, `entry_iv`), and
   - Δ_long uses the long leg's own IV, solved from its mid at S₀.
   - Δ > 0.
3. **Exit spot S₁:** the close of the Yahoo 1-hour bar containing `exit_date`'s
   timestamp. Use the last bar at or before it if the timestamp falls outside
   regular hours. If no intraday bar exists, use the daily close of the exit date.
4. **Delta-equivalent P&L:** H = Δ · 100 · q · (S₁ − S₀). This is a STATIC
   hedge sized at entry. It is the first-order beta of the position, not a
   dynamically rebalanced replica.
5. **Excess:** X = `pnl_usd` − H, the part of the P&L not explained by the
   entry delta (theta + vega + gamma + costs).
6. **On capital:** x = X / `capital_at_risk`.

**Amendment, before computation:** 16 of the 59 trades have NULL leg quotes. For
these:
- S₀ = the daily close of the entry date.
- The long leg is priced at the short leg's `entry_iv` (flat).

They are flagged `s0_fallback`, and the sensitivity that drops fallback rows
drops them too. Times are America/Toronto (= New York), the ledger's local
clock.

## Estimator and decision rule

- Primary: **mean x**, with a CI from 10,000 bootstrap resamples **clustered by
  entry ISO week**, seed 20261015.
- Also reported: Σ`pnl_usd`, ΣH, ΣX in dollars, and the share of the book P&L
  that ΣH accounts for.

| result | verdict |
|---|---|
| CI lower > 0 | **beats its delta** — earns something beyond beta |
| CI upper < 0 | **costly beta** — worse than holding its own delta |
| otherwise | **inconclusive** |

Sensitivity, with no authority: cluster by symbol instead of week, and drop
trades where S₀ fell back to the daily close.

**Authority:** none on its own. It feeds the 2027-01-15 decision rule (task 5).
The proposed condition there is that the book must **beat the delta-matched
benchmark**.

## Prior

ΣH is likely a large share of +$878, because the market rose. The verdict is
likely inconclusive: 59 trades in ~8 weekly clusters.

Estimator: `scripts/delta_matched_benchmark.py`.
