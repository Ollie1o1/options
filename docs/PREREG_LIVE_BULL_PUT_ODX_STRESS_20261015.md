# The live Bull Put rule on optionsDX SPY 2010-2023 — REGISTRATION

**Frozen 2026-10-15, before this rule was run on this data.** Task 4 of the 10-15
strategic reset.

## Provenance

- `docs/OPTIONSDX_RESULTS_20260811.md` has already been read. Different rules on
  the same data:
  - `bull_put`: 25-60 DTE, 25 delta, $5 wide, held to expiry. n=1,591, +2.30%
    RoC, DSR 0.699 (not significant).
  - H3: by tenor band. The apparent tenor edge was the bull market, and friction
    is non-stationary past 120 DTE.
- Tasks 2 and 3 (same day) are the motivation:
  - The shadow corpus cannot test index vs single-name.
  - The live single-name book's P&L is all delta. The premium leg came out
    −4.0% of capital at risk per trade (inconclusive).
- This specific rule — ~14 DTE, ~32 delta, ~1.5%-of-spot wide, TP/stop, 3-day
  time exit — has never been run on optionsDX. Nothing below was tuned on it.

## The live rule, as specified

From `config.json` → `exit_rules` and `paper_manager._evaluate_multileg_exit`,
plus the observed live entries (66 sized-era Bull Puts: DTE 8-31, median ~14,
mean 16; short delta 0.19-0.40, mean 0.32; width ~1.5% of spot; credit/width
mean 0.32):

| parameter | value | source |
|---|---|---|
| structure | bull put vertical | |
| expiry | nearest listed with DTE in **[14, 31]** | live median DTE ≈ 14 |
| short leg | put nearest **0.32** delta, within ±0.10 | live mean short delta |
| long leg | nearest listed strike **1.5% of spot** below the short | live widths ≈ 1.5-1.9% of spot |
| credit / width floor | **≥ 0.20** | `filters.credit_spreads.min_credit_to_width` |
| friction ceiling | crossing cost **≤ 25%** of credit | `auto_log.max_friction_to_credit` |
| take profit | **+50%** of credit | `exit_rules.spread.take_profit` |
| stop | **−100%** of credit | `exit_rules.spread.stop_loss` |
| time exit | DTE ≤ 21 **and held ≥ 3 days** | `time_exit_dte`, `min_days_held` |
| check order | TP, stop, then time | `_evaluate_multileg_exit` |
| fills | at the touch, entry AND exit (engine `fill_with_reason`) | conservative vs the live `limit` policy |
| entry | **every trading day**, no concurrency cap binding (`max_concurrent` 1000) | per-trade question |

Engine support (default-off, no existing behaviour changes):
- `exit.min_days_held` gates only the time exit, as in the paper manager.
- `entry.width_pct_of_spot` overrides `width` with pct × implied spot.
- `entry.min_credit_to_width` refuses entries below the floor.

## Population

SPY, every `available_dates` day from 2010-01-04 to 2023-12-29. Each day opens
at most one position. Excluded from the return series: positions closed as
`ticker_ended`, and positions with no capital at risk.

## Outcomes

Per trade i, with c = capital at risk per contract and P&L per contract:

- **RoC** = pnl / c.
- **Delta-equivalent** H = Δ_net · 100 · (S_exit − S_entry), where
  - Δ_net = −(Δ_short) + (Δ_long), from the chain's own deltas at entry (sign
    for a long-delta position: positive);
  - S comes from `implied_spot_any` on the entry and exit dates.
- **Excess** x = (pnl − H) / c.
- **PF on capital at risk** = Σ positive RoC / |Σ negative RoC|.

**Cluster:** calendar month of entry. Holds are ~3-10 days, so adjacent months
barely overlap. Bootstrap 10,000 resamples of months, seed 20261015, percentile
95%.

## H-A (primary) — does the live rule on INDEX earn anything beyond its delta?

Statistic: mean x across all trades, month-clustered.

| result | verdict |
|---|---|
| CI lower > 0 | **index premium exists** under the live rule |
| CI upper < 0 | **costly beta** on index too |
| otherwise | **inconclusive** |

Reported with it, with no override:
- mean RoC and PF on capital at risk, each with a CI;
- win rate;
- per-year table (n, mean RoC, mean x, PF, SPY total return);
- the stress years **2011, 2015, 2018, 2020, 2022** called out;
- worst drawdown of a fixed-fractional book. Each trade risks 2% of a constant
  $100k notional, and P&L is booked on the exit date, not compounded. The
  drawdown is reported both in % of notional and in days to recover.

## H-B (secondary, own decision line) — does the gate's flat-RV EV separate on index?

For each trade, recompute the live negative-EV verdict from the entry-day chain:
- per leg: gross_instrument = 100·(BS_put(S, K, σ_rv, T) − mid), with r = 2%,
  q = 1.5%, T = DTE/365;
- σ_rv = 0.5·hv_252d + 0.5·hv_30d, close-to-close, annualised, from the
  implied-spot series up to and including the entry date. The live hv_30d also
  blends EWMA and Parkinson, which needs OHLC this data lacks — a stated
  deviation;
- gross = g_long − g_short; cost = Σ 100·(ask − bid); net = gross − cost;
  noise = 0.25·cost (the structure fallback);
- **TAKE** iff gross > 0 and net > noise; otherwise refused.

Statistic: mean x(refused) − mean x(TAKE), with a month-clustered bootstrap
that resamples months and recomputes both arms' means.

| result | verdict |
|---|---|
| CI upper < 0 | **gate separates correctly** on index (refused worse) |
| CI lower > 0 | **gate inverted** on index (refused better) |
| otherwise | **no separation** |

Also report the TAKE share, and RoC by gate arm.

**Mechanism cut (no authority):** skew penalty quintiles — (flat-IV-at-short
value − credit)/width — mean x by quintile.

## Sensitivities (no authority)

1. DTE window [8, 31] (picks ~8 DTE).
2. Hold to expiry (no TP, stop or time exit), everything else the same.
3. 25 delta instead of 32.
4. Mid fills (half the crossing cost back on entry and exit), as an upper bound
   for the live `limit` policy.

## Authority

Neither verdict changes live config. Their role in task 5's decision rule:
- **H-A positive and H-B "inverted":** grounds for a gate-redesign proposal (a
  skew-aware real-world distribution) and an index-only shadow arm. A design
  doc, not a config flip.
- **H-A inconclusive or negative:** the "index premium" case for the book is
  not supported under the live rule, and task 5 proceeds without it.

## Prior

- H-A: weakly positive point estimate (index VRP is well documented), but
  inconclusive at a 3-day hold. TP/stop at the touch on 14-DTE spreads gives
  back a lot of credit.
- H-B: no separation, or inverted.

Estimator: `scripts/live_bull_put_odx_stress.py`.
