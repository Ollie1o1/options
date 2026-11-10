# Earnings premium + Bear Call absolute — REGISTRATION

**Frozen 2026-11-09, before either estimator was run.** Three tests: E1 (Dolt
history, runnable now), E2 (forward, archive, judged later), B1 (shadow corpus,
runnable now). Decision-rule context: `docs/DECISION_RULE_20261015.md` §3 — a
pass motivates a forward step; it authorises no money and no config change.

## Provenance

- E1: a June 2026 Dolt short-strangle earnings test (`src/dolt_earnings_sell.py`)
  gave n=24, PF 1.10 on credit, "~breakeven", undefined risk, predating the
  08-14 Dolt defect fixes. That is the only earnings-selling outcome seen. The
  event count below (2,509 usable events, 104 symbols) was measured from
  snapshot DATES only, no prices.
- B1: the registered gate separation test (forward re-run 11-09) shows passed
  Bear Calls beat refused ones within (symbol, expiry, day), d_auc +0.19. That
  is relative. No absolute Bear Call return, and no delta-adjusted figure, has
  been computed.

## E1 — selling earnings vol, defined risk (Dolt 2020-01 .. 2026-06)

- **Events:** `dolt_options.earnings_cal`, `whn` before-open or after-close.
  Announcement effect lands on session D = event date (before-open) or the next
  session (after-close). **Pre** snapshot = last Dolt snapshot on or before
  D−1 (before-open) / the event date (after-close), within 4 calendar days.
  **Post** snapshot = first snapshot on or after D, within 4 calendar days.
- **Expiry:** nearest expiration present in the pre snapshot that is strictly
  after the post snapshot date.
- **Quote filter (each leg, at entry):** bid > 0, ask ≥ bid, (ask−bid)/mid ≤ 0.5.
- **Structures (claimants):**
  - **E1-S** short iron condor: short call/put nearest ±0.25 delta, long
    call/put nearest ±0.10 delta beyond the shorts.
  - **E1-F** short iron fly: short call and put at the call strike nearest
    0.50 delta (put must exist at that strike), long wings nearest ±0.10 delta.
  - **E1-L** long straddle at that same 0.50-delta strike (the opposite bet).
- **Fills:** cross at both ends (entry sell at bid / buy at ask; exit the
  reverse). An event with any leg unquoted at exit is dropped and counted.
- **Outcome:** r = pnl / CAR. Short structures: CAR = 100 × (max wing width −
  credit). Long straddle: CAR = debit × 100.
- **Clustering:** ISO week of D (earnings seasons move together). Cluster
  bootstrap 10,000, seed 20261109.
- **Pass (each claimant):** mean r and PF on r, Bonferroni over 3
  (98.33% two-sided): mean lower > 0 AND PF lower > 1.
- **Sensitivities (no authority):** mid fills; by year; before-open vs
  after-close; excluding 2020; dropped-at-exit share.

## E2 — the same rule forward on our own chain archive (judged later)

E1-S and E1-F exactly as above, applied to `data/chain_archive.db` (daily
snapshots, 32 symbols since 2026-10-16) for earnings dated 2026-10-01 ..
2027-02-28, earnings dates from `src.earnings_provider` fetched AFTER that
window closes. **Judge on or after 2027-03-15.** Expected n is small
(~30-60 events); the forward step's role is to confirm sign and cost, not to
reach significance on its own.

## B1 — gate-passed Bear Calls, absolute and delta-adjusted (shadow corpus)

- **Population:** `candidates` ⨝ `candidate_positions` on (scan_id, board,
  contract_key); `strategy_name='Bear Call'`, `gate_passed=1`, status CLOSED,
  pnl_pct NOT NULL, **expiration ≤ 2026-11-06** (every such path is resolved;
  OPEN remainders reported). Exits dated 2026-08-29/30 dropped (weekend-quote
  artifacts). **One observation per (contract_key, entry_date)**: the first
  scan that day.
- **Return on capital at risk:** r = pnl_pct × credit / (width − credit),
  credit = position entry_price, width = `features_json.spread_width`.
- **Delta-equivalent:** a bear call is short delta. Spot S0 at entry is backed
  out of the short-call mid at `entry_iv` (Black-Scholes, r=4%), rejected if
  > 3% from that day's close (then the close is used); long-leg delta from its
  own implied vol. S1 = daily close on exit_date (Yahoo, unadjusted).
  h = −(Δs − Δl) × (S1 − S0) / (width − credit). x = r − h.
- **Clustering:** ISO week of entry. 10,000 resamples, seed 20261109.
- **Pass (both, the decision rule's A and B):** x CI lower > 0 AND PF on r CI
  lower > 1, 95%.
- **Reference arms (no authority):** refused_by `friction` and `negative_ev`
  Bear Calls, same estimator.
- **Sensitivities (no authority):** cluster by symbol; drop exits dated
  2026-10-12 (Sunday-repair batch); index ETFs vs single names.

## Not claimed

None of these tests is a profitability claim for the live system; passes feed
§3 step 3 (≥ 8 forward weeks). Data caveats stated in advance: Dolt quotes are
EOD on Mon/Wed/Fri and pre-08-14 Dolt results had defects (spots); E1 uses no
spot. B1's shadow fills are paper (execution truth ≈ 27% of credit).

## AMENDMENT 1 (2026-11-09, before the E1 estimator was run on any event set)

A one-event mechanics check (AAPL 2025-10-30) showed Dolt stores a sparse,
day-varying subset of ~27 strikes per expiry, so exact leg strikes are often
absent at exit. The registered "drop if unquoted at exit" would drop events
**selectively by the size of the move** (the stored strikes re-centre after a
gap), biasing toward the seller. Only that one event's outcome (a straddle)
was seen. Amended rule:

- A leg missing at exit is priced by **linear interpolation of bid and of ask
  in strike** between the nearest stored strikes on each side (same expiry,
  type, post snapshot). No bracket on both sides → beyond the stored range:
  price = intrinsic-free bound 0 for an OTM leg is NOT assumed; the event is
  dropped and counted.
- Option prices are convex in strike, so linear interpolation is an upper
  bound on the price: conservative for buying back short legs, generous for
  selling back long legs.
- Sensitivity added (no authority): exact-quoted events only.

## AMENDMENT 2 (2026-11-09, after B1 was run) — B1-F forward replication

B1 ran with A fail (+2.84% [−1.98, +6.33]) and B pass (PF 1.891 [1.087,
3.670]) on only 11 week clusters, so it is NOT a pass. Registered now: **B1-F**,
the identical estimator (`scripts/bear_call_absolute.py`, unchanged except
these two constants) on **entries dated ≥ 2026-10-30 only** (none of which B1
used) and **expiration ≤ 2027-01-15**. Judge on or after **2027-01-20**. Pass =
A and B, 95%, week clusters. The index-ETF sub-population is reported
alongside with no authority (it was seen to carry B1).

## AMENDMENT 3 (2026-11-10, before any B1-F or E2 outcome was computed)

Made after a methodology audit of B1 (`docs/RESEARCH_FINDINGS.md`, 2026-11-10).
No forward outcome was read: B1-F rows were counted (4,600 so far, 3 entry
weeks), E2 events were counted (`--forward --coverage`), nothing more.

**B1-F — the pass rule is unchanged; one line is added beside it.**

- B1's `r` is the shadow position's `pnl_pct`: entry at the `limit` policy
  (35% of the half-spread conceded) and **exit at the mid**. It charges about
  a sixth of a round trip. E1 and the 10-15 SPY test both cross at both ends,
  and on SPY the crossing was the whole deficit.
- Re-priced at the touch at both ends, B1 is PF 1.185 [0.678, 2.235] and excess
  −1.42% [−6.27, +2.15] (index ETFs: PF 1.413 [0.693, 3.142], excess −0.11%).
  B1's "B pass" does not survive the fills E1 was judged on.
- `scripts/bear_call_absolute.py` now prints that line for B1-F
  ("Amendment 3"). **No authority**: the registered A and B decide B1-F.
  Definition, in `cross_fills`: credit = short bid − long ask at scan; exit
  cost = exit mid + both legs' half-spreads **at entry** (per-leg exit quotes
  are not stored); capital at risk = width − crossed credit.
- **Open, the user's call, and only in the stricter direction:** whether a
  B1-F pass must also hold on that line before it counts toward §3.

**B1-F — two instrument changes land with this amendment's PR.** Both apply to
positions opened or resolved after the merge; nothing is back-filled.

- `candidate_positions.entry_date` is the scan day. It was the day the mark
  job next ran, 1 or 3 days later for a third of Bear Call rows. B1-F's
  population bound (`entry_date ≥ 2026-10-30`), its one-per-(contract,
  entry_date) rule and its week clusters all read this column, and
  `min_days_held` counts from it, so the late-scanned rows now exit on the
  book's schedule instead of 1-3 days after it.
- An exit needs a mark taken that day. 5% of gate-passed Bear Call time exits
  were booked on a mark 0-2 days after entry.
- B1-F is run with `--split-at <merge date>`, which prints entries before and
  from that date beside the registered result (no authority).

**E2 — source and estimator fixed.**

- Earnings dates come from the Dolt calendar cache (`earnings_cal`, the table
  E1 used, refreshed weekly since 11-09), not `src.earnings_provider`: that
  module only returns a symbol's NEXT date on or after today, with no session,
  so it cannot supply a past window "fetched after the window closes".
- Estimator: `scripts.earnings_premium_dolt --forward` — E1-S and E1-F
  unchanged, chains from `data/chain_archive.db` (`mid` = quote midpoint,
  which is what Dolt's `mid` is). It refuses to run before 2027-03-15 without
  `--early`; `--coverage` prints event counts and no outcome.
- Coverage on 2026-11-10: 30 timed events on archived symbols so far, 15 with
  a snapshot within 4 days on both sides. Seven events after 10-16 are already
  lost to archive gaps (BA, UPS, GILD, MRK, MPC, CVS, QCOM).
