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
