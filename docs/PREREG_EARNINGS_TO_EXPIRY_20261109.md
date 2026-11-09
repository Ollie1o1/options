# Earnings premium held to expiry (E3) — REGISTRATION

**Frozen 2026-11-09, before E3 was run.** Same day as, and AFTER, E1
(`docs/RESULT_EARNINGS_AND_BEAR_CALL_20261109.md`).

## Provenance — what is known

E1 showed: selling earnings vol earns about +4% of CAR per event at mid, but crossing four
legs at both ends turns that into −20% (condor) / −33% (fly). Break-even needs
paying ≤ 12-15% of the half-spread per leg-crossing. **E3 is chosen because of
that result:** hold to the front expiry and settle at intrinsic, so the exit
crossing disappears. This is a rule picked after seeing related data, and the
verdict carries that. What E3 adds (post-event drift and gap risk until expiry)
has not been looked at.

## Rule

- Events, expiry, quote filter, legs, entry fills: **identical to E1**
  (`scripts/earnings_premium_dolt.py` functions reused unchanged): E3-S short
  25/10 condor, E3-F short ATM iron fly with 10Δ wings, E3-L long ATM straddle.
- **Exit:** none. Settle every leg at intrinsic on the expiry, using the raw
  close from `dolt_options.stocks_close` on the last date ≤ expiration.
- **Spot guard:** at the pre snapshot, put-call parity spot at the ATM strike
  (K + C_mid − P_mid) must be within 2% of `stocks_close` on that date
  (a check against the old dividend-adjusted spot defect). Fails are dropped
  and counted.
- r, CAR, week clustering, bootstrap, Bonferroni 98.33%, pass rule: as E1.

## Sensitivities (no authority)

Mid entry fills; expiry ≤ 10 vs > 10 calendar days after the event; by year;
excluding 2020.
