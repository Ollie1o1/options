# Earnings premium (E1) + Bear Call absolute (B1) — RESULT

Run 2026-11-09 against `docs/PREREG_EARNINGS_AND_BEAR_CALL_20261109.md`
(frozen `facf425`; Amendment 1 `178cc07` before E1 ran; Amendment 2 registers
the B1 forward re-run). Scripts: `scripts/earnings_premium_dolt.py`,
`scripts/bear_call_absolute.py`.

## E1 — earnings vol, Dolt 2020-01 .. 2026-06: 0/3 pass

| claimant | n | mean r on CAR, crossing both ends | PF | at mid (no authority) |
|---|---|---|---|---|
| E1-S short 25/10 condor | 728 | −20.2% [−31.2, −13.8] | 0.114 | **+4.0% [+2.2, +5.8], PF 1.57** |
| E1-F short ATM iron fly | 1,329 | −33.4% [−36.6, −30.5] | 0.071 | **+4.0% [+1.6, +6.7], PF 1.37** |
| E1-L long ATM straddle | 1,684 | −16.1% [−18.1, −14.1] | 0.329 | +0.8% [−1.4, +3.0] |

Clusters are ISO weeks (186-252). Every slice (by year, open/close
announcements, excluding 2020, exact-quoted only) has the same sign.

**Reading:**
- **The earnings premium exists before costs.** At mid, short condors and flies
  earn about +4% of capital at risk per event, with 98.33% lower bounds above
  zero and PF above 1. The long straddle at mid is about zero. So sellers, not
  buyers, are paid. This matches the "~24% IV crush" seen in June.
- **Crossing kills it.** Four legs crossed twice, on EOD Dolt quotes, costs a
  median $100 (condor) to $180 (fly) per trade. The mid profit is $28-40. Break-even
  requires paying only **12-15% of the half-spread on every leg-crossing**
  (on 50 mega/large caps the fly needs 12% and the condor 2%). Paper execution
  here has measured about 27% of credit lost (execution truth), so a realistic
  fill is far above break-even.
- The condor's worst trade (−35× CAR) is an exit ask far above the wing width:
  Dolt's untradeable-quote defect at exit. It inflates the touch-fill loss but
  doesn't change the sign: median r is −9%.
- **Not tested (would need its own registration disclosing this result):**
  holding through to the front expiry instead of exiting the next day. That
  halves the crossings, but adds post-event gap and drift risk.

## B1 — gate-passed Bear Calls, shadow corpus: A fail, B pass → NOT a pass

Entries 2026-08-19 .. 10-29, expiry ≤ 11-06, 4,456 observations, 81 symbols,
**11 week clusters**, none still open.

| arm | r on CAR | delta-equivalent | A excess | B PF |
|---|---|---|---|---|
| **passed** | +5.87% [+0.77, +11.74] | +3.03% | **+2.84% [−1.98, +6.33]** fail | **1.891 [1.087, 3.670]** pass |
| refused friction | −1.89% | +1.21% | −3.10% [−4.92, −1.79] | 0.802 |
| refused negative_ev | +1.89% | +2.60% | −0.71% [−3.06, +1.03] | 1.221 |

- Passed Bear Calls made money over this window (PF CI > 1). About half of it is the
  short delta paying off in a window where these names fell. The excess over
  delta is positive but its CI spans zero.
- The friction-refused arm is reliably worse than its delta (−3.1%), which is
  consistent with the gate separation result: cost control is the part that works.
- Index ETFs carry the result (13 symbols, PF 2.21, excess +4.1% [−1.9,
  +8.4]); single names are flat (excess −0.5%).
- Only 11 weekly clusters, from one market regime. **B1-F** (Amendment 2)
  re-runs the identical estimator on entries ≥ 2026-10-30, expiry ≤ 2027-01-15,
  judged ≥ 2027-01-20:
  `python -m scripts.bear_call_absolute --cache DIR --forward`.

## Status of §3

Still no rule passes. Leads, in order: B1-F (forward, January), and earnings
premium if real fills can be shown to cost ≤ ~12% of the half-spread. Only
real orders can show that.
