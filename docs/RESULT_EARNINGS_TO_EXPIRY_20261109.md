# Earnings premium held to expiry (E3) — RESULT

Run 2026-11-09 against `docs/PREREG_EARNINGS_TO_EXPIRY_20261109.md` (frozen
`6347f91` before running). Script: `scripts/earnings_to_expiry_dolt.py`.

## Verdict: 0/3 pass. Holding to expiry makes it WORSE.

| claimant | n | mean r on CAR (entry at touch) | PF | entry at mid (no authority) |
|---|---|---|---|---|
| E3-S short 25/10 condor | 1,106 | −10.1% [−14.9, −6.0] | 0.599 | −0.7% [−4.0, +2.7], PF 0.97 |
| E3-F short ATM iron fly | 1,807 | −14.9% [−17.7, −12.3] | 0.532 | −2.3% [−5.4, +0.8], PF 0.92 |
| E3-L long ATM straddle | 2,042 | −6.0% [−10.1, −1.8] | 0.815 | +0.8% [−3.4, +5.3], PF 1.03 |

Spot guard dropped 82-105 events per structure. 2 condor rows are impossible for a
defined-risk position (r < −1: IOSP 2025-08, REG 2025-02, junk quotes with
crossed short strikes). Excluding them: condor −8.3% [−11.4, −5.3]. Same verdict.

## Reading

- **E1's +4% at mid disappears when held.** Next-day exit at mid: +4%. Hold to
  expiry, mid entry: about 0 or negative. The premium is the overnight IV crush;
  what's left after it (post-event drift and follow-through) belongs to
  the buyer, not the seller.
- Halving the crossings doesn't help, because the thing being sold is no
  longer mispriced after day one.
- **Closes earnings-vol selling on Dolt data** in both forms. The only version
  with a positive expectation is E1 at mid, a next-day exit, which needs fills
  ≤ 12-15% of the half-spread.
