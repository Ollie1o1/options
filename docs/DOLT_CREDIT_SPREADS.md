# Credit spreads on real marks — 2026-07-29

Can the two lines that carry the paper book be tested on history instead of
waiting a year for live accrual? `gate-is-time-not-data` says the binding
constraint is calendar rather than data, and for signals nothing historical can
express that is true. Bull Put and Bear Call are not such signals: the cached
DoltHub chains carry real bid and ask, so they can be priced with no slippage
assumption at all.

```bash
python -m src.dolt_spread --side put  --symbols SPY,AAPL,MSFT,NVDA,AMD,META,AMZN,GOOG \
    --start 2022-01-01 --end 2025-12-31 --weekly
python -m src.dolt_spread --side call --symbols ... # same
```

Entry: short leg nearest 0.25 delta, wing nearest 0.10, same expiry, DTE floor
of `time_exit_dte + 7`. Sold at the bid, wing bought at the ask — the real
quote, crossed. Exits run through the ledger's own
`_evaluate_multileg_exit`, and expiries settle at intrinsic through the ledger's
own `_legs_intrinsic_close_value`. Returns are on **max risk** (width − credit).

## Result

208 weekly entry dates × 8 megacaps, 2022-01 → 2025-12:

| | n | win% | mean ret on risk | median | total |
|---|---:|---:|---:|---:|---:|
| Bull Put | 1,528 | 66.6% | **−1.89%** | +10.26% | −2,893% |
| Bear Call | 1,517 | 61.9% | **−2.95%** | +8.11% | −4,481% |

Both lines are net negative on capital at risk despite winning about two trades
in three. That shape — high win rate, positive median, negative mean — is the
premium seller's payoff, and here the tail wins. Note this is with commissions
at the configured US$0 and no currency conversion charged, so the cost side is
if anything flattering.

Bear Call, by exit:

| exit | n | mean | median | worst | contribution |
|---|---:|---:|---:|---:|---:|
| stop_loss | 385 | −28.1% | −23.1% | −96.0% | −10,830% |
| time_exit | 199 | −1.9% | −0.7% | −29.0% | −370% |
| expiry | 565 | +4.7% | +14.4% | −100.0% | +2,682% |
| take_profit | 368 | +11.0% | +10.5% | +1.6% | +4,037% |

29 trades (1.9%) took a full max-risk loss and account for −2,900% of the
−4,481% — two thirds of the damage from under two percent of the book.

## The artifact that isn't

Every one of those 29 full losses settled at expiry rather than being stopped
out, which looks like the exit rules never firing because the chain had no
quotes. `marks_seen` records how many days each position was actually
observable, and it confirms the mechanism — 25 of the 29 saw one mark or none.

It does **not** rescue the result:

| | n | mean ret | median | full losses |
|---|---:|---:|---:|---:|
| unobserved (≤1 mark) | 782 | −1.2% | +10.8% | 25 |
| managed (>1 mark) | 735 | **−4.8%** | +6.1% | 4 |

Restricting to positions the exit rules could actually act on makes the result
**worse**, not better. The unobserved cohort is mostly spreads that quietly
expired worthless and kept the credit, plus a few catastrophes; the managed
cohort pays the stop-loss repeatedly at −28% a time. Whatever else is true, the
negative mean is not an artifact of missing quotes.

One uncomfortable implication, offered as a question rather than a finding: the
unmanaged cohort outperformed the managed one. That is the direction
`hold-to-expiry-small-account` predicts, but it is badly confounded — positions
go unobserved because their strikes and expiries are thinly quoted, which is not
a random sample.

## Why this does not transfer to the live book

Stated up front in the idea entry, and it holds:

- **Universe is 8 megacaps plus SPY.** The live scanner trades a DISCOVER
  universe of ~82 names. Megacap credit spreads are their own animal.
- **Widths are wrong by an order of magnitude.** Strike spacing in this corpus
  is $10–20, giving a mean max risk of $1,753 (Bear Call) and $1,812 (Bull Put).
  The live book's median max risk is **$54** for Bear Call and **$128** for Bull
  Put. These are 14–32× the structures actually being traded, and gamma near the
  short strike does not scale linearly with width.
- **Half the sample was unmanageable.** 52% of positions were quoted on at most
  one day, so for half the book the exit rules under test never ran.

## Verdict

The corpus cannot answer the question that was asked — whether the live book's
$1–5 wide credit spreads survive their costs. Reporting these numbers as
validation of the live lines would be exactly the "number that looks like
validation" the idea entry warned against, so they are not offered as such.

What it does deliver is one usable negative: **wide credit spreads on megacaps,
2022–2025, under this book's own exit rules, lost money on capital at risk even
with zero commissions.** That is worth knowing before anyone widens the live
structures toward these sizes on the intuition that more width means more
safety.

The gate-shortening ambition fails on its own terms. The calendar constraint
stands.
