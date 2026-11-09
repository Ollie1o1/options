# Research findings — dated register

What we've learned about making money with this system, **newest first**. Each
entry has the date it was found, its status, and where the detail lives.

Status key: **LEAD** = promising, being re-checked · **PENDING** = forward test
scheduled · **CLOSED** = answered, don't re-run · **BASELINE** = the default.

---

## 2026-11-09 (newest)

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | **Gate-passed Bear Calls made money on paper.** 4,456 trades, Aug-Oct 2026: profit factor 1.89 [1.09, 3.67]. About half of it came from the underlying falling, though, and the excess over delta (+2.8% [−2.0, +6.3] of capital at risk) isn't proven. **Index/sector ETFs carry it** (PF 2.2); single names are flat. | **LEAD**, re-check on fresh trades **≥ 2027-01-20** (`python -m scripts.bear_call_absolute --cache DIR --forward`) | `RESULT_EARNINGS_AND_BEAR_CALL_20261109.md` |
| 2 | **The earnings premium is real, but trading costs eat it.** Over 2,466 earnings events (2020-26), selling a defined-risk condor or fly the day before and buying back the day after earns about **+4% of capital at risk per event at mid**. Crossing the bid-ask on all 4 legs, both ends, turns that into −20% to −33%. Break-even needs paying ≤ 12-15% of the half-spread. Buying straddles into earnings loses even at mid. | CLOSED as a next-day trade. Real-fill question open. | same |
| 3 | **Holding the earnings trade to expiry makes it worse.** One crossing instead of two, but the result goes to −10% (condor) / −15% (fly), and even at mid it is about zero or negative. The premium is the overnight IV crush only; after that the move belongs to the buyer. | CLOSED | `RESULT_EARNINGS_TO_EXPIRY_20261109.md` |
| 4 | **Delta-hedged SPY short vol** (2010-23, monthly): +1.2% of capital at risk per trade [−0.8, +3.3], Sharpe ~0.3. Not significant; it would take ~39 years of data to confirm. Hedging cuts risk (worst trade −100% → −46%) but doesn't add return. "Sell when IV is rich vs realised" pointed the **wrong** way. | CLOSED on this data; needs more data (SPX 2004+, more index underlyings) | `RESULT_HEDGED_INDEX_VOL_20261109.md` |
| 5 | **The gate still picks better credit spreads** on fresh data (Oct 13 - Nov 6): Bear Call +0.19, Bull Put +0.17, 73-76 symbols. That is relative (passed vs refused), not profit. It does nothing for Long Call/Long Put. | CLOSED (replicated) | `reports/gate_separation/gate_separation_20261109T215401Z.json` |
| 6 | **Friction-refused Bear Calls are reliably worse than their delta** (−3.1%). Cost control is the part of the system that clearly works. | supports the gate | B1 result |

## 2026-10-15

| Finding | Status |
|---|---|
| **The live Bull Put rule is "costly beta"**: on SPY 2010-23 it trails holding its delta by 2.1% per trade [−2.7, −1.6]. The deficit is the bid-ask crossing on a ~3-day hold. Vetoed for real money; auto-logging stopped. | CLOSED |
| **A SPY put spread held to expiry** (one crossing) also fails to beat its delta (−0.8%). Every put-spread measurement lands at or below its delta: on this data a put spread is just a way to hold the market. | CLOSED |
| **Decision rule agreed**: the only path to real money is a pre-registered rule that passes history, then ≥ 8 forward weeks, then a $1,500 pilot to measure real fills. | **BASELINE** (`DECISION_RULE_20261015.md`) |
| The EV gate's skew treatment puts index puts at a disadvantage (index pass rate 5.2% vs single names 13.2%). | CLOSED |

## 2026-10-10 .. 10-14 — directional signals

All null or below the cost of trading options: earnings-surprise drift (+0.14% at 2 sessions, real but ~5× too small), news event types, analyst up/downgrades, post-shock drift (MRNA was the right tail, which can't be picked in advance), long options at 1-3 sessions (lose ~8% of premium, mostly the spread). **Predicting direction from public data is exhausted here.** CLOSED.

## 2026-09-28 .. 09-30

- The paper book trailed SPY by about $7,700 over 5.4 months; no edge was measurable (PF 1.04 [0.87, 1.24]). CLOSED.
- 750 exit-rule combinations: none promoted. Exits are second-order. CLOSED.
- Bull Put scarcity is caused by the EV gate, not the universe. CLOSED.

## 2026-08 and earlier

- Friction (bid-ask cost) filter: −6.76% → −0.64% return on capital. **The largest effect measured anywhere.** BASELINE.
- Single-name short vol: PF 0.30, no edge. CLOSED.
- optionsDX SPY bull put 2010-23: +2.30% per trade before delta adjustment. Superseded by the 10-15 delta-adjusted results.

---

## The one-paragraph summary (as of 2026-11-09)

Nothing has yet passed the decision rule, so real money stays off and the
default is to hold an index fund. Every edge found is either **real but eaten
by trading costs** (earnings premium, index short vol at mid) or **too new to
trust** (ETF Bear Calls). The system's clearest value is **cost control**. The
next verdicts are scheduled: Bear Call forward ≥ 2027-01-20, earnings forward
on our own archive ≥ 2027-03-15.

## Pending checks

| Date | Check | Command / doc |
|---|---|---|
| ≥ 2027-01-15 | Bull Put §2 (for the record) | `DECISION_RULE_20261015.md` |
| mid-Jan 2027 | Analyst estimate revisions | estimates archive |
| ≥ 2027-01-20 | **B1-F Bear Call forward** | `scripts/bear_call_absolute.py --forward` |
| ≥ 2027-03-15 | E2 earnings forward (own archive) | `PREREG_EARNINGS_AND_BEAR_CALL_20261109.md` |
