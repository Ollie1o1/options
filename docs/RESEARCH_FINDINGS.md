# Research findings — dated register

What we've learned about making money with this system, **newest first**. Each
entry has the date it was found, its status, and where the detail lives.

Status key: **LEAD** = promising, being re-checked · **PENDING** = forward test
scheduled · **CLOSED** = answered, don't re-run · **BASELINE** = the default.

---

## 2026-11-10 (newest) — audit of the findings below

Each finding was re-read against the script that produced it, except where a row says otherwise. The Bear Call
and gate-separation numbers were re-run and reproduce to the digit.

| Finding | As recorded | Could it flip? | What the check showed |
|---|---|---|---|
| #1 Bear Calls made money (PF 1.89) | LEAD | **Yes — it does** | The paper P&L exits at the **mid** and concedes 35% of the half-spread at entry. Crossing both ends: PF 1.19 [0.68, 2.24], excess −1.4% [−6.3, +2.2]. Index ETFs: PF 1.41 [0.69, 3.14], excess −0.1%. Single names: excess −4.8% [−6.9, −2.8]. **Not profitable at the fills the earnings and SPY tests were judged on.** Still a lead only in the sense that mid-fill profit exists. |
| #6 Friction-refused Bear Calls trail their delta | supports the gate | No | −3.1% as recorded; −20% once crossed. Stronger, same sign. |
| #5 The gate separates credit spreads | CLOSED (replicated) | No | Reproduces (+0.19 / +0.17). Holds with the 08-29/30 weekend exits removed (original window +0.24 / +0.23) and at mid-to-mid prices (forward +0.10 / +0.07, both CIs above zero). The negative-EV arm alone separates at mid (+0.19 / +0.13), so it is not only the spread being charged to the wide contracts. One exception: forward Bull Put, passed vs friction-refused at mid, +0.04 [−0.01, +0.09]. |
| #2 Earnings premium real at mid, dead at the touch | CLOSED | No | Entry/exit days are correct for before-open and after-close reports; no look-ahead. The touch result (−20% to −33%) is far from zero. "Real at mid" rests on mids of quotes up to 50% wide and is the weaker half. |
| #3 Earnings held to expiry is worse | CLOSED | **Not audited** | `scripts/earnings_to_expiry_dolt.py` was not read. |
| #4 Hedged SPY short vol | CLOSED | No | Hedge uses the prior close's delta; the VRP split uses only prior trades. Null stands. |
| #8 Nine Long Puts lagged their delta | descriptive | No | No duplicates or weekend exits in the ledger; the +$241 / +$786 figures were not recomputed. |
| 10-15 live Bull Put rule is costly beta | CLOSED | No | Crosses both ends, month clusters, absolute deltas. Negative in 12 of 13 years. |
| 10-15 SPY put spread held to expiry | CLOSED | **Not audited** | `scripts/index_hold_to_expiry_odx.py` was not read. |
| 10-10..14 direction signals | CLOSED | No (not re-run) | All null or below the cost hurdle with tight CIs; a bug would have to create an effect 5× larger. |
| 09-28 book vs SPY, exit sweep | CLOSED | No (not re-run) | Read from the real ledger on capital at risk. |

**Bugs found in the code behind the shadow corpus** (fixed, each with a failing
test; apply from the merge on, nothing back-filled):

- A quarter of simulated positions (206,221) carried an entry date 1 or 3 days
  after the scan that priced them. Their minimum hold ran from the later day.
- 31,559 simulated time exits (6%) were booked on a quote from before the
  minimum hold, because an exit could be decided on a mark of any age.
- The "top quintile" gate had never fired on a discovery board: 0 refusals in
  840,691 candidates; eight MRNA contracts above the cutoff were shown as
  cleared on 09-22 and 09-25. The auto-log board was unaffected.
- `delta_matched_benchmark` would price exits from a stale price cache without
  saying so. It now stops.

None of these moves a recorded verdict: gate separation is a within-day rank
comparison and both arms shared the defects.

## 2026-11-09

| # | Finding | Status | Detail |
|---|---|---|---|
| 1 | **Gate-passed Bear Calls made money on paper — at a mid exit only (see 11-10: PF 1.19 [0.68, 2.24] once the spread is crossed).** 4,456 trades, Aug-Oct 2026: profit factor 1.89 [1.09, 3.67]. About half of it came from the underlying falling, though, and the excess over delta (+2.8% [−2.0, +6.3] of capital at risk) isn't proven. **Index/sector ETFs carry it** (PF 2.2); single names are flat. | **LEAD**, re-check on fresh trades **≥ 2027-01-20** (`python -m scripts.bear_call_absolute --cache DIR --forward`) | `RESULT_EARNINGS_AND_BEAR_CALL_20261109.md` |
| 2 | **The earnings premium is real, but trading costs eat it.** Over 2,466 earnings events (2020-26), selling a defined-risk condor or fly the day before and buying back the day after earns about **+4% of capital at risk per event at mid**. Crossing the bid-ask on all 4 legs, both ends, turns that into −20% to −33%. Break-even needs paying ≤ 12-15% of the half-spread. Buying straddles into earnings loses even at mid. | CLOSED as a next-day trade. Real-fill question open. | same |
| 3 | **Holding the earnings trade to expiry makes it worse.** One crossing instead of two, but the result goes to −10% (condor) / −15% (fly), and even at mid it is about zero or negative. The premium is the overnight IV crush only; after that the move belongs to the buyer. | CLOSED | `RESULT_EARNINGS_TO_EXPIRY_20261109.md` |
| 4 | **Delta-hedged SPY short vol** (2010-23, monthly): +1.2% of capital at risk per trade [−0.8, +3.3], Sharpe ~0.3. Not significant; it would take ~39 years of data to confirm. Hedging cuts risk (worst trade −100% → −46%) but doesn't add return. "Sell when IV is rich vs realised" pointed the **wrong** way. | CLOSED on this data; needs more data (SPX 2004+, more index underlyings) | `RESULT_HEDGED_INDEX_VOL_20261109.md` |
| 5 | **The gate still picks better credit spreads** on fresh data (Oct 13 - Nov 6): Bear Call +0.19, Bull Put +0.17, 73-76 symbols. That is relative (passed vs refused), not profit. It does nothing for Long Call/Long Put. | CLOSED (replicated) | `reports/gate_separation/gate_separation_20261109T215401Z.json` |
| 7 | **The earnings filter had been mostly blind since 08-20.** Its calendar cache was never refreshed, and a report on the entry day after the close counted as "already priced in". The XYZ Long Put bought on report day (11-03) lost 99.6% overnight. **Fixed:** weekly calendar refresh, same-day after-close reports count as held through, and long options are now gated too (buying into earnings lost −16% on average, E1-L). | FIXED (live) | `config.json` → `auto_log.refuse_long_through_earnings` |
| 8 | **The 9 trades while you were away (10-16 .. 11-09):** all Long Puts, +$241. Holding the same market exposure would have made +$786 (SPY −2.5%), so the picks lagged their own direction. Two winners (INTC, SLV) carried it. The gate doesn't separate Long Puts, so these were effectively random draws. | CLOSED (descriptive) | `paper_trades.db` ids 1116-1124 |
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
by trading costs** (earnings premium, index short vol at mid, and — as of the
11-10 audit — ETF Bear Calls, whose paper profit is a mid-exit figure). The system's clearest value is **cost control**. The
next verdicts are scheduled: Bear Call forward ≥ 2027-01-20, earnings forward
on our own archive ≥ 2027-03-15.

## Pending checks

| Date | Check | Command / doc |
|---|---|---|
| ≥ 2027-01-15 | Bull Put §2 (for the record) | `python -m scripts.delta_matched_benchmark --cache NEW_EMPTY_DIR --exit-through 2027-01-15` |
| mid-Jan 2027 | Analyst estimate revisions | **not runnable yet: no registration and no estimator.** Write the hypothesis down before opening the archive. |
| ≥ 2027-01-20 | **B1-F Bear Call forward** | `python -m scripts.bear_call_absolute --cache DIR --forward` (prints the crossed-fill line, Amendment 3) |
| ≥ 2027-03-15 | E2 earnings forward (own archive) | `python -m scripts.earnings_premium_dolt --forward`; `--coverage` any time. **At risk: 7 events already lost to chain-archive gaps.** |
