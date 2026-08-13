# Paper Trading Track Record

_Generated 2026-08-13 13:59 • 866 closed trades_

> **Methodology & caveats.** These are **paper trades**, not live fills. Entries and exits use **delayed retail data** (Yahoo Finance) and a **modeled friction** assumption (spread/slippage), so realized results would differ. The descriptive stats below are real; the **predictive edge of the ranking model is still under out-of-sample evaluation** and is *not* established — see [docs/VALIDATION_POWER.md](../docs/VALIDATION_POWER.md).

_Ranking model: EXPERIMENTAL — OOS IC -0.14 (p=0.09, n=192) | gate: STOP (n=120/50)
OOS walk-forward as of 2026-08-02 (11d old) | cohort IC -0.064 pearson / -0.134 rank_

## Headline

- Net P&L: **+$5,639.30** across 866 closed trades with a recorded dollar result
- Return on capital risked: **+0.2%** (+$5,639.30 of $3,295,780 risked across 866 trades with capital_at_risk recorded)
- Within the $4,000 per-position ceiling: **+3.8%** of capital risked (+$20,324.62 of $536,774 risked, 772 trades) — the subset the account could actually have held

Secondary, size-blind figures:

- Win rate: **48.5%** = 420 wins / 866 closed trades with a recorded return; 0 closed trades excluded for missing returns
- Mean return per trade: **-0.9%** (unweighted mean of per-trade returns **on entry premium** — a $28 spread counts the same as a $27,000 cash-secured put, and the premium denominator is a debit on long structures but a credit on short ones; not the headline for either reason)
- Median return per trade: **-0.3%** of capital risked (typical trade, size-blind)

## By strategy

| Strategy | Closed | Win rate | Net $ | Return on risk (aggregate, of capital risked) | Median return on risk (per trade, of capital risked) | Mean return per trade (unweighted) |
|----------|-------:|---------:|------:|------:|------:|------:|
| Bear Call | 135 | 59.3% | +$108.04 | +1.1% | +6.9% | -7.1% |
| Bull Put | 131 | 66.4% | +$5,315.25 | +27.7% | +24.6% | +16.9% |
| Iron Condor | 142 | 50.0% | +$4,323.81 | +2.2% | +0.2% | -7.5% |
| Long Call | 275 | 38.2% | -$1,376.58 | -0.5% | -26.4% | -1.3% |
| Long Put | 75 | 32.0% | +$5,611.77 | +12.4% | -17.3% | -5.0% |
| Short Put | 108 | 49.1% | -$8,342.99 | -0.3% | -0.0% | -2.3% |

_Win rate counts trades with a recorded return; aggregate return on risk and median return on risk count trades with capital_at_risk recorded. Where aggregate and median disagree in sign, see the methodology notes._

## Credit structures: return on credit collected

| Strategy | Closed | Credit collected | Net $ | Return on credit (of credit collected) | Median return on credit (per trade) | Return on risk (of capital risked) |
|----------|-------:|-----------------:|------:|------:|------:|------:|
| Bear Call | 135 | $8,528 | +$108.04 | +1.3% | +8.4% | +1.1% |
| Bull Put | 131 | $17,152 | +$5,315.25 | +31.0% | +32.0% | +27.7% |
| Iron Condor | 142 | $106,236 | +$4,323.81 | +4.1% | +0.2% | +2.2% |
| Short Put (cash-secured) | 108 | $73,063 | -$8,342.99 | -11.4% | -1.2% | -0.3% |

## Forward-cohort gate

- Gate decision: **STOP**
- Cohort size: **120** (closed cohort trades accumulated since the gate window opened)

## Methodology notes

### Weighting and bases

Every percentage in this document names its basis. **Of capital risked** means dollars of P&L over dollars of capital_at_risk (the ledger's own per-position risk figure: premium paid on debits, collateral or width less credit on credits). **Of credit collected** is P&L over premium taken in. **Unweighted mean** is the arithmetic mean of per-trade percentage returns, which counts every trade equally no matter its size and is reported only as a secondary line.

Size dominates the raw aggregate: of $3,295,780 risked across the book, only $536,774 sat inside the $4,000 per-position ceiling the ledger now enforces (`auto_log.max_capital_at_risk`). The oversized positions are a sizing artifact of an unbounded feeder, not a strategy result, which is why the affordable subset is published beside the whole book.

### Median vs aggregate

Aggregate return on risk is a dollar-weighted number: one large contract can carry a whole line. The median per-trade return on risk is published beside it so the typical trade is visible. Where the two disagree in sign, the aggregate is a story about one or two positions.

- **Long Put**: aggregate +12.4% of capital risked but median -17.3% per trade — one GS trade (+$3,255.70) is +58% of the line's net.

### Cash-secured collateral denominator

A cash-secured short posts the whole strike as collateral, so its capital_at_risk denominator is roughly fifty to a hundred times the premium at stake. Return on risk therefore reads as a near-flat line however the trade actually went — that flatness is the denominator, not the result. Return on credit collected is published as the companion figure and is the one that moves.

- **Short Put**: -0.3% of capital risked vs -11.4% of $73,063 credit collected (108 closed).

### Stops overshot because exits were checked by hand

Exit checks run inside `update_positions`, which runs when the screener is opened — not on a timer. The scheduled LaunchAgents stopped running on **2026-06-15**, so from that date exits were checked at irregular, manual intervals. A stop rule cannot fire on a day nobody looked, so stopped-out trades in that window record the loss they had drifted to by the next check, not the loss the rule specified.

Measured over 121 closed trades whose exit reason states a numeric stop level (of 155 stop exits in total; 30 more stopped on a strike breach, which has no numeric level to overshoot). Overshoot is the realized loss minus the stated stop, in percent of entry premium:

| window | trades | median overshoot | p90 | worst | share past their stop |
|---|---:|---:|---:|---:|---:|
| Before 2026-06-15 | 32 | +8.3% | +24.3% | +66.7% | 75% |
| After 2026-06-15 (manual cadence) | 89 | +11.1% | +33.5% | +57.5% | 96% |
| All | 121 | +10.7% | +32.5% | +66.7% | 90% |

**The recorded exits are not corrected for this and never will be.** The record stays as-traded; this note is how it is read. Losses on stopped trades in the manual-cadence window are overstated relative to the rules that were supposed to govern them, and because defined-risk credit structures state their stops as a multiple of a small credit, the overstatement falls hardest on exactly the lines the credit-vs-debit comparison depends on.

This note is removed only once the scheduler has been verifiably alive for a full window — not when it is merely fixed.

## Closed trades


`P&L % of premium` is the per-trade return on the **entry premium** — the debit paid on Long Call/Long Put, and the credit received on Short Put and every spread. Those are different denominators, so this column is not comparable across structures; `P&L $` and `Capital at risk` are.

| Date | Ticker | Structure | Entry | Exit | P&L % of premium | P&L $ | Capital at risk | Exit reason |
|------|--------|-----------|------:|-----:|------:|------:|----------------:|-------------|
| 2026-04-18 | WFC | Long Put | $1.90 | $2.18 | +8.1% | +$15.30 | $190 | Time Exit (21d to expiry) |
| 2026-04-18 | WMT | Long Call | $2.34 | $3.45 | +40.9% | +$95.66 | $234 | — |
| 2026-04-18 | GLD | Long Call | $6.80 | $4.80 | -35.6% | -$242.10 | $680 | — |
| 2026-04-18 | ORCL | Long Call | $5.35 | $6.00 | +5.9% | +$31.60 | $535 | — |
| 2026-04-18 | SLV | Long Call | $2.50 | $1.59 | -42.9% | -$107.30 | $250 | — |
| 2026-04-18 | ORCL | Long Call | $6.90 | $7.70 | +5.4% | +$37.30 | $690 | — |
| 2026-04-18 | GE | Long Put | $11.00 | $26.90 | +138.4% | +$1,522.70 | $1,100 | — |
| 2026-04-18 | BA | Long Call | $6.65 | $9.40 | +35.2% | +$233.80 | $665 | — |
| 2026-04-18 | ORCL | Long Put | $7.45 | $6.85 | -14.2% | -$106.00 | $745 | Time Exit (21d to expiry) |
| 2026-04-18 | MU | Long Call | $25.05 | $42.10 | +64.0% | +$1,603.70 | $2,505 | — |
| 2026-04-18 | TSLA | Long Call | $14.70 | $10.15 | -37.0% | -$544.50 | $1,470 | — |
| 2026-04-18 | TSLA | Long Call | $12.80 | $8.70 | -38.1% | -$488.10 | $1,280 | — |
| 2026-04-18 | MU | Long Call | $18.80 | $14.61 | -27.7% | -$520.30 | $1,880 | — |
| 2026-04-18 | LRCX | Long Call | $14.90 | $7.80 | -53.7% | -$800.70 | $1,490 | Time Exit (21d to expiry) |
| 2026-04-18 | MU | Long Call | $22.25 | $17.40 | -26.4% | -$586.30 | $2,225 | — |
| 2026-04-19 | ORCL | Long Call | $5.35 | $11.67 | +111.9% | +$598.60 | $535 | — |
| 2026-04-19 | ORCL | Long Call | $6.90 | $13.35 | +87.3% | +$602.30 | $690 | — |
| 2026-04-19 | ORCL | Long Put | $7.45 | $6.85 | -14.2% | -$106.00 | $745 | Time Exit (21d to expiry) |
| 2026-04-19 | NKE | Long Call | $1.10 | $0.48 | -66.6% | -$73.30 | $110 | — |
| 2026-04-19 | MU | Long Call | $18.80 | $35.52 | +83.5% | +$1,570.70 | $1,880 | — |
| 2026-04-19 | MU | Long Call | $22.25 | $38.85 | +70.1% | +$1,558.70 | $2,225 | — |
| 2026-04-19 | MU | Long Call | $25.05 | $42.10 | +64.0% | +$1,603.70 | $2,505 | — |
| 2026-04-19 | GS | Long Call | $21.00 | $27.35 | +25.4% | +$533.70 | $2,100 | — |
| 2026-04-19 | AXP | Long Call | $8.35 | $2.28 | -78.9% | -$658.40 | $835 | — |
| 2026-04-19 | INTC | Long Put | $5.05 | $5.75 | +7.6% | +$38.40 | $505 | Time Exit (21d to expiry) |
| 2026-04-20 | ORCL | Long Call | $6.65 | $13.35 | +94.6% | +$628.80 | $665 | — |
| 2026-04-20 | ORCL | Long Call | $7.95 | $14.99 | +82.4% | +$655.00 | $795 | — |
| 2026-04-20 | WMT | Long Call | $2.71 | $3.45 | +20.8% | +$56.44 | $271 | — |
| 2026-04-20 | PFE | Long Call | $0.51 | $0.23 | -77.1% | -$39.30 | $51 | — |
| 2026-04-20 | ORCL | Long Call | $7.15 | $12.04 | +62.2% | +$444.80 | $715 | — |
| 2026-04-20 | BA | Long Call | $7.00 | $9.40 | +28.1% | +$196.70 | $700 | — |
| 2026-04-20 | NKE | Long Call | $1.24 | $0.48 | -70.4% | -$87.30 | $124 | — |
| 2026-04-20 | ORCL | Long Call | $7.05 | $11.15 | +52.0% | +$366.40 | $705 | — |
| 2026-04-20 | T | Long Put | $0.73 | $0.37 | -64.8% | -$47.30 | $73 | Time Exit (21d to expiry) |
| 2026-04-20 | COST | Long Call | $16.15 | $23.38 | +38.7% | +$624.80 | $1,615 | — |
| 2026-04-20 | MU | Long Call | $24.45 | $47.18 | +88.8% | +$2,171.70 | $2,445 | — |
| 2026-04-23 | WFC | Long Put | $1.87 | $1.86 | -7.2% | -$13.52 | $187 | Stop Loss (strike breached) |
| 2026-04-23 | NKE | Long Put | $0.91 | $0.81 | -23.4% | -$21.30 | $91 | Time Exit (12d to expiry) |
| 2026-04-23 | WFC | Long Put | $1.44 | $1.76 | +14.4% | +$20.70 | $144 | Time Exit (12d to expiry) |
| 2026-04-23 | NKE | Long Put | $1.12 | $1.08 | -13.7% | -$15.30 | $112 | Time Exit (19d to expiry) |
| 2026-04-23 | NFLX | Long Put | $2.13 | $2.28 | +0.4% | +$0.92 | $213 | Time Exit (19d to expiry) |
| 2026-04-23 | NFLX | Long Call | $1.81 | $1.55 | -21.1% | -$38.16 | $181 | Time Exit (19d to expiry) |
| 2026-04-23 | WFC | Long Put | $2.25 | $2.18 | -9.7% | -$21.80 | $225 | Stop Loss (strike breached) |
| 2026-04-23 | MS | Long Call | $3.65 | $2.89 | -27.2% | -$99.20 | $365 | Time Exit (19d to expiry) |
| 2026-04-24 | WFC | Long Put | $1.57 | $1.21 | -30.1% | -$47.30 | $157 | Time Exit (18d to expiry) |
| 2026-04-24 | UAL | Long Put | $3.85 | $3.85 | -6.3% | -$24.40 | $385 | Time Exit (18d to expiry) |
| 2026-04-24 | UAL | Long Put | $4.25 | $4.23 | -6.8% | -$28.80 | $425 | Time Exit (18d to expiry) |
| 2026-04-24 | WFC | Long Put | $1.82 | $1.34 | -33.1% | -$60.22 | $182 | Time Exit (11d to expiry) |
| 2026-04-24 | WMT | Long Call | $2.01 | $1.36 | -39.0% | -$78.36 | $201 | Time Exit (11d to expiry) |
| 2026-04-24 | WMT | Long Call | $2.24 | $1.58 | -36.0% | -$80.74 | $224 | Time Exit (18d to expiry) |
| 2026-04-24 | WFC | Long Put | $1.76 | $1.42 | -26.1% | -$45.86 | $176 | Time Exit (18d to expiry) |
| 2026-04-24 | PYPL | Long Call | $1.89 | $1.63 | -20.4% | -$38.64 | $189 | Time Exit (18d to expiry) |
| 2026-04-24 | NKE | Long Put | $1.06 | $0.88 | -27.6% | -$29.30 | $106 | Time Exit (18d to expiry) |
| 2026-04-24 | T | Long Put | $0.53 | $0.48 | -30.8% | -$16.30 | $53 | Time Exit (18d to expiry) |
| 2026-04-24 | GS | Long Call | $22.25 | $22.00 | -5.7% | -$126.30 | $2,225 | Time Exit (18d to expiry) |
| 2026-04-24 | CRM | Long Put | $6.35 | $4.60 | -33.8% | -$214.40 | $635 | Time Exit (18d to expiry) |
| 2026-04-24 | MA | Long Put | $12.50 | $11.10 | -17.3% | -$216.30 | $1,250 | Time Exit (18d to expiry) |
| 2026-04-24 | PFE | Long Call | $0.42 | $0.34 | -46.0% | -$19.30 | $42 | Time Exit (18d to expiry) |
| 2026-04-24 | ORCL | Long Call | $6.55 | $5.45 | -23.0% | -$150.60 | $655 | Time Exit (18d to expiry) |
| 2026-04-24 | MA | Long Put | $10.50 | $9.00 | -20.4% | -$214.30 | $1,050 | Time Exit (18d to expiry) |
| 2026-04-24 | BAC | Long Call | $0.81 | $1.03 | +13.2% | +$10.70 | $81 | Time Exit (11d to expiry) |
| 2026-04-24 | GS | Long Call | $17.55 | $17.98 | -3.3% | -$58.30 | $1,755 | Time Exit (18d to expiry) |
| 2026-04-25 | WFC | Long Put | $1.52 | $0.76 | -57.4% | -$87.30 | $152 | Take Profit (35% @ 17d) |
| 2026-04-25 | ORCL | Long Call | $6.55 | $3.38 | -54.6% | -$357.60 | $655 | Take Profit (35% @ 17d) |
| 2026-04-25 | WFC | Long Put | $1.73 | $1.12 | -42.0% | -$72.68 | $173 | Take Profit (35% @ 11d) |
| 2026-04-25 | GS | Long Call | $22.25 | $19.95 | -14.9% | -$331.30 | $2,225 | Time Exit (17d to expiry) |
| 2026-04-25 | WFC | Long Put | $1.72 | $0.97 | -50.4% | -$86.62 | $172 | Take Profit (35% @ 17d) |
| 2026-04-25 | ORCL | Long Put | $7.60 | $10.10 | +26.7% | +$203.10 | $760 | Time Exit (17d to expiry) |
| 2026-04-25 | WFC | Long Put | $2.00 | $1.20 | -40.0% | -$80.00 | $200 | manual_thesis_broken |
| 2026-04-25 | ORCL | Long Call | $8.00 | $3.70 | -59.9% | -$479.30 | $800 | Stop Loss (-50%) |
| 2026-04-26 | T | Long Put | $0.53 | $0.48 | -30.8% | -$16.30 | $53 | Stop Loss (strike breached) |
| 2026-04-26 | ORCL | Long Call | $6.55 | $3.38 | -54.6% | -$357.60 | $655 | Take Profit (35% @ 17d) |
| 2026-04-26 | GS | Long Call | $22.25 | $10.22 | -58.6% | -$1,304.30 | $2,225 | Time Exit (16d to expiry) |
| 2026-04-26 | GE | Long Call | $8.40 | $6.08 | -33.8% | -$283.70 | $840 | Time Exit (16d to expiry) |
| 2026-04-26 | INTC | Long Call | $4.05 | $10.80 | +160.3% | +$649.40 | $405 | Take Profit (100%) |
| 2026-04-26 | INTC | Long Call | $4.70 | $11.12 | +130.3% | +$612.50 | $470 | Take Profit (100%) |
| 2026-04-26 | MU | Long Call | $27.25 | $40.69 | +45.6% | +$1,242.70 | $2,725 | Stop Loss (strike breached) |
| 2026-04-26 | WFC | Short Put | $1.52 | $0.76 | +42.6% | +$64.70 | $7,598 | Take Profit (35% @ 17d) |
| 2026-04-26 | WMT | Short Put | $2.14 | $2.66 | -30.9% | -$66.14 | $12,486 | Time Exit (16d to expiry) |
| 2026-04-26 | WMT | Short Put | $2.56 | $3.18 | -30.7% | -$78.66 | $12,544 | Stop Loss (strike breached) |
| 2026-04-26 | GE | Short Put | $7.00 | $4.50 | +29.5% | +$206.70 | $27,300 | Take Profit (35% @ 10d) |
| 2026-04-26 | XLU | Short Put | $0.41 | $0.31 | -3.2% | -$1.30 | $4,459 | Time Exit (16d to expiry) |
| 2026-04-26 | MRNA | Short Put | $2.99 | $3.48 | -22.8% | -$68.24 | $4,601 | Stop Loss (strike breached) |
| 2026-04-26 | WMT | Short Put | $1.89 | $2.53 | -40.6% | -$76.64 | $12,611 | Stop Loss (strike breached) |
| 2026-04-26 | MU | Short Put | $22.60 | $13.70 | +34.9% | +$788.70 | $45,740 | Take Profit (35% @ 11d) |
| 2026-04-26 | INTC | Bull Put | $0.50 | $0.20 | +14.8% | +$7.40 | $50 | Take Profit (50% of credit) |
| 2026-04-26 | RIVN | Bull Put | $0.21 | $0.24 | -100.0% | -$21.50 | $29 | Time Exit (16d to expiry) |
| 2026-04-26 | INTC | Bull Put | $0.47 | $0.12 | +27.2% | +$12.90 | $53 | Take Profit (50% of credit) |
| 2026-04-26 | INTC | Bull Put | $0.45 | $0.25 | -5.8% | -$2.60 | $55 | Time Exit (16d to expiry) |
| 2026-04-26 | INTC | Bull Put | $0.90 | $0.40 | +30.4% | +$27.40 | $110 | Take Profit (50% of credit) |
| 2026-04-26 | INTC | Bull Put | $0.45 | $0.15 | +16.4% | +$7.40 | $55 | Take Profit (50% of credit) |
| 2026-04-27 | UNH | Long Call | $7.00 | $15.75 | +118.8% | +$831.70 | $700 | Stop Loss (strike breached) |
| 2026-04-27 | META | Long Call | $20.30 | $18.25 | -15.1% | -$306.30 | $2,030 | Time Exit (15d to expiry) |
| 2026-04-27 | GLD | Long Put | $8.10 | $13.38 | +59.0% | +$478.10 | $810 | Stop Loss (strike breached) |
| 2026-04-27 | BA | Long Call | $5.85 | $3.75 | -42.1% | -$246.40 | $585 | Take Profit (35% @ 17d) |
| 2026-04-27 | SLV | Long Put | $2.07 | $3.20 | +54.6% | +$113.00 | $207 | manual_take_profit_50pct |
| 2026-04-27 | SLV | Long Put | $2.50 | $3.35 | +27.5% | +$68.70 | $250 | Stop Loss (strike breached) |
| 2026-04-27 | SLV | Long Put | $2.74 | $3.78 | +31.5% | +$86.26 | $274 | Stop Loss (strike breached) |
| 2026-04-27 | ORCL | Long Put | $8.00 | $10.10 | +20.1% | +$160.70 | $800 | Stop Loss (strike breached) |
| 2026-04-27 | NKE | Long Put | $1.31 | $1.15 | -20.8% | -$27.30 | $131 | Time Exit (21d to expiry) |
| 2026-04-27 | GLD | Iron Condor | $8.43 | $7.94 | +3.1% | +$25.90 | $1,658 | Time Exit (17d to expiry) |
| 2026-04-27 | TLT | Iron Condor | $0.56 | $0.17 | +29.9% | +$16.90 | $144 | Take Profit (50% of credit) |
| 2026-04-27 | NFLX | Iron Condor | $2.08 | $2.36 | -24.3% | -$50.60 | $392 | Time Exit (17d to expiry) |
| 2026-04-27 | SPY | Iron Condor | $7.06 | $0.00 | +100.0% | +$706.00 | $1,794 | Expired (settled at intrinsic) |
| 2026-04-27 | AAPL | Iron Condor | $4.57 | $0.07 | +93.5% | +$427.40 | $1,043 | Take Profit (50% of credit) |
| 2026-04-27 | QQQ | Iron Condor | $9.70 | $0.21 | +95.5% | +$925.90 | $2,553 | Take Profit (50% of credit) |
| 2026-04-27 | AMZN | Iron Condor | $6.10 | $0.00 | +100.0% | +$610.00 | $1,390 | Expired (settled at intrinsic) |
| 2026-04-27 | MSFT | Bear Call | $1.10 | $0.16 | +64.9% | +$71.40 | $140 | Take Profit (50% of credit) |
| 2026-04-27 | SPY | Bear Call | $0.46 | $0.17 | +13.9% | +$6.40 | $54 | Take Profit (50% of credit) |
| 2026-04-27 | SPY | Bear Call | $0.88 | $0.89 | -26.8% | -$23.60 | $112 | Time Exit (8d to expiry) |
| 2026-04-27 | INTC | Bull Put | $0.50 | $0.10 | +34.8% | +$17.40 | $50 | Take Profit (50% of credit) |
| 2026-04-27 | AMD | Bull Put | $2.32 | $1.09 | +43.4% | +$100.90 | $268 | Take Profit (50% of credit) |
| 2026-04-27 | NFLX | Bull Put | $0.42 | $0.21 | -3.8% | -$1.60 | $58 | Take Profit (50% of credit) |
| 2026-04-27 | INTC | Bull Put | $0.48 | $0.22 | +6.1% | +$2.90 | $52 | Take Profit (50% of credit) |
| 2026-04-27 | COIN | Bull Put | $2.08 | $3.40 | -63.9% | -$132.50 | $292 | manual_stop_loss_breach |
| 2026-04-28 | UNH | Iron Condor | $5.85 | $0.63 | +85.4% | +$498.90 | $1,416 | Take Profit (50% of credit) |
| 2026-04-28 | AAPL | Iron Condor | $5.04 | $0.07 | +94.1% | +$473.90 | $996 | Take Profit (50% of credit) |
| 2026-04-28 | GLD | Iron Condor | $8.50 | $3.83 | +52.3% | +$444.40 | $1,650 | Take Profit (50% of credit) |
| 2026-04-28 | QQQ | Iron Condor | $11.77 | $0.17 | +96.6% | +$1,137.40 | $2,823 | Take Profit (50% of credit) |
| 2026-04-28 | SPY | Iron Condor | $9.84 | $0.26 | +95.1% | +$935.90 | $2,516 | Take Profit (50% of credit) |
| 2026-04-28 | TLT | Iron Condor | $0.82 | $0.20 | +48.0% | +$39.40 | $218 | Take Profit (50% of credit) |
| 2026-04-28 | AMAT | Bull Put | $1.63 | $0.70 | +43.0% | +$69.90 | $87 | Take Profit (50% of credit) |
| 2026-04-28 | SPY | Bear Call | $0.50 | $0.25 | +4.8% | +$2.40 | $50 | Take Profit (50% of credit) |
| 2026-04-28 | UBER | Bull Put | $0.24 | $0.42 | -78.7% | -$18.50 | $26 | manual_stop_loss_breach |
| 2026-04-28 | PYPL | Bull Put | $0.25 | $0.41 | -60.8% | -$15.50 | $25 | manual_stop_loss_breach |
| 2026-04-28 | BA | Bull Put | $1.25 | $0.61 | +33.1% | +$41.40 | $125 | Take Profit (50% of credit) |
| 2026-04-28 | ORCL | Bull Put | $1.18 | $0.75 | +16.9% | +$19.90 | $132 | Time Exit (21d to expiry) |
| 2026-04-28 | QQQ | Bear Call | $0.40 | $0.04 | +34.3% | +$13.90 | $60 | Take Profit (50% of credit) |
| 2026-04-29 | QQQ | Iron Condor | $14.10 | $26.32 | -89.9% | -$1,267.20 | $2,590 | Time Exit (20d to expiry) |
| 2026-04-29 | GLD | Iron Condor | $9.56 | $8.82 | +7.7% | +$74.00 | $1,544 | Manual Close (derisk to clear stress gate) |
| 2026-04-29 | SPY | Iron Condor | $12.18 | $20.13 | -69.1% | -$840.70 | $2,282 | Time Exit (20d to expiry) |
| 2026-04-29 | IWM | Iron Condor | $6.19 | $6.96 | -19.7% | -$122.20 | $1,081 | Time Exit (20d to expiry) |
| 2026-04-29 | AAPL | Iron Condor | $7.71 | $8.32 | -7.9% | -$61.00 | $1,229 | Manual Close (derisk to clear stress gate) |
| 2026-04-29 | SLV | Long Put | $1.82 | $1.11 | -39.0% | -$71.00 | $182 | manual_thesis_broken |
| 2026-04-29 | SLV | Long Put | $2.24 | $1.46 | -34.8% | -$78.00 | $224 | manual_thesis_broken |
| 2026-04-29 | SLV | Long Put | $1.62 | $1.01 | -37.7% | -$61.00 | $162 | manual_thesis_broken |
| 2026-04-29 | SLV | Long Put | $2.02 | $1.27 | -37.1% | -$75.00 | $202 | manual_thesis_broken |
| 2026-04-29 | NVDA | Long Call | $5.85 | $1.97 | -72.5% | -$424.40 | $585 | Stop Loss (-50%) |
| 2026-04-29 | NVDA | Long Call | $4.85 | $1.55 | -74.3% | -$360.40 | $485 | Stop Loss (-50%) |
| 2026-04-29 | WMT | Long Call | $3.40 | $5.24 | +54.1% | +$184.00 | $340 | manual_take_profit_50pct |
| 2026-04-29 | SPY | Bear Call | $0.43 | $0.17 | +9.0% | +$3.90 | $57 | Take Profit (50% of credit) |
| 2026-04-29 | META | Bull Put | $2.42 | $5.55 | -100.0% | -$242.50 | $258 | Stop Loss (100% of credit) |
| 2026-04-29 | AMD | Bull Put | $1.42 | $0.69 | +35.7% | +$50.90 | $108 | Take Profit (50% of credit) |
| 2026-04-29 | GLD | Bear Call | $0.52 | $1.00 | -90.5% | -$47.50 | $48 | Stop Loss (100% of credit) |
| 2026-04-29 | INTC | Bull Put | $0.90 | $0.32 | +39.3% | +$35.40 | $110 | Take Profit (50% of credit) |
| 2026-04-29 | MU | Bull Put | $2.38 | $1.05 | +46.3% | +$109.90 | $262 | Take Profit (50% of credit) |
| 2026-04-29 | ORCL | Bull Put | $0.97 | $0.48 | +27.6% | +$26.90 | $153 | Take Profit (50% of credit) |
| 2026-04-29 | SLV | Bull Put | $0.20 | $0.14 | -83.0% | -$16.60 | $30 | Time Exit (20d to expiry) |
| 2026-04-29 | QQQ | Bear Call | $0.38 | $0.19 | +50.0% | +$19.50 | $62 | Take Profit (50% of credit) |
| 2026-04-29 | AMZN | Bear Call | $1.05 | $0.49 | +31.8% | +$33.40 | $145 | Take Profit (50% of credit) |
| 2026-04-30 | AMZN | Long Call | $5.70 | $9.00 | +51.7% | +$294.50 | $570 | Time Exit (19d to expiry) |
| 2026-04-30 | ORCL | Long Call | $5.00 | $10.70 | +107.7% | +$538.70 | $500 | Take Profit (100%) |
| 2026-04-30 | MSFT | Iron Condor | $11.06 | $10.19 | +7.9% | +$87.50 | $1,894 | Manual Close (derisk to clear stress gate) |
| 2026-04-30 | AMZN | Iron Condor | $8.36 | $9.03 | -8.1% | -$67.50 | $1,164 | Manual Close (derisk to clear stress gate) |
| 2026-04-30 | SPY | Iron Condor | $11.30 | $12.27 | -8.6% | -$97.00 | $2,370 | Manual Close (derisk to clear stress gate) |
| 2026-04-30 | AAPL | Iron Condor | $7.69 | $8.32 | -8.3% | -$63.50 | $1,232 | Manual Close (derisk to clear stress gate) |
| 2026-04-30 | NFLX | Iron Condor | $2.96 | $3.56 | -35.3% | -$104.70 | $504 | Time Exit (20d to expiry) |
| 2026-04-30 | SPY | Bear Call | $0.48 | $0.90 | -108.3% | -$52.00 | $52 | Time Exit (12d to expiry) |
| 2026-04-30 | INTC | Bull Put | $0.50 | $0.26 | +2.8% | +$1.40 | $50 | Time Exit (12d to expiry) |
| 2026-04-30 | AMD | Bull Put | $1.25 | $0.50 | +41.9% | +$52.40 | $125 | Take Profit (50% of credit) |
| 2026-04-30 | SLV | Bull Put | $0.19 | $0.13 | -82.6% | -$16.10 | $31 | Time Exit (12d to expiry) |
| 2026-04-30 | QQQ | Bear Call | $0.44 | $0.54 | -72.1% | -$32.10 | $56 | Time Exit (5d to expiry) |
| 2026-05-01 | SPY | Bear Call | $0.94 | $0.74 | -2.8% | -$2.60 | $106 | Time Exit (11d to expiry) |
| 2026-05-01 | DIA | Bear Call | $0.47 | $0.19 | +11.5% | +$5.40 | $53 | Take Profit (50% of credit) |
| 2026-05-01 | SLV | Bull Put | $0.24 | $0.00 | +100.0% | +$24.00 | $26 | Take Profit (50% of credit) |
| 2026-05-01 | GLD | Bull Put | $0.45 | $0.08 | +32.0% | +$14.40 | $55 | Take Profit (50% of credit) |
| 2026-05-01 | IWM | Bear Call | $0.40 | $0.59 | -104.0% | -$41.60 | $60 | Time Exit (11d to expiry) |
| 2026-05-01 | TXN | Long Call | $5.10 | $5.55 | +2.6% | +$13.10 | $510 | Time Exit (11d to expiry) |
| 2026-05-01 | NVDA | Long Call | $4.15 | $3.45 | -23.2% | -$96.20 | $415 | Time Exit (11d to expiry) |
| 2026-05-01 | QCOM | Long Call | $4.90 | $2.01 | -65.2% | -$319.70 | $490 | Time Exit (11d to expiry) |
| 2026-05-01 | SPY | Long Call | $6.13 | $4.48 | -33.1% | -$203.08 | $613 | Time Exit (11d to expiry) |
| 2026-05-04 | ORCL | Long Call | $6.35 | $13.25 | +102.5% | +$650.60 | $635 | Take Profit (100%) |
| 2026-05-04 | MU | Long Call | $24.45 | $64.77 | +160.8% | +$3,930.70 | $2,445 | Take Profit (100%) |
| 2026-05-04 | AVGO | Long Call | $10.10 | $7.60 | -30.9% | -$311.90 | $1,010 | Time Exit (8d to expiry) |
| 2026-05-04 | AAPL | Long Call | $4.60 | $9.40 | +98.1% | +$451.10 | $460 | Take Profit (100%) |
| 2026-05-04 | QCOM | Long Call | $5.40 | $15.80 | +186.4% | +$1,006.30 | $540 | Take Profit (100%) |
| 2026-05-05 | QQQ | Bear Call | $0.44 | $1.00 | -129.9% | -$56.50 | $56 | Stop Loss (100% of credit) |
| 2026-05-05 | SPY | Bear Call | $0.46 | $0.20 | +8.4% | +$3.90 | $54 | Take Profit (50% of credit) |
| 2026-05-05 | AMD | Bull Put | $1.28 | $0.16 | +69.7% | +$88.90 | $122 | Take Profit (50% of credit) |
| 2026-05-05 | GLD | Bull Put | $0.45 | $0.21 | +3.1% | +$1.40 | $55 | Take Profit (50% of credit) |
| 2026-05-05 | AVGO | Bull Put | $1.42 | $3.02 | -75.4% | -$107.50 | $108 | Stop Loss (100% of credit) |
| 2026-05-06 | GE | Long Call | $6.95 | $3.45 | -56.5% | -$393.00 | $695 | Time Exit (11d to expiry) |
| 2026-05-06 | ORCL | Bear Call | $1.15 | $0.43 | +43.0% | +$49.40 | $135 | Take Profit (50% of credit) |
| 2026-05-06 | AMD | Long Call | $21.00 | $41.00 | +90.4% | +$1,898.70 | $2,100 | Time Exit (18d to expiry) |
| 2026-05-06 | SPY | Long Call | $6.58 | $8.07 | +16.4% | +$108.22 | $658 | Time Exit (11d to expiry) |
| 2026-05-07 | ORCL | Long Call | $9.15 | $7.55 | -23.6% | -$216.20 | $915 | Time Exit (18d to expiry) |
| 2026-05-07 | SPY | Bear Call | $0.50 | $0.77 | -101.2% | -$50.10 | $50 | Time Exit (4d to expiry) |
| 2026-05-07 | XLE | Iron Condor | $0.70 | $0.37 | -18.3% | -$12.70 | $180 | Time Exit (20d to expiry) |
| 2026-05-08 | ORCL | Long Call | $7.70 | $7.23 | -12.3% | -$94.50 | $770 | Time Exit (11d to expiry) |
| 2026-05-08 | TSLA | Bear Call | $1.25 | $0.92 | +8.3% | +$10.40 | $125 | Time Exit (11d to expiry) |
| 2026-05-08 | SPY | Iron Condor | $9.95 | $9.54 | -0.4% | -$4.20 | $2,205 | Time Exit (20d to expiry) |
| 2026-05-08 | AMD | Long Call | $19.05 | $24.30 | +22.2% | +$423.70 | $1,905 | Time Exit (11d to expiry) |
| 2026-05-08 | GOOGL | Long Call | $7.65 | $5.60 | -33.0% | -$252.20 | $765 | Time Exit (11d to expiry) |
| 2026-05-08 | DAL | Long Call | $1.70 | $1.18 | -37.4% | -$63.50 | $170 | Time Exit (11d to expiry) |
| 2026-05-08 | SLV | Long Call | $1.96 | $4.71 | +133.6% | +$261.94 | $196 | Take Profit (100%) |
| 2026-05-08 | SPY | Bear Call | $0.48 | $0.48 | -47.1% | -$22.60 | $52 | Time Exit (7d to expiry) |
| 2026-05-08 | AMD | Bull Put | $1.17 | $0.00 | +100.0% | +$117.50 | $133 | Take Profit (50% of credit) |
| 2026-05-08 | NVDA | Bull Put | $1.02 | $0.90 | -9.9% | -$10.10 | $148 | Time Exit (11d to expiry) |
| 2026-05-08 | SLV | Bull Put | $0.21 | $0.12 | -64.8% | -$13.60 | $29 | Time Exit (11d to expiry) |
| 2026-05-08 | AVGO | Bull Put | $1.22 | $1.01 | -0.9% | -$1.10 | $128 | Time Exit (11d to expiry) |
| 2026-05-08 | MRK | Iron Condor | $1.69 | $1.59 | -21.2% | -$35.70 | $332 | Time Exit (20d to expiry) |
| 2026-05-08 | AAPL | Iron Condor | $7.04 | $7.84 | -17.7% | -$124.70 | $1,296 | Time Exit (20d to expiry) |
| 2026-05-08 | XLE | Iron Condor | $0.69 | $0.37 | -20.0% | -$13.70 | $182 | Time Exit (20d to expiry) |
| 2026-05-08 | NFLX | Iron Condor | $2.67 | $2.08 | +5.0% | +$13.30 | $434 | Time Exit (20d to expiry) |
| 2026-05-08 | GLD | Iron Condor | $8.71 | $6.97 | +14.8% | +$128.80 | $1,629 | Time Exit (20d to expiry) |
| 2026-05-11 | TSLA | Long Call | $11.60 | $29.35 | +146.9% | +$1,704.10 | $1,160 | Take Profit (100%) |
| 2026-05-11 | TSLA | Bear Call | $2.53 | $2.53 | -9.1% | -$23.10 | $247 | Time Exit (15d to expiry) |
| 2026-05-11 | AMZN | Iron Condor | $6.64 | $4.53 | +25.0% | +$165.80 | $1,336 | Time Exit (20d to expiry) |
| 2026-05-11 | SBUX | Long Call | $1.84 | $1.64 | -17.6% | -$32.34 | $184 | Time Exit (15d to expiry) |
| 2026-05-11 | MS | Long Call | $4.50 | $4.70 | -1.8% | -$8.30 | $450 | Time Exit (21d to expiry) |
| 2026-05-11 | ORCL | Long Call | $7.55 | $3.43 | -60.7% | -$458.60 | $755 | Stop Loss (-50%) |
| 2026-05-12 | INTC | Bull Put | $0.52 | $0.07 | +43.6% | +$22.90 | $48 | Take Profit (50% of credit) |
| 2026-05-12 | AMZN | Iron Condor | $5.85 | $4.22 | +20.2% | +$118.30 | $914 | Time Exit (20d to expiry) |
| 2026-05-13 | COIN | Long Call | $8.65 | $18.00 | +101.9% | +$881.80 | $865 | Take Profit (100%) |
| 2026-05-13 | ORCL | Long Call | $5.85 | $3.40 | -48.1% | -$281.40 | $585 | Time Exit (11d to expiry) |
| 2026-05-13 | IWM | Long Call | $4.45 | $2.16 | -57.8% | -$257.00 | $445 | Stop Loss (-50%) |
| 2026-05-13 | NVDA | Long Call | $9.35 | $4.90 | -53.7% | -$502.40 | $935 | Time Exit (11d to expiry) |
| 2026-05-13 | F | Bear Call | $0.28 | $0.18 | -47.6% | -$13.10 | $22 | Time Exit (4d to expiry) |
| 2026-05-13 | SPY | Bear Call | $0.50 | $0.33 | -12.3% | -$6.10 | $50 | Time Exit (4d to expiry) |
| 2026-05-13 | GLD | Bull Put | $0.52 | $5.70 | -90.5% | -$47.50 | $48 | Stop Loss (100% of credit) |
| 2026-05-13 | META | Bear Call | $1.23 | $1.03 | -2.5% | -$3.10 | $127 | Time Exit (4d to expiry) |
| 2026-05-13 | GOOGL | Bear Call | $1.20 | $1.08 | -8.8% | -$10.60 | $130 | Time Exit (4d to expiry) |
| 2026-05-13 | AAPL | Iron Condor | $5.73 | $5.70 | -7.3% | -$41.70 | $926 | Time Exit (20d to expiry) |
| 2026-05-13 | AMZN | Iron Condor | $6.23 | $4.22 | +25.1% | +$156.30 | $876 | Time Exit (20d to expiry) |
| 2026-05-13 | QQQ | Iron Condor | $13.68 | $13.70 | -3.5% | -$47.20 | $2,132 | Time Exit (20d to expiry) |
| 2026-05-13 | SPY | Iron Condor | $10.94 | $9.74 | +6.8% | +$74.80 | $2,006 | Time Exit (20d to expiry) |
| 2026-05-13 | MSFT | Iron Condor | $11.39 | $18.26 | -64.2% | -$731.70 | $1,860 | Time Exit (20d to expiry) |
| 2026-05-14 | AMD | Long Call | $19.65 | $6.20 | -73.6% | -$1,446.30 | $1,965 | Time Exit (11d to expiry) |
| 2026-05-14 | SBUX | Long Call | $1.85 | $1.32 | -35.4% | -$65.40 | $185 | Time Exit (11d to expiry) |
| 2026-05-14 | ORCL | Long Call | $8.35 | $2.77 | -73.0% | -$609.40 | $835 | Time Exit (11d to expiry) |
| 2026-05-14 | INTC | Long Call | $8.60 | $3.26 | -68.2% | -$586.90 | $860 | Stop Loss (-50%) |
| 2026-05-14 | NFLX | Bull Put | $0.47 | $0.25 | -1.3% | -$0.60 | $53 | Time Exit (18d to expiry) |
| 2026-05-14 | INTC | Bear Call | $0.48 | $0.00 | +71.4% | +$33.90 | $52 | Take Profit (50% of credit) |
| 2026-05-14 | TMO | Bull Put | $2.25 | $0.73 | +57.5% | +$129.40 | $275 | Take Profit (50% of credit) |
| 2026-05-14 | GLD | Bull Put | $1.15 | $0.00 | +100.0% | +$115.00 | $185 | Take Profit (50% of credit) |
| 2026-05-14 | SLV | Bull Put | $0.27 | $0.00 | +72.4% | +$19.90 | $23 | Take Profit (50% of credit) |
| 2026-05-14 | XLE | Iron Condor | $0.70 | $0.59 | -47.8% | -$33.70 | $180 | Time Exit (20d to expiry) |
| 2026-05-14 | TLT | Iron Condor | $0.56 | $0.28 | -29.6% | -$16.70 | $144 | Take Profit (50% of credit) |
| 2026-05-14 | XLF | Iron Condor | $0.68 | $0.42 | -29.2% | -$19.70 | $132 | Time Exit (20d to expiry) |
| 2026-05-14 | MSFT | Iron Condor | $9.83 | $12.97 | -36.5% | -$359.20 | $1,517 | Time Exit (20d to expiry) |
| 2026-05-14 | AMZN | Iron Condor | $8.06 | $5.85 | +21.8% | +$175.80 | $1,194 | Time Exit (20d to expiry) |
| 2026-05-15 | MSFT | Long Call | $10.25 | $8.10 | -27.1% | -$277.80 | $1,025 | Time Exit (18d to expiry) |
| 2026-05-15 | ORCL | Long Call | $6.90 | $2.77 | -66.0% | -$455.70 | $690 | Time Exit (11d to expiry) |
| 2026-05-15 | INTC | Bull Put | $0.50 | $0.00 | +94.8% | +$47.40 | $50 | Take Profit (50% of credit) |
| 2026-05-15 | GLD | Bear Call | $0.37 | $0.80 | -166.7% | -$62.50 | $63 | Stop Loss (100% of credit) |
| 2026-05-15 | ADBE | Bull Put | $1.95 | $2.13 | -20.8% | -$40.60 | $305 | Time Exit (21d to expiry) |
| 2026-05-15 | IWM | Bear Call | $0.41 | $0.32 | -33.2% | -$13.60 | $59 | Time Exit (11d to expiry) |
| 2026-05-15 | AVGO | Bull Put | $2.23 | $0.75 | +56.1% | +$124.90 | $277 | Take Profit (50% of credit) |
| 2026-05-15 | AAPL | Iron Condor | $5.12 | $4.43 | +4.6% | +$23.30 | $988 | Time Exit (20d to expiry) |
| 2026-05-15 | MSFT | Iron Condor | $10.67 | $10.09 | +1.2% | +$13.30 | $1,933 | Time Exit (20d to expiry) |
| 2026-05-15 | TLT | Iron Condor | $0.84 | $0.94 | -64.7% | -$54.70 | $216 | Time Exit (20d to expiry) |
| 2026-05-15 | AMZN | Iron Condor | $7.13 | $6.22 | +6.5% | +$46.30 | $1,287 | Time Exit (20d to expiry) |
| 2026-05-15 | META | Iron Condor | $16.18 | $11.25 | +27.7% | +$447.30 | $2,882 | Time Exit (20d to expiry) |
| 2026-05-19 | MSFT | Long Call | $7.45 | $6.30 | -21.6% | -$161.00 | $745 | Time Exit (14d to expiry) |
| 2026-05-19 | GOOGL | Long Call | $8.05 | $5.85 | -33.5% | -$269.60 | $805 | Time Exit (14d to expiry) |
| 2026-05-19 | AMD | Long Call | $19.45 | $48.38 | +143.5% | +$2,791.70 | $1,945 | Take Profit (100%) |
| 2026-05-20 | SPY | Bear Call | $0.59 | $0.26 | +16.9% | +$9.90 | $41 | Take Profit (50% of credit) |
| 2026-05-20 | QQQ | Bull Put | $0.54 | $0.10 | +39.1% | +$20.90 | $46 | Take Profit (50% of credit) |
| 2026-05-20 | IWM | Bear Call | $0.46 | $0.19 | +8.6% | +$3.90 | $54 | Take Profit (50% of credit) |
| 2026-05-20 | SLV | Bull Put | $0.45 | $0.39 | -36.9% | -$16.60 | $55 | Time Exit (18d to expiry) |
| 2026-05-20 | COIN | Bull Put | $2.13 | $1.32 | +27.2% | +$57.90 | $287 | Time Exit (20d to expiry) |
| 2026-05-20 | QQQ | Iron Condor | $18.70 | $15.12 | +16.7% | +$312.80 | $3,130 | Time Exit (21d to expiry) |
| 2026-05-20 | SPY | Iron Condor | $10.14 | $7.30 | +23.6% | +$238.80 | $1,986 | Time Exit (21d to expiry) |
| 2026-05-20 | TLT | Iron Condor | $0.72 | $0.61 | -47.5% | -$34.20 | $128 | Time Exit (21d to expiry) |
| 2026-05-20 | AAPL | Iron Condor | $7.88 | $10.75 | -42.2% | -$332.20 | $1,212 | Time Exit (21d to expiry) |
| 2026-05-20 | BAC | Iron Condor | $1.34 | $1.80 | -68.1% | -$91.20 | $266 | Time Exit (21d to expiry) |
| 2026-05-20 | AMD | Long Call | $25.60 | $56.84 | +118.1% | +$3,022.70 | $2,560 | Take Profit (100%) |
| 2026-05-21 | AMD | Long Call | $24.15 | $63.55 | +159.0% | +$3,838.70 | $2,415 | Take Profit (100%) |
| 2026-05-21 | NVDA | Long Call | $7.25 | $4.25 | -47.6% | -$344.80 | $725 | Time Exit (20d to expiry) |
| 2026-05-21 | MSFT | Long Call | $11.15 | $23.85 | +107.8% | +$1,201.80 | $1,115 | Take Profit (100%) |
| 2026-05-21 | INTC | Long Call | $7.55 | $6.69 | -17.6% | -$132.60 | $755 | Time Exit (20d to expiry) |
| 2026-05-22 | NVDA | Long Call | $6.90 | $5.40 | -27.9% | -$192.70 | $690 | Time Exit (20d to expiry) |
| 2026-05-22 | AMD | Long Call | $25.85 | $39.35 | +48.3% | +$1,248.70 | $2,585 | Time Exit (20d to expiry) |
| 2026-05-22 | MSFT | Long Call | $10.25 | $23.85 | +126.6% | +$1,297.20 | $1,025 | Take Profit (100%) |
| 2026-05-22 | SLV | Bull Put | $0.23 | $0.23 | -98.3% | -$22.60 | $27 | Time Exit (20d to expiry) |
| 2026-05-22 | NVDA | Bull Put | $0.43 | $0.34 | -33.2% | -$14.10 | $57 | Time Exit (20d to expiry) |
| 2026-05-22 | SPY | Bear Call | $0.43 | $0.14 | +14.9% | +$6.40 | $57 | Take Profit (50% of credit) |
| 2026-05-22 | AMD | Bull Put | $2.32 | $2.00 | +4.3% | +$9.90 | $268 | Time Exit (18d to expiry) |
| 2026-05-22 | INTC | Bull Put | $0.50 | $0.67 | -79.2% | -$39.60 | $50 | Time Exit (18d to expiry) |
| 2026-05-22 | AAPL | Iron Condor | $4.39 | $3.88 | +1.4% | +$6.30 | $1,060 | Time Exit (21d to expiry) |
| 2026-05-22 | MSFT | Iron Condor | $13.20 | $26.80 | -106.5% | -$1,405.20 | $2,180 | Stop Loss (100% of credit) |
| 2026-05-22 | XLE | Iron Condor | $1.32 | $1.68 | -61.5% | -$81.20 | $268 | Time Exit (21d to expiry) |
| 2026-05-22 | QQQ | Iron Condor | $16.41 | $13.54 | +14.7% | +$241.30 | $2,860 | Time Exit (21d to expiry) |
| 2026-05-22 | SPY | Iron Condor | $12.52 | $9.01 | +24.5% | +$306.30 | $2,748 | Time Exit (21d to expiry) |
| 2026-05-27 | INTC | Long Call | $7.05 | $2.85 | -65.8% | -$463.60 | $705 | Time Exit (11d to expiry) |
| 2026-05-27 | QCOM | Long Call | $14.40 | $14.18 | -7.6% | -$109.70 | $1,440 | Time Exit (17d to expiry) |
| 2026-05-27 | AMD | Long Call | $24.20 | $25.46 | +1.0% | +$24.70 | $2,420 | Time Exit (11d to expiry) |
| 2026-05-29 | CHTR | Long Put | $6.30 | $5.90 | -12.6% | -$79.10 | $630 | Time Exit (17d to expiry) |
| 2026-05-29 | INTC | Long Call | $6.15 | $3.00 | -57.4% | -$353.20 | $615 | Time Exit (11d to expiry) |
| 2026-05-29 | QCOM | Long Call | $14.45 | $6.81 | -59.0% | -$852.00 | $1,445 | Time Exit (17d to expiry) |
| 2026-05-29 | AMD | Long Call | $21.15 | $20.00 | -10.2% | -$216.30 | $2,115 | Time Exit (11d to expiry) |
| 2026-05-29 | AMAT | Long Call | $13.40 | $18.85 | +34.6% | +$463.30 | $1,340 | Time Exit (11d to expiry) |
| 2026-06-01 | NVDA | Long Call | $6.35 | $2.99 | -59.1% | -$375.40 | $635 | Stop Loss (-50%) |
| 2026-06-01 | LLY | Long Call | $24.10 | $65.23 | +166.5% | +$4,011.70 | $2,410 | Take Profit (100%) |
| 2026-06-01 | CRM | Long Call | $7.85 | $2.88 | -69.5% | -$545.40 | $785 | Stop Loss (-50%) |
| 2026-06-01 | SLV | Long Put | $2.03 | $5.26 | +152.5% | +$309.52 | $203 | Take Profit (100%) |
| 2026-06-01 | CVX | Long Put | $4.15 | $3.20 | -29.2% | -$121.20 | $415 | Time Exit (13d to expiry) |
| 2026-06-02 | CHTR | Long Put | $6.80 | $10.80 | +52.6% | +$357.90 | $680 | Time Exit (13d to expiry) |
| 2026-06-02 | CRM | Long Call | $5.80 | $1.30 | -83.8% | -$486.10 | $580 | Time Exit (13d to expiry) |
| 2026-06-02 | QCOM | Long Call | $13.00 | $4.63 | -70.5% | -$916.30 | $1,300 | Time Exit (13d to expiry) |
| 2026-06-02 | INTC | Long Put | $8.60 | $9.85 | +8.4% | +$72.10 | $860 | Time Exit (21d to expiry) |
| 2026-06-02 | NVDA | Long Call | $6.55 | $2.68 | -65.3% | -$427.60 | $655 | Stop Loss (-50%) |
| 2026-06-02 | AMD | Long Call | $22.50 | $8.30 | -67.6% | -$1,521.30 | $2,250 | Time Exit (13d to expiry) |
| 2026-06-02 | SLV | Long Put | $1.92 | $5.26 | +167.3% | +$321.18 | $192 | Take Profit (100%) |
| 2026-06-02 | AMD | Long Call | $24.30 | $9.00 | -67.1% | -$1,631.30 | $2,430 | Time Exit (13d to expiry) |
| 2026-06-02 | INTC | Long Put | $10.00 | $9.30 | -13.1% | -$131.30 | $1,000 | Time Exit (21d to expiry) |
| 2026-06-03 | CRM | Long Call | $6.05 | $2.16 | -70.5% | -$426.60 | $605 | Time Exit (18d to expiry) |
| 2026-06-03 | QCOM | Long Call | $13.55 | $3.20 | -82.5% | -$1,117.60 | $1,355 | Stop Loss (-50%) |
| 2026-06-03 | INTC | Long Put | $8.15 | $7.67 | -12.0% | -$98.20 | $815 | Time Exit (18d to expiry) |
| 2026-06-03 | XYZ | Long Put | $2.35 | $1.79 | -30.4% | -$71.40 | $235 | Time Exit (10d to expiry) |
| 2026-06-03 | TXN | Long Call | $8.90 | $3.45 | -67.4% | -$599.70 | $890 | Stop Loss (-50%) |
| 2026-06-03 | WMT | Long Put | $2.49 | $1.17 | -59.5% | -$148.24 | $249 | Stop Loss (-50%) |
| 2026-06-03 | AAPL | Long Put | $7.70 | $5.39 | -36.2% | -$278.50 | $770 | Time Exit (18d to expiry) |
| 2026-06-08 | SPY | Bear Call | $0.50 | $0.17 | +20.8% | +$10.40 | $50 | Take Profit (50% of credit) |
| 2026-06-08 | QQQ | Bear Call | $0.61 | $0.00 | +100.0% | +$61.00 | $39 | Take Profit (50% of credit) |
| 2026-06-08 | GOOGL | Bull Put | $2.10 | $0.10 | +84.5% | +$177.40 | $290 | Take Profit (50% of credit) |
| 2026-06-08 | INTC | Bull Put | $0.50 | $0.00 | +64.8% | +$32.40 | $50 | Take Profit (50% of credit) |
| 2026-06-08 | NVDA | Bear Call | $0.43 | $0.21 | -2.6% | -$1.10 | $57 | Take Profit (50% of credit) |
| 2026-06-08 | META | Bull Put | $2.25 | $2.70 | -30.0% | -$67.60 | $275 | Time Exit (7d to expiry) |
| 2026-06-08 | GOOGL | Bull Put | $2.05 | $2.60 | -37.9% | -$77.60 | $295 | Time Exit (7d to expiry) |
| 2026-06-08 | TSLA | Bull Put | $2.08 | $3.18 | -64.1% | -$133.10 | $292 | Time Exit (15d to expiry) |
| 2026-06-08 | INTC | Bull Put | $0.50 | $0.00 | +58.8% | +$29.40 | $50 | Take Profit (50% of credit) |
| 2026-06-08 | QQQ | Bear Call | $0.49 | $0.04 | +45.7% | +$22.40 | $51 | Take Profit (50% of credit) |
| 2026-06-08 | TSLA | Bull Put | $1.08 | $0.53 | +29.7% | +$31.90 | $142 | Take Profit (50% of credit) |
| 2026-06-08 | AMD | Bull Put | $2.62 | $0.60 | +68.5% | +$179.90 | $238 | Take Profit (50% of credit) |
| 2026-06-08 | SPY | Bear Call | $0.50 | $1.31 | -102.0% | -$50.50 | $50 | Stop Loss (100% of credit) |
| 2026-06-08 | INTC | Bull Put | $0.48 | $0.08 | +35.6% | +$16.90 | $52 | Take Profit (50% of credit) |
| 2026-06-08 | NVDA | Bear Call | $0.23 | $0.00 | +8.4% | +$1.90 | $27 | Take Profit (50% of credit) |
| 2026-06-08 | QQQ | Bear Call | $0.45 | $0.00 | +54.2% | +$24.40 | $55 | Take Profit (50% of credit) |
| 2026-06-08 | AMD | Bear Call | $1.30 | $2.75 | -92.3% | -$120.00 | $120 | Stop Loss (100% of credit) |
| 2026-06-08 | SPY | Bear Call | $0.51 | $0.06 | +43.9% | +$22.40 | $49 | Take Profit (50% of credit) |
| 2026-06-08 | QQQ | Bear Call | $0.49 | $0.21 | +10.1% | +$4.90 | $51 | Take Profit (50% of credit) |
| 2026-06-08 | DIA | Bear Call | $0.50 | $0.00 | +60.8% | +$30.40 | $50 | Take Profit (50% of credit) |
| 2026-06-08 | IWM | Bull Put | $0.40 | $0.19 | -4.0% | -$1.60 | $60 | Take Profit (50% of credit) |
| 2026-06-08 | AAPL | Bull Put | $1.33 | $2.93 | -88.7% | -$117.50 | $117 | Stop Loss (100% of credit) |
| 2026-06-08 | QQQ | Bull Put | $0.48 | $0.10 | +32.1% | +$15.40 | $52 | Take Profit (50% of credit) |
| 2026-06-08 | INTC | Bear Call | $0.50 | $0.00 | +54.8% | +$27.40 | $50 | Take Profit (50% of credit) |
| 2026-06-08 | SPY | Bear Call | $0.46 | $0.18 | +11.7% | +$5.40 | $54 | Take Profit (50% of credit) |
| 2026-06-08 | MSFT | Bear Call | $1.18 | $0.50 | +38.2% | +$44.90 | $132 | Take Profit (50% of credit) |
| 2026-06-08 | QQQ | Bull Put | $0.56 | $0.00 | +93.6% | +$52.40 | $44 | Take Profit (50% of credit) |
| 2026-06-08 | AAPL | Bull Put | $3.30 | $3.76 | -20.8% | -$68.60 | $170 | Time Exit (7d to expiry) |
| 2026-06-08 | UNH | Bear Call | $1.17 | $0.80 | +12.7% | +$14.90 | $133 | Time Exit (7d to expiry) |
| 2026-06-08 | SPY | Bull Put | $0.39 | $0.13 | +8.7% | +$3.40 | $61 | Take Profit (50% of credit) |
| 2026-06-08 | SPY | Bear Call | $0.46 | $0.00 | +100.0% | +$45.50 | $54 | Take Profit (50% of credit) |
| 2026-06-08 | ORCL | Bull Put | $1.22 | $0.70 | +24.4% | +$29.90 | $128 | Time Exit (7d to expiry) |
| 2026-06-08 | NVDA | Bull Put | $1.77 | $2.12 | -32.2% | -$57.10 | $323 | Time Exit (15d to expiry) |
| 2026-06-09 | QQQ | Bull Put | $0.56 | $0.00 | +93.6% | +$52.40 | $44 | Take Profit (50% of credit) |
| 2026-06-09 | SPY | Bear Call | $0.46 | $0.00 | +100.0% | +$45.50 | $54 | Take Profit (50% of credit) |
| 2026-06-09 | ORCL | Bull Put | $1.22 | $3.05 | -104.1% | -$127.50 | $128 | Stop Loss (100% of credit) |
| 2026-06-09 | INTC | Bull Put | $0.50 | $0.00 | +100.0% | +$50.00 | $50 | Take Profit (50% of credit) |
| 2026-06-09 | NVDA | Bull Put | $1.77 | $1.64 | -5.1% | -$9.10 | $323 | Time Exit (11d to expiry) |
| 2026-06-09 | SPY | Bear Call | $2.66 | $1.30 | +42.6% | +$113.40 | $234 | Take Profit (50% of credit) |
| 2026-06-09 | XLF | Bear Call | $0.24 | $0.48 | -108.3% | -$26.00 | $26 | Stop Loss (100% of credit) |
| 2026-06-09 | F | Bear Call | $0.19 | $0.09 | -66.3% | -$12.60 | $31 | Take Profit (50% of credit) |
| 2026-06-09 | WMT | Bear Call | $1.80 | $1.42 | +8.6% | +$15.40 | $320 | Time Exit (11d to expiry) |
| 2026-06-09 | SLV | Bear Call | $0.22 | $0.08 | -39.1% | -$8.60 | $28 | Take Profit (50% of credit) |
| 2026-06-09 | COIN | Bull Put | $2.02 | $0.90 | +44.4% | +$89.90 | $298 | Take Profit (50% of credit) |
| 2026-06-09 | QQQ | Bear Call | $0.60 | $0.28 | +15.7% | +$9.40 | $40 | Take Profit (50% of credit) |
| 2026-06-09 | NVDA | Bear Call | $0.27 | $0.12 | -25.8% | -$7.10 | $23 | Take Profit (50% of credit) |
| 2026-06-09 | SPY | Bear Call | $0.56 | $0.26 | +12.4% | +$6.90 | $44 | Take Profit (50% of credit) |
| 2026-06-09 | AAPL | Bear Call | $1.20 | $0.46 | +42.8% | +$51.40 | $130 | Take Profit (50% of credit) |
| 2026-06-09 | META | Bear Call | $1.57 | $0.35 | +63.4% | +$99.90 | $93 | Take Profit (50% of credit) |
| 2026-06-10 | QQQ | Bull Put | $0.48 | $0.00 | +100.0% | +$48.50 | $52 | Take Profit (50% of credit) |
| 2026-06-10 | SPY | Bull Put | $0.51 | $0.00 | +100.0% | +$51.00 | $49 | Take Profit (50% of credit) |
| 2026-06-10 | INTC | Bull Put | $2.45 | $0.95 | +52.0% | +$127.40 | $255 | Take Profit (50% of credit) |
| 2026-06-10 | NVDA | Bull Put | $0.97 | $0.42 | +33.7% | +$32.90 | $153 | Take Profit (50% of credit) |
| 2026-06-10 | ORCL | Bull Put | $1.28 | $2.82 | -96.1% | -$122.50 | $122 | Stop Loss (100% of credit) |
| 2026-06-10 | AAPL | Iron Condor | $5.67 | $5.61 | -6.9% | -$39.20 | $933 | Time Exit (21d to expiry) |
| 2026-06-10 | META | Iron Condor | $18.10 | $15.18 | +13.6% | +$246.80 | $2,690 | Time Exit (21d to expiry) |
| 2026-06-10 | SPY | Iron Condor | $12.71 | $9.51 | +21.7% | +$275.30 | $2,428 | Time Exit (21d to expiry) |
| 2026-06-10 | GOOGL | Iron Condor | $11.73 | $12.35 | -9.1% | -$107.20 | $1,827 | Time Exit (21d to expiry) |
| 2026-06-10 | IWM | Iron Condor | $7.01 | $8.60 | -29.0% | -$203.70 | $999 | Time Exit (21d to expiry) |
| 2026-06-11 | IWM | Long Call | $3.65 | $9.70 | +159.1% | +$581.38 | $365 | Take Profit (100%) |
| 2026-06-11 | GM | Long Put | $1.96 | $1.36 | -37.4% | -$73.43 | $196 | Time Exit (11d to expiry) |
| 2026-06-11 | CSCO | Long Put | $3.39 | $1.91 | -50.1% | -$169.80 | $339 | Time Exit (11d to expiry) |
| 2026-06-11 | SCHW | Long Call | $2.68 | $2.65 | -7.5% | -$20.11 | $268 | Time Exit (21d to expiry) |
| 2026-06-11 | UNH | Long Call | $6.98 | $3.57 | -55.1% | -$384.45 | $698 | Time Exit (11d to expiry) |
| 2026-06-11 | AVGO | Long Put | $16.15 | $8.50 | -53.4% | -$863.20 | $1,615 | Time Exit (17d to expiry) |
| 2026-06-11 | ORCL | Long Put | $6.70 | $3.10 | -59.9% | -$401.50 | $670 | Stop Loss (-50%) |
| 2026-06-11 | SPY | Long Call | $8.13 | $18.55 | +122.0% | +$991.92 | $813 | Take Profit (100%) |
| 2026-06-11 | ORCL | Long Put | $14.40 | $7.16 | -56.4% | -$811.70 | $1,440 | Stop Loss (-50%) |
| 2026-06-11 | WMT | Long Put | $3.25 | $4.20 | +22.8% | +$74.20 | $325 | Time Exit (21d to expiry) |
| 2026-06-11 | DIA | Long Call | $6.90 | $15.00 | +111.2% | +$767.30 | $690 | Take Profit (100%) |
| 2026-06-11 | F | Long Put | $0.59 | $0.27 | -73.4% | -$43.30 | $59 | Stop Loss (-50%) |
| 2026-06-11 | IWM | Long Call | $6.24 | $13.72 | +113.7% | +$709.26 | $624 | Take Profit (100%) |
| 2026-06-11 | INTC | Bull Put | $0.50 | $0.00 | +100.0% | +$50.00 | $50 | Take Profit (50% of credit) |
| 2026-06-11 | ORCL | Bull Put | $1.25 | $3.72 | -100.0% | -$125.00 | $125 | Stop Loss (100% of credit) |
| 2026-06-11 | MU | Bull Put | $5.40 | $0.00 | +100.0% | +$540.00 | $460 | Take Profit (50% of credit) |
| 2026-06-11 | SPY | Bear Call | $0.50 | $0.06 | +43.4% | +$21.90 | $50 | Take Profit (50% of credit) |
| 2026-06-11 | AMD | Bull Put | $4.33 | $1.80 | +53.2% | +$229.90 | $567 | Take Profit (50% of credit) |
| 2026-06-13 | AVGO | Long Put | $21.35 | $10.29 | -56.5% | -$1,207.30 | $2,135 | Stop Loss (-50%) |
| 2026-06-13 | CSCO | Long Call | $3.95 | $1.93 | -57.5% | -$227.00 | $395 | Stop Loss (-50%) |
| 2026-06-13 | DIA | Long Put | $5.80 | $2.78 | -58.3% | -$338.10 | $580 | Stop Loss (-50%) |
| 2026-06-13 | COST | Long Put | $18.70 | $26.20 | +34.7% | +$648.70 | $1,870 | Time Exit (21d to expiry) |
| 2026-06-13 | ORCL | Long Put | $10.10 | $25.05 | +141.9% | +$1,433.10 | $1,010 | Take Profit (100%) |
| 2026-06-15 | COST | Long Put | $24.45 | $40.00 | +59.5% | +$1,453.70 | $2,445 | Time Exit (21d to expiry) |
| 2026-06-15 | VZ | Long Call | $1.06 | $0.52 | -61.6% | -$65.30 | $106 | Stop Loss (-50%) |
| 2026-06-15 | C | Long Put | $4.65 | $2.28 | -57.2% | -$266.20 | $465 | Stop Loss (-50%) |
| 2026-06-15 | AVGO | Long Put | $20.50 | $26.43 | +24.0% | +$491.70 | $2,050 | Time Exit (17d to expiry) |
| 2026-06-15 | AMZN | Long Put | $8.00 | $18.20 | +121.3% | +$970.70 | $800 | Take Profit (100%) |
| 2026-06-15 | META | Bear Call | $1.20 | $0.25 | +60.3% | +$72.40 | $130 | Take Profit (50% of credit) |
| 2026-06-15 | SPY | Bear Call | $0.48 | $0.23 | +5.0% | +$2.40 | $52 | Take Profit (50% of credit) |
| 2026-06-15 | QQQ | Bear Call | $0.97 | $0.48 | +26.8% | +$25.90 | $103 | Take Profit (50% of credit) |
| 2026-06-15 | AAPL | Bear Call | $1.12 | $0.51 | +34.6% | +$38.90 | $138 | Take Profit (50% of credit) |
| 2026-06-15 | ORCL | Bull Put | $1.08 | $1.32 | -43.8% | -$47.10 | $142 | Time Exit (8d to expiry) |
| 2026-06-16 | AMD | Long Call | $32.05 | $24.67 | -26.2% | -$839.30 | $3,205 | Time Exit (21d to expiry) |
| 2026-06-16 | UAL | Long Call | $5.35 | $10.72 | +94.1% | +$503.60 | $535 | Take Profit (100%) |
| 2026-06-16 | GS | Long Call | $34.05 | $10.33 | -72.6% | -$2,473.30 | $3,405 | Time Exit (21d to expiry) |
| 2026-06-16 | DE | Long Call | $15.60 | $32.28 | +100.8% | +$1,573.10 | $1,560 | Take Profit (100%) |
| 2026-06-16 | LRCX | Long Call | $22.20 | $24.95 | +7.8% | +$173.70 | $2,220 | Time Exit (21d to expiry) |
| 2026-06-16 | AAPL | Iron Condor | $4.56 | $6.83 | -59.5% | -$271.70 | $1,044 | Time Exit (21d to expiry) |
| 2026-06-16 | AMZN | Iron Condor | $6.21 | $8.60 | -45.9% | -$284.70 | $880 | Time Exit (21d to expiry) |
| 2026-06-16 | GOOGL | Iron Condor | $9.16 | $13.77 | -55.3% | -$506.20 | $1,584 | Time Exit (21d to expiry) |
| 2026-06-16 | WMT | Iron Condor | $1.17 | $1.18 | -39.5% | -$46.20 | $383 | Time Exit (21d to expiry) |
| 2026-06-16 | IWM | Iron Condor | $5.31 | $5.66 | -15.0% | -$79.70 | $768 | Time Exit (21d to expiry) |
| 2026-06-16 | RTX | Long Call | $4.15 | $4.81 | +9.6% | +$39.80 | $415 | Time Exit (21d to expiry) |
| 2026-06-16 | RIVN | Long Call | $0.70 | $1.03 | +31.0% | +$21.70 | $70 | Time Exit (21d to expiry) |
| 2026-06-16 | UBER | Long Call | $1.52 | $0.79 | -55.5% | -$84.30 | $152 | Time Exit (13d to expiry) |
| 2026-06-16 | UBER | Long Put | $2.37 | $1.98 | -23.0% | -$54.52 | $237 | Time Exit (21d to expiry) |
| 2026-06-16 | DAL | Long Put | $3.70 | $1.63 | -62.3% | -$230.50 | $370 | Stop Loss (-50%) |
| 2026-06-16 | UAL | Long Call | $5.40 | $15.63 | +183.2% | +$989.30 | $540 | Take Profit (100%) |
| 2026-06-16 | GS | Long Put | $38.35 | $71.92 | +84.9% | +$3,255.70 | $3,835 | Time Exit (21d to expiry) |
| 2026-06-17 | SPY | Bear Call | $0.45 | $0.15 | +16.4% | +$7.40 | $55 | Take Profit (50% of credit) |
| 2026-06-17 | QQQ | Bear Call | $0.46 | $0.00 | +50.3% | +$22.90 | $54 | Take Profit (50% of credit) |
| 2026-06-17 | INTC | Bull Put | $1.97 | $0.96 | +39.9% | +$78.90 | $303 | Take Profit (50% of credit) |
| 2026-06-17 | IWM | Bear Call | $0.42 | $0.09 | +23.9% | +$9.90 | $58 | Take Profit (50% of credit) |
| 2026-06-17 | AAPL | Bear Call | $0.95 | $0.81 | -9.1% | -$8.60 | $155 | Time Exit (12d to expiry) |
| 2026-06-17 | QQQ | Iron Condor | $18.33 | $11.38 | +35.5% | +$649.80 | $3,167 | Time Exit (21d to expiry) |
| 2026-06-17 | SPY | Iron Condor | $11.67 | $6.12 | +43.7% | +$509.30 | $2,334 | Time Exit (21d to expiry) |
| 2026-06-17 | AMZN | Iron Condor | $6.13 | $4.78 | +14.7% | +$90.30 | $1,386 | Time Exit (17d to expiry) |
| 2026-06-17 | AAPL | Iron Condor | $6.20 | $8.29 | -41.0% | -$254.20 | $1,380 | Time Exit (17d to expiry) |
| 2026-06-17 | TLT | Iron Condor | $0.58 | $0.60 | -81.4% | -$47.20 | $142 | Time Exit (17d to expiry) |
| 2026-06-17 | NVDA | Long Call | $4.25 | $6.17 | +38.9% | +$165.20 | $425 | Time Exit (12d to expiry) |
| 2026-06-17 | SPY | Long Call | $4.39 | $4.12 | -12.4% | -$54.64 | $439 | Time Exit (9d to expiry) |
| 2026-06-17 | AAPL | Long Call | $4.20 | $4.55 | +2.0% | +$8.50 | $420 | Time Exit (12d to expiry) |
| 2026-06-17 | MSFT | Long Call | $4.80 | $2.37 | -56.9% | -$273.10 | $480 | Stop Loss (-50%) |
| 2026-06-18 | TLT | Long Call | $0.85 | $0.11 | -100.0% | -$85.00 | $85 | Stop Loss (-50%) |
| 2026-06-18 | UAL | Long Call | $6.50 | $15.00 | +124.6% | +$809.70 | $650 | Take Profit (100%) |
| 2026-06-18 | JPM | Long Call | $7.20 | $7.42 | -3.1% | -$22.50 | $720 | Time Exit (17d to expiry) |
| 2026-06-18 | C | Long Call | $4.65 | $1.93 | -64.8% | -$301.20 | $465 | Stop Loss (-50%) |
| 2026-06-18 | XLF | Long Call | $0.92 | $2.12 | +118.2% | +$108.70 | $92 | Take Profit (100%) |
| 2026-06-18 | INTC | Long Call | $7.50 | $10.75 | +37.2% | +$278.70 | $750 | Time Exit (18d to expiry) |
| 2026-06-18 | AMD | Long Call | $32.70 | $21.50 | -37.3% | -$1,221.30 | $3,270 | Time Exit (21d to expiry) |
| 2026-06-18 | UAL | Long Call | $5.45 | $15.63 | +180.6% | +$984.00 | $545 | Take Profit (100%) |
| 2026-06-18 | RTX | Long Call | $5.15 | $2.53 | -57.1% | -$294.20 | $515 | Stop Loss (-50%) |
| 2026-06-18 | WFC | Long Call | $1.70 | $1.51 | -17.9% | -$30.50 | $170 | Time Exit (10d to expiry) |
| 2026-06-22 | QQQ | Bull Put | $0.48 | $0.98 | -106.2% | -$51.50 | $52 | Stop Loss (100% of credit) |
| 2026-06-22 | SPY | Bear Call | $0.90 | $0.35 | +36.0% | +$32.40 | $110 | Take Profit (50% of credit) |
| 2026-06-22 | IWM | Bear Call | $0.44 | $0.00 | +82.9% | +$36.90 | $56 | Take Profit (50% of credit) |
| 2026-06-22 | INTC | Bull Put | $0.97 | $0.40 | +35.8% | +$34.90 | $103 | Take Profit (50% of credit) |
| 2026-06-22 | ORCL | Bull Put | $2.18 | $1.05 | +41.3% | +$89.90 | $282 | Take Profit (50% of credit) |
| 2026-06-23 | CSCO | Long Call | $3.15 | $0.90 | -77.8% | -$245.20 | $315 | Time Exit (21d to expiry) |
| 2026-06-23 | HD | Long Call | $7.00 | $14.15 | +96.0% | +$671.70 | $700 | Take Profit (100%) |
| 2026-06-23 | AMD | Long Call | $31.10 | $15.00 | -55.0% | -$1,711.30 | $3,110 | Stop Loss (-50%) |
| 2026-06-23 | UPS | Long Call | $2.30 | $2.45 | -0.0% | -$0.10 | $230 | Time Exit (21d to expiry) |
| 2026-06-23 | AVGO | Long Call | $13.30 | $4.50 | -72.3% | -$961.10 | $1,330 | Time Exit (21d to expiry) |
| 2026-06-24 | AAPL | Iron Condor | $8.70 | $4.60 | +39.5% | +$343.42 | $1,630 | Time Exit (21d to expiry) |
| 2026-06-24 | GLD | Iron Condor | $9.01 | $4.17 | +48.7% | +$438.80 | $1,599 | Take Profit (50% of credit) |
| 2026-06-24 | XOM | Iron Condor | $4.23 | $6.09 | -56.4% | -$238.69 | $577 | Time Exit (21d to expiry) |
| 2026-06-24 | QQQ | Iron Condor | $23.74 | $11.93 | +45.1% | +$1,069.78 | $3,626 | Time Exit (21d to expiry) |
| 2026-06-24 | TLT | Iron Condor | $0.96 | $2.02 | -157.5% | -$151.20 | $204 | Stop Loss (100% of credit) |
| 2026-06-24 | SPY | Bear Call | $0.49 | $1.81 | -104.1% | -$51.00 | $51 | Stop Loss (100% of credit) |
| 2026-06-24 | QQQ | Bear Call | $0.55 | $1.13 | -81.8% | -$45.00 | $45 | Stop Loss (100% of credit) |
| 2026-06-24 | IWM | Bear Call | $0.56 | $1.13 | -80.2% | -$44.50 | $44 | Stop Loss (100% of credit) |
| 2026-06-24 | EEM | Bear Call | $0.26 | $0.92 | -88.7% | -$23.50 | $24 | Stop Loss (100% of credit) |
| 2026-06-24 | AMZN | Bull Put | $1.77 | $0.88 | +37.7% | +$66.90 | $323 | Take Profit (50% of credit) |
| 2026-06-24 | MU | Short Put | $44.35 | $19.95 | +52.7% | +$2,338.70 | $83,565 | Take Profit (50% @ 22d) |
| 2026-06-24 | IWM | Short Put | $2.98 | $0.71 | +69.7% | +$207.82 | $28,202 | Take Profit (35% @ 10d) |
| 2026-06-24 | DIA | Short Put | $3.30 | $0.36 | +82.7% | +$272.90 | $50,170 | Take Profit (35% @ 10d) |
| 2026-06-24 | UNH | Short Put | $4.85 | $2.12 | +50.0% | +$242.60 | $36,515 | Take Profit (50% @ 21d) |
| 2026-06-24 | BAC | Short Put | $0.65 | $0.13 | +62.6% | +$40.70 | $5,435 | Take Profit (35% @ 10d) |
| 2026-06-25 | QQQ | Bull Put | $0.68 | $0.12 | +48.7% | +$32.90 | $32 | Take Profit (50% of credit) |
| 2026-06-25 | SPY | Bear Call | $1.01 | $0.15 | +62.8% | +$63.40 | $99 | Take Profit (50% of credit) |
| 2026-06-25 | IWM | Bear Call | $0.57 | $0.16 | +32.9% | +$18.90 | $43 | Take Profit (50% of credit) |
| 2026-06-25 | META | Bull Put | $2.43 | $5.73 | -106.2% | -$257.50 | $257 | Stop Loss (100% of credit) |
| 2026-06-25 | INTC | Bull Put | $0.55 | $1.37 | -81.8% | -$45.00 | $45 | Stop Loss (100% of credit) |
| 2026-06-25 | DIA | Short Put | $2.73 | $0.43 | +77.8% | +$212.32 | $50,727 | Take Profit (35% @ 10d) |
| 2026-06-25 | C | Short Put | $2.20 | $2.95 | -40.7% | -$89.50 | $13,780 | Time Exit (10d to expiry) |
| 2026-06-25 | SPY | Short Put | $6.50 | $0.68 | +83.3% | +$541.70 | $71,350 | Take Profit (35% @ 10d) |
| 2026-06-25 | META | Short Put | $9.70 | $0.58 | +87.9% | +$852.50 | $52,030 | Take Profit (35% @ 10d) |
| 2026-06-25 | COIN | Short Put | $3.50 | $0.49 | +79.6% | +$278.70 | $12,650 | Take Profit (35% @ 10d) |
| 2026-06-25 | AAPL | Iron Condor | $8.76 | $17.97 | -110.3% | -$966.20 | $1,624 | Stop Loss (100% of credit) |
| 2026-06-25 | SPY | Iron Condor | $10.78 | $8.51 | +16.9% | +$181.80 | $2,022 | Time Exit (21d to expiry) |
| 2026-06-25 | NVDA | Iron Condor | $5.74 | $5.41 | -2.2% | -$12.70 | $926 | Time Exit (21d to expiry) |
| 2026-06-25 | QQQ | Iron Condor | $22.36 | $14.06 | +32.3% | +$722.92 | $3,764 | Time Exit (21d to expiry) |
| 2026-06-25 | TLT | Iron Condor | $0.98 | $1.87 | -134.6% | -$131.94 | $302 | Time Exit (21d to expiry) |
| 2026-06-26 | TLT | Long Call | $0.82 | $0.11 | -100.0% | -$82.00 | $82 | Stop Loss (-50%) |
| 2026-06-26 | BAC | Long Call | $1.43 | $2.00 | +32.0% | +$45.70 | $143 | Time Exit (21d to expiry) |
| 2026-06-26 | CAT | Long Call | $48.10 | $8.93 | -83.5% | -$4,018.30 | $4,810 | Stop Loss (-50%) |
| 2026-06-26 | BA | Long Call | $8.05 | $7.09 | -18.1% | -$145.60 | $805 | Time Exit (21d to expiry) |
| 2026-06-26 | TGT | Long Call | $4.00 | $0.30 | -98.8% | -$395.30 | $400 | Stop Loss (-50%) |
| 2026-06-26 | SPY | Bear Call | $0.49 | $0.74 | -99.2% | -$48.10 | $51 | Time Exit (10d to expiry) |
| 2026-06-26 | TSLA | Bull Put | $2.40 | $0.96 | +50.6% | +$121.40 | $260 | Take Profit (50% of credit) |
| 2026-06-26 | QQQ | Bear Call | $0.47 | $0.22 | +5.1% | +$2.40 | $53 | Take Profit (50% of credit) |
| 2026-06-26 | AVGO | Bull Put | $2.30 | $1.13 | +41.0% | +$94.40 | $270 | Take Profit (50% of credit) |
| 2026-06-26 | IWM | Bear Call | $0.46 | $0.13 | +21.8% | +$9.90 | $54 | Take Profit (50% of credit) |
| 2026-06-26 | UNH | Short Put | $9.45 | $8.05 | +8.7% | +$82.00 | $40,055 | Time Exit (10d to expiry) |
| 2026-06-26 | NFLX | Short Put | $1.57 | $0.88 | +36.8% | +$57.70 | $6,843 | Take Profit (35% @ 10d) |
| 2026-06-26 | IWM | Short Put | $4.75 | $2.65 | +37.9% | +$180.20 | $29,025 | Take Profit (35% @ 10d) |
| 2026-06-26 | SCHW | Short Put | $1.99 | $0.05 | +90.8% | +$180.76 | $8,551 | Take Profit (35% @ 10d) |
| 2026-06-26 | DIA | Short Put | $3.75 | $0.14 | +89.9% | +$337.20 | $51,125 | Take Profit (25% @ 3d) |
| 2026-06-26 | SPY | Iron Condor | $11.65 | $5.71 | +47.1% | +$548.80 | $2,335 | Take Profit (50% of credit) |
| 2026-06-26 | QQQ | Iron Condor | $22.70 | $11.48 | +44.7% | +$1,013.90 | $3,230 | Time Exit (21d to expiry) |
| 2026-06-26 | AAPL | Iron Condor | $8.38 | $16.90 | -107.2% | -$897.70 | $1,162 | Stop Loss (100% of credit) |
| 2026-06-26 | IWM | Iron Condor | $8.91 | $4.44 | +45.1% | +$401.80 | $2,109 | Take Profit (50% of credit) |
| 2026-06-26 | GLD | Iron Condor | $5.57 | $2.73 | +42.9% | +$238.80 | $1,943 | Take Profit (50% of credit) |
| 2026-07-07 | ABBV | Long Call | $8.30 | $3.75 | -61.0% | -$506.10 | $830 | Stop Loss (-50%) |
| 2026-07-07 | BA | Long Call | $8.80 | $4.10 | -59.6% | -$524.10 | $880 | Stop Loss (-50%) |
| 2026-07-07 | META | Long Call | $29.75 | $71.02 | +135.3% | +$4,025.70 | $2,975 | Take Profit (100%) |
| 2026-07-07 | C | Long Call | $5.10 | $1.41 | -78.6% | -$400.90 | $510 | Stop Loss (-50%) |
| 2026-07-07 | V | Long Call | $9.10 | $10.95 | +11.3% | +$103.10 | $910 | Time Exit (21d to expiry) |
| 2026-07-07 | SPY | Bear Call | $0.48 | $0.18 | +14.5% | +$6.90 | $52 | Take Profit (50% of credit) |
| 2026-07-07 | IWM | Bear Call | $0.42 | $0.14 | +13.9% | +$5.90 | $58 | Take Profit (50% of credit) |
| 2026-07-07 | QQQ | Bear Call | $0.48 | $0.15 | +20.8% | +$9.90 | $52 | Take Profit (50% of credit) |
| 2026-07-07 | MU | Bull Put | $2.50 | $0.00 | +100.0% | +$250.00 | $250 | Take Profit (50% of credit) |
| 2026-07-07 | C | Bear Call | $0.44 | $0.18 | +7.7% | +$3.40 | $56 | Take Profit (50% of credit) |
| 2026-07-07 | AAPL | Short Put | $5.80 | $4.93 | +8.8% | +$50.90 | $29,920 | Time Exit (21d to expiry) |
| 2026-07-07 | NVDA | Short Put | $4.80 | $2.23 | +47.3% | +$226.90 | $18,520 | Take Profit (50% @ 21d) |
| 2026-07-07 | AMZN | Short Put | $8.15 | $7.90 | -3.1% | -$25.20 | $23,185 | Time Exit (21d to expiry) |
| 2026-07-07 | TSLA | Short Put | $7.75 | $5.94 | +17.2% | +$133.20 | $37,225 | Time Exit (14d to expiry) |
| 2026-07-07 | SLV | Short Put | $0.90 | $2.13 | -149.2% | -$134.30 | $5,110 | Stop Loss (2.0× premium) |
| 2026-07-07 | AAPL | Iron Condor | $7.76 | $6.46 | +8.5% | +$66.24 | $1,224 | Time Exit (21d to expiry) |
| 2026-07-07 | IWM | Iron Condor | $5.09 | $3.56 | +19.2% | +$97.73 | $991 | Time Exit (21d to expiry) |
| 2026-07-07 | TLT | Iron Condor | $0.69 | $0.96 | -100.1% | -$69.07 | $131 | Time Exit (21d to expiry) |
| 2026-07-07 | GLD | Iron Condor | $7.56 | $5.97 | +12.8% | +$96.81 | $1,244 | Time Exit (21d to expiry) |
| 2026-07-07 | QQQ | Iron Condor | $18.07 | $15.42 | +9.5% | +$170.79 | $3,193 | Time Exit (21d to expiry) |
| 2026-07-08 | ABBV | Long Call | $8.30 | $3.75 | -61.0% | -$506.10 | $830 | Stop Loss (-50%) |
| 2026-07-08 | AAPL | Long Call | $7.65 | $16.50 | +109.5% | +$837.80 | $765 | Take Profit (100%) |
| 2026-07-08 | XLF | Long Call | $1.01 | $1.56 | +41.6% | +$41.97 | $101 | Time Exit (21d to expiry) |
| 2026-07-08 | WFC | Long Call | $2.71 | $1.29 | -61.4% | -$166.39 | $271 | Stop Loss (-50%) |
| 2026-07-08 | JPM | Long Call | $9.35 | $19.05 | +97.6% | +$912.60 | $935 | Take Profit (100%) |
| 2026-07-08 | TGT | Long Call | $5.00 | $10.04 | +91.8% | +$459.00 | $500 | Take Profit (100%) |
| 2026-07-08 | SPY | Bull Put | $0.50 | $0.00 | +67.1% | +$33.90 | $50 | Take Profit (50% of credit) |
| 2026-07-08 | QQQ | Bull Put | $0.50 | $0.03 | +48.3% | +$23.90 | $50 | Take Profit (50% of credit) |
| 2026-07-08 | XOM | Bear Call | $0.42 | $0.19 | +1.0% | +$0.40 | $58 | Take Profit (50% of credit) |
| 2026-07-08 | MU | Bull Put | $5.43 | $1.84 | +61.9% | +$335.90 | $457 | Take Profit (50% of credit) |
| 2026-07-08 | AMD | Bull Put | $2.45 | $0.00 | +100.0% | +$245.00 | $255 | Take Profit (50% of credit) |
| 2026-07-08 | TSLA | Short Put | $11.45 | $7.10 | +31.9% | +$365.00 | $36,855 | Take Profit (35% @ 15d) |
| 2026-07-08 | UNH | Short Put | $10.80 | $10.22 | -0.8% | -$8.10 | $40,420 | Time Exit (13d to expiry) |
| 2026-07-08 | CHTR | Short Put | $8.60 | $9.00 | -10.8% | -$92.90 | $12,140 | Stop Loss (strike breached) |
| 2026-07-08 | NFLX | Short Put | $2.65 | $3.75 | -48.0% | -$127.20 | $7,135 | Stop Loss (strike breached) |
| 2026-07-08 | WMT | Short Put | $1.23 | $0.75 | +29.8% | +$36.70 | $10,777 | Take Profit (35% @ 20d) |
| 2026-07-08 | SPY | Iron Condor | $11.34 | $6.11 | +39.6% | +$448.49 | $2,366 | Time Exit (21d to expiry) |
| 2026-07-08 | QQQ | Iron Condor | $18.79 | $11.66 | +32.8% | +$617.11 | $3,120 | Time Exit (21d to expiry) |
| 2026-07-08 | AAPL | Iron Condor | $8.51 | $6.81 | +12.2% | +$103.99 | $1,150 | Time Exit (21d to expiry) |
| 2026-07-08 | WMT | Iron Condor | $3.90 | $3.25 | +3.4% | +$13.30 | $610 | Time Exit (21d to expiry) |
| 2026-07-08 | IWM | Iron Condor | $6.04 | $2.95 | +41.5% | +$250.39 | $1,396 | Take Profit (50% of credit) |
| 2026-07-09 | GS | Long Call | $33.35 | $12.50 | -65.6% | -$2,186.30 | $3,335 | Stop Loss (-50%) |
| 2026-07-09 | NVDA | Long Call | $8.10 | $3.94 | -57.5% | -$465.90 | $810 | Stop Loss (-50%) |
| 2026-07-09 | EEM | Long Call | $2.35 | $1.15 | -57.6% | -$135.40 | $235 | Stop Loss (-50%) |
| 2026-07-09 | AMZN | Long Call | $10.40 | $4.72 | -60.7% | -$631.70 | $1,040 | Stop Loss (-50%) |
| 2026-07-09 | XLI | Long Call | $4.45 | $1.97 | -62.0% | -$276.00 | $445 | Stop Loss (-50%) |
| 2026-07-09 | QQQ | Bear Call | $0.46 | $0.47 | -51.3% | -$23.60 | $54 | Time Exit (5d to expiry) |
| 2026-07-09 | SPY | Bear Call | $0.46 | $0.55 | -68.7% | -$31.60 | $54 | Time Exit (5d to expiry) |
| 2026-07-09 | IWM | Bear Call | $0.41 | $0.20 | -2.7% | -$1.10 | $59 | Take Profit (50% of credit) |
| 2026-07-09 | META | Bull Put | $2.18 | $0.19 | +80.9% | +$175.90 | $282 | Take Profit (50% of credit) |
| 2026-07-09 | DIA | Bear Call | $0.43 | $0.43 | -52.6% | -$22.60 | $57 | Time Exit (12d to expiry) |
| 2026-07-09 | UNH | Short Put | $12.75 | $14.00 | -15.9% | -$202.80 | $40,725 | Time Exit (19d to expiry) |
| 2026-07-09 | GOOGL | Short Put | $9.30 | $6.78 | +21.0% | +$194.90 | $33,570 | Time Exit (12d to expiry) |
| 2026-07-09 | NVDA | Short Put | $3.65 | $2.06 | +37.2% | +$135.80 | $19,135 | Take Profit (35% @ 14d) |
| 2026-07-09 | BAC | Short Put | $0.85 | $0.58 | +18.5% | +$15.70 | $5,715 | Time Exit (12d to expiry) |
| 2026-07-09 | ABBV | Short Put | $5.95 | $7.45 | -31.4% | -$187.00 | $23,905 | Time Exit (19d to expiry) |
| 2026-07-09 | DIS | Iron Condor | $1.83 | $1.26 | +6.3% | +$11.51 | $317 | Time Exit (21d to expiry) |
| 2026-07-09 | TLT | Iron Condor | $0.66 | $0.96 | -109.1% | -$71.98 | $134 | Time Exit (21d to expiry) |
| 2026-07-09 | QQQ | Iron Condor | $17.14 | $15.56 | +3.9% | +$66.58 | $2,786 | Time Exit (21d to expiry) |
| 2026-07-09 | AAPL | Iron Condor | $8.45 | $8.13 | -3.9% | -$33.35 | $1,155 | Time Exit (21d to expiry) |
| 2026-07-09 | IWM | Iron Condor | $5.08 | $3.56 | +19.0% | +$96.76 | $992 | Time Exit (21d to expiry) |
| 2026-07-09 | QCOM | Long Put | $15.30 | $26.05 | +64.2% | +$981.90 | $1,530 | Time Exit (21d to expiry) |
| 2026-07-09 | WMT | Long Put | $2.12 | $1.03 | -58.0% | -$123.02 | $212 | Stop Loss (-50%) |
| 2026-07-09 | IWM | Long Put | $2.67 | $2.83 | -0.5% | -$1.32 | $267 | Time Exit (8d to expiry) |
| 2026-07-09 | CRM | Long Put | $5.75 | $5.19 | -16.0% | -$91.80 | $575 | Time Exit (19d to expiry) |
| 2026-07-10 | AAPL | Long Call | $8.70 | $18.45 | +105.9% | +$921.50 | $870 | Take Profit (100%) |
| 2026-07-10 | TGT | Long Call | $5.60 | $8.15 | +36.5% | +$204.60 | $560 | Time Exit (21d to expiry) |
| 2026-07-10 | AVGO | Long Call | $19.65 | $9.15 | -58.6% | -$1,151.30 | $1,965 | Stop Loss (-50%) |
| 2026-07-10 | TSLA | Long Call | $19.90 | $9.65 | -56.6% | -$1,126.30 | $1,990 | Stop Loss (-50%) |
| 2026-07-10 | IWM | Long Call | $5.91 | $2.64 | -61.5% | -$363.76 | $591 | Stop Loss (-50%) |
| 2026-07-10 | SPY | Bear Call | $0.86 | $0.69 | -5.9% | -$5.10 | $114 | Time Exit (11d to expiry) |
| 2026-07-10 | INTC | Bull Put | $0.90 | $1.00 | -36.2% | -$32.60 | $110 | Time Exit (11d to expiry) |
| 2026-07-10 | RIVN | Bull Put | $0.24 | $0.10 | -38.7% | -$9.10 | $26 | Take Profit (50% of credit) |
| 2026-07-10 | QQQ | Bull Put | $0.40 | $0.52 | -84.2% | -$34.10 | $60 | Time Exit (7d to expiry) |
| 2026-07-10 | DIA | Bear Call | $0.43 | $0.44 | -54.9% | -$23.60 | $57 | Time Exit (11d to expiry) |
| 2026-07-10 | NVDA | Bear Call | $0.97 | $0.84 | -9.3% | -$9.10 | $153 | Time Exit (11d to expiry) |
| 2026-07-10 | SPY | Bear Call | $0.43 | $0.30 | -20.9% | -$9.10 | $57 | Time Exit (11d to expiry) |
| 2026-07-10 | QCOM | Bull Put | $1.22 | $0.86 | +11.3% | +$13.90 | $128 | Time Exit (18d to expiry) |
| 2026-07-10 | SPY | Bear Call | $0.49 | $0.46 | -40.0% | -$19.60 | $51 | Time Exit (7d to expiry) |
| 2026-07-10 | INTC | Bull Put | $0.45 | $0.50 | -61.3% | -$27.60 | $55 | Time Exit (11d to expiry) |
| 2026-07-10 | LRCX | Bull Put | $4.82 | $0.00 | +100.0% | +$482.50 | $518 | Take Profit (50% of credit) |
| 2026-07-10 | WFC | Short Put | $1.55 | $1.90 | -29.9% | -$46.30 | $8,345 | Time Exit (11d to expiry) |
| 2026-07-10 | GOOGL | Short Put | $9.60 | $9.46 | -4.7% | -$44.90 | $34,040 | Time Exit (11d to expiry) |
| 2026-07-10 | NVDA | Short Put | $4.00 | $4.30 | -13.8% | -$55.30 | $20,100 | Time Exit (11d to expiry) |
| 2026-07-10 | JPM | Short Put | $4.60 | $5.75 | -31.3% | -$143.90 | $32,540 | Time Exit (11d to expiry) |
| 2026-07-10 | BAC | Short Put | $1.07 | $1.37 | -38.6% | -$41.30 | $5,793 | Time Exit (18d to expiry) |
| 2026-07-10 | C | Short Put | $2.77 | $3.00 | -14.8% | -$40.92 | $13,523 | Time Exit (11d to expiry) |
| 2026-07-10 | TLT | Iron Condor | $0.65 | $0.96 | -113.9% | -$73.44 | $136 | Time Exit (21d to expiry) |
| 2026-07-10 | AAPL | Iron Condor | $7.01 | $7.84 | -20.5% | -$144.03 | $1,299 | Time Exit (21d to expiry) |
| 2026-07-10 | WMT | Iron Condor | $3.73 | $3.25 | -1.0% | -$3.68 | $628 | Time Exit (21d to expiry) |
| 2026-07-10 | SPY | Iron Condor | $9.60 | $7.42 | +15.5% | +$148.71 | $1,840 | Time Exit (21d to expiry) |
| 2026-07-10 | NVDA | Iron Condor | $6.11 | $4.90 | +12.3% | +$75.30 | $890 | Time Exit (19d to expiry) |
| 2026-07-10 | TSLA | Short Put | $12.65 | $16.81 | -39.0% | -$493.20 | $38,735 | Time Exit (11d to expiry) |
| 2026-07-10 | SPY | Short Put | $4.08 | $4.87 | -25.7% | -$104.78 | $74,592 | Time Exit (11d to expiry) |
| 2026-07-10 | LCID | Short Put | $0.33 | $0.30 | -25.2% | -$8.30 | $517 | Time Exit (11d to expiry) |
| 2026-07-13 | AAPL | Long Call | $8.10 | $16.30 | +95.1% | +$770.10 | $810 | Take Profit (100%) |
| 2026-07-13 | JPM | Long Call | $8.70 | $17.75 | +97.9% | +$851.50 | $870 | Take Profit (100%) |
| 2026-07-13 | V | Long Call | $8.05 | $8.10 | -8.4% | -$67.45 | $805 | Time Exit (21d to expiry) |
| 2026-07-13 | XLE | Long Call | $1.28 | $2.65 | +98.2% | +$125.70 | $128 | Take Profit (100%) |
| 2026-07-13 | IWM | Long Call | $5.47 | $2.64 | -58.0% | -$317.12 | $547 | Stop Loss (-50%) |
| 2026-07-13 | SPY | Bear Call | $0.53 | $0.01 | +55.0% | +$28.90 | $47 | Take Profit (50% of credit) |
| 2026-07-13 | QQQ | Bear Call | $1.53 | $0.34 | +63.1% | +$96.90 | $147 | Take Profit (50% of credit) |
| 2026-07-13 | MU | Bull Put | $3.00 | $0.00 | +100.0% | +$300.00 | $200 | Take Profit (50% of credit) |
| 2026-07-13 | GOOGL | Bull Put | $1.15 | $0.06 | +75.1% | +$86.40 | $135 | Take Profit (50% of credit) |
| 2026-07-13 | INTC | Bull Put | $0.47 | $0.00 | +52.4% | +$24.90 | $53 | Take Profit (50% of credit) |
| 2026-07-13 | UNH | Short Put | $12.30 | $4.08 | +60.7% | +$746.90 | $40,770 | Take Profit (35% @ 15d) |
| 2026-07-13 | PYPL | Short Put | $1.60 | $0.15 | +83.6% | +$133.70 | $4,340 | Take Profit (50% @ 23d) |
| 2026-07-13 | TSLA | Short Put | $9.80 | $10.37 | -11.9% | -$117.10 | $37,520 | Time Exit (8d to expiry) |
| 2026-07-13 | AAPL | Short Put | $5.45 | $2.70 | +44.2% | +$241.00 | $30,455 | Take Profit (35% @ 16d) |
| 2026-07-13 | WMT | Short Put | $1.88 | $1.50 | +13.5% | +$25.42 | $11,112 | Time Exit (15d to expiry) |
| 2026-07-13 | IWM | Iron Condor | $4.81 | $2.66 | +33.4% | +$161.05 | $1,018 | Time Exit (21d to expiry) |
| 2026-07-13 | AAPL | Iron Condor | $6.75 | $7.84 | -25.2% | -$169.73 | $1,326 | Time Exit (21d to expiry) |
| 2026-07-13 | QQQ | Iron Condor | $14.73 | $13.76 | +0.9% | +$13.30 | $2,526 | Time Exit (21d to expiry) |
| 2026-07-13 | SPY | Iron Condor | $12.37 | $11.19 | +6.3% | +$78.00 | $2,763 | Time Exit (21d to expiry) |
| 2026-07-13 | WMT | Iron Condor | $3.56 | $3.25 | -5.5% | -$19.68 | $644 | Time Exit (21d to expiry) |
| 2026-07-13 | WMT | Long Put | $1.98 | $1.98 | -6.7% | -$13.18 | $198 | Time Exit (21d to expiry) |
| 2026-07-13 | JPM | Long Call | $7.50 | $10.10 | +28.5% | +$213.70 | $750 | Time Exit (21d to expiry) |
| 2026-07-13 | BAC | Long Call | $1.44 | $2.39 | +58.1% | +$83.70 | $144 | Time Exit (21d to expiry) |
| 2026-07-13 | TSLA | Long Put | $18.40 | $21.72 | +12.5% | +$230.70 | $1,840 | Time Exit (21d to expiry) |
| 2026-07-13 | WMT | Long Put | $1.64 | $0.86 | -54.5% | -$89.30 | $164 | Time Exit (21d to expiry) |
| 2026-07-13 | DIA | Long Put | $5.20 | $4.45 | -20.7% | -$107.50 | $520 | Time Exit (21d to expiry) |
| 2026-07-14 | WFC | Long Call | $2.10 | $0.80 | -70.9% | -$148.90 | $210 | Stop Loss (-50%) |
| 2026-07-14 | IWM | Long Call | $5.18 | $2.00 | -70.4% | -$364.62 | $518 | Stop Loss (-50%) |
| 2026-07-14 | V | Long Call | $9.65 | $4.80 | -56.4% | -$544.20 | $965 | Stop Loss (-50%) |
| 2026-07-14 | AMGN | Long Call | $11.35 | $22.00 | +87.7% | +$995.60 | $1,135 | Take Profit (Δ 0.83 deep ITM) |
| 2026-07-14 | GS | Long Call | $40.05 | $19.55 | -53.7% | -$2,151.30 | $4,005 | Stop Loss (-50%) |
| 2026-07-14 | QQQ | Bear Call | $0.47 | $0.19 | +11.5% | +$5.40 | $53 | Take Profit (50% of credit) |
| 2026-07-14 | SPY | Bear Call | $0.46 | $0.22 | +2.0% | +$0.90 | $54 | Take Profit (50% of credit) |
| 2026-07-14 | IWM | Bear Call | $0.21 | $0.45 | -132.6% | -$28.50 | $29 | Stop Loss (100% of credit) |
| 2026-07-14 | WMT | Bull Put | $0.45 | $0.14 | +19.6% | +$8.90 | $55 | Take Profit (50% of credit) |
| 2026-07-14 | UBER | Bull Put | $0.45 | $0.20 | +5.3% | +$2.40 | $55 | Take Profit (50% of credit) |
| 2026-07-14 | AAPL | Short Put | $5.90 | $2.70 | +48.0% | +$283.30 | $30,410 | Take Profit (35% @ 16d) |
| 2026-07-14 | UBER | Short Put | $1.18 | $0.67 | +33.6% | +$39.70 | $6,782 | Take Profit (35% @ 16d) |
| 2026-07-14 | LCID | Short Put | $2.89 | $2.75 | -1.7% | -$4.87 | $211 | Stop Loss (strike breached) |
| 2026-07-14 | NVDA | Short Put | $4.30 | $6.03 | -46.5% | -$200.10 | $20,070 | Time Exit (14d to expiry) |
| 2026-07-14 | AMZN | Short Put | $3.75 | $2.13 | +36.9% | +$138.20 | $22,625 | Take Profit (35% @ 16d) |
| 2026-07-14 | AMZN | Short Put | $5.30 | $2.89 | +39.2% | +$207.90 | $22,970 | Take Profit (35% @ 16d) |
| 2026-07-14 | IWM | Iron Condor | $3.98 | $2.22 | +31.2% | +$124.06 | $602 | Time Exit (21d to expiry) |
| 2026-07-14 | AAPL | Iron Condor | $8.09 | $8.13 | -8.4% | -$68.27 | $1,191 | Time Exit (21d to expiry) |
| 2026-07-14 | BAC | Iron Condor | $0.98 | $0.93 | -38.7% | -$37.94 | $152 | Time Exit (21d to expiry) |
| 2026-07-14 | DIS | Iron Condor | $1.76 | $1.26 | +2.7% | +$4.72 | $324 | Time Exit (21d to expiry) |
| 2026-07-14 | SPY | Iron Condor | $9.63 | $5.86 | +32.0% | +$308.60 | $1,937 | Time Exit (21d to expiry) |
| 2026-07-14 | IWM | Iron Condor | $4.52 | $2.66 | +29.4% | +$132.92 | $1,048 | Time Exit (21d to expiry) |
| 2026-07-14 | QQQ | Iron Condor | $14.91 | $21.21 | -47.9% | -$714.25 | $3,008 | Time Exit (21d to expiry) |
| 2026-07-15 | AAPL | Long Call | $8.70 | $0.54 | -100.0% | -$870.00 | $870 | Time Exit (21d to expiry) |
| 2026-07-15 | VLO | Long Call | $11.90 | $25.89 | +111.5% | +$1,326.30 | $1,190 | Take Profit (100%) |
| 2026-07-15 | NVDA | Long Call | $7.30 | $2.65 | -69.9% | -$510.10 | $730 | Stop Loss (-50%) |
| 2026-07-15 | PFE | Long Call | $0.53 | $0.69 | +8.3% | +$4.41 | $53 | Time Exit (21d to expiry) |
| 2026-07-15 | QQQ | Long Call | $16.11 | $7.25 | -61.1% | -$983.96 | $1,611 | Stop Loss (-50%) |
| 2026-07-15 | AVGO | Long Call | $12.80 | $7.10 | -50.6% | -$648.10 | $1,280 | Time Exit (11d to expiry) |
| 2026-07-15 | WFC | Long Call | $1.61 | $1.00 | -44.9% | -$72.30 | $161 | Time Exit (11d to expiry) |
| 2026-07-15 | AAPL | Long Call | $6.90 | $6.30 | -14.9% | -$102.70 | $690 | Time Exit (18d to expiry) |
| 2026-07-15 | UNH | Long Call | $12.70 | $5.10 | -65.9% | -$837.50 | $1,270 | Time Exit (11d to expiry) |
| 2026-07-15 | WMT | Long Call | $2.32 | $1.32 | -49.7% | -$115.22 | $232 | Time Exit (18d to expiry) |
| 2026-07-15 | QQQ | Bear Call | $1.01 | $0.44 | +34.1% | +$34.40 | $99 | Take Profit (50% of credit) |
| 2026-07-15 | SPY | Bear Call | $0.46 | $0.09 | +32.0% | +$14.90 | $54 | Take Profit (50% of credit) |
| 2026-07-15 | ORCL | Bull Put | $0.95 | $2.16 | -110.5% | -$105.00 | $105 | Stop Loss (100% of credit) |
| 2026-07-15 | DIA | Bear Call | $0.45 | $0.90 | -122.2% | -$55.00 | $55 | Stop Loss (100% of credit) |
| 2026-07-15 | GLD | Bear Call | $0.40 | $2.45 | -150.0% | -$60.00 | $60 | Stop Loss (100% of credit) |
| 2026-07-15 | AAPL | Short Put | $3.95 | $4.00 | -7.6% | -$30.00 | $31,105 | Time Exit (11d to expiry) |
| 2026-07-15 | AMZN | Short Put | $7.80 | $10.37 | -39.1% | -$305.10 | $24,220 | Stop Loss (strike breached) |
| 2026-07-15 | SPY | Short Put | $5.74 | $7.57 | -38.1% | -$218.74 | $73,926 | Stop Loss (strike breached) |
| 2026-07-15 | TSLA | Short Put | $12.00 | $20.85 | -79.9% | -$958.30 | $36,800 | Time Exit (11d to expiry) |
| 2026-07-15 | META | Short Put | $26.20 | $39.50 | -54.6% | -$1,431.30 | $63,380 | Stop Loss (strike breached) |
| 2026-07-15 | AAPL | Iron Condor | $7.09 | $9.71 | -45.5% | -$322.78 | $1,290 | Time Exit (21d to expiry) |
| 2026-07-15 | IWM | Iron Condor | $3.92 | $2.22 | +30.2% | +$118.24 | $608 | Time Exit (21d to expiry) |
| 2026-07-15 | QQQ | Iron Condor | $15.81 | $15.05 | -0.7% | -$10.94 | $2,918 | Time Exit (21d to expiry) |
| 2026-07-15 | SPY | Iron Condor | $9.37 | $5.94 | +29.3% | +$274.41 | $1,863 | Time Exit (21d to expiry) |
| 2026-07-15 | WMT | Iron Condor | $3.61 | $3.25 | -4.1% | -$14.83 | $639 | Time Exit (21d to expiry) |
| 2026-07-15 | CVX | Iron Condor | $3.65 | $4.49 | -37.0% | -$134.95 | $635 | Time Exit (21d to expiry) |
| 2026-07-16 | UNH | Long Call | $12.15 | $4.90 | -65.8% | -$799.20 | $1,215 | Stop Loss (-50%) |
| 2026-07-16 | AAPL | Long Call | $9.00 | $0.64 | -100.0% | -$900.00 | $900 | Stop Loss (-50%) |
| 2026-07-16 | GS | Long Call | $33.65 | $11.81 | -67.9% | -$2,285.30 | $3,365 | Stop Loss (-50%) |
| 2026-07-16 | NVDA | Long Call | $7.15 | $2.65 | -69.1% | -$494.20 | $715 | Stop Loss (-50%) |
| 2026-07-16 | PFE | Long Call | $0.55 | $1.14 | +88.2% | +$48.50 | $55 | Take Profit (100%) |
| 2026-07-16 | QQQ | Bear Call | $0.47 | $0.15 | +20.0% | +$9.40 | $53 | Take Profit (50% of credit) |
| 2026-07-16 | SPY | Bear Call | $0.43 | $0.20 | +0.9% | +$0.40 | $57 | Take Profit (50% of credit) |
| 2026-07-16 | TSLA | Bull Put | $1.35 | $1.70 | -42.7% | -$57.60 | $115 | Time Exit (4d to expiry) |
| 2026-07-16 | MU | Bull Put | $2.45 | $1.01 | +49.6% | +$121.40 | $255 | Take Profit (50% of credit) |
| 2026-07-16 | C | Bear Call | $0.46 | $0.16 | +15.2% | +$6.90 | $54 | Take Profit (50% of credit) |
| 2026-07-16 | AMZN | Short Put | $7.75 | $10.37 | -40.0% | -$309.80 | $24,225 | Stop Loss (strike breached) |
| 2026-07-16 | PYPL | Short Put | $1.57 | $1.22 | +15.1% | +$23.70 | $5,343 | Time Exit (11d to expiry) |
| 2026-07-16 | AAPL | Short Put | $4.80 | $5.52 | -21.3% | -$102.10 | $31,520 | Time Exit (11d to expiry) |
| 2026-07-16 | IWM | Short Put | $2.43 | $3.34 | -44.0% | -$106.88 | $29,257 | Stop Loss (strike breached) |
| 2026-07-16 | META | Short Put | $26.50 | $39.50 | -52.9% | -$1,401.30 | $63,350 | Stop Loss (strike breached) |
| 2026-07-16 | QQQ | Iron Condor | $14.79 | $12.74 | +8.2% | +$120.63 | $2,521 | Time Exit (21d to expiry) |
| 2026-07-16 | AAPL | Iron Condor | $8.62 | $14.00 | -70.1% | -$603.86 | $1,138 | Time Exit (21d to expiry) |
| 2026-07-16 | TLT | Iron Condor | $0.75 | $1.01 | -91.0% | -$68.25 | $125 | Time Exit (21d to expiry) |
| 2026-07-16 | WMT | Iron Condor | $3.59 | $3.25 | -4.5% | -$16.29 | $640 | Time Exit (21d to expiry) |
| 2026-07-16 | PFE | Iron Condor | $0.42 | $0.31 | -72.0% | -$30.26 | $58 | Time Exit (21d to expiry) |
| 2026-07-17 | GS | Long Call | $34.00 | $12.50 | -66.2% | -$2,251.30 | $3,400 | Stop Loss (-50%) |
| 2026-07-17 | AAPL | Long Call | $8.90 | $0.64 | -100.0% | -$890.00 | $890 | Stop Loss (-50%) |
| 2026-07-17 | WFC | Long Call | $1.96 | $0.92 | -59.7% | -$117.06 | $196 | Stop Loss (-50%) |
| 2026-07-17 | JPM | Long Call | $7.25 | $9.72 | +25.1% | +$181.75 | $725 | Time Exit (21d to expiry) |
| 2026-07-17 | IWM | Long Call | $4.40 | $1.75 | -69.2% | -$304.60 | $440 | Stop Loss (-50%) |
| 2026-07-17 | SPY | Bear Call | $0.50 | $0.10 | +34.1% | +$16.90 | $50 | Take Profit (50% of credit) |
| 2026-07-17 | IWM | Bear Call | $0.43 | $0.32 | -27.0% | -$11.60 | $57 | Time Exit (9d to expiry) |
| 2026-07-17 | QQQ | Bear Call | $0.46 | $0.22 | +3.0% | +$1.40 | $54 | Take Profit (50% of credit) |
| 2026-07-17 | DIA | Bear Call | $0.46 | $0.00 | +100.0% | +$46.50 | $54 | Take Profit (50% of credit) |
| 2026-07-17 | NVDA | Bull Put | $0.98 | $1.05 | -30.9% | -$30.10 | $152 | Time Exit (11d to expiry) |
| 2026-07-17 | PYPL | Short Put | $1.35 | $1.22 | +1.3% | +$1.70 | $5,365 | Time Exit (11d to expiry) |
| 2026-07-17 | AAPL | Short Put | $4.90 | $5.52 | -18.9% | -$92.70 | $31,510 | Time Exit (11d to expiry) |
| 2026-07-17 | COIN | Short Put | $9.05 | $7.40 | +12.1% | +$109.40 | $14,595 | Time Exit (11d to expiry) |
| 2026-07-17 | LRCX | Short Put | $20.70 | $22.95 | -15.8% | -$326.30 | $28,430 | Time Exit (11d to expiry) |
| 2026-07-17 | NVDA | Short Put | $4.35 | $4.25 | -4.0% | -$17.40 | $19,565 | Time Exit (11d to expiry) |
| 2026-07-17 | TLT | Iron Condor | $0.56 | $0.57 | -72.6% | -$41.00 | $94 | Time Exit (21d to expiry) |
| 2026-07-17 | AAPL | Iron Condor | $8.98 | $14.00 | -63.4% | -$569.43 | $1,102 | Time Exit (21d to expiry) |
| 2026-07-17 | PFE | Iron Condor | $0.41 | $0.31 | -78.3% | -$31.71 | $60 | Time Exit (21d to expiry) |
| 2026-07-17 | WMT | Iron Condor | $3.70 | $3.25 | -1.6% | -$6.10 | $630 | Time Exit (21d to expiry) |
| 2026-07-17 | SPY | Iron Condor | $9.71 | $6.29 | +28.1% | +$272.87 | $2,129 | Time Exit (21d to expiry) |
| 2026-07-21 | AAPL | Long Call | $8.55 | $0.54 | -100.0% | -$855.00 | $855 | Time Exit (21d to expiry) |
| 2026-07-21 | WFC | Long Call | $1.69 | $0.62 | -72.3% | -$122.21 | $169 | Stop Loss (-50%) |
| 2026-07-21 | GS | Long Call | $36.25 | $14.83 | -61.9% | -$2,243.30 | $3,625 | Stop Loss (-50%) |
| 2026-07-21 | V | Long Call | $8.30 | $8.10 | -11.4% | -$94.70 | $830 | Time Exit (21d to expiry) |
| 2026-07-21 | MMM | Long Call | $3.80 | $9.97 | +156.0% | +$592.90 | $380 | Take Profit (100%) |
| 2026-07-21 | SPY | Bear Call | $0.53 | $0.21 | +17.7% | +$9.40 | $47 | Take Profit (50% of credit) |
| 2026-07-21 | QQQ | Bear Call | $0.44 | $0.07 | +32.7% | +$14.40 | $56 | Take Profit (50% of credit) |
| 2026-07-21 | GLD | Bear Call | $0.46 | $0.26 | -5.7% | -$2.60 | $54 | Time Exit (5d to expiry) |
| 2026-07-21 | IWM | Bear Call | $0.35 | $0.17 | -11.5% | -$4.10 | $64 | Take Profit (50% of credit) |
| 2026-07-21 | UAL | Bull Put | $1.88 | $1.21 | +21.8% | +$40.87 | $312 | Time Exit (21d to expiry) |
| 2026-07-21 | DIA | Bear Call | $0.85 | $0.65 | -3.0% | -$2.55 | $115 | Time Exit (21d to expiry) |
| 2026-07-21 | MSFT | Bull Put | $2.17 | $0.25 | +76.3% | +$165.97 | $283 | Take Profit (50% of credit) |
| 2026-07-21 | AAPL | Bear Call | $0.95 | $0.47 | +26.7% | +$25.40 | $155 | Take Profit (50% of credit) |
| 2026-07-21 | NFLX | Bull Put | $0.39 | $0.17 | -1.5% | -$0.60 | $61 | Take Profit (50% of credit) |
| 2026-07-21 | MU | Bull Put | $3.07 | $0.00 | +100.0% | +$307.50 | $193 | Take Profit (50% of credit) |
| 2026-07-21 | LCID | Short Put | $1.16 | $1.24 | -16.6% | -$19.30 | $584 | Stop Loss (strike breached) |
| 2026-07-21 | CAT | Short Put | $38.20 | $60.38 | -60.7% | -$2,319.30 | $81,180 | Stop Loss (strike breached) |
| 2026-07-21 | MPC | Short Put | $8.60 | $8.30 | -5.5% | -$47.40 | $29,140 | Time Exit (21d to expiry) |
| 2026-07-21 | AMZN | Short Put | $8.45 | $14.32 | -75.6% | -$639.00 | $23,155 | Stop Loss (strike breached) |
| 2026-07-21 | MA | Short Put | $9.60 | $3.40 | +58.4% | +$561.10 | $51,040 | Take Profit (50% @ 24d) |
| 2026-07-21 | QQQ | Iron Condor | $14.00 | $12.74 | +3.1% | +$44.00 | $2,600 | Time Exit (21d to expiry) |
| 2026-07-21 | AAPL | Iron Condor | $6.14 | $9.68 | -67.2% | -$412.42 | $886 | Time Exit (21d to expiry) |
| 2026-07-21 | GLD | Iron Condor | $5.76 | $4.55 | +11.1% | +$63.72 | $924 | Time Exit (21d to expiry) |
| 2026-07-21 | NFLX | Iron Condor | $1.69 | $1.74 | -29.6% | -$50.07 | $331 | Time Exit (21d to expiry) |
| 2026-07-22 | AAPL | Long Call | $8.35 | $0.89 | -98.3% | -$821.15 | $835 | Stop Loss (-50%) |
| 2026-07-22 | AVGO | Long Call | $19.35 | $27.80 | +38.5% | +$745.00 | $1,935 | Time Exit (21d to expiry) |
| 2026-07-22 | IWM | Long Call | $5.06 | $2.52 | -59.2% | -$299.54 | $506 | Stop Loss (-50%) |
| 2026-07-22 | TLT | Long Call | $0.94 | $0.34 | -77.5% | -$72.82 | $94 | Stop Loss (-50%) |
| 2026-07-22 | PFE | Long Call | $0.43 | $0.87 | +79.1% | +$34.00 | $43 | Take Profit (100%) |
| 2026-07-22 | GS | Long Call | $31.25 | $13.70 | -59.4% | -$1,856.30 | $3,125 | Stop Loss (-50%) |
| 2026-07-22 | V | Long Call | $7.10 | $8.10 | +5.1% | +$36.10 | $710 | Time Exit (21d to expiry) |
| 2026-07-22 | JNJ | Long Call | $5.65 | $11.92 | +104.7% | +$591.80 | $565 | Take Profit (100%) |
| 2026-07-22 | WFC | Long Call | $1.43 | $1.06 | -33.8% | -$48.30 | $143 | Time Exit (12d to expiry) |
| 2026-07-22 | IWM | Long Call | $3.71 | $2.54 | -37.9% | -$140.56 | $371 | Time Exit (12d to expiry) |
| 2026-07-22 | QQQ | Bear Call | $0.49 | $0.17 | +19.2% | +$9.40 | $51 | Take Profit (50% of credit) |
| 2026-07-22 | SPY | Bear Call | $0.45 | $0.17 | +11.0% | +$4.90 | $55 | Take Profit (50% of credit) |
| 2026-07-22 | DIA | Bear Call | $0.48 | $0.22 | +6.1% | +$2.90 | $52 | Take Profit (50% of credit) |
| 2026-07-22 | IWM | Bear Call | $0.45 | $0.20 | +4.3% | +$1.90 | $55 | Take Profit (50% of credit) |
| 2026-07-22 | COST | Bull Put | $2.40 | $0.30 | +78.1% | +$187.40 | $260 | Take Profit (50% of credit) |
| 2026-07-22 | CAT | Short Put | $41.65 | $44.81 | -10.0% | -$417.30 | $82,835 | Stop Loss (strike breached) |
| 2026-07-22 | SPY | Short Put | $6.52 | $10.97 | -74.5% | -$485.42 | $73,848 | Stop Loss (strike breached) |
| 2026-07-22 | AAPL | Short Put | $7.55 | $4.70 | +31.6% | +$238.40 | $31,245 | Take Profit (35% @ 19d) |
| 2026-07-22 | CRM | Short Put | $6.65 | $9.35 | -46.8% | -$311.20 | $15,335 | Stop Loss (strike breached) |
| 2026-07-22 | XOM | Short Put | $4.10 | $3.19 | +13.2% | +$54.10 | $14,590 | Time Exit (21d to expiry) |
| 2026-07-23 | SPY | Bear Call | $0.50 | $0.39 | -24.4% | -$12.10 | $50 | Time Exit (5d to expiry) |
| 2026-07-23 | QQQ | Bull Put | $0.46 | $0.17 | +13.9% | +$6.40 | $54 | Take Profit (50% of credit) |
| 2026-07-23 | DIA | Bear Call | $0.47 | $0.14 | +22.9% | +$10.90 | $53 | Take Profit (50% of credit) |
| 2026-07-23 | IWM | Bear Call | $0.46 | $0.22 | +2.0% | +$0.90 | $54 | Take Profit (50% of credit) |
| 2026-07-23 | MU | Bull Put | $4.70 | $1.79 | +57.1% | +$268.40 | $530 | Take Profit (50% of credit) |
| 2026-07-23 | AAPL | Short Put | $7.65 | $3.35 | +50.0% | +$382.80 | $30,735 | Take Profit (50% @ 25d) |
| 2026-07-23 | LCID | Short Put | $2.98 | $0.65 | +71.8% | +$213.82 | $252 | Take Profit (50% @ 26d) |
| 2026-07-23 | SPY | Short Put | $6.45 | $5.60 | +7.0% | +$45.00 | $72,355 | Time Exit (12d to expiry) |
| 2026-07-23 | CAT | Short Put | $43.50 | $68.08 | -58.8% | -$2,559.30 | $83,650 | Stop Loss (strike breached) |
| 2026-07-23 | AMZN | Short Put | $6.45 | $6.04 | +0.2% | +$1.00 | $21,855 | Time Exit (12d to expiry) |
| 2026-07-23 | AAPL | Long Call | $8.35 | $16.84 | +95.5% | +$797.60 | $835 | Take Profit (100%) |
| 2026-07-23 | XLF | Long Call | $0.84 | $1.77 | +97.3% | +$81.70 | $84 | Take Profit (100%) |
| 2026-07-23 | AVGO | Long Call | $17.95 | $27.80 | +49.3% | +$885.00 | $1,795 | Time Exit (21d to expiry) |
| 2026-07-23 | IWM | Long Call | $4.99 | $6.70 | +28.2% | +$140.56 | $499 | Time Exit (21d to expiry) |
| 2026-07-23 | TLT | Long Call | $0.73 | $0.34 | -70.1% | -$51.19 | $73 | Stop Loss (-50%) |
| 2026-07-27 | WMT | Long Call | $2.97 | $2.64 | -17.1% | -$50.82 | $297 | Time Exit (21d to expiry) |
| 2026-07-27 | IWM | Long Call | $5.03 | $6.70 | +27.1% | +$136.32 | $503 | Time Exit (21d to expiry) |
| 2026-07-27 | SLV | Long Call | $2.01 | $4.03 | +94.2% | +$189.44 | $201 | Take Profit (100%) |
| 2026-07-27 | UNH | Long Call | $10.50 | $4.72 | -61.0% | -$640.50 | $1,050 | Stop Loss (-50%) |
| 2026-07-27 | SPY | Bear Call | $0.48 | $0.01 | +50.8% | +$24.40 | $52 | Take Profit (50% of credit) |
| 2026-07-27 | DIA | Bear Call | $0.47 | $2.15 | -110.5% | -$52.50 | $53 | Stop Loss (100% of credit) |
| 2026-07-27 | GLD | Bear Call | $0.23 | $0.55 | -122.2% | -$27.50 | $27 | Stop Loss (100% of credit) |
| 2026-07-27 | NFLX | Bull Put | $0.40 | $0.17 | +3.3% | +$1.31 | $60 | Take Profit (50% of credit) |
| 2026-07-27 | QQQ | Bear Call | $1.34 | $0.35 | +56.1% | +$75.47 | $166 | Take Profit (50% of credit) |
| 2026-07-27 | LCID | Short Put | $0.80 | $0.35 | +40.8% | +$32.60 | $520 | Take Profit (50% @ 22d) |
| 2026-07-27 | BAC | Short Put | $0.63 | $0.48 | +4.9% | +$3.11 | $5,937 | Time Exit (21d to expiry) |
| 2026-07-27 | CAT | Short Put | $27.40 | $50.45 | -90.8% | -$2,487.20 | $77,260 | Stop Loss (strike breached) |
| 2026-07-27 | RIVN | Short Put | $1.02 | $1.33 | -43.2% | -$44.06 | $1,498 | Time Exit (21d to expiry) |
| 2026-07-27 | AAPL | Short Put | $6.40 | $7.55 | -27.0% | -$172.60 | $32,360 | Time Exit (15d to expiry) |
| 2026-07-27 | TLT | Iron Condor | $0.59 | $0.48 | -49.2% | -$29.00 | $91 | Time Exit (21d to expiry) |
| 2026-07-28 | SPY | Bear Call | $0.51 | $0.25 | +8.8% | +$4.47 | $49 | Take Profit (50% of credit) |
| 2026-07-28 | INTC | Bull Put | $1.03 | $0.45 | +33.6% | +$34.42 | $97 | Take Profit (50% of credit) |
| 2026-07-28 | MU | Bull Put | $2.80 | $0.00 | +100.0% | +$280.00 | $220 | Take Profit (50% of credit) |
| 2026-07-28 | QQQ | Bear Call | $0.89 | $0.39 | +30.3% | +$26.85 | $111 | Take Profit (50% of credit) |
| 2026-07-28 | ORCL | Bull Put | $0.42 | $0.00 | +100.0% | +$42.50 | $58 | Take Profit (50% of credit) |
| 2026-07-28 | XYZ | Short Put | $3.95 | $4.46 | -21.9% | -$86.55 | $7,605 | Time Exit (21d to expiry) |
| 2026-07-28 | XLF | Short Put | $0.41 | $0.40 | -25.0% | -$10.23 | $5,559 | Time Exit (21d to expiry) |
| 2026-07-28 | AAPL | Short Put | $6.40 | $30.15 | -380.1% | -$2,432.60 | $32,360 | Time Exit (21d to expiry) |
| 2026-07-28 | NVDA | Short Put | $4.20 | $3.55 | +6.5% | +$27.20 | $18,330 | Time Exit (21d to expiry) |
| 2026-07-28 | WMT | Short Put | $1.95 | $2.88 | -56.7% | -$110.55 | $10,805 | Time Exit (21d to expiry) |
| 2026-07-28 | SPY | Iron Condor | $8.99 | $10.38 | -19.9% | -$179.00 | $2,001 | Time Exit (21d to expiry) |
| 2026-07-28 | HD | Long Call | $10.05 | $5.00 | -59.2% | -$595.45 | $1,005 | Stop Loss (-50%) |
| 2026-07-28 | TLT | Long Call | $0.82 | $0.39 | -67.6% | -$55.46 | $82 | Stop Loss (-50%) |
| 2026-07-28 | AMZN | Long Call | $8.20 | $32.62 | +288.8% | +$2,368.20 | $820 | Take Profit (100%) |
| 2026-07-28 | IWM | Long Call | $4.90 | $6.70 | +30.6% | +$150.10 | $490 | Time Exit (21d to expiry) |
| 2026-07-28 | XLF | Long Call | $0.90 | $0.45 | -64.1% | -$57.70 | $90 | Stop Loss (-50%) |
| 2026-07-29 | AAPL | Long Call | $9.20 | $0.50 | -100.0% | -$920.00 | $920 | Stop Loss (-50%) |
| 2026-07-29 | TLT | Long Call | $0.75 | $0.30 | -76.3% | -$57.25 | $75 | Stop Loss (-50%) |
| 2026-07-29 | QQQ | Long Call | $13.68 | $27.95 | +98.3% | +$1,344.92 | $1,368 | Take Profit (100%) |
| 2026-07-29 | QQQ | Bull Put | $0.50 | $0.00 | +100.0% | +$50.50 | $50 | Take Profit (50% of credit) |
| 2026-07-29 | SPY | Bear Call | $0.96 | $2.02 | -108.3% | -$104.00 | $104 | Stop Loss (100% of credit) |
| 2026-07-29 | GLD | Bear Call | $0.45 | $1.20 | -122.2% | -$55.00 | $55 | Stop Loss (100% of credit) |
| 2026-07-29 | MU | Bull Put | $4.77 | $0.88 | +74.4% | +$355.17 | $523 | Take Profit (50% of credit) |
| 2026-07-29 | DIA | Bear Call | $0.50 | $0.00 | +81.0% | +$40.50 | $50 | Take Profit (50% of credit) |
| 2026-07-29 | AAPL | Iron Condor | $9.37 | $18.79 | -104.7% | -$981.50 | $1,563 | Stop Loss (100% of credit) |
| 2026-07-30 | TLT | Long Call | $0.81 | $0.40 | -63.6% | -$51.50 | $81 | Stop Loss (-50%) |
| 2026-07-30 | XLE | Long Call | $1.38 | $0.68 | -58.0% | -$80.00 | $138 | Stop Loss (-50%) |
| 2026-07-30 | MSFT | Long Call | $12.50 | $32.22 | +151.8% | +$1,897.50 | $1,250 | Take Profit (100%) |
| 2026-07-30 | SLV | Long Call | $1.89 | $4.08 | +109.6% | +$207.16 | $189 | Take Profit (100%) |
| 2026-07-30 | QQQ | Bull Put | $0.52 | $0.00 | +83.5% | +$43.44 | $48 | Take Profit (50% of credit) |
| 2026-07-30 | IWM | Bear Call | $0.49 | $0.44 | -29.3% | -$14.50 | $51 | Time Exit (19d to expiry) |
| 2026-07-30 | DIA | Bull Put | $0.25 | $0.00 | +49.0% | +$12.25 | $25 | Take Profit (50% of credit) |
| 2026-07-30 | INTC | Bull Put | $1.50 | $0.70 | +40.0% | +$60.00 | $150 | Take Profit (50% of credit) |
| 2026-07-30 | SPY | Bear Call | $0.90 | $2.02 | -121.0% | -$109.50 | $110 | Stop Loss (100% of credit) |
| 2026-07-30 | LCID | Short Put | $0.87 | $0.72 | +6.3% | +$5.50 | $613 | Time Exit (19d to expiry) |
| 2026-07-30 | RIVN | Short Put | $0.82 | $0.98 | -34.7% | -$28.46 | $1,468 | Stop Loss (strike breached) |
| 2026-07-30 | NKE | Short Put | $1.05 | $1.00 | -5.2% | -$5.50 | $3,995 | Time Exit (19d to expiry) |
| 2026-07-30 | LYFT | Short Put | $0.51 | $0.36 | +9.8% | +$5.00 | $1,349 | Time Exit (12d to expiry) |
| 2026-07-30 | CMCSA | Short Put | $0.40 | $0.24 | +12.0% | +$4.80 | $2,260 | Take Profit (35% @ 14d) |
| 2026-07-31 | AMZN | Long Call | $7.45 | $3.38 | -60.7% | -$452.20 | $745 | Stop Loss (-50%) |
| 2026-07-31 | GOOGL | Long Call | $10.50 | $21.10 | +95.0% | +$997.00 | $1,050 | Take Profit (100%) |
| 2026-07-31 | XLE | Long Call | $1.58 | $0.68 | -63.3% | -$100.00 | $158 | Stop Loss (-50%) |
| 2026-07-31 | SPY | Bear Call | $0.49 | $1.30 | -102.0% | -$50.50 | $51 | Stop Loss (100% of credit) |
| 2026-07-31 | ORCL | Bull Put | $0.47 | $0.21 | +10.7% | +$5.07 | $53 | Take Profit (50% of credit) |
| 2026-07-31 | IWM | Bear Call | $0.46 | $0.58 | -71.4% | -$32.50 | $54 | Time Exit (18d to expiry) |
| 2026-07-31 | QQQ | Bear Call | $0.84 | $1.20 | -66.1% | -$55.50 | $116 | Time Exit (18d to expiry) |
| 2026-07-31 | INTC | Bull Put | $0.25 | $0.08 | -15.0% | -$3.75 | $25 | Take Profit (50% of credit) |
| 2026-07-31 | LCID | Short Put | $0.60 | $0.46 | +5.8% | +$3.50 | $640 | Time Exit (11d to expiry) |
| 2026-07-31 | AAL | Short Put | $0.58 | $0.30 | +30.2% | +$17.50 | $1,442 | Take Profit (35% @ 18d) |
| 2026-07-31 | NIO | Short Put | $0.12 | $0.11 | -70.8% | -$8.50 | $438 | Time Exit (18d to expiry) |
| 2026-07-31 | RIVN | Short Put | $0.36 | $0.21 | +12.5% | +$4.50 | $1,364 | Take Profit (35% @ 18d) |
| 2026-07-31 | NKE | Short Put | $0.55 | $0.41 | +8.2% | +$4.50 | $3,895 | Time Exit (18d to expiry) |
| 2026-08-03 | QQQ | Long Call | $13.51 | $27.23 | +95.6% | +$1,291.44 | $1,351 | Take Profit (100%) |
| 2026-08-03 | AMZN | Long Call | $8.75 | $4.20 | -58.0% | -$507.50 | $875 | Stop Loss (-50%) |
| 2026-08-04 | AMZN | Long Call | $8.10 | $3.52 | -62.5% | -$506.10 | $810 | Stop Loss (-50%) |
| 2026-08-04 | COP | Long Call | $4.20 | $8.85 | +104.7% | +$439.80 | $420 | Take Profit (100%) |
| 2026-08-04 | XLE | Long Call | $0.86 | $0.52 | -51.2% | -$44.00 | $86 | Time Exit (14d to expiry) |
| 2026-08-04 | NFLX | Long Call | $1.69 | $1.55 | -14.3% | -$24.14 | $169 | Time Exit (14d to expiry) |
| 2026-08-04 | NVDA | Long Call | $8.10 | $13.15 | +56.3% | +$456.40 | $810 | Time Exit (21d to expiry) |
| 2026-08-04 | AMZN | Long Call | $7.00 | $4.90 | -36.0% | -$252.00 | $700 | Time Exit (21d to expiry) |
| 2026-08-04 | XOM | Long Call | $2.40 | $1.98 | -23.7% | -$56.90 | $240 | Time Exit (14d to expiry) |
| 2026-08-05 | AMZN | Long Call | $9.80 | $4.65 | -58.6% | -$573.80 | $980 | Stop Loss (-50%) |
| 2026-08-05 | MRK | Long Call | $3.55 | $1.70 | -58.1% | -$206.30 | $355 | Stop Loss (-50%) |
| 2026-08-05 | GS | Long Call | $32.85 | $15.23 | -56.7% | -$1,862.50 | $3,285 | Stop Loss (-50%) |
| 2026-08-05 | NFLX | Long Call | $1.45 | $1.15 | -27.9% | -$40.50 | $145 | Time Exit (12d to expiry) |
| 2026-08-05 | C | Long Call | $2.58 | $1.24 | -57.9% | -$149.48 | $258 | Stop Loss (-50%) |
| 2026-08-05 | GS | Long Call | $27.30 | $8.60 | -72.2% | -$1,970.00 | $2,730 | Stop Loss (-50%) |
| 2026-08-05 | AAPL | Long Call | $5.75 | $4.97 | -19.5% | -$112.00 | $575 | Time Exit (12d to expiry) |
| 2026-08-05 | VZ | Long Call | $1.11 | $1.94 | +65.8% | +$73.00 | $111 | Time Exit (19d to expiry) |
| 2026-08-06 | GOOGL | Long Call | $10.95 | $4.42 | -65.6% | -$718.20 | $1,095 | Stop Loss (-50%) |
| 2026-08-07 | AMZN | Long Call | $4.65 | $3.97 | -20.5% | -$95.40 | $465 | Time Exit (11d to expiry) |
| 2026-08-07 | AAPL | Long Call | $5.75 | $2.71 | -58.8% | -$338.00 | $575 | Stop Loss (-50%) |
| 2026-08-07 | TSLA | Long Call | $8.30 | $9.80 | +12.1% | +$100.20 | $830 | Time Exit (11d to expiry) |
| 2026-08-07 | NVDA | Long Call | $5.00 | $4.10 | -24.0% | -$120.00 | $500 | Time Exit (11d to expiry) |
| 2026-08-10 | MSFT | Long Call | $13.25 | $5.92 | -61.3% | -$812.00 | $1,325 | Stop Loss (-50%) |
| 2026-08-11 | SLB | Long Call | $1.75 | $0.81 | -59.7% | -$104.50 | $175 | Stop Loss (-50%) |
