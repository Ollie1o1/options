# Preregistration — Study 2E: `VM_OUTLOOK_TOP3` Rebalance Execution Clock & Overnight/Intraday Decomposition

**Registered:** 2026-11-27 (frozen before `scripts/backtest_rebalance_timing_and_overnight.py` is executed).
**Data:** Local Alpaca SIP 5-minute split+dividend adjusted bars (`data/intraday_5m/`, all 16 `VM_OUTLOOK_TOP3` ETFs, 2016-01 .. 2026-11, ~2,720 full sessions, early closes excluded via exchange calendar).

## Why

1. Your live `VALIDATED` TFSA sleeve (`VM_OUTLOOK_TOP3`) signals at close `t` and is executed manually on Wealthsimple during session `t+1`. We have never measured whether executing at `09:35`, `10:00`, `12:00`, `14:00`, or `15:55` ET systematically gains or loses bps on rebalance days.
2. Lou, Polk & Skouras (2019) document that overnight (close-to-open) and intraday (open-to-close) returns are driven by distinct investor clienteles (institutions/auctions vs intraday flows). Unlike the closed SPY late-day fade (`I1`), this tests (a) whether `VM_OUTLOOK_TOP3`'s alpha lives overnight vs intraday and (b) whether conditioning uptrend pullback entries on **intraday selloff vs overnight gap-down** improves 3-day reversion in shares.

## Frozen Hypotheses (zero tuning)

| ID | Question | Metric & Test | Pass Bar |
|---|---|---|---|
| **E1 (Rebalance Clock)** | On the ~129 monthly `VM_OUTLOOK_TOP3` rebalance days (`t+1`), compare executing at `09:35`, `10:00`, `12:00`, `14:00` vs baseline `15:55` ET (close of `t+1`). | Paired monthly return difference (`arm - close_1555`) in bps/rebalance + full-window Sharpe, 3-month block bootstrap (10,000 draws, seed `20261127`), Bonferroni across 4 clocks (`98.75%` CI). | `98.75%` CI lower bound `> 0` AND same sign in `2016-2020` and `2021-2026`. |
| **E2a (Overnight vs Intraday Attribution)** | Descriptive attribution of `VM_OUTLOOK_TOP3` monthly returns into overnight (`15:55 -> 09:30`) vs intraday (`09:30 -> 15:55`) components across all 16 ETFs. | Annualized contribution + t-stat (descriptive, no authority). | Descriptive. |
| **E2b (Intraday Selloff vs Gap-Down Pullback)** | Across the 16 ETFs in uptrend (`Close > SMA200`), compare 3-session forward excess return (net of 6 bps round-trip) after: (i) **Intraday Selloff** (`open->15:55 <= -1.5%` and `overnight > -0.5%`) vs (ii) **Overnight Gap-Down** (`prev_15:55->open <= -1.5%`), entered at `15:55` close of signal day. | Mean 3d return & paired difference (`intraday_selloff - gap_down`), 10-day block bootstrap (`97.5%` CI across 2 arms). | `97.5%` CI lower bound `> 0` AND positive in both `2016-2020` and `2021-2026`. |

## Operational Consequence
- If **E1** passes, record the optimal rebalance execution time in `opt rebalance` / `docs/GO_LIVE_RUNBOOK.md`. If null, confirm any liquid mid-day/close Wealthsimple fill is equivalent.
- If **E2b** passes, register a follow-up single-name pullback filter test; if null, close cleanly in `docs/RESEARCH_FINDINGS.md`.
