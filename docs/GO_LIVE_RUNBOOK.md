# Go-Live Runbook — First Real-Money Long Call

**Mode:** Mirror only. The system prints an order ticket; **you** place it in your
broker. There is no broker API. Real money is OFF until every box below is checked.

---

## Pre-flight (do NOT skip)

0. **Run the machine-checked preflight first:**
   ```
   PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.execution.preflight
   ```
   It checks every item below automatically (gate, arming, risk caps,
   checkpoint freshness, automation health, slippage DB) and prints
   `CLEARED ✅` or `NOT CLEARED 🔒` with reasons. If it says NOT CLEARED,
   stop — the rest of this runbook explains *what* it verified.

1. **Gate is READY.** Run the screener; the startup banner / `reports/GATE_STATUS.md`
   must read `READY`. Confirm with:
   ```
   PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.execution.pipeline
   ```
   It must print `ARMED` only after step 2. If it says `DISARMED` because
   `gate=...`, you are not cleared — stop.
2. **Read `docs/VALIDATION_POWER.md` first.** Understand *why* the gate fired:
   - At n=50 a READY means observed IC ≳ 0.28 (a strong-edge signal), OR
   - you adopted the Bayesian rule (n≥50 AND P(true IC≥0.08) ≥ 0.85).
   If the edge is modest (~0.10), size **small** — the power analysis says you
   cannot have frequentist certainty at this sample size. Risk caps are the point.
3. **Flip the switch.** Set `config.json → live_execution.enabled = true`.
   Re-run the pipeline command; it must now print `ARMED ✅`.
4. **Set your account value** honestly (the number you are willing to trade, not
   net worth). Sizing caps at **2% risk** and **10% cost** per position.

## Placing the trade

5. Generate the ticket for the chosen pick (mirror mode). It will show:
   `BUY Nx TICKER STRIKE C exp DATE @ limit $X.XX`, plus take-profit, stop, and
   time-exit date.
6. **Sanity-check the ticket** before placing:
   - [ ] Contracts > 0 and risk_pct ≤ 2%.
   - [ ] Limit price is inside the current bid/ask.
   - [ ] Expiration ≥ 30 DTE (swing runway; matches the cohort rule).
   - [ ] You understand the stop and take-profit in dollars.
7. Place the order in your broker as a **limit** order at the ticket price. Do not
   chase; if it doesn't fill near mid, let it go.

## Measured Execution Windows (Empirically Verified 2026-11-27)

* **Options spreads & single legs (IBKR `[TAXABLE_ONLY]`, Study H4 — 3.72M OPRA `cbbo-1m` quotes):**
  * **Best windows:** **`11:30–13:30 ET`** (midday, `4.4%` median O1 friction, deepest resting books `87` lots) and **`15:15–15:50 ET`** (`2.1%` near-ATM spread, `5.0%` median O1 friction).
  * **Avoid `09:30–10:30 ET`:** `09:45` spreads are `2.2×` wider on `60–120 DTE` spreads (`14.3%` vs `6.3%`) and `4×` wider on ATM options (`8.2%` vs `2.1%`); even `10:30 ET` remains `+2.2 pp` wider than `15:45 ET`.
  * **Avoid `14:00 ET` on FOMC / Fed Minutes Wednesdays:** market makers widen/pull quotes around `14:00 ET` releases (`mean` O1 friction spikes to `12.8%`).
  * **Multi-ETF O1 OPRA Liquidity Map (`15:45 ET`):** Tier 1 (`100%` pass rate): `SPY` (`2.3%`), `QQQ` (`3.8%`), `GLD` (`4.9%`, `252`-lot depth), `IWM` (`5.0%`), `TLT` (`7.2%`); Tier 2 caution: `XLE` (`12.9%` med, `p90 26.0%`).
* **TFSA core ETF sleeve (`VM_OUTLOOK_TOP3` on Wealthsimple USD shares, Study 2E):**
  * Executing monthly rebalances at any liquid window between **`10:00` and `15:55 ET`** is statistically equivalent to the close (`±1.6 bps/mo`). Never flatten the sleeve intraday: `100%+` of `VM_OUTLOOK_TOP3`'s return accrues overnight (`+15.4%` ann, Sharpe `1.57` vs `-3.0%` intraday).

## After the fill

8. Record the actual fill so we can measure slippage:
   ```python
   from src.execution.slippage import record_fill
   record_fill("data/fills.db", ticker="AAPL", intended_price=4.20,
               actual_price=4.25, contracts=2, ticket_id="<from ticket>")
   ```
9. Set your broker exits to the ticket's stop / take-profit, and put the
   time-exit date in your calendar.

## Ongoing

10. After ~10 fills, check slippage:
    ```python
    from src.execution.slippage import slippage_report
    print(slippage_report("data/fills.db"))
    ```
    If `avg_slippage_per_contract` is a large fraction of your edge, your real
    fills are worse than paper — shrink size or widen entry discipline before
    scaling up.
11. Keep running the screener daily so the cohort and gate stay current.

## Kill switch

At any time, set `config.json → live_execution.enabled = false`. The next ticket
reverts to DRY RUN. If the gate later flips to `STOP`, honor it: stop opening new
positions, manage open ones to their exits.
