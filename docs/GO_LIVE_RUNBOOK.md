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
