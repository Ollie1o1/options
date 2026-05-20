# Journal

## 2026-05-20 - crypto P&L backfill applied

**Trigger:** First proving test of the shared core (Plan 1 P0.5).
Fixes the x100 inflation legacy bug (commit 91dc97a only fixed code path, not stored data) AND applies the new $1,000 max-bet cap.

**Command:** `python -m scripts.backfill_crypto_pnl --db paper_trades_crypto.db --apply`

**Snapshots taken (revert path):**
- `paper_trades_crypto.db.bak.20260520-124027` (24KB)
- `paper_trades_crypto.db.bak.20260520-124118` (post-apply restore point)

**Before -> After:**

| Metric | Before | After |
|---|---|---|
| Closed total pnl_usd | $83,361 | **$6,437** |
| Rows updated | - | 25 |
| Win rate (closed) | 44.1% (mixed inflated) | **48.0%** |

**Notable corrections:**
- id 30 (BTC $60k call, +311%): $15,587 -> $3,114 (qty 0.1998, bet ~$999)
- id 38 (BTC $80k put, +21.6%): $55,402 -> $216 (qty 0.39, bet ~$999)
- id 41 (Long Put loss): -$29,097 -> -$96 (qty 0.33, bet ~$999)

**Method:** `quantity = capped_quantity(unit_risk=999/unit_risk floored to 4dp)` where
unit_risk = entry_price for debit, (spread_width - net_credit) for credit. New pnl_usd = pnl_pct * entry_price * quantity. Idempotent: re-running the script after apply produces the same fixed point (pnl_pct and entry_price are never modified).

**Operational note:** Backfill script opens raw sqlite; the real DB was at user_version=10 and needed a one-shot PaperManager() instantiation to run the v11 migration before the first apply attempt. Future script reuse on any unmigrated DB should add a migration step internally.

---

