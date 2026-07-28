# Signal overlays — validation feasibility

Assessed 2026-07-28. The UOA and EDGAR-insider overlays are display-only: they
never touch `quality_score` and never suppress a pick. Neither has ever been
validated. This note records what it would take, and why one of them cannot be
done today.

## UOA (unusual options activity) — BLOCKED, no data

`src/uoa.py` is built on **open-interest deltas between chain snapshots**
(`oi_deltas`, `uoa.py:40`): it compares today's OI against the previous
snapshot for the same contract. Validating it therefore needs a point-in-time
history of open interest.

Neither store has one:

| store | rows | coverage | has OI? |
|---|---:|---|---|
| `data/chain_archive.db` | 300,726 | 2026-06-10 → 2026-07-27, **19 snapshot dates** | yes |
| `data/dolt_options.db` (`dolt_chain`) | 7 years EOD | 2019 → present | **no** — `symbol, date, expiration, strike, type, bid, ask, mid, iv, delta, gamma, theta, vega, rho` |

The DoltHub table carries full Greeks but no open interest and no volume, so
the one source with real history cannot express the signal at all. The chain
archive can, but 19 snapshot dates spanning seven weeks leaves almost no room
for a forward window — a 21-day horizon is measurable for only the earliest
handful of dates.

For scale: the squeeze study needed 205 settlement dates and 480,744 graded
observations to resolve a ~3pp effect. UOA has roughly three orders of
magnitude less.

**Conclusion: not testable today, and not for want of effort.** The archive
accrues ~1 snapshot per trading day, so a study becomes possible in roughly a
year of accumulation, or immediately with a paid OI history vendor. Until then
the overlay should keep saying nothing about edge — which, being display-only,
it currently does.

## EDGAR insider clusters — feasible, but a project of its own

`pick_context.insider_summary` fetches Form 4 filings live and caches per
(ticker, day). Nothing is stored historically, but unlike OI this is a
solvable problem: EDGAR full-text search and the daily index files are free and
go back decades, and Form 4 carries transaction date, price, shares and the
insider's role.

A validation would need the same shape as `src/squeeze/backtest/`:

1. **Backfill** — walk EDGAR daily indexes, parse Form 4 XML, store
   `(symbol, filed_date, transaction_date, role, shares, price, code)`.
2. **Point-in-time discipline** — enter on the *filed* date, never the
   transaction date. Form 4 is due within two business days, and that gap is
   exactly where look-ahead would flatter the result.
3. **Price panel** — reuse `data/squeeze_prices.db`, which already holds
   survivorship-free daily closes.
4. **Event study** — forward returns at 21/42/63d versus a matched control,
   clustered by filing date, with a moving-block bootstrap.

Estimated scope: comparable to the squeeze harness. It is a real study, not a
fix, and should be planned on its own rather than bolted onto other work.

**Recommendation: do this one next if the overlays matter, and leave UOA alone
until the archive is deep enough to say anything honest.**
