# Broker costs — what this system charges, and why

**Broker:** Wealthsimple, self-directed, Ontario, CAD account.
**Verified:** 2026-07-29 against Wealthsimple's published fee schedule.
**Configured in:** `config.json` → `paper_trading`.

If you change broker, plan tier, or open a USD account, change the numbers here
and in `config.json` together. Nothing in the code hardcodes a fee.

---

## The schedule

| Cost | Amount | Modelled? |
|---|---|---|
| Options commission | **US$0** | yes — `commission_per_contract: 0.0` |
| Equity / ETF contract fee | **US$0** (all tiers) | yes — same field |
| Options assignment | **US$0** | n/a, no charge |
| Auto-exercise at expiry | **US$0** | n/a, no charge |
| Early exercise | **US$45** | **no** — see gaps below |
| Do-not-exercise instruction | **US$45** | **no** — see gaps below |
| CAD→USD / USD→CAD conversion | **1.5%** each way | yes — `fx_conversion_rate: 0.015` |
| Index options (SPX, SPXW, VIX) | **US$1.00**/contract | **no** — see gaps below |
| Index options (RUT, RUTW) | US$0.50/contract | no |
| Index options (XSP) | US$0.25/contract | no |
| Bid-ask spread | not a fee, but the largest cost | yes — measured, see below |

The conversion rate steps down with volume: 1.5% below US$10k of conversions,
1.0% to $25k, 0.5% to $100k, 0% above. A USD account removes it entirely for
$10/month. The configured 0.015 assumes the bottom tier, which is the right
default for a small account.

Source: <https://www.wealthsimple.com/en-ca/legal/fees/trade>

---

## The two costs that actually matter

**Bid-ask spread.** No fee schedule lists it and it dwarfs everything that is
listed. `src/execution_costs.py` measures it rather than assuming it: each
logged contract is joined to the archived quote of that same contract on its
own entry date (`data/chain_archive.db`). Measured medians, n=172 contract-days:

| Structure | Real half-spread | Previously charged |
|---|---:|---:|
| Bull Put | $0.163 | $0.050 |
| Short Put | $0.100 | $0.050 |
| Long Call | $0.073 | $0.050 |
| Iron Condor | $0.055 | $0.050 |
| Bear Call | $0.025 | $0.050 |

The old flat $0.05 was not uniformly wrong — it was wrong per structure and in
opposite directions, undercharging Bull Put 3.2x while overcharging Bear Call
2x. Buckets with fewer than 10 observations fall back to the flat default
rather than setting a constant from a handful of quotes.

**Currency conversion.** Unlike a contract fee it scales with money moved, not
with contract count, so it cannot be reduced by trading fewer legs and it falls
hardest on long premium. Across 187 closed credit trades in the cohort window
it came to $1,098.

---

## Known gaps

These are unmodelled, deliberately, and each will flatter results until fixed:

1. **Early exercise (US$45).** Assignment and auto-exercise are free, so holding
   to expiry costs nothing — but being forced to act early costs $45 against a
   median credit of $122. Not modelled because the ledger has no early-exercise
   events to calibrate against.
2. **Index option contract fees.** `commission_per_contract` is a single number
   and cannot express per-symbol pricing. Harmless today (the ledger trades SPY,
   QQQ and IWM, which are ETFs and genuinely free) but wrong the moment any SPX
   work starts. Tracked as `index-options-are-not-free` in `ideas.json`.
3. **Spread measured on the anchor leg** is applied to every leg of a structure.
   A Bull Put's long leg is further out of the money and need not quote like its
   short leg.
4. **Medians hide the tail.** Getting out of a losing spread in a fast market
   does not cost the median quote of a calm one.

---

## Seeing it yourself

```bash
# What the book looks like re-priced on real fees and measured spreads
python scripts/cost_model_report.py

# The costs charged on each auto-close are printed inline by the ledger,
# e.g.  [costs: $0.05/share spread x2 + 1.5% CAD/USD conversion x2 — see docs/BROKER_COSTS.md]
python -m src.check_pnl
```

The measurement improves on its own: `data/chain_archive.db` gains roughly one
snapshot per trading day, so re-running the report monthly tightens every number
in the table above.
