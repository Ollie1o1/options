# The #1 slot: refuse, don't rank

2026-08-09. Harness: `scripts/validate_gates.py`. Module: `src/pick_ranking.py`.

## The question

"When I'm told this is the number 1 contract, it should be the best contract."

Fair demand. This is what the ledger said when it was asked.

## What was actually there

Fifteen call sites, six sort keys. "#1" meant a different thing on every screen:

| board / surface | sorted by |
|---|---|
| discovery, ticker, spreads | `ev_per_contract` (via `rank_by_verdict`) |
| iron condors | `return_on_risk` |
| squeeze, unusual activity | `quality_score` |
| lottery | `lottery_ticket_score` |
| EXECUTIVE SUMMARY top 3 | `quality_score` |
| "top pick" → tearsheet, 3D surface | `overall_score` = `quality_score` |
| CSV export, log-trades menu | `quality_score` |
| comparison table | `quality_score` |

`rank_by_verdict` had migrated four of them. The other eleven — including the
most prominent block in the report and the pick that feeds every tearsheet —
never moved off the metric whose top quintile lost $10,173.

`print_best_setup_callout` deserves its own line: it rendered a prominent
framed box for the single highest `quality_score` pick, and fired **only when
that score was ≥ 0.75**. The gate now refits its own cutoff from the ledger and
gets 0.756. The callout was a reliable pointer at the losing bucket, drawn more
prominently than anything else on screen.

## What the ledger says (851 closed, non-duplicate, 2026-04-18..2026-08-05)

Nothing ranks. Spearman against return on capital:

| key | population | rho |
|---|---|---|
| `ev_score` | Iron Condor, n=121 | **-0.325** (p=0.0003) |
| `pop_score` | Iron Condor, n=121 | -0.302 (p=0.0008) |
| `return_on_risk` | Iron Condor, n=139 | -0.216 (p=0.011) |
| `quality_score` | Long Call, n=263 | -0.131 (p=0.034) |
| `quality_score` | whole book, n=851 | -0.017 (p=0.63) |

Long single-leg quintiles by `quality_score` — the top bucket is the worst cell:

| bucket | n | win | mean ret | total |
|---|---|---|---|---|
| Q1 (worst score) | 68 | 37% | -3.0% | -$99 |
| Q2 | 67 | 40% | +4.3% | +$5,920 |
| Q3 | 68 | 41% | +5.6% | +$5,195 |
| Q4 | 67 | 43% | +4.5% | +$6,624 |
| **Q5 (best score)** | **68** | **27%** | **-15.7%** | **-$10,173** |

Not a range-restriction artifact: scores span 0.00–1.00, median 0.586.

**Why Q5 loses is not mysterious.** Same delta, same DTE as the rest — but
higher IV (0.370 vs 0.334), more expensive premium ($8.35 vs $5.82) and a
heavier theta bill (-0.206 vs -0.158/day). It buys the expensive version of the
same trade. `composite_weights` explains it exactly: eight vol-surface
components (vrp, iv_velocity, term_structure, iv_edge, vega_risk, iv_rank,
skew_align, iv_mispricing) sum to **0.680**, while theta carries **0.020** and
ev **0.007**. Eight correlated votes for "implied vol is interesting here" and
a rounding error for what that costs per day. On long premium, high IV is the
price, not the edge.

`spread_score` looked like the one real signal on condors at +0.536. Demeaned
within ticker it is +0.055 (p=0.55). It was never a ranking signal — it was the
universe rule wearing a component's clothes, which is why it became a gate.

## The two questions, answered separately

**REMOVAL — does refusing measured losers improve what is left? Yes.**
Walk-forward, every threshold fitted strictly before the fold it is applied to:

| fold | window | n | kept | keep mean | refuse mean | avoided |
|---|---|---|---|---|---|---|
| 1 | 06-08..06-18 | 100 | 87 | +0.150 | +0.025 | +$1,178 |
| 2 | 06-18..07-08 | 101 | 81 | +0.034 | -0.198 | -$1,286 |
| 3 | 07-08..07-14 | 102 | 79 | -0.002 | -0.121 | +$1,907 |
| 4 | 07-14..07-22 | 110 | 73 | +0.009 | -0.335 | +$15,644 |
| 5 | 07-22..08-05 | 92 | 88 | +0.058 | -0.219 | +$195 |

5 of 5 folds. **Caveats, stated rather than buried:** individual folds are
underpowered (p = 0.39, 0.11, 0.10, 0.0000), so this rests on sign consistency
(two-sided sign test p=0.06), and $15,644 of the $17,639 is one week in
mid-July. The direction is consistent; the magnitude is one bad week the gates
sat out.

**RANKING — does any ordering of the survivors beat `quality_score` at #1? No.**
A theta-cost ranker, out of sample, per board per day: won **23 of 48** paired
(day, board) cells, Wilcoxon **p=0.89**. Trimmed of outliers the old ranker was
ahead (+0.050 vs +0.023). This ranker is kept in `scripts/validate_gates.py`
(`rank_survivors`) so the negative result stays checkable.

## What shipped

Refuse, don't rank. Each gate is scoped to the population it was measured on —
G5 says nothing about single legs, G6 says nothing about condors, and neither is
applied outside its evidence.

| gate | applies to | evidence class |
|---|---|---|
| unquotable | all | ARITHMETIC |
| friction > 25% of reward | all | ARITHMETIC |
| credit disappears when crossed | credit structures | ARITHMETIC |
| condor off broad index | condors only | MEASURED |
| composite top quintile | long single-leg only | MEASURED |
| EV verdict is SKIP | all | CONSISTENCY |

G6's cutoff is refit from the ledger on every scan (80th percentile of closed
long single legs, currently 0.756) rather than frozen in source, and disables
itself below 30 rows rather than letting a handful of trades judge a board.

The EV gate defers to `decide_verdict` — the rule the top-N table and tearsheet
already render — rather than testing `ev < 0`. A raw sign test refused -$1 while
admitting +$9, two numbers inside the same error bar; a consistency gate that
disagrees with the number printed beside it would be the defect it exists to
remove.

Survivors come back ordered by carry cost, disclosed on the board as *"an
ordering, not a ranking; no row is a recommendation"*. The column header is `#`,
not `Rank`. There is no #1, because the evidence does not support one.

The `DO` line is withheld when the card cannot justify an order. It used to
print unconditionally: on a live scan this morning seven of nine cards read
NEGATIVE EV and all nine printed `DO BUY 1 @ $x`.

Gating is failure-safe like the rest of the scan path: if anything raises, every
row is kept and the board renders. A board that shows everything is a bug; a
board that shows nothing because of a bug is worse.

## Before and after, same scan

| | before | after |
|---|---|---|
| cards reading NEGATIVE EV | 7 of 9 | 0 of 9 |
| cards printing `DO BUY` | 9 of 9 | 8 of 9 (1 withheld, no EV basis) |
| discovery board | 275 shown | 50 of 275, 225 refused |
| most prominent block | TOP 3 OPPORTUNITIES by `quality_score` | CLEARED THE GATES, board order |
| best-setup callout | fired at score ≥ 0.75 | retired |

## What this does not claim

That the first row is the best contract. It is not, and no evidence in this
repo supports any ordering that would make it so. What is claimed is narrower
and testable: nothing on the board is a candidate the ledger measures as a
loser, and nothing on it contradicts the numbers printed beside it.

G6 was found in-sample over roughly three months of one book. Its walk-forward
support is real but thin. Re-run `scripts/validate_gates.py` as the ledger grows
and delete the gate if it stops holding.

---

# Addendum, 2026-08-10: the WORTH grade

`POSITIVE EV +18/ct` and `POSITIVE EV +6/ct` read identically on a card. The
sign of the edge was the only thing shown, and the sign is not the question.

## What was tested first

p* against your own history looked like the answer — a per-contract margin,
pure arithmetic, no model. Measured strictly causally (each trade's prior window
uses only trades that **closed** before it **opened**), n=302:

| | rho | p |
|---|---|---|
| pooled | **+0.246** | 0.00001 |
| time third 1 | +0.200 | 0.047 |
| time third 2 | +0.134 | 0.184 |
| time third 3 | +0.386 | 0.000 |
| **demeaned within structure** | **+0.104** | **0.072** |

Pooled it is strong and stable. Within structure it is not significant. What it
detects is Bull Put (margin +0.106, +22.0% return) against Iron Condor (-0.137,
-8.1%) — a structure-selection signal wearing a per-contract disguise, the third
time in two days that pattern has appeared. So it demotes a grade on arithmetic
grounds and is never presented as predicting which spread wins.

One finding fell out and is **not acted on**: within structure, p* measures
**-0.206 (p=0.0003)** against return — the safer-looking end of the credit
ladder underperforms, i.e. credit-to-width relates positively to return on
verticals. That bears on the open c2w question and needs its own validation.

## What shipped

`src/worth.py`. Three margins, and **the grade is the weakest of them, never a
blend** — the same discipline as the gates, for the same reason.

| margin | STRONG | CLEAR | basis |
|---|---|---|---|
| edge vs its own error bar (`net_ev / noise`) | ≥2.0 | ≥1.0 | model confidence |
| round-trip friction as share of reward | ≤5% | ≤10% | arithmetic |
| p* against your win rate on that structure | ≥+10pp | ≥0 | arithmetic |

"Limited by X" prints only when one margin is genuinely holding the grade down.
A live scan first read `STRONG ... (limited by edge vs its error bar)` when every
margin agreed — a constraint that was not constraining.

## What it is not

Not a profit forecast, and not a claim that STRONG beats THIN. The primary axis
reports how confident the model is in its own EV estimate, and that estimate has
no demonstrated out-of-sample edge in this repo.

**It could not be validated, and that is now fixed going forward.** Schema 21
persists `entry_ev_net`, `entry_ev_gross`, `entry_ev_cost`, `entry_ev_noise`.
Until 2026-08-10 the ledger kept only `ev_score` — a within-scan rank, which
cannot say how large an edge was, only where it sat that day. That is why 851
closed trades could not answer whether a STRONG contract beat a THIN one. From
now the question accumulates an answer. NULL means "not recorded", never zero.

## Also fixed

The noise formula existed twice — `tearsheet/collect.py` with named constants and
`cli_display.py` with the same numbers as literals. They agreed by luck. The
verdict, the EV gate, the WORTH grade and the persisted `entry_ev_noise` all
claim to be that number, so a drift would have made a card disagree with the
ledger row it produced. One implementation now, `tearsheet.collect.ev_noise`,
with a test asserting the sigma table appears in exactly one module.

---

# Addendum 2, 2026-08-10: the coverage sweep

The first pass wired the gate into the board dispatch. It did not reach
everything, and an AST guard found the rest.

## What was still showing the discredited metric

`format_decision_zone` — the only caller of the WORTH line — serves single-leg
boards. Spreads, condors and the squeeze board rendered their own cards, and
every one of them still led with `★★★☆☆` stars from `quality_score`.

Worse, the **single-leg card header itself** opened with those stars. Retiring
the best-setup callout and the TOP 3 block left the brightest element on every
card still driven by the metric whose top quintile lost $10,173.

Producer functions also stamped an order before the gate ever saw the frame:

| site | was | now |
|---|---|---|
| `find_credit_spreads` return | `sort_values("quality_score")` | unsorted |
| `find_iron_condors` return | `sort_values("return_on_risk")` | unsorted |
| `squeeze/board._rank_calls` fallback | `sort_values("quality_score")` | incoming order |
| `print_comparison_table` default | `quality_score` | board order |
| card header, Valuation row | `★★★☆☆ 0.62` | `●●○○ CLEAR`, `score 0.62` muted |
| spread + condor cards | `Score 0.62 ★★★★☆` | WORTH line, score muted |
| exec summary rows, per-bucket callout | stars | WORTH pips |

A producer that stamps an order is claiming to rank. Anything imposed there is
either overwritten downstream or, worse, survives into a consumer that does not
re-order.

## The guard

`tests/test_ranking_coverage.py` parses the source with `ast` and fails if any
display module sorts a frame by `quality_score`, `return_on_risk`, `ev_score` or
`overall_score`, or if any card renders the composite as stars.

The first version of this guard used regex and matched its own explanatory
comments and the docstrings describing the defect — modules that had been fixed
still failed, and modules that were broken passed. Parsing was not a
refinement; the regex version was wrong in both directions.

## What may still consult the composite

Two places, both deliberate, both documented inline, both asserted by the guard:

1. **`filter_and_score`'s funnel sort.** This is SELECTION, not display: it
   decides which leg per symbol survives dedup and which symbols reach the
   top-N, so a candidate it drops is never seen again. Replacing it is a
   behaviour change to what enters the funnel and there is no measured
   alternative — the replacement key tested on 2026-08-09 was a coin flip.
   Swapping a bad key for an unmeasured one is not an improvement. To settle
   it: log both orderings' selections and compare outcomes.
2. **`rank_by_verdict`'s except branch.** The failure-safe. A board rendered in
   a discredited order beats a board that does not render.

## Boards deliberately NOT gated on EV

Lottery and the squeeze long side are convexity plays whose premise is a small
negative expectation bought for a large tail. Applying the EV consistency gate
would empty them on every run — deleting the feature rather than improving it.
They keep the arithmetic gates and their existing display-only framing.

## Verified live

`--ticker AAPL`, 2026-08-10:

```
─ TICKER — 1 of 2 shown ──────────────────────
  1 refused

━ #1/1  AAPL PUT $300  exp 2026-08-28  17d  ◆ WHALE ━━━
  ●○○○ THIN   Prem $3.75   IV 23.9%   OI 2,516 ...
  VERDICT     POSITIVE EV +23/ct  VRP HIGH_PREMIUM
  WORTH       ●○○○ THIN    edge 0.6x its error bar · costs 5% of reward
  History     (Long Put, DTE±10, |Δ|±0.10): n=55 | win 29% | avg -5.9%
```

`POSITIVE EV +23/ct` alone reads as a green light. The card now says the edge is
0.6 of its own error bar, and that your book has run 29% on 55 similar setups.
Three lines that used to disagree now agree.
