# Gate separation — RESULT, 2026-09-29

Registration: `docs/PREREG_GATE_SEPARATION_20260929.md` (two amendments, both
recorded there).
Reproduce: `PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.gate_separation_report`

Corpus `160b723b06ede2b3` — 416,914 candidate rows, 282,563 closed positions,
2026-08-19 .. 2026-09-28. **A run against a different fingerprint is not
comparable to this one.**

## Headline

The live gate **does** separate forward outcomes, on every strategy that can be
measured. Family-wise p = 0.0000 by max-statistic sign-flip permutation over
the four claimants.

| strategy | cells | symbols | d_auc | 95% CI | win.mean | 1st half | 2nd half | LOSO | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Long Put | 179 | 54 | 0.2143 | [0.126, 0.296] | 0.062 | 0.182 | 0.273 | yes | **separates** |
| Bear Call | 398 | 53 | 0.2435 | [0.187, 0.296] | 0.090 | 0.213 | 0.231 | yes | **separates** |
| Long Call | 134 | 50 | 0.1302 | [0.031, 0.228] | 0.028 | 0.206 | 0.006 | yes | **separates** |
| Bull Put | 287 | 49 | 0.2327 | [0.182, 0.282] | 0.089 | 0.204 | 0.222 | yes | **separates** |
| Iron Condor | 4 | 2 | 0.3333 | [0.167, 0.500] | 0.096 | −0.500 | 0.500 | — | insufficient |
| Short Put | 0 | 0 | — | — | — | — | — | — | **not measurable at all** |

`d_auc` = P(passed beats refused) − 0.5, within `(symbol, expiration, scan
day)`, averaged over symbols. `+0.23` means a passed contract beats a refused
one on the same underlying, same expiry, same day about **73%** of the time.

Guards: negative control PASS (worst |null mean| 0.00344 against a 0.01 bar);
masked-permutation equivalence asserted in `tests/test_gate_separation.py`.

## The Long Put "inversion" was an artifact. It is retracted.

The finding that motivated this whole test — Long Put's gate passing worse
contracts than it refused — **does not survive a matched comparison.** It
reverses.

| Long Put | passed | refused |
|---|---|---|
| pooled, all 4,720 cells | −0.1845 | −0.0214 |
| matched, 93 both-arm cells | **+0.0878** | **−0.1544** |

Simpson's paradox, and the mechanism is visible in the cell counts: of 4,720
Long Put cells, 1,049 are passed-only and 3,578 refused-only. Only 93 hold both
arms. So ~98% of the pooled difference is a comparison between *different
underlyings on different days* — it measures where and when the gate chooses to
act, not which contract it picks. Condition on the opportunity and the sign
flips.

**Anyone re-deriving this from `AVG(pnl_pct) GROUP BY strategy_name,
gate_passed` will get the retracted answer.** That query is in this repo's
history as the thing that produced a false alarm.

## What this does and does not establish

**Does:** within an opportunity the gate is already acting on, it picks the
better contract, consistently, across four strategies, stable to dropping any
single symbol, replicating in both halves of the window.

**Does not:** say the gate's *choice of opportunities* helps. The matched design
deliberately conditions that away — it is the confound, so it cannot also be
the result. The two jobs are separable and only the first is measured here.

**Does not:** say any of these strategies make money. The book still trails SPY
by ~$7,700 and the sized era's profit factor is 0.63. A gate that ranks well
inside a losing population produces a better-chosen loss.

**Does not:** authorise a configuration change. See the registration's
provenance section — the hypothesis was formed on this same corpus. The only
clean sample is scans recorded after 2026-09-29.

## Caveats a reader should hold

1. **`pnl_pct` is return on PREMIUM, not on capital at risk.** Unbounded below
   for credit structures: Bull Put reaches −168.231, Bear Call +659.0. The
   primary statistic is rank-based and immune; `win.mean` is winsorized at
   1/99%. Magnitudes are **not comparable across strategies** — 0.09 of premium
   on a Bear Call and 0.06 on a Long Put are different amounts of money.
2. **Long Call is the weakest result.** Its second half is 0.006 — positive,
   but barely. Treat it as the one most likely to evaporate.
3. **Matched cells are a small slice**: 179 of 4,720 Long Put cells, 287 of
   4,045 Bull Put cells. The result describes the gate's behaviour on contested
   opportunities, which is not a random sample of all opportunities.
4. **Iron Condor cannot be measured** (2 symbols) and is excluded from the
   family. Including it set the family-wise p to 0.50 single-handedly, because
   a 2-symbol sign flip leaves its |mean| at its observed value 49% of the time
   — measured, see Amendment 1.

## One defect found on the way — fixed

`candidate_marks.legs_for` and `marking_legs` decided a single leg's side by
`strategy.startswith("Short")`, reading `strategy_name` alone. That column is
NULL on every single-leg row **by design** — `candidate_record` says discovery
boards carry `type='call'|'put'`, an option type and not a strategy, and `mode`
exists in the schema precisely so the strategy can be derived. `family_for`
already derives it; these two functions did not.

So a Premium Selling short put fell through to the `buy` default and
`entry_price_for` returned a **negative** price — a debit paid, for a position
that receives a credit. `pnl_pct` branches on exactly that sign, so it would
have been scored as a long put. Measured on real rows: `DIA side=buy
entry=-5.585` before, `side=sell entry=+5.515` after.

**Latent, not active**: `open_positions` routes `family == "short_premium"` to
`UNSUPPORTED` with reason `needs_spot_and_delta` before anything is priced, so
no corrupted row exists today. It becomes real the moment short-premium marking
is enabled. Same shape as the inverted `pop_score` for sellers and the bear
calls named "Bull Put" — a label and a geometry that disagree, with nothing
comparing them.

Fixed on `fix/candidate-marks-side-from-mode`, which makes both call sites
follow `family_for`'s discipline. Behaviour-preserving for every row currently
simulated.

**An earlier draft of this document called the blank `strategy_name` itself a
write-side defect. It is not — it is the recorder's intended design, and
Amendment 2's derivation is the recovery the schema was built for.**

## Short Put is completely dark

All 37,788 Premium Selling candidates are `UNSUPPORTED` — 100%, by design, the
shadow engine cannot derive exit rules for naked short premium. Combined with
zero real Short Put entries in the ledger, the strategy has **no forward
outcomes from either corpus**, while being the second-largest refusal
population (8,103 `negative_ev` refusals in the week after 2026-09-22).

It is not doing badly. It is not doing well. It is unmeasured, and nothing
currently in the system will change that.

## Registered forward test

The only uncontaminated sample is scans after the freeze. Re-run against a
corpus whose `ts_min` is ≥ 2026-09-30 and require the same four verdicts. Until
that replicates, this result describes a corpus, not a gate.
