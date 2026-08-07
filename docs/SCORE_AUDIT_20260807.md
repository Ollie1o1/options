# Score, EV and squeeze-board audit — 2026-08-07

An end-to-end read of how `quality_score`, `ev_per_contract` and the squeeze
board's columns are computed, against what each is presented as. The question
was not "is the scorer profitable" — that is settled and negative, see
`docs/TRUST_AUDIT_20260803.md` — but the narrower one: **does any displayed
number mean something other than its label says?**

Six did. Each is fixed below with the measurement that justified it, and the
tests that pin it are in `tests/test_score_audit_20260807.py`.

One finding is much larger than the others and is listed first.

---

## 1. A quarter of `quality_score` was assigned by an accident of arithmetic

`load_ic_adjusted_weights` blends the config weights with weights derived from
component ICs measured on the paper ledger:

```
final = 0.7 * config_weight + 0.3 * (component_IC / ic_total)
```

`ic_total` was the sum of IC **over the components that cleared p < 0.10**.
On 2026-08-07 exactly one component cleared it — `theta`, at IC +0.082,
p = 0.021. With one survivor that ratio is `ic/ic = 1.0` by construction, so
theta collected the entire 0.30 reallocation:

| | base weight | live weight | share of composite |
|---|---|---|---|
| theta | 0.0197 | **0.3138** | **24.3%** |

A 16× lift. The code's own comment two hundred lines below still describes the
intended balance as "profitability (pop+rr+ev=30%)"; live, `pop+rr+ev` came to
**7.2%** while theta alone came to 24.3%.

Two properties follow, and both are disqualifying for something calling itself
an IC-derived weight:

```
LIVE   theta weight = 0.3138  (24.3% of composite)

with momentum also crossing p=0.10 (theta's own evidence UNCHANGED):
       theta weight = 0.2323  (18.4%)
       theta lost 26.0% of its weight because an unrelated component became eligible

with theta's measured IC DOUBLED (0.082 -> 0.165), nothing else changed:
       theta weight = 0.3138   (change: +0.0000)
```

The weight does not respond to its own evidence, and does respond to somebody
else's. It was transmitting the survivor count, not the IC.

**Fix.** The denominator now spans every candidate component rather than only
the eligible ones. The p-gate keeps deciding *eligibility*; the IC magnitudes
decide *the split*. Both properties are restored, and theta falls to 13.3% of
the composite.

**What this is and is not proved to do.** It is proved to make the weight a
function of the evidence. It is **not** proved to improve returns, and the
honest measurement says it changes them very little:

```
walk-forward, expanding train window, scored on trades that came after
cut date      n_tr  n_te   OOS IC survivors   OOS IC all-cand
2026-05-27     245   564           -0.0037           -0.0037
2026-06-10     323   486           +0.0542           +0.0417
2026-06-18     397   412           +0.0486           +0.0611
2026-07-07     467   342           +0.0330           +0.0322
2026-07-16     636   173           +0.0596           +0.0401

mean OOS IC:  survivor-sum +0.0383   all-candidate +0.0343   difference -0.0040
```

The difference is far inside noise — the test sets are nested and overlapping,
and an IC standard error at n = 173–564 is roughly 0.04–0.08. The two rules are
indistinguishable out of sample. Note also what an in-sample comparison would
have shown: the old weights score **better** on the full ledger (+0.094 vs
+0.081), which is exactly what you expect from weights fitted to maximise IC on
that ledger. That number is circular and is not evidence for the old rule.

Rank correlation between the old and new composite over the 809 closed trades
with stored components is **0.933**, so roughly 7% of the ordering moves.

### The larger thing this exposed, which is not fixed here

The weights being blended are themselves fitted on a sample whose own headline
verdict, stored in `ic_weights_cache.json`, is:

> `"NO SIGNIFICANT EDGE detected (IC=-0.03, p=0.433)"`

and whose quintile table reports the **top** quintile of the shipped score as
the worst performer:

| quintile | n | win rate | avg return |
|---|---|---|---|
| 1 (lowest score) | 165 | 43.6% | −1.1% |
| 5 (highest score) | 165 | **41.2%** | **−5.2%** |

Fitting component weights on a sample where the composite does not rank is
fitting to noise, whichever denominator is used. That is a calibration-policy
decision, not a bug, so it is reported rather than changed. The existing
guidance stands: rank by net-of-cost EV (`rank_by_verdict`), and treat
`quality_score` as a tiebreak.

---

## 2. A refused EV displayed as a neutral one

`ev_per_contract` is set to `NaN` deliberately when the basis cannot be trusted
— HV missing, or a realized/implied vol gap too wide for both to describe one
market (`trade_analysis.implausible_vol_gap`, the guard that caught the MSFT
+$4,664-on-a-$5-edge case). That is an **absent** basis, not a zero edge.

`format_decision_zone` pushed it through a sign test anyway:

```
  VERDICT     FLAT EV +nan/ct
```

"FLAT" reads as *no edge either way*. On the very same row, `_verdict_for_row`
returns `INDETERMINATE` — so the two surfaces of one pick disagreed, and the
louder one was wrong.

**Fix.** A refused EV now renders `EV UNAVAILABLE (no trustworthy vol basis)`,
in the muted role, matching the INDETERMINATE verdict. Real values are
unchanged.

---

## 3. The squeeze board's spread cost was 41% impossible

The calls board prints `BE vol` and beside it a `+N`, captioned as *what
crossing the spread costs in vol points*. The breakeven was solved from
`premium + costs`; the `+N` subtracted the **vendor's** reported
`impliedVolatility`, which is quoted against a different price than the mid the
breakeven is built on. So `+N` was the spread cost **plus the vendor's
disagreement with the mid**.

Measured over 7,994 archived CBOE calls at DTE ≥ 60 — the board's own floor:

| | old (vs vendor IV) | new (vs own premium) |
|---|---|---|
| share printing a **negative** cost | **41.4%** | **0.0%** |
| median printed `+N` | 0.32vp | 0.79vp |
| p95 printed `+N` | 4.45vp | 5.62vp |

Old value's absolute error against the spread cost it claimed to be: median
0.48vp, mean 0.83vp, p95 2.71vp. **25.4%** of contracts were off by more than a
vol point, and **42.5%** were off by more than the true cost itself.

Crossing a spread cannot be a discount. Two contracts in five said it was.

**Fix.** `board.breakeven_vol_premium_ref` solves both numbers from the same
premium, so the gap is the spread and nothing else — zero negatives by
construction, and zero when the spread is zero. The live scan reads yfinance
rather than CBOE, where agreement with the mid is weaker still, so the live
error is likely larger than measured here.

---

## 4. The short-interest bonus was pointed at both tails

`quality_score` gained `+0.05` on every contract of a name with short interest
above 20% of float — calls, puts and short premium alike. The squeeze study
measures what heavy short interest does to the *shape* of the forward
distribution, and the shape is one-sided. Rebuilt panel, 810,266 rows, 42
trading days, float at 80% of shares outstanding:

| cohort | n | P(+2σ) | P(−2σ) | median end |
|---|---|---|---|---|
| SI > 20% (bonus fired) | 10,635 | **10.26%** | **3.77%** | −2.55% |
| SI ≤ 20% | 780,964 | 6.87% | 5.05% | +0.08% |

Up-tail **+3.39pp**. Down-tail **−1.28pp**, 95% CI [−2.07, −0.51] on a
moving-block bootstrap over 200 settlement dates — clear of zero.

A long call is paid by the up tail and a long put by the down tail. The bonus
was rewarding puts for a tail measurably *thinner* than the base rate, on
exactly the names that triggered it.

(The raw P(−20%) rate on these names is higher, +18.67pp — they are simply more
volatile, and that volatility is already in the premium via IV. The
σ-normalised tail is the one that decides the sign, for the same reason the
squeeze study uses it.)

**Fix.** `_short_interest_bonus` gives the bonus to long calls only. Premium
Selling, Credit Spreads and Iron Condor get nothing — they are short the tail
that is measurably fatter.

---

## 5. The score explanation named components that carry no score

`explain_quality_score` ranked "top drivers" by a weight table hardcoded inside
itself — `PoP 1.0, EV 1.0, RR 0.8, Liquidity 0.7, Catalyst 0.5` — with no
relation to the weights in force. Against the live weights:

| shown as | its weight here | actual share of the score |
|---|---|---|
| EV | 1.0 (joint top) | **0.5%** |
| Catalyst | 0.5 | **0.4%** |
| EM Real | 0.6 | 2.1% |
| iv_velocity | *no row at all* | **10.6%** (3rd largest) |

So the line explaining the score could name a component contributing four parts
in a thousand, and could never name the third-largest contributor. It also
disagreed with `score_drivers`, computed a few hundred lines away from the real
weights, on the same row.

**Fix.** Drivers are ranked by `weight × value` using the weights actually
loaded, components absent from the weight map are dropped rather than given an
invented weight, and the table gained rows for the vol components that now
carry most of the score.

A second defect in the same function: negatives were ranked by `val * weight`
ascending, which named the *lowest-weight* component as the worst offender —
precisely because it could not matter. They are now ranked by `weight ×
shortfall`, the score actually being given up.

---

## 6. Three different squeezes shared one label

`squeeze_play` is `is_squeezing AND Unusual_Whale`: a **TTM squeeze**
(Bollinger bands inside Keltner channels — volatility *compression*) on a
contract with volume/OI over 1.5. The trade thesis rendered that as:

> Gamma squeeze setup [SQUEEZE]

It is not a gamma squeeze, which is a dealer hedging feedback loop. It is also
not a short squeeze, which is `src/squeeze`, keys off short interest, and has
its own 810k-row evidence base. Three unrelated signals under one word, one of
which carries a `+0.10` score bonus.

**Fix.** Relabelled `Vol compressed + heavy flow [COILED]`. Display only; the
bonus is unchanged, since a vol-expansion read is not directional and the
bonus's direction-blindness is defensible in a way that item 4's was not.

---

## 7. A documented tiebreaker that never ran

`_cross_section_normalize`'s docstring described an EV tiebreaker worth ±0.015,
reading `df["ev"]`. No path in the repo creates a bare `ev` column — the scan
carries `ev_per_contract`, `ev_score` and `ev_gross_per_contract` — so the
branch never executed once.

**Fix.** Removed, rather than repointed at `ev_per_contract`. EV already enters
the raw score through `ev_score`; wiring a second, unmeasured EV tilt into the
display scale would be adding a signal under cover of a bug fix. The mapping is
now exactly what its docstring says: a pure function of the raw score.

---

## Reproduce

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_score_audit_20260807 -v
./scripts/test.sh            # 3,394 tests
```

The panel used in item 4 is rebuilt from local tables in ~84s:

```python
from src.squeeze.backtest import panel as P, DEFAULT_DB, PRICES_DB
rows = P.build(DEFAULT_DB, PRICES_DB)      # 810,266 rows
P.grade(rows, si_scale=1.25, ret5d_scale='as_intended')
```

Note `data/squeeze_panel.pkl` (292MB) and `data/squeeze_panel_partial.pkl`
(256MB) are **iCloud-evicted placeholders** on this machine — reading either
raises `TimeoutError: [Errno 60]` rather than loading. `--rebuild` is the
reliable path until they are materialised.

## Not changed

- **The additive adjustment stack.** ~20 hand-set bonuses and penalties
  (`-0.20` decay, `+0.10` seasonal, `-0.15` gamma ramp, …) are applied after the
  27-component weighted composite, and several are larger than the entire
  spread the composite produces. That is a design question with no measurement
  behind it either way, and changing it is a calibration decision.
- **`quality_score` weights themselves.** Item 1 fixes how the blend arithmetic
  works, not what the weights should be. Re-fitting them needs a sample where
  the composite ranks, which this ledger is not.
- **The squeeze mode remains display-only**, and the study still measures the
  underlying's move, not the option trade.
