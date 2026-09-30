# Friction-to-credit admission sweep — RESULT

**Prereg:** `docs/PREREG_FRICTION_SWEEP_20260928.md` (committed `17cfce8`, before any return was computed)
**Run:** 2026-09-28. Accepted Bull Put cohort, n=2,566, 218 ticker-day clusters.

## Result: H1 fails, H2 fails decisively. DO NOT tighten the gate.

| threshold | n | clusters | mean ret/CAR | 95% CI (clustered) | total return per unit of original capacity |
|---|---|---|---|---|---|
| 0.075 | 298 | 45 | +3.14% | [-8.76%, +12.70%] | **0.36%** |
| 0.100 | 660 | 83 | +4.59% | [-4.60%, +12.22%] | **1.18%** |
| 0.150 | 1,402 | 151 | +4.29% | [-3.33%, +10.63%] | **2.34%** |
| 0.200 | 2,070 | 193 | +3.73% | [-3.18%, +10.09%] | **3.01%** |
| **none (current 0.25 gate)** | **2,566** | **218** | **+4.24%** | [-2.29%, +9.73%] | **4.24%** |

**H1 — tightening raises the per-trade mean: FAILS.** The means are flat and
non-monotone (3.14 / 4.59 / 4.29 / 3.73 / 4.24). Every CI contains zero and
they overlap heavily. Friction does not discriminate *within* the accepted
cohort.

**H2 — tightening raises total return: FAILS DECISIVELY.** Total return per
unit of original capacity falls **monotonically** as the threshold tightens:
4.24% -> 3.01% -> 2.34% -> 1.18% -> 0.36%. Tightening strictly destroys return
by removing trades without improving the ones that remain.

## Why, and why this is good news

**The existing gate at 0.25 has already removed the harmful cases.** Measured
separately: candidates refused by the friction gate average **-4.54%** across
80,200 rows and 1,919 clusters — the worst cohort in the corpus. The gate
catches them. What survives is homogeneous with respect to friction (p10 0.051,
median 0.100, p90 0.211), and that residual variation carries no information
about returns.

So the finding is that **`auto_log.max_friction_to_credit = 0.25` is
well-calibrated and should be left exactly as it is.** The hypothesis that
drove this sweep — that friction-to-payoff ratio was the untapped lever — is
wrong, and it was wrong in a specific way worth recording: the lever had
already been pulled.

## Provenance and limits, as preregistered

- The hypothesis came from inspecting this data, so this was never going to be
  confirmatory. It is a powered falsification of a mechanism, which is the
  useful direction for a wrong idea.
- The 0.075 cell has 45 clusters and is underpowered; its 3.14% is not
  evidence of anything. The H2 conclusion does not depend on it — the
  monotone decline in total return is driven by trade count, which is exact.
- Friction is computed from stored entry quotes, not realised fills.
- One regime, 26 trading days.

## What this authorises

Nothing. No config change. Specifically: **do not tighten
`max_friction_to_credit`**, and do not loosen it either — this sweep tested
only tightening.
