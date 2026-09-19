# Hypothesis Sweep Preregistration

Locked 2026-09-19. Copied verbatim from
`docs/superpowers/specs/2026-09-19-hypothesis-sweep-design.md` — that spec
is the source of truth; this file is the pointer `src/hypothesis_sweep.py`
reads by convention (matching `docs/PREREG_RANKER_TEST.md`'s role for
`src/prereg_ranker.py`). Not hand-edited after generation: a new hypothesis
is a new preregistration, not an edit to this one.

Family size: N_TRIALS = 6.

| ID | Category | Question | Current | Variant | Statistic |
|----|----------|----------|---------|---------|-----------|
| H1 | Strategy | Does a Bull Put spread show OOS Sharpe distinguishable from zero-skill under current spread-surface friction? | — | — | DSR(pnl_pct series) |
| H2 | Strategy | Does a Bear Call spread show OOS Sharpe distinguishable from zero-skill under current spread-surface friction? (Currently excluded from auto-log because it measurably fails its own managed-exit breakeven — 59.3% delivered win rate against a 66.7% required rate, per `config.json`'s `filters.credit_spreads` note. This asks whether that gap still holds in a fresh synthetic sample, not whether friction has changed.) | — | — | DSR(pnl_pct series) |
| H3 | Weights | Does replacing the synthetic engine's flat neutral `spread_score` (currently hardcoded 0.5) with a spread-surface-derived score change the `spread` factor's per-ticker-clustered rank IC vs. realized P&L? | `spread_score = 0.5` | `spread_score` from `SpreadSurface.oi_collapsed_relative`, normalized to [0,1] | cluster-bootstrap IC (Bonferroni alpha = 0.05/6) |
| H4 | Entry/exit | Does entry DTE 30 change mean EV-net-of-friction per Bull Put trade vs. entry DTE 45? | `ENTRY_DTE=45` | `entry_dte=30` | DSR, independent comparison |
| H5 | Entry/exit | Does stop-loss multiple 1.5x change realized P&L per Bull Put trade vs. 2.0x? | `STOP_LOSS_MULT=2.0` | `stop_mult=1.5` | DSR(paired difference series) |
| H6 | Gates | Does raising the credit-to-width floor to 0.25 change EV-net-of-friction of gate survivors vs. the current 0.20 floor? | `floor=0.20` | `floor=0.25` | DSR, nested-subset comparison; survivor-count change is descriptive only |

"Survives": DSR ≥ 0.95 (H1, H2, H5) or DSR(variant) ≥ 0.95 **and** exceeds
DSR(current) (H4, H6), or a Bonferroni-adjusted CI excluding 0 (H3).

Iron Condor is explicitly out of scope for this batch. Any hypothesis added
after seeing H1-H6's results is a new batch with its own preregistration
and its own N_TRIALS — never an amendment to this one.

Implementation: `docs/superpowers/plans/2026-09-19-hypothesis-sweep.md`.
