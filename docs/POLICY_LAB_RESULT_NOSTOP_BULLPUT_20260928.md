# Policy Lab Result

- git: `aa672f13b33b712856a2c84ffada62bfcc2338b1`
- generated: 2026-09-28T19:29:25.574246+00:00
- corpus: A  |  costs: cross  |  seed: 0
- trials in grid (n_trials for DSR): **6**
- clustering unit: (symbol, entry_date)
- rows: {'loaded': 113442, 'dropped': {'fewer_than_two_marks': 1935, 'sign_contradicts_family': 4, 'not_closed': 5060}, 'spread_imputed_frac': 1.0}

- **calibration: 89439/113442 positions reproduce their recorded P&L (21.2% fail)**
  - by reason/strategy:
    - `Bull Put`: 89439/113442 reproduce (21.2% fail)

- **100.0% of path points had no observed two-sided quote** and were widened from mid by the modelled half-spread instead of a real quote (see `CostModel.imputed_half_spread`).

- **Detection floor: ~+0.012 of capital at risk per trade.** Validated by planting a synthetic effect: +0.008 CAR/trade gives family-wise p=0.474 (invisible), +0.012 gives p=0.000 (detected). A null result below this line is not "no effect" — see the limitations below.

- whole-grid max-stat p ("is the single best policy in this grid real"): **0.0135**

## Results

| policy | verdict | clusters | mean d (cross) | mean d (surface) | 95% CI | fwer_p | dsr | pbo | skew_d | var ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| `nostop_tp050_slnone` | **reject** | 2053 | +0.0011 | +0.0011 | not computed | 0.2530 | 0.095 | 0.000 | +8.92 | 0.28 |
| `nostop_tp050_sl3.0` | **reject** | 2053 | +0.0001 | +0.0001 | not computed | 0.9940 | 0.100 | 0.000 | -2.64 | 0.23 |
| `nostop_tp050_sl1.5` | **reject** | 2053 | -0.0012 | -0.0012 | not computed | 0.3610 | 0.106 | 0.000 | +6.19 | 0.39 |
| `nostop_tp065_slnone` | **reject** | 2053 | +0.0010 | +0.0010 | not computed | 0.4620 | 0.099 | 0.000 | +8.37 | 0.30 |
| `nostop_tp065_sl2.0` | **reject** | 2053 | -0.0000 | -0.0000 | not computed | 0.9715 | 0.051 | 0.000 | -8.27 | 0.10 |
| `nostop_tp035_slnone` | **reject** | 2053 | +0.0013 | +0.0013 | [-0.0013, +0.0040] | 0.0135 | 0.102 | 0.000 | +7.41 | 0.33 |

## What this run cannot tell you

- It is **not** a selection result. Nothing here ranks candidates, and a
  positive verdict is evidence that a policy knob paid — **not evidence that
  the system has an edge**.
- Corpus A spans 26 trading days of a single low-vol regime, and Corpus B
  covers the same period. This is **one regime**: SPY's worst drawdown
  across the whole life of the book was -4.49%, against a long-run drawdown
  of -55.2%, so no result here says anything about a volatility event.
- Corpus A's own recorded P&L reproduces at only ~78% under mid costs
  (calibrate.py), with stop-loss exits failing at roughly double the rate of
  other exit reasons. Corpus A's recorded P&L must **never** be quoted as
  ground truth.
- Path depth averages 5.75 marks per contract, so coarse threshold questions
  are answerable and intraday timing questions are not.
- Short Put has zero terminal priced rows in Corpus A and n=20 in Corpus B.
- A null result here means "no knob worth more than ~1.2% of capital at
  risk per trade was found at this grid size" — it does **not** mean "no
  effect". See the detection floor above.
- Grid cells with `time_exit_dte` of 21 or 28 on Corpus A are labelled
  `[first-mark exit]` below: at a median entry DTE of 17, those cells exit
  at the first available mark for most positions rather than measuring
  time decay.
