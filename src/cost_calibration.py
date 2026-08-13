"""The tenor range this system's cost thresholds were actually measured on.

Every friction threshold here was calibrated on the Dolt universe, which is
DTE 10-67: WORTH's 5%/10% bands (`worth.py`, measured 2026-08-04), the
auto-log gate's 25% (`config.auto_log.max_friction_to_credit`, tightened
2026-08-06 off a 400-trade sample), and `candidate_verdict`'s 25%. None of them
declared a range, so all three were applied at any tenor and read as
authoritative everywhere.

`docs/OPTIONSDX_RESULTS_20260811.md` measured what happens outside it, on SPY
2010-2023 with all expirations available for the first time:

    band        fric median%   % over the 25% gate
    10-25            3.6              0.8
    25-60            4.8              1.2
    60-120           4.2              0.3
    120-250          7.3              3.5
    250-500         13.0             26.9
    500-1000        23.6             46.0

Two different regimes. Inside the band the gate is a backstop that almost never
binds; past 250 DTE it silently becomes the dominant filter, and the WORTH
bands become *unreachable* — a median candidate is capped at THIN by cost
alone, so STRONG and CLEAR cannot be earned there at any quality. A grade that
quietly stops meaning anything is worse than one that declines to answer.

WHAT THIS DOES NOT DO
---------------------
It changes no threshold. A 23.6% round trip is a real cost and is still
refused. This only stops an out-of-range reading being reported as though the
model applied to it.

WHY THE CEILING IS 67 AND NOT 250
---------------------------------
The H3 measurement above suggests 68-250 behaves like the calibrated band, and
it is tempting to extend the ceiling there. It is SPY-only. SPY is the
tightest-spread instrument in existence — 3.6% median bid-ask against 16-21%
for the rest of the universe — and there is no long-dated data for the other
122 symbols anywhere on disk. Extending a multi-symbol threshold on
single-symbol evidence is the same generalisation this guard exists to prevent.

Raise the ceiling when multi-symbol long-dated data exists, not before.

Costs nothing today: all 972 logged trades are DTE 8-59, and the live filters
cap at 45 (60 for irons). The guard fires only where the book does not
currently go — which is exactly where `docs/OPTIONSDX_RESULTS_20260811.md` H3
suggests someone might next be tempted to.

Dependency-free (stdlib only), for the same reason `paths` and
`ledger_filters` are: it is imported by the ledger and by the grader, and a
calibration fact must never drag a heavy import into a light module.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional

# The thresholds were fitted on DTE 10-67. Only the CEILING is enforced.
#
# A floor was written first and removed: there is no evidence the cost model
# breaks at short tenors — 10-25 DTE carries the LOWEST friction of any band
# measured (3.6% median) — and `config.filters.min_days_to_expiration` already
# sets a minimum for its own reasons. A second floor here refused near-dated
# candidates under a cost-model heading that the data does not support, which
# is the same "authoritative number outside its evidence" this module exists
# to stop.
CALIBRATED_MAX_DTE = 67
CALIBRATED_DTE = (10, CALIBRATED_MAX_DTE)   # the measured band, for reference

OUT_OF_RANGE_REASON = (
    f"tenor beyond the cost model's calibrated range "
    f"(measured to DTE {CALIBRATED_MAX_DTE}); friction thresholds were never "
    f"measured here — see docs/OPTIONSDX_RESULTS_20260811.md"
)

_DTE_KEYS = ("dte", "days_to_expiration", "entry_dte")


def _as_date(value: Any) -> Optional[_dt.date]:
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def entry_dte(row: Dict[str, Any]) -> Optional[int]:
    """Days to expiration at ENTRY, from whatever the row happens to carry.

    Returns None when it cannot be told. None is not zero: zero would read as
    an expiring contract and ungrade the entire book.
    """
    for key in _DTE_KEYS:
        v = row.get(key)
        if v is not None:
            try:
                return int(float(v))
            except (TypeError, ValueError):
                pass
    exp = _as_date(row.get("expiration"))
    if exp is None:
        return None
    # `date` is the entry date on a ledger row. Falling back to today matches
    # a live scan row, which is priced for entry now.
    start = _as_date(row.get("date")) or _dt.date.today()
    return (exp - start).days


def in_calibration(dte: Optional[int]) -> bool:
    """True unless the tenor is LONGER than anything the thresholds were fitted on.

    Ceiling only — see `CALIBRATED_MAX_DTE`. Short and even negative tenors
    pass: an already-expired contract is a different defect with its own
    validation, and this guard has nothing to say about it.

    An unknown tenor counts as in-range. Ungrading everything whose DTE cannot
    be read would be a far larger behaviour change than this guard is entitled
    to make, and a row carrying no tenor is usually a synthetic or a test
    fixture rather than a long-dated trade.
    """
    if dte is None:
        return True
    return int(dte) <= CALIBRATED_MAX_DTE
