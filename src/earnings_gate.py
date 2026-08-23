"""Refuse to sell premium across a dated binary event.

Why this exists: on 2026-08-18 the feeder opened a WMT Bull Put two days before
Walmart reported. It was stopped out on the morning of the report for -$274.50,
-254% of the credit received. **The earnings date was already in this repo's
own cache** (`src/dolt_earnings.py` had WMT 2026-08-20) and the correct
predicate already existed (`holds_through_earnings`) — but only
`src/dolt_cohort.py`, an offline analysis module, ever called it. The live scan
path had no earnings gate at all.

What it did have was `earnings_buffer_days`, which sets a DISPLAY flag on the
test ``|expiration - earnings| <= 5``. That asks whether the CONTRACT EXPIRES
near the event, not whether the event happens while the position is held; for
WMT the gap was 29 days, so nothing lit up. And when that flag does fire it
*raises* the candidate's score in Premium Selling mode — "sellers: high crush =
opportunity". The nearest thing to an earnings guard ranked such a candidate
higher.

Measured 2026-08-20 over 543 closed credit trades since 2026-05-01, splitting
on whether an earnings date fell between entry and expiration:

    held through   n=78   win 43.6%   equal-weighted PF 0.842  CI [0.29, 2.08]
    clear of it    n=73   win 67.1%   equal-weighted PF 2.652  CI [1.41, 5.83]
    UNKNOWN       n=305   win 55.7%   equal-weighted PF 0.888  CI [0.64, 1.24]

The CLEAR interval is one of only two in this repo that exclude 1. Read it as a
lead, not a verdict: it is in-sample, the split was chosen after seeing the WMT
loss, and 78 trades cluster in time. What justifies the gate is not that
interval — it is that a dated, public, binary event is an uncompensated risk
this system was taking without ever looking, on data it already had.

**UNKNOWN is 72% of the population**, which is the number that shaped this
module. The cache covers about a quarter of the book's symbols, so a two-state
gate would be silently inert on three trades in four while looking like it was
working. That is the partial-silence failure this repo has hit before, so the
verdict is deliberately three-valued and the unknowns are counted and logged.
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .utils import is_short_position

logger = logging.getLogger(__name__)

#: The position is open when a known earnings date passes.
THROUGH = "through_earnings"
#: The cache reaches past this trade and reports no event inside it.
CLEAR = "clear_of_earnings"
#: Nothing is known about this holding period — NOT the same as clear.
UNKNOWN = "earnings_unknown"
#: No announced date, but the symbol's own cadence puts its next report inside
#: the holding period. An ESTIMATE, kept as its own verdict so it can never be
#: read as an observation.
PROJECTED_THROUGH = "projected_through_earnings"
#: Cadence is regular and the projected report falls outside the window.
PROJECTED_CLEAR = "projected_clear_of_earnings"

# Projection guards, all measured 2026-08-21 against the 16 symbols whose next
# report was already announced, with the answer hidden:
#
#   regular cadence   n=10  median |error| 1 day    worst 8   9/10 within 7d
#   irregular/stale   n= 6  median |error| 19 days  worst 77  1/6  within 7d
#
# The two properties below are what separate those buckets, and both are
# knowable in advance from the symbol's own history.
_MIN_HISTORY = 8          # fewer reports is a coincidence, not a cadence
_MAX_GAP_SPREAD_DAYS = 20  # SJM 25 -> off by 15; GME 91 -> off by 77
_MAX_STALE_DAYS = 120     # past a quarter, a report has gone unrecorded
_BUFFER_DAYS = 7          # covered 9 of 10 regular symbols

#: Where `src/dolt_earnings.py` caches the calendar.
DEFAULT_CACHE = "data/dolt_options.db"

#: How far ahead to look. ``expiration`` is the conservative reading: a
#: position CAN be held to expiry, and no exit rule is a guarantee.
#: ``time_exit`` stops where the DTE rule force-closes the position, which is
#: the only exit that fires without reading a mark. On the book the two differ
#: sharply — to expiration refuses 14.7% of credit trades, to the time exit
#: 1.3% — so the choice is a real one and is recorded in config.
_HORIZONS = ("expiration", "time_exit")

_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "horizon": "expiration",
    "cache_path": DEFAULT_CACHE,
    "projection": "off",
}

# Credit structures name themselves; none of these words is one
# `is_short_position` detects, so they are listed explicitly — the same list
# capital_risk.py keeps, and for the same reason.
_CREDIT_KEYS = ("bull put", "bear call", "iron condor", "credit",
                "cash-secured put", "cash secured put")
# A debit spread contains "spread" and is long premium; match on the structure
# name, never on a bare word.
_LONG_KEYS = ("bull call", "bear put", "long call", "long put", "calendar")


def applies_to(strategy_name: Optional[str]) -> bool:
    """True for short-premium structures, which is all this gate claims about.

    Selling a spread across an event is being short a gap. BUYING one is being
    long it — whatever is wrong with that (IV crush) is a different trade with
    different evidence, and this gate does not pretend to have measured it.
    """
    name = (strategy_name or "").strip().lower()
    if not name:
        return False
    if any(key in name for key in _LONG_KEYS):
        return False
    if any(key in name for key in _CREDIT_KEYS):
        return True
    return bool(is_short_position(name))


def load_earnings_gate_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Read `auto_log.refuse_through_earnings` and friends, falling back per key.

    A config without the key has not opted in, so the gate is OFF: absent
    configuration must never change what the ledger does.
    """
    out = dict(_DEFAULTS)
    block: Mapping[str, Any] = {}
    if isinstance(config, Mapping) and isinstance(config.get("auto_log"), Mapping):
        block = config["auto_log"]
    out["enabled"] = bool(block.get("refuse_through_earnings", False))
    horizon = str(block.get("earnings_horizon", _DEFAULTS["horizon"]))
    # An unrecognised horizon falls back to the CONSERVATIVE one rather than
    # to the permissive one: a typo must not quietly widen what gets logged.
    out["horizon"] = horizon if horizon in _HORIZONS else "expiration"
    path = block.get("earnings_cache_path") or DEFAULT_CACHE
    out["cache_path"] = str(path)
    # off / report / refuse. An unrecognised value falls back to OFF, never to
    # refuse: a typo must not silently start turning trades away on an estimate.
    mode = str(block.get("earnings_projection", "off")).strip().lower()
    out["projection"] = mode if mode in ("off", "report", "refuse") else "off"
    return out


def horizon_end(expiration: Optional[str], time_exit_dte: Any,
                horizon: str) -> Optional[str]:
    """Last date the position can still be open, as ISO yyyy-mm-dd.

    ``None`` when the expiration cannot be parsed — an unreadable date is not a
    reason to conclude anything, and the caller turns it into UNKNOWN.
    """
    try:
        exp = _dt.date.fromisoformat(str(expiration)[:10])
    except (TypeError, ValueError):
        return None
    if horizon != "time_exit":
        return exp.isoformat()
    try:
        dte = int(time_exit_dte)
    except (TypeError, ValueError):
        dte = 0
    return (exp - _dt.timedelta(days=max(0, dte))).isoformat()


def cached_earnings_dates(symbol: str, cache_path: str = DEFAULT_CACHE) -> List[str]:
    """Earnings dates already cached for a symbol. NEVER fetches.

    `dolt_earnings.earnings_dates` queries DoltHub on a cache miss. This runs
    inside `log_trade`, so it reads the table directly and returns nothing
    rather than putting an HTTP call inside a ledger write. A missing file or a
    missing table is empty, which the caller reads as UNKNOWN.
    """
    if not symbol:
        return []
    try:
        conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            "SELECT date FROM earnings_cal WHERE UPPER(symbol) = ? "
            "AND date IS NOT NULL ORDER BY date", (symbol.upper(),)).fetchall()
        return [str(r[0]) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def classify(dates: Sequence[str], entry_date: str,
             end_date: Optional[str]) -> str:
    """THROUGH, CLEAR or UNKNOWN for one holding period.

    The window is ``(entry_date, end_date]`` — an event ON the entry date is
    already public and priced in, an event on the last day is still held
    through.

    UNKNOWN when no cached date reaches the entry date. That covers both "never
    fetched" and "cache ends before this trade", and keeping them together is
    the point: a stale cache reports no events inside any future window, which
    is indistinguishable from safety unless it is named.
    """
    entry = str(entry_date)[:10]
    if not end_date:
        return UNKNOWN
    end = str(end_date)[:10]
    known = [str(d)[:10] for d in dates if d]
    if any(entry < d <= end for d in known):
        return THROUGH
    if not any(d >= entry for d in known):
        return UNKNOWN
    return CLEAR


def project_next_earnings(dates: Sequence[str],
                          today: Optional[_dt.date] = None,
                          min_history: int = _MIN_HISTORY,
                          max_spread_days: int = _MAX_GAP_SPREAD_DAYS,
                          max_stale_days: int = _MAX_STALE_DAYS) -> Optional[str]:
    """The symbol's next report, projected from its own cadence, or None.

    None whenever the estimate would not be trustworthy, and the three reasons
    are all properties of the history rather than of the outcome:

    * fewer than ``min_history`` reports — a cadence needs to be demonstrated;
    * gaps that vary by more than ``max_spread_days`` — the irregular bucket
      missed by a median of 19 days and by 77 at worst;
    * a last known report more than ``max_stale_days`` ago — at least one
      report has happened without being recorded, so the anchor is wrong.

    Returns the first projected date strictly after ``today``: stepping forward
    by whole quarters matters when the calendar is a little behind.
    """
    today = today or _dt.date.today()
    parsed: List[_dt.date] = []
    for value in dates:
        try:
            parsed.append(_dt.date.fromisoformat(str(value)[:10]))
        except (TypeError, ValueError):
            continue
    parsed.sort()
    past = [d for d in parsed if d <= today]
    if len(past) < min_history:
        return None

    window = past[-min_history:]
    gaps = [(window[i + 1] - window[i]).days for i in range(len(window) - 1)]
    if not gaps or (max(gaps) - min(gaps)) > max_spread_days:
        return None

    anchor = past[-1]
    if (today - anchor).days > max_stale_days:
        return None

    step = sorted(gaps)[len(gaps) // 2]
    if step <= 0:
        return None
    projected = anchor + _dt.timedelta(days=step)
    while projected <= today:
        projected += _dt.timedelta(days=step)
    return projected.isoformat()


def classify_with_projection(dates: Sequence[str], entry_date: str,
                             end_date: Optional[str],
                             today: Optional[_dt.date] = None,
                             buffer_days: int = _BUFFER_DAYS,
                             enabled: bool = True) -> str:
    """`classify`, with a cadence projection filling in for a silent calendar.

    Precedence, and the order matters:

    1. An ANNOUNCED event inside the window is THROUGH. An observation always
       beats an estimate.
    2. Otherwise the projection is consulted — including when the announced
       check would have said CLEAR only because the calendar reaches just past
       the entry date. A trade long enough to span the NEXT quarter is exposed
       whether or not that report has been announced yet, and that case is
       invisible to the announced check alone.
    3. Then the announced CLEAR, which is authoritative when the calendar
       carries a real future date.
    4. Otherwise UNKNOWN.
    """
    announced = classify(dates, entry_date, end_date)
    if announced == THROUGH or not enabled or not end_date:
        return announced

    today = today or _dt.date.today()
    projected = project_next_earnings(dates, today=today)
    if projected is None:
        return announced

    entry = str(entry_date)[:10]
    try:
        end = _dt.date.fromisoformat(str(end_date)[:10]) + _dt.timedelta(
            days=max(0, int(buffer_days)))
    except (TypeError, ValueError):
        return announced
    if entry < projected <= end.isoformat():
        return PROJECTED_THROUGH
    # An announced future date is a real observation about the next event and
    # outranks a projection that agrees with it.
    return announced if announced == CLEAR else PROJECTED_CLEAR


def verdict_for_trade(strategy_name: Optional[str], symbol: Optional[str],
                      entry_date: str, expiration: Optional[str],
                      time_exit_dte: Any, cfg: Mapping[str, Any]) -> Optional[str]:
    """The gate's answer for one candidate, or None when it does not apply.

    None means "this gate has nothing to say" — disabled, or long premium —
    and is deliberately distinct from CLEAR, which is a measured all-clear.
    """
    if not cfg.get("enabled") or not applies_to(strategy_name):
        return None
    end = horizon_end(expiration, time_exit_dte, str(cfg.get("horizon")))
    dates = cached_earnings_dates(str(symbol or ""),
                                  str(cfg.get("cache_path") or DEFAULT_CACHE))
    mode = str(cfg.get("projection", "off"))
    return classify_with_projection(dates, entry_date, end,
                                    enabled=mode in ("report", "refuse"))


def refuses(verdict: Optional[str], cfg: Mapping[str, Any]) -> bool:
    """Whether a verdict should stop the trade.

    An announced event always does. A PROJECTION only does under
    ``earnings_projection: "refuse"`` — it is an estimate, and the deliberate
    default is to count and print it for a while first, so its behaviour on the
    live board is observed before it starts turning trades away.
    """
    if verdict == THROUGH:
        return True
    if verdict == PROJECTED_THROUGH:
        return str(cfg.get("projection", "off")) == "refuse"
    return False
