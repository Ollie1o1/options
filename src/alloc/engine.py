"""Replay a strategy over historical chains, one day at a time.

The engine may only read the chain for the date it is acting on. Nothing else in
this system prevents look-ahead, so that discipline lives here and is pinned by a
test that deliberately tries to break it.

Three hazards this handles explicitly, all observed in the real data:

  * A ticker can STOP EXISTING mid-sample. FB became META on 2022-06-03; PBCT was
    acquired 2022-04-01. A position open at that point is closed at its last
    available mark and flagged `ticker_ended` — never dropped, because dropping
    would delete exactly the acquisition and delisting outcomes and quietly
    reintroduce survivorship bias into unbiased data.
  * A date can have NO DATA even though the market was open. Those days are
    skipped, not scored as "no opportunity", which would flatter an always-on
    benchmark for free.
  * A quote can be missing or crossed. Those are counted and skipped, never
    modelled at some assumed price.
"""
from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from src.alloc.fills import (Leg, SKIP_CROSSED, SKIP_MISSING, _key as _fkey,
                             fill_with_reason, quotes_from_chain, reverse)
from src.alloc.settle import implied_spot_any, settle
from src.alloc.signals import SignalHistory, passes, snapshot
from src.strategies.spec import StrategySpec

# Structures whose net fill should be a credit. Everything else is a debit.
CREDIT_STRUCTURES = ("bull_put", "bear_call", "iron_condor", "short_put")


@dataclass
class Trade:
    symbol: str
    entry_date: str
    entry_price: float
    capital_at_risk: float
    legs: List[Leg]
    expiration: str
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    stratum: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None


class ChainSource(Protocol):
    def chain(self, symbol: str, date: str) -> List[Dict[str, Any]]: ...


class SqliteChainSource:
    """Reads chains from the Dolt cache. db_path MUST be absolute.

    dolt_options.DEFAULT_CACHE is relative, so a caller running from anywhere
    but the repo root would silently read an empty database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def chain(self, symbol: str, date: str) -> List[Dict[str, Any]]:
        key = (symbol, date)
        if key not in self._cache:
            from src.dolt_options import _cache_read
            self._cache[key] = _cache_read(self.db_path, symbol, date) or []
        return self._cache[key]


def _dte(date: str, expiration: str) -> int:
    try:
        return (_dt.date.fromisoformat(expiration[:10])
                - _dt.date.fromisoformat(date[:10])).days
    except ValueError:
        return 0


def _pick_expiry(chain, date: str, lo: int, hi: int) -> Optional[str]:
    """Nearest expiry inside the DTE window. None when nothing qualifies."""
    exps = {str(c["expiration"])[:10] for c in chain}
    ok = [e for e in exps if lo <= _dte(date, e) <= hi]
    return min(ok, key=lambda e: _dte(date, e)) if ok else None


def _nearest_delta(rows, target: float):
    scored = [r for r in rows if r.get("delta") is not None]
    if not scored:
        return None
    return min(scored, key=lambda r: abs(abs(float(r["delta"])) - target))


def _nearest_wing(rows, short_strike: float, width: float, below: bool):
    """The listed wing strike nearest `short_strike -/+ width`.

    Must lie strictly beyond the short strike on the protective side. Without
    that constraint the nearest listed strike can be the short strike itself,
    producing a zero-width "spread" that has no risk and collects no credit.
    """
    side = [r for r in rows
            if (float(r["strike"]) < short_strike if below
                else float(r["strike"]) > short_strike)]
    target = short_strike - width if below else short_strike + width
    # Tolerance is generous because strike spacing scales with price: SPY's
    # listed strikes gap by up to $15 and NVDA's by $30, so a $5 tolerance
    # excluded exactly the tightest-spread names in the universe. Risk is
    # priced off actual_width(), so taking a wider wing than requested is
    # measured honestly rather than assumed away.
    return _nearest_strike(side, target, max(width * 3.0, width + 1.0))


def _nearest_strike(rows, target: float, tolerance: float):
    """The listed strike closest to `target`, within `tolerance`.

    Requiring an exact `short_strike - width` silently excluded every
    high-priced underlying: this dataset carries roughly 150-200 contracts per
    symbol-day, so on a $500 name the exact strike five dollars away is often
    simply not listed. That biased the study toward cheap, wide-spread names —
    the opposite of what a real book would trade.

    A trader buys the wing that exists, so the engine does too, and the actual
    width is recomputed from the strike it got rather than the one it wanted.
    """
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(float(r["strike"]) - target))
    return best if abs(float(best["strike"]) - target) <= tolerance else None


def select_legs(spec: StrategySpec, chain, date: str,
                rng: Optional[random.Random] = None) -> Optional[List[Leg]]:
    """Choose this structure's legs from the day's chain.

    Returns None whenever the chain cannot support the structure — a missing
    wing is a skipped trade, never a substituted strike.
    """
    entry = spec.entry
    lo, hi = entry.get("dte", [25, 45])
    expiry = _pick_expiry(chain, date, int(lo), int(hi))
    if expiry is None:
        return None
    at_exp = [c for c in chain if str(c["expiration"])[:10] == expiry]
    puts = [c for c in at_exp if str(c["type"]).lower() == "put"]
    calls = [c for c in at_exp if str(c["type"]).lower() == "call"]
    width = float(entry.get("width", 5.0))
    struct = spec.structure

    def _short(rows, target):
        if entry.get("strike_selection") == "random":
            pool = [r for r in rows if r.get("delta") is not None]
            return (rng or random).choice(pool) if pool else None
        return _nearest_delta(rows, target)

    if struct in ("bull_put", "short_put"):
        s = _short(puts, float(entry.get("short_delta", 0.25)))
        if s is None:
            return None
        legs = [Leg(expiry, float(s["strike"]), "put", "sell")]
        if struct == "bull_put":
            wing = _nearest_wing(puts, float(s["strike"]), width, below=True)
            if wing is None:
                return None
            legs.append(Leg(expiry, float(wing["strike"]), "put", "buy"))
        return legs

    if struct == "bear_call":
        s = _short(calls, float(entry.get("short_delta", 0.25)))
        if s is None:
            return None
        wing = _nearest_wing(calls, float(s["strike"]), width, below=False)
        if wing is None:
            return None
        return [Leg(expiry, float(s["strike"]), "call", "sell"),
                Leg(expiry, float(wing["strike"]), "call", "buy")]

    if struct == "iron_condor":
        d = float(entry.get("short_delta", 0.16))
        sp, sc = _short(puts, d), _short(calls, d)
        if sp is None or sc is None:
            return None
        pw = _nearest_wing(puts, float(sp["strike"]), width, below=True)
        cw = _nearest_wing(calls, float(sc["strike"]), width, below=False)
        if pw is None or cw is None:
            return None
        return [Leg(expiry, float(sp["strike"]), "put", "sell"),
                Leg(expiry, float(pw["strike"]), "put", "buy"),
                Leg(expiry, float(sc["strike"]), "call", "sell"),
                Leg(expiry, float(cw["strike"]), "call", "buy")]

    if struct in ("long_call", "long_put"):
        typ = "call" if struct == "long_call" else "put"
        rows = calls if typ == "call" else puts
        s = _nearest_delta(rows, float(entry.get("target_delta", 0.40)))
        return [Leg(expiry, float(s["strike"]), typ, "buy")] if s is not None else None

    return None


def crossing_cost(legs: List[Leg], quotes) -> Optional[float]:
    """What crossing the spread costs, per share, versus trading at mid.

    Measured at 50% of the median credit across 115 symbols — the single
    largest drag found in this system, larger than every prior estimate.
    Exposing it lets a strategy refuse trades whose friction eats the credit,
    which is the same guard the live auto-log applies.
    """
    total = 0.0
    for leg in legs:
        quote = quotes.get(_fkey(leg.expiration, leg.strike, leg.type))
        if quote is None:
            return None
        bid, ask = quote
        if bid is None or ask is None or ask < bid:
            return None
        total += (float(ask) - float(bid)) / 2.0
    return round(total, 4)


def actual_width(legs: List[Leg]) -> float:
    """Widest same-type strike gap in the structure.

    An iron condor's risk is the wider of its two sides, not their sum: only one
    side can be breached at expiry.
    """
    widths = []
    for typ in ("put", "call"):
        strikes = [float(l.strike) for l in legs if str(l.type).lower() == typ]
        if len(strikes) >= 2:
            widths.append(max(strikes) - min(strikes))
    return max(widths) if widths else 0.0


def capital_at_risk(spec: StrategySpec, legs: List[Leg], price: float) -> float:
    """Cash actually tied up, per contract.

    Defined-risk structures risk width minus credit. A naked short put is
    cash-secured at the full strike, which is why it prices most names out of a
    $4,000 account entirely.
    """
    struct = spec.structure
    if struct in ("bull_put", "bear_call", "iron_condor"):
        # The width ACTUALLY obtained, not the one requested. Wings snap to the
        # nearest listed strike, so a $5 request can fill $2.50 or $10 wide, and
        # using the requested width would misstate risk in both directions.
        width = actual_width(legs)
        if width <= 0:
            return 0.0
        return max(0.0, width * 100 - price * 100)
    if struct == "short_put":
        return float(legs[0].strike) * 100 - price * 100
    return abs(price) * 100          # debit paid up front


def _should_exit(spec: StrategySpec, trade: Trade, close_price: float,
                 date: str) -> Optional[str]:
    """Which exit rule fires, if any. Held-to-expiry ignores the managed rules."""
    ex = spec.exit
    dte = _dte(date, trade.expiration)
    if dte <= 0:
        return "expiry"
    if ex.get("hold_to_expiry"):
        return None

    credit = trade.entry_price
    if credit > 0:                            # credit structure
        captured = (credit + close_price) / credit if credit else 0.0
        if ex.get("profit_target") and captured >= float(ex["profit_target"]):
            return "profit_target"
        loss = -(credit + close_price)
        if ex.get("stop") and loss >= float(ex["stop"]) * credit:
            return "stop"
    else:                                     # debit structure
        paid = -credit
        value = -close_price
        if paid > 0:
            if ex.get("profit_target") and value >= paid * (1 + float(ex["profit_target"])):
                return "profit_target"
            if ex.get("stop") and value <= paid * 0.5:
                return "stop"
    if ex.get("time_exit_dte") and dte <= int(ex["time_exit_dte"]):
        return "time_exit"
    return None


def replay(spec: StrategySpec, symbols: Sequence[str], dates: Sequence[str],
           source: ChainSource,
           terminal: Optional[Dict[str, str]] = None,
           splits: Optional[Dict[str, Any]] = None,
           stratum_of: Optional[Dict[str, str]] = None,
           seed: int = 20260806) -> Tuple[List[Trade], Dict[str, int]]:
    """Walk the calendar forward, opening and managing positions.

    `terminal` maps symbol -> last date with data, so a position whose ticker
    stops existing is closed rather than lost.
    """
    rng = random.Random(seed)
    signals = SignalHistory()
    conditions = {k: v for k, v in spec.entry.items()
                  if k.endswith(('_min', '_max'))
                  and k != 'max_friction_pct_of_credit'}
    terminal = terminal or {}
    splits = splits or {}
    stratum_of = stratum_of or {}
    trades: List[Trade] = []
    open_by_symbol: Dict[str, List[Trade]] = {}
    stats = {"opened": 0, "closed": 0, "skipped_missing": 0,
             "skipped_crossed": 0, "skipped_no_legs": 0,
             "skipped_capital": 0, "skipped_friction": 0, "skipped_signal": 0,
             "split_closed": 0,
             "ticker_ended": 0}

    cap = float(spec.sizing.get("max_capital_at_risk", 4000))
    max_open = int(spec.sizing.get("max_concurrent", 5))
    entry_days = spec.entry.get("entry_days")

    for date in sorted(dates):
        for sym in symbols:
            chain = source.chain(sym, date)          # only today's data, ever
            quotes = quotes_from_chain(chain) if chain else {}
            live = open_by_symbol.setdefault(sym, [])

            # A split lands today: this dataset is not split-adjusted, so the
            # chain's strikes and spot jump by the split factor while the real
            # CONTRACTS adjust too. That cannot be reconstructed here, so open
            # positions are closed at their last clean mark and the signal
            # history is dropped rather than reading the jump as a 95% crash.
            if date in splits.get(sym, ()):
                for t in list(live):
                    t.exit_date = date
                    t.exit_reason = "split"
                    t.exit_price = t.exit_price if t.exit_price is not None else 0.0
                    t.pnl = (t.entry_price + (t.exit_price or 0.0)) * 100
                    live.remove(t)
                    stats["split_closed"] += 1
                signals.forget(sym)
                continue
            # Portfolio-wide, not per symbol: an account holds ONE book. Keeping
            # the cap per symbol let 14 names run 3 positions each — 42 open
            # against a $4,000 account — and made every account-level return
            # figure meaningless.
            total_open = sum(len(v) for v in open_by_symbol.values())

            # ── manage what is already open ──
            for t in list(live):
                if chain:
                    at_expiry = _dte(date, t.expiration) <= 0
                    if at_expiry:
                        # Settle at intrinsic value. At expiry there is no
                        # spread to cross — ITM contracts are assigned, OTM ones
                        # expire — and pricing off stale quotes let a defined-
                        # risk spread post a loss larger than its own width.
                        spot = implied_spot_any(chain, t.expiration)
                        price = settle(t.legs, spot) if spot is not None else None
                    else:
                        price, _reason = fill_with_reason(reverse(t.legs),
                                                          quotes)
                    if price is not None:
                        why = _should_exit(spec, t, price, date)
                        if why:
                            t.exit_date, t.exit_price = date, price
                            t.exit_reason = why
                            t.pnl = (t.entry_price + price) * 100
                            live.remove(t)
                            stats["closed"] += 1
                            continue
                # the ticker's data ends here: close at the last known mark
                if terminal.get(sym) and date >= terminal[sym]:
                    t.exit_date = date
                    t.exit_reason = "ticker_ended"
                    t.exit_price = t.exit_price if t.exit_price is not None else 0.0
                    t.pnl = (t.entry_price + (t.exit_price or 0.0)) * 100
                    live.remove(t)
                    stats["ticker_ended"] += 1

            if not chain:
                continue                             # no data: skip, do not score

            # Record today BEFORE deciding, so the signal may use today's own
            # close but nothing later. History only ever moves forward.
            signals.update(sym, snapshot(chain, date))
            if conditions and not passes(signals.features(sym), conditions):
                stats["skipped_signal"] += 1
                continue
            if total_open >= max_open:
                continue
            if entry_days == "random" and rng.random() > 0.5:
                continue

            legs = select_legs(spec, chain, date, rng)
            if legs is None:
                stats["skipped_no_legs"] += 1
                continue
            price, reason = fill_with_reason(legs, quotes)
            if reason == SKIP_MISSING:
                stats["skipped_missing"] += 1
                continue
            if reason == SKIP_CROSSED:
                stats["skipped_crossed"] += 1
                continue
            if spec.structure in CREDIT_STRUCTURES and price <= 0:
                stats["skipped_missing"] += 1
                continue

            max_fric = spec.entry.get("max_friction_pct_of_credit")
            if max_fric is not None and price > 0:
                fric = crossing_cost(legs, quotes)
                if fric is None or fric > float(max_fric) * price:
                    stats["skipped_friction"] += 1
                    continue

            car = capital_at_risk(spec, legs, price)
            if car <= 0 or car > cap:
                stats["skipped_capital"] += 1
                continue

            exp = _pick_expiry(chain, date, *spec.entry.get("dte", [25, 45]))
            t = Trade(symbol=sym, entry_date=date, entry_price=float(price),
                      capital_at_risk=car, legs=legs, expiration=exp or date,
                      stratum=stratum_of.get(sym))
            trades.append(t)
            live.append(t)
            stats["opened"] += 1

    return trades, stats
