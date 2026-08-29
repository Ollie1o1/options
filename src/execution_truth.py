"""What a fill actually costs, and whether the structure survives paying it.

The live scan path prices every leg at the bid/ask MID (`options_screener`
builds ``premium`` from ``mid``), and `paper_trading.slippage_per_share` is
applied only on the way OUT. Entry friction is therefore modelled as exactly
zero. Measured against archived quotes for 29 logged Bull Puts, the real cost
of getting in is $0.35/share of credit — 27% of the credit collected — which
moves the breakeven win rate on that structure from 58.0% to 73.3%. The book
wins 70.4%. That gap is the difference between an edge and a loss, and nothing
in the system currently sees it.

This module is the one place that answers "what price do I actually get". It
takes quotes and returns fills; it does not know about the ledger, the scanner
or the scoring. Callers depend on it, it depends on nothing: a leg without a
real two-sided quote is refused, never repaired from a modelled spread.

CLI:
    python -m src.execution_truth --explain 1.00 1.20   # one leg, all policies
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# Fill policies, cheapest-for-the-trader first.
POLICIES = ("mid", "limit", "cross")

# How much of the half-spread a worked limit order concedes. 0.0 fills at the
# mid (optimistic — this is today's implicit assumption), 1.0 crosses fully.
DEFAULT_LIMIT_K = 0.35


def leg_fill(bid: float, ask: float, side: str, policy: str,
             k: Optional[float] = None) -> float:
    """Price one leg actually fills at.

    `side` is "buy" or "sell" from the trader's perspective; the concession
    always moves against the trader, so a buyer pays up and a seller gives up.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}, expected one of {POLICIES}")
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0

    if policy == "mid":
        frac = 0.0
    elif policy == "cross":
        frac = 1.0
    else:
        frac = DEFAULT_LIMIT_K if k is None else k

    return mid + frac * half if side == "buy" else mid - frac * half


@dataclass(frozen=True)
class FillResult:
    """What a structure fills at. `price` is signed from the trader's cash
    perspective: positive is a net credit received, negative a net debit paid.
    `slip_vs_mid` is always the (non-negative) cost of this policy against a
    mid fill — the number the screener currently assumes is zero."""
    price: float
    slip_vs_mid: float
    per_leg: Sequence[float]
    policy: str


def _quote(leg: Dict[str, Any]) -> Optional[tuple]:
    """(bid, ask) if the leg carries a usable two-sided quote, else None.

    A zero or absent side, or a crossed book (ask < bid), is refused rather
    than repaired. The synthetic `lastPrice +/- 5%` fallback in the scan path
    is how the mid-fill assumption entered the ledger in the first place; this
    module will not reproduce it."""
    bid, ask = leg.get("bid"), leg.get("ask")
    if bid is None or ask is None:
        return None
    try:
        b, a = float(bid), float(ask)
    except (TypeError, ValueError):
        return None
    if a <= 0.0 or b < 0.0 or a < b:
        return None
    return b, a


def structure_fill(legs: Sequence[Dict[str, Any]], policy: str,
                   k: Optional[float] = None) -> Optional[FillResult]:
    """Net fill for a multi-leg structure, or None if any leg is unpriceable.

    Refusing the whole structure on one bad leg is deliberate: a spread priced
    from one real quote and one guess is not a price."""
    if not legs:
        return None

    per_leg: List[float] = []
    price = 0.0
    mid_price = 0.0
    for leg in legs:
        q = _quote(leg)
        if q is None:
            return None
        bid, ask = q
        side = str(leg.get("side", "")).lower()
        fill = leg_fill(bid, ask, side, policy, k=k)
        mid = leg_fill(bid, ask, side, "mid")
        sign = -1.0 if side == "buy" else 1.0
        per_leg.append(fill)
        price += sign * fill
        mid_price += sign * mid

    return FillResult(price=price, slip_vs_mid=mid_price - price,
                      per_leg=tuple(per_leg), policy=policy)


def breakeven_win_rate(credit: float, width: float) -> Optional[float]:
    """The fraction of trades a credit spread must win to break even.

        p* = 1 - credit/width

    Derived from EV = p*credit - (1-p)*(width-credit) = 0. Friction does not
    appear as a separate term because `credit` is already the *filled* credit
    — pay the spread by collecting less, not by adding a fee. Returns None for
    a zero/absent width, 1.0 when no credit survives the fill (unreachable),
    and clamps to 0.0 when the credit covers the full width (unloseable)."""
    if not width or width <= 0:
        return None
    if credit <= 0:
        return 1.0
    return max(0.0, 1.0 - credit / width)


@dataclass(frozen=True)
class EdgeReport:
    """Breakeven win rate under each fill policy, plus the fills behind them."""
    breakeven: Dict[str, Optional[float]]
    fills: Dict[str, FillResult]
    width: float


def edge_report(legs: Sequence[Dict[str, Any]], width: float,
                k: Optional[float] = None) -> Optional[EdgeReport]:
    """p* under every policy at once — what the pre-trade gate reads."""
    fills: Dict[str, FillResult] = {}
    breakeven: Dict[str, Optional[float]] = {}
    for policy in POLICIES:
        f = structure_fill(legs, policy, k=k)
        if f is None:
            return None
        fills[policy] = f
        breakeven[policy] = breakeven_win_rate(f.price, width)
    return EdgeReport(breakeven=breakeven, fills=fills, width=width)


@dataclass(frozen=True)
class GateVerdict:
    """Whether a candidate survives paying to get in, and why."""
    passed: bool
    reason: str
    breakeven: Optional[float] = None
    report: Optional[EdgeReport] = None


def gate(legs: Sequence[Dict[str, Any]], width: float,
         max_breakeven: Optional[float] = None, policy: str = "limit",
         k: Optional[float] = None) -> GateVerdict:
    """Refuse a candidate whose breakeven win rate is unreachable once filled.

    Judged at `policy` (default 'limit') rather than at the mid, because the
    mid is the price nobody gets. `max_breakeven` of None disables the
    threshold but still refuses anything that cannot be quoted — a candidate
    with no two-sided market is not a candidate."""
    report = edge_report(legs, width, k=k)
    if report is None:
        return GateVerdict(False, "no two-sided quote on every leg")

    be = report.breakeven.get(policy)
    if be is None:
        return GateVerdict(False, "width is zero or absent", report=report)

    if max_breakeven is None:
        return GateVerdict(True, f"breakeven {be:.1%} at {policy} (no threshold set)",
                           breakeven=be, report=report)

    if be > max_breakeven:
        return GateVerdict(
            False,
            f"breakeven {be:.1%} at {policy} exceeds the {max_breakeven:.1%} limit",
            breakeven=be, report=report)

    return GateVerdict(True, f"breakeven {be:.1%} at {policy}",
                       breakeven=be, report=report)
