"""Long options for a hold of one to three sessions: what they cost to trade.

Measured 2026-10-13 on `data/candidates.db` (19,310 long calls and puts from
the discovery board over 33 entry days, bought at the ask and sold at the bid
two sessions later):

  spread under 3% of mid    -4.5% mean return on premium
  3-6%                      -7.4%
  6-10%                    -10.7%
  10-20%                   -13.0%

Priced at the mid those buckets are flat, so the gradient is the spread itself
— arithmetic, not a prediction. Nothing else separated: the negative-EV gate's
interval spans zero at this horizon, `ev_net` does not order names, and delta
and days to expiry flip sign with whether the trade worked, which is leverage
rather than edge.

So this view refuses on round-trip cost and on nothing else, orders what is
left by that same cost, and says so. Beside each contract it prints the move
the underlying must make over the hold to get back to zero, against the move
that name has typically made. It holds no opinion on direction.

It is built from every candidate the scan found, including the ones the
board's own gates refused, because on the discovery board that gate is the
negative-EV one and it is not established at this horizon. Each row says
whether the board kept it, so the two tables cannot silently disagree.

DISPLAY ONLY. It is not a gate: it does not touch `pick_ranking.gate_board`,
the auto-logger, or what `candidate_record` persists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd

try:
    from . import formatting as fmt
    HAS_ENHANCED_CLI = True
except ImportError:
    HAS_ENHANCED_CLI = False

# Round-trip cost above which a contract is not shown: (ask - bid) / mid.
MAX_COST_PCT = 0.06
HORIZON_SESSIONS = 3
MAX_ROWS = 12

SINGLE_LEG = frozenset({"call", "put"})
TRADING_DAYS = 252
# Theta is quoted per calendar day and a three-session hold spans a weekend
# as often as not, so decay is charged on calendar days.
CALENDAR_PER_SESSION = 7 / 5


@dataclass
class SwingResult:
    """What is cheap enough to swing, and how much was not."""
    kept: pd.DataFrame
    scanned: int = 0
    over_cap: int = 0
    unpriced: int = 0
    max_cost_pct: float = MAX_COST_PCT
    horizon_sessions: int = HORIZON_SESSIONS


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. None means NOT RECORDED, never zero."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def round_trip_cost_pct(bid: Any, ask: Any) -> Optional[float]:
    """Share of the premium lost buying at the ask and selling at the bid.

    (ask - bid) / mid, the same definition the 2026-10-13 buckets were cut on.
    None without a two-sided quote; a locked market is a real 0.0.
    """
    b, a = _num(bid), _num(ask)
    if b is None or a is None or b <= 0 or a < b:
        return None
    return (a - b) / ((a + b) / 2)


def breakeven_move(bid: Any, ask: Any, delta: Any, gamma: Any, theta: Any,
                   horizon_sessions: int = HORIZON_SESSIONS) -> Optional[float]:
    """Dollars the underlying must move, the right way, to exit at no loss.

    The option has to gain back the spread paid plus the decay over the hold:
    gain = (ask - bid) + |theta| * calendar days. The move m solves
    |delta| * m + gamma * m^2 / 2 = gain. Assumes the spread is as wide at the
    exit as at the entry and implied volatility does not change.
    """
    b, a, d, t = _num(bid), _num(ask), _num(delta), _num(theta)
    if b is None or a is None or a < b or d is None or d == 0 or t is None:
        return None
    gain = (a - b) + abs(t) * horizon_sessions * CALENDAR_PER_SESSION
    d = abs(d)
    g = _num(gamma)
    if g is None or g <= 0:
        return gain / d
    return (-d + math.sqrt(d * d + 2 * g * gain)) / g


def typical_move(underlying: Any, hist_vol: Any,
                 horizon_sessions: int = HORIZON_SESSIONS) -> Optional[float]:
    """One standard deviation of the underlying's move over the hold, in dollars.

    From annualised realised volatility, not implied: the question is what the
    name has been doing, not what the option already charges for.
    """
    s, v = _num(underlying), _num(hist_vol)
    if s is None or s <= 0 or v is None or v <= 0:
        return None
    return s * v * math.sqrt(horizon_sessions / TRADING_DAYS)


def _contract_keys(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """(symbol, call/put, strike, expiry) per row, or None if it cannot be built."""
    if df is None or any(c not in df.columns
                         for c in ("symbol", "type", "strike", "expiration")):
        return None
    return pd.Series(
        list(zip(df["symbol"].astype(str),
                 df["type"].astype(str).str.lower(),
                 pd.to_numeric(df["strike"], errors="coerce").round(4),
                 df["expiration"].astype(str).str[:10])),
        index=df.index, dtype="object")


def swing_board(df: Optional[pd.DataFrame], *,
                max_cost_pct: float = MAX_COST_PCT,
                horizon_sessions: int = HORIZON_SESSIONS,
                board_kept: Optional[pd.DataFrame] = None) -> SwingResult:
    """Keep the single-leg calls and puts whose round trip costs at most the cap.

    One contract per (symbol, call/put): the cheapest to trade. Ordered by
    cost, cheapest first — an ordering, not a ranking.

    `board_kept` is the frame the board's gates let through. When given, each
    row carries `swing_board_kept`; when it is not, or the contracts cannot be
    matched, the column is absent rather than guessed.
    """
    empty = SwingResult(kept=pd.DataFrame(), max_cost_pct=max_cost_pct,
                        horizon_sessions=horizon_sessions)
    if df is None or len(df) == 0 or "type" not in df.columns:
        return empty

    kind = df["type"].astype(str).str.lower()
    work = df[kind.isin(SINGLE_LEG)].copy()
    if work.empty:
        return empty
    work["type"] = kind[work.index]

    def col(name: str) -> pd.Series:
        if name not in work.columns:
            return pd.Series([None] * len(work), index=work.index, dtype="object")
        return work[name]

    keys, kept_keys = _contract_keys(work), _contract_keys(board_kept)
    if board_kept is not None and keys is not None and (
            kept_keys is not None or len(board_kept) == 0):
        allowed = set(kept_keys) if kept_keys is not None else set()
        work["swing_board_kept"] = [k in allowed for k in keys]

    cost: List[Optional[float]] = []
    need: List[Optional[float]] = []
    typical: List[Optional[float]] = []
    for bid, ask, delta, gamma, theta, spot, vol in zip(
            col("bid"), col("ask"), col("delta"), col("gamma"), col("theta"),
            col("underlying"), col("hv_30d")):
        cost.append(round_trip_cost_pct(bid, ask))
        move = breakeven_move(bid, ask, delta, gamma, theta, horizon_sessions)
        usual = typical_move(spot, vol, horizon_sessions)
        s = _num(spot)
        if s is None or s <= 0:
            need.append(None)
            typical.append(None)
            continue
        need.append(move / s if move is not None else None)
        typical.append(usual / s if usual is not None else None)

    work["swing_cost_pct"] = pd.Series(cost, index=work.index, dtype="float64")
    work["swing_need_pct"] = pd.Series(need, index=work.index, dtype="float64")
    work["swing_typical_pct"] = pd.Series(typical, index=work.index, dtype="float64")
    work["swing_need_vs_typical"] = work["swing_need_pct"] / work["swing_typical_pct"]

    unpriced = work["swing_cost_pct"].isna()
    # A quote exactly at the cap is kept; the epsilon absorbs float division.
    over = work["swing_cost_pct"] > max_cost_pct + 1e-9
    kept = work[~unpriced & ~over].sort_values(
        ["swing_cost_pct", "swing_need_vs_typical"], kind="stable")
    if "symbol" in kept.columns:
        kept = kept.drop_duplicates(["symbol", "type"], keep="first")
    return SwingResult(kept=kept.reset_index(drop=True), scanned=len(work),
                       over_cap=int(over.sum()), unpriced=int(unpriced.sum()),
                       max_cost_pct=max_cost_pct,
                       horizon_sessions=horizon_sessions)


def _style(text: str, name: str, bold: Optional[bool] = None) -> str:
    return fmt.style(text, name, bold=bold) if HAS_ENHANCED_CLI else text


def _cell(value: Any, spec: str) -> str:
    v = _num(value)
    return "n/a" if v is None else format(v, spec)


def print_swing_view(result: SwingResult, max_rows: int = MAX_ROWS) -> None:
    """Print the swing table under the board it was built from.

    Prints nothing when no single-leg contract was scanned. A board on which
    every contract was refused still prints its refusal line: "none shown" and
    "none found" are different answers.
    """
    if result is None or result.scanned == 0:
        return
    h = result.horizon_sessions
    cap = f"{result.max_cost_pct * 100:.0f}%"

    print()
    print("  " + _style(f"SWING BUYS  —  hold 1 to {h} sessions", 'label', bold=True))
    print("  " + _style(
        f"round-trip cost of {cap} or less, cheapest to trade first — an "
        "ordering, not a ranking; one contract per name and side", 'muted'))
    print("  " + _style(
        "this is not a view on direction: it shows which contracts leave a "
        "correct call room to pay", 'muted'))

    shown = result.kept.head(max_rows)
    flagged = "swing_board_kept" in result.kept.columns
    if len(shown):
        header = (f"  {'Ticker':<7} {'Type':<5} {'Strike':>8} {'Expiry':<11} "
                  f"{'DTE':>4} {'Ask':>8} {'Cost%':>6} {'Need':>7} "
                  f"{'Typical':>8} {'Need/Typ':>9}")
        if flagged:
            header += "  Board"
        print(_style(header, 'label', bold=True))
        for _, r in shown.iterrows():
            expiry = str(r.get("expiration") or "")[:10]
            print(f"  {str(r.get('symbol') or ''):<7} {str(r.get('type')).upper():<5} "
                  f"{_cell(r.get('strike'), '.2f'):>8} {expiry:<11} "
                  f"{_cell(r.get('dte'), '.0f'):>4} {_cell(r.get('ask'), '.2f'):>8} "
                  f"{_cell(r['swing_cost_pct'] * 100, '.1f') + '%':>6} "
                  f"{_pct(r['swing_need_pct']):>7} {_pct(r['swing_typical_pct']):>8} "
                  f"{_ratio(r['swing_need_vs_typical']):>9}"
                  + (("  kept" if r["swing_board_kept"] else "  refused")
                     if flagged else ""))
        hidden = len(result.kept) - len(shown)
        if hidden > 0:
            print("  " + _style(f"{hidden} more within the cap not shown", 'muted'))
        print("  " + _style(
            f"Need = move in the underlying, your way, to exit flat after {h} "
            "sessions (spread plus decay, implied vol unchanged)", 'muted'))
        print("  " + _style(
            f"Typical = one standard deviation of its {h}-session move at "
            "30-day realised vol", 'muted'))
        if flagged and not shown["swing_board_kept"].all():
            print("  " + _style(
                "Board = refused means the board above dropped it, for "
                "negative expected value or your budget; that gate did not "
                f"separate outcomes at 1 to {h} sessions (measured 2026-10-13)",
                'muted'))

    if result.over_cap:
        print("  " + _style(
            f"{result.over_cap} of {result.scanned} long options cost more "
            f"than {cap} round trip and are not shown", 'muted'))
    if result.unpriced:
        print("  " + _style(
            f"{result.unpriced} had no two-sided quote and are not shown", 'muted'))


def _pct(value: Any) -> str:
    v = _num(value)
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _ratio(value: Any) -> str:
    v = _num(value)
    return "n/a" if v is None else f"{v:.2f}x"
