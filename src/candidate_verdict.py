"""What a candidate is worth once you have paid to get into it.

Three measured facts drive everything here, all from 2026-08-04:

  1. Friction is structure-dependent and enormous. One crossing costs 0.7-1.7%
     of a single leg's premium and 33% of a two-leg credit spread's credit,
     because a spread's friction is denominated in its legs while its reward is
     denominated in their difference. The scan path never showed this.
  2. `quality_score` cannot rank. Spearman against return on capital is -0.132
     on the long-premium book and +0.047 on short premium; its top bucket is
     the worst cell in the ledger (31.6% win, -19.9% return on capital). It is
     usable as hygiene, not as an ordering.
  3. p* = 1 - credit/width is the decision. A credit spread must beat that win
     rate or it loses, and it is arithmetic rather than a 35-metric composite.

So candidates are ordered by what survives their own costs, and `quality_score`
is demoted to a tie-breaker. See docs/EXECUTION_TRUTH.md and docs/LAB_FINDINGS.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

from . import execution_truth as et

# Legs per structure, for reading quotes off a candidate row.
_TWO_LEG = {"Bull Put": ("put", "put"), "Bear Call": ("call", "call")}
_FOUR_LEG = {"Iron Condor"}
_SINGLE = {"Long Call", "Long Put", "Short Put", "Short Call"}

# A candidate paying more than this share of its own reward to trade is refused.
# Well above the measured single-leg burden (1-2%) and below the measured
# two-leg spread burden (33%), so it bites exactly where the evidence says it
# should. Round trip, so a 33%-per-crossing spread reads as 67% here.
DEFAULT_MAX_FRICTION = 0.25


@dataclass(frozen=True)
class Verdict:
    """Whether a candidate survives its costs, and the numbers behind it."""
    priced: bool
    passed: bool
    reason: str
    friction_pct: Optional[float] = None      # one crossing / reward
    round_trip_pct: Optional[float] = None    # both crossings / reward
    breakeven: Optional[float] = None         # p*, credit structures only
    net_reward: Optional[float] = None        # credit or premium, after filling

    @property
    def sort_key(self):
        """Passing candidates first, then cheapest to trade."""
        return (1 if self.passed else 0,
                -(self.round_trip_pct if self.round_trip_pct is not None else 9e9))


def _q(row: Dict[str, Any], bid_key: str, ask_key: str) -> Optional[tuple]:
    b, a = row.get(bid_key), row.get(ask_key)
    if b is None or a is None:
        return None
    try:
        b, a = float(b), float(a)
    except (TypeError, ValueError):
        return None
    if a <= 0 or b < 0 or a < b:
        return None
    return b, a


def _legs_of(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Quoted legs for a candidate row, or None if it cannot be priced."""
    strategy = row.get("strategy_name") or row.get("type") or ""

    if strategy in _FOUR_LEG:
        # Both wings, or nothing. A condor priced from three real quotes and
        # one guess is not a price, and the friction it under-reports is the
        # whole reason the structure is marginal: four crossings against one
        # credit, twice the two-leg burden.
        legs = []
        for prefix, side in (("short_put", "sell"), ("long_put", "buy"),
                             ("short_call", "sell"), ("long_call", "buy")):
            q = _q(row, f"{prefix}_bid", f"{prefix}_ask")
            if q is None:
                return None
            legs.append({"bid": q[0], "ask": q[1], "side": side})
        return legs

    if strategy in _TWO_LEG:
        s = _q(row, "short_bid", "short_ask")
        l = _q(row, "long_bid", "long_ask")
        if s is None or l is None:
            return None
        return [{"bid": s[0], "ask": s[1], "side": "sell"},
                {"bid": l[0], "ask": l[1], "side": "buy"}]

    q = _q(row, "bid", "ask")
    if q is None:
        return None
    side = "sell" if str(strategy).startswith("Short") else "buy"
    return [{"bid": q[0], "ask": q[1], "side": side}]


def verdict_for(row: Dict[str, Any],
                historical_win_rate: Optional[float] = None,
                max_friction_pct: float = DEFAULT_MAX_FRICTION) -> Verdict:
    """Price one candidate at what it would actually fill for and judge it.

    An unquotable candidate is refused rather than assumed tradeable: the scan
    path's `lastPrice +/- 5%` fallback is how a mid-price fill entered the
    ledger in the first place.

    `historical_win_rate` no longer gates anything — see the note beside the
    removed breakeven refusal below. It is kept in the signature because
    `worth.assess` passes it and because a caller may want it recorded, not
    because it changes a verdict."""
    legs = _legs_of(row)
    if legs is None:
        return Verdict(priced=False, passed=False,
                       reason="no two-sided quote on every leg")

    mid = et.structure_fill(legs, "mid")
    crossed = et.structure_fill(legs, "cross")
    if mid is None or crossed is None:
        return Verdict(priced=False, passed=False, reason="could not be priced")

    # Reward is the credit collected, or the premium at risk for a debit.
    reward = abs(mid.price)
    if reward <= 0:
        return Verdict(priced=False, passed=False, reason="no premium to speak of")

    friction = crossed.slip_vs_mid / reward
    round_trip = 2.0 * friction

    filled = et.structure_fill(legs, "limit")
    net_reward = filled.price if filled else None

    # A credit structure is one the market pays you to open. A debit's net
    # price is legitimately negative — you pay for a long call — so the
    # "credit vanished" check below applies only to the credit side.
    is_credit = mid.price > 0

    width = row.get("spread_width")
    breakeven = None
    if width and is_credit and net_reward is not None and net_reward > 0:
        breakeven = et.breakeven_win_rate(net_reward, float(width))

    if is_credit and net_reward is not None and net_reward <= 0:
        return Verdict(True, False, "credit disappears once the spread is crossed",
                       friction, round_trip, breakeven, net_reward)

    if round_trip > max_friction_pct:
        return Verdict(True, False,
                       f"friction {round_trip:.0%} of reward exceeds the "
                       f"{max_friction_pct:.0%} ceiling",
                       friction, round_trip, breakeven, net_reward)

    # A breakeven REFUSAL used to sit here: reject when `1 - credit/width`
    # exceeded the historical win rate. Removed 2026-08-17 for three reasons,
    # measured before acting.
    #
    # It never fired. Instrumented across four scan modes: 482 verdict calls,
    # `historical_win_rate` supplied on ZERO of them, because
    # `rank_by_verdict(df, win_rates=None)` defaults it and neither caller
    # passes one. Single legs could not reach it regardless — no
    # `spread_width`, so no breakeven.
    #
    # Wiring it up would have been worse than leaving it dead. The comparison
    # is a category error: `1 - credit/width` is what you would need HELD TO
    # EXPIRY, while the win rate it is judged against was achieved WITH
    # management. It points the wrong way, too — the NVDA Bull Put 215/205
    # needs 77.2% held to expiry against a 66.4% history and would have been
    # refused, while the MANAGED requirement for that family is 50.9% against
    # the same 66.4%: a +15.5pp margin on the one structure family this ledger
    # says works.
    #
    # And a per-CONTRACT breakeven gate cannot be right under managed exits at
    # all. If a winner takes tp x credit and a loser gives up sl x credit, then
    # p* = sl / (sl + tp), independent of that contract's credit/width. The
    # requirement is a property of the STRATEGY. That is why the measured
    # managed rate is per-strategy, and why the comparison now lives in the
    # Breakeven column and in `allowed_strategies` rather than here.
    #
    # `breakeven` is still computed and still reported below. It is a real
    # number and worth seeing; it simply no longer refuses anything.

    detail = f"friction {round_trip:.0%} of reward"
    if breakeven is not None:
        detail += f", breakeven {breakeven:.0%}"
    return Verdict(True, True, detail, friction, round_trip, breakeven, net_reward)


def rank(rows: Sequence[Dict[str, Any]],
         win_rates: Optional[Dict[str, float]] = None,
         max_friction_pct: float = DEFAULT_MAX_FRICTION) -> List[Dict[str, Any]]:
    """Order candidates by what survives their costs.

    `quality_score` breaks ties and nothing more. It correlates -0.132 with
    return on the long-premium book, so ordering by it actively selects the
    worst candidates — the top score bucket is the worst cell in the ledger."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        # A row with no strategy name (or a non-string one) has no historical
        # win rate to look up — it must not fall back to an arbitrary entry.
        name = r.get("strategy_name")
        wr = (win_rates or {}).get(name) if isinstance(name, str) else None
        r["verdict"] = verdict_for(r, historical_win_rate=wr,
                                   max_friction_pct=max_friction_pct)
        out.append(r)

    def _ev(r) -> float:
        """Net-of-cost EV per contract, or -inf when it was never computed.

        An absent EV must never outrank a measured positive one — treating
        'unknown' as 'zero' would float unpriced candidates above the ones
        that demonstrably clear their costs."""
        v = r.get("ev_per_contract")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float("-inf")
        return f if f == f else float("-inf")     # NaN is not a number either

    # Cost is a GATE, not the sort key. A live scan made this obvious: among
    # single legs the round-trip cost sits at 1-4% across the whole board and
    # separates almost nothing, so ranking on it put a pick with net EV -36 and
    # a SKIP verdict above the only board pick that cleared its costs (+253,
    # TAKE). Order: survives the gate, then net-of-cost EV, then cheapest to
    # trade, then quality_score purely to break ties.
    out.sort(key=lambda r: (1 if r["verdict"].passed else 0,
                            _ev(r),
                            -(r["verdict"].round_trip_pct
                              if r["verdict"].round_trip_pct is not None else 9e9),
                            float(r.get("quality_score") or 0.0)), reverse=True)
    return out


@lru_cache(maxsize=8)
def required_win_rates_from_ledger(db_path: str = "paper_trades.db",
                                   min_n: int = 20) -> Dict[str, float]:
    """The win rate each structure MUST clear, given how it actually closes.

        p* = mean_loss / (mean_loss + mean_win)

    over CLOSED trades, both legs in percent of capital at risk. This is the
    MANAGED rate: the trades it is measured on were themselves managed by the
    configured exits (take_profit 0.5, stop_loss -0.25, time exit 21 DTE), so
    the number already carries whatever those rules do to the tails.

    It replaces `1 - credit/width` as the benchmark the WORTH margin is judged
    against. That formula is the HOLD-TO-EXPIRY rate, which this repo ruled
    against on 2026-08-13, and it is only computable for rows carrying a
    `spread_width` — which is why every single-leg board showed `n/a`.

    Extending the hold-to-expiry formula to single legs instead is the obvious
    move and is wrong: a cash-secured put's risk is its collateral and its
    reward is its credit, giving p* = 98.7-99.0%. Arithmetically correct, and
    the right answer to the wrong question, because it assumes the put is
    ridden to total loss.

    Reproduces the recorded figures exactly: Bull Put 50.9% needed against
    66.4% delivered (+15.6pp), Bear Call -7.4pp, Iron Condor -10.4pp.

    Structures with fewer than `min_n` closed trades are omitted, and so are
    those with no losers or no winners — the ratio is undefined there and a
    number driven by one side is worse than an absent one. Same discipline as
    `win_rates_from_ledger`.

    NOT for the refusal gate. This is per-STRATEGY, so gating on it refuses
    whole families rather than individual candidates.
    """
    import sqlite3
    out: Dict[str, float] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        rows = conn.execute(
            "SELECT strategy_name, pnl_pct FROM trades "
            "WHERE status='CLOSED' AND pnl_pct IS NOT NULL "
            "AND strategy_name IS NOT NULL").fetchall()
    except sqlite3.Error:
        return out
    finally:
        conn.close()

    buckets: Dict[str, list] = {}
    for name, pnl in rows:
        try:
            buckets.setdefault(str(name), []).append(float(pnl))
        except (TypeError, ValueError):
            continue
    for name, vals in buckets.items():
        if len(vals) < min_n:
            continue
        wins = [v for v in vals if v > 0]
        losses = [-v for v in vals if v <= 0]   # a scratch is not a win
        if not wins or not losses:
            continue
        mean_win = sum(wins) / len(wins)
        mean_loss = sum(losses) / len(losses)
        if mean_win + mean_loss <= 0:
            continue
        out[name] = mean_loss / (mean_loss + mean_win)
    return out


@lru_cache(maxsize=8)
def win_rates_from_ledger(db_path: str = "paper_trades.db",
                          since: Optional[str] = None,
                          min_n: int = 20) -> Dict[str, float]:
    """Observed win rate per structure, for the breakeven comparison.

    Structures with fewer than `min_n` closed trades are omitted rather than
    reported on thin evidence — an absent gate is safer than a gate driven by
    five trades."""
    import sqlite3
    out: Dict[str, float] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        q = ("SELECT strategy_name, COUNT(*), "
             "SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) "
             "FROM trades WHERE status!='OPEN' AND duplicate_of IS NULL")
        args: List[Any] = []
        if since:
            q += " AND exit_date >= ?"
            args.append(since)
        q += " GROUP BY strategy_name"
        for name, n, wins in conn.execute(q, args):
            if name and n and n >= min_n:
                out[name] = float(wins or 0) / float(n)
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out
