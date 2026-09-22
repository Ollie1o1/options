"""Corpus loaders. Each emits the same `PricePath`, so the replay engine never
learns which database a position came from.

Every connection is opened read-only. A path that fails an invariant is DROPPED
AND COUNTED, never interpolated: a silently repaired path is a fabricated
observation.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from src.policy_lab.types import PathPoint, PricePath

# Strategies closed by buying the structure back.
_CREDIT_STRATEGIES = {"Bull Put", "Bear Call", "Iron Condor", "Short Put"}

# candidate_positions.family values whose entry_price sign is checkable.
# 'short_premium' carries no fixed expectation here.
_EXPECTED_CREDIT_FAMILY = "spread"       # entry_price must be > 0
_EXPECTED_DEBIT_FAMILY = "long_option"   # entry_price must be < 0


@dataclass
class LoadReport:
    """What the loader kept and what it threw away, by reason."""
    loaded: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _ro(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _gap_ok(dates: List[str], max_gap_days: int) -> bool:
    for a, b in zip(dates, dates[1:]):
        if (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days > max_gap_days:
            return False
    return True


def _capital_at_risk(features_json: Optional[str], entry_price: float) -> Optional[float]:
    """Capital at risk in dollars per contract, from `candidates.features_json`.

    Verified on 2,000 real Bull Put rows: `max_loss == (spread_width -
    entry_price) * 100` to within $1 on 2000/2000, never <= 0. Prefer the
    recorded `max_loss`; fall back to `(spread_width - abs(entry_price)) *
    100` only when `max_loss` is absent or non-positive. Return None when
    neither is usable — the caller drops the row rather than guessing.
    """
    if not features_json:
        return None
    try:
        feats = json.loads(features_json)
    except (TypeError, ValueError):
        return None

    max_loss = feats.get("max_loss")
    if max_loss is not None:
        try:
            max_loss = float(max_loss)
        except (TypeError, ValueError):
            max_loss = None
        if max_loss is not None and max_loss > 0:
            return max_loss

    spread_width = feats.get("spread_width")
    if spread_width is not None:
        try:
            car = (float(spread_width) - abs(float(entry_price))) * 100.0
        except (TypeError, ValueError):
            return None
        if car > 0:
            return car

    return None


def load_corpus_a(db_path: str, strategy: str,
                  max_gap_days: int = 5) -> Tuple[List[PricePath], LoadReport]:
    """Closed candidate positions plus their forward marks.

    Corpus A is wide but spans only 26 trading days — one regime — and its
    spread exits are blended mid, so exit crossing cost must be modelled
    rather than read. Both facts belong in any report built on it.

    `candidate_marks` rows with `source='live_quote_structure'` carry bid and
    ask as 100% NULL for every mark attached to a closed Bull Put position.
    Those marks are loaded with `bid=ask=mid` and `spread_imputed=True` so
    `CostModel` widens off mid instead of crossing a fabricated quote (see
    costs.py). Marks with a real two-sided quote keep `spread_imputed=False`.

    `actual_pnl_frac` is realised P&L as a fraction of capital at risk, but
    `candidate_positions.pnl_pct` is a fraction of premium
    (`(entry_price - exit_price) / entry_price`). It is rescaled here by
    `abs(entry_price) * 100 / capital_at_risk` — skipping this conversion
    mixes two denominators that differ by roughly 2.5x on a typical position.

    `entry_price` is positive for credit spreads and negative for debit
    (long option) structures. Any row whose sign contradicts its
    `candidate_positions.family` ('spread' -> credit, 'long_option' -> debit)
    is dropped as `sign_contradicts_family` rather than silently flipped.
    """
    rep = LoadReport()
    out: List[PricePath] = []
    con = _ro(db_path)
    try:
        rows = con.execute(
            """SELECT p.scan_id, p.board, p.contract_key, c.symbol,
                      c.strategy_name, p.family, p.entry_date, p.entry_price,
                      p.status, p.exit_date, p.pnl_pct, c.features_json
                 FROM candidate_positions p
                 JOIN candidates c
                   ON c.scan_id = p.scan_id AND c.board = p.board
                  AND c.contract_key = p.contract_key
                WHERE c.strategy_name = ?""", (strategy,)).fetchall()

        for (scan_id, board, key, symbol, strat, family, entry_date,
             entry_price, status, exit_date, pnl_pct,
             features_json) in rows:
            if status != "CLOSED":
                rep.drop("not_closed")
                continue
            if exit_date is None or pnl_pct is None or entry_price is None:
                rep.drop("missing_terminal_fields")
                continue

            entry_price = float(entry_price)
            if family == _EXPECTED_CREDIT_FAMILY and entry_price <= 0:
                rep.drop("sign_contradicts_family")
                continue
            if family == _EXPECTED_DEBIT_FAMILY and entry_price >= 0:
                rep.drop("sign_contradicts_family")
                continue

            marks = con.execute(
                """SELECT mark_date, bid, ask, mid FROM candidate_marks
                    WHERE contract_key = ? AND mark_date >= ? AND mark_date <= ?
                    ORDER BY mark_date""",
                (key, entry_date[:10], exit_date[:10])).fetchall()
            if len(marks) < 2:
                rep.drop("fewer_than_two_marks")
                continue
            if not _gap_ok([m[0] for m in marks], max_gap_days):
                rep.drop("gap_too_large")
                continue

            car = _capital_at_risk(features_json, entry_price)
            if car is None:
                rep.drop("unsizable")
                continue

            exp_day = date.fromisoformat(exit_date[:10])
            points = []
            for md, bid, ask, mid in marks:
                if mid is None:
                    continue
                imputed = bid is None or ask is None
                points.append(PathPoint(
                    date=md,
                    bid=(mid if imputed else bid),
                    ask=(mid if imputed else ask),
                    mid=mid,
                    spot=None,
                    dte=(exp_day - date.fromisoformat(md[:10])).days,
                    spread_imputed=imputed,
                ))
            if len(points) < 2:
                rep.drop("marks_without_mid")
                continue

            actual_pnl_frac = float(pnl_pct) * abs(entry_price) * 100.0 / car

            out.append(PricePath(
                position_id=f"{scan_id}|{board}|{key}",
                symbol=symbol, strategy=strat,
                entry_date=entry_date[:10],
                entry_price=abs(entry_price),
                capital_at_risk=car,
                is_credit=strat in _CREDIT_STRATEGIES,
                points=tuple(points),
                actual_exit_date=exit_date[:10],
                actual_pnl_frac=actual_pnl_frac,
                corpus="A",
            ))
            rep.loaded += 1
    finally:
        con.close()
    return out, rep


def load_corpus_b(ledger_path: str, archive_path: str, strategy: str,
                  max_gap_days: int = 5) -> Tuple[List[PricePath], LoadReport]:
    """Live ledger trades joined to archived chain snapshots of their short leg.

    Small — only trades with two or more archived snapshots of their short
    leg between entry and exit survive — but every point carries a real
    two-sided quote, so this is the corpus that checks a cost assumption
    instead of believing it. Its role is validation: a policy winning in
    Corpus A and reversing here is not promoted.

    The short-leg strike to join on is `trades.short_put_strike` ONLY for
    Iron Condor (the only strategy where that column is populated); every
    other strategy — including Bull Put, whose 163 closed rows are 100% NULL
    in `short_put_strike` — carries its short leg in the generic `strike`
    column instead. `COALESCE(short_put_strike, strike)` picks the right one
    for every strategy, since `short_put_strike` is NULL exactly where
    `strike` is the column to use. Joining on `short_put_strike` alone
    returns zero rows for every non-Iron-Condor strategy.

    `chain_snapshots` carries zero NULL and zero both-zero bid/ask across the
    whole archive, so every `PathPoint` built here keeps the default
    `spread_imputed=False` — unlike Corpus A, there is nothing to impute.

    `capital_at_risk` comes straight from the ledger's own column (dropped as
    `no_capital_at_risk` when NULL or <= 0). `actual_pnl_frac` is
    `pnl_usd / capital_at_risk` with NO rescale — the ledger already records
    both in dollars, unlike Corpus A's premium-fraction `pnl_pct`.
    """
    rep = LoadReport()
    out: List[PricePath] = []
    con = _ro(ledger_path)
    try:
        con.execute("ATTACH ? AS ca", (f"file:{archive_path}?mode=ro",))
    except sqlite3.OperationalError:
        con.close()
        con = _ro(ledger_path)
        con.execute(f"ATTACH 'file:{archive_path}?mode=ro' AS ca")
    try:
        rows = con.execute(
            """SELECT entry_id, date, ticker, expiration, strategy_name,
                      exit_date, pnl_usd, capital_at_risk, net_credit,
                      spread_width, COALESCE(short_put_strike, strike)
                 FROM trades
                WHERE status = 'CLOSED' AND strategy_name = ?
                  AND exit_date IS NOT NULL""", (strategy,)).fetchall()

        for (eid, entry_date, ticker, expiration, strat, exit_date, pnl_usd,
             car, credit, width, short_leg_strike) in rows:
            if car is None or car <= 0:
                rep.drop("no_capital_at_risk")
                continue
            if (credit is None or width is None or short_leg_strike is None
                    or expiration is None or entry_date is None):
                rep.drop("missing_structure_fields")
                continue
            if pnl_usd is None:
                rep.drop("no_pnl_recorded")
                continue

            snaps = con.execute(
                """SELECT snap_date, bid, ask, spot FROM ca.chain_snapshots
                    WHERE symbol = ? AND expiration = ? AND strike = ?
                      AND snap_date >= ? AND snap_date <= ?
                    ORDER BY snap_date""",
                (ticker, expiration[:10], float(short_leg_strike),
                 entry_date[:10], exit_date[:10])).fetchall()
            if len(snaps) < 2:
                rep.drop("fewer_than_two_snapshots")
                continue
            if not _gap_ok([s[0] for s in snaps], max_gap_days):
                rep.drop("gap_too_large")
                continue

            exp_day = date.fromisoformat(expiration[:10])
            points = tuple(
                PathPoint(date=sd, bid=float(bid), ask=float(ask),
                          mid=(float(bid) + float(ask)) / 2.0,
                          spot=(float(spot) if spot is not None else None),
                          dte=(exp_day - date.fromisoformat(sd[:10])).days)
                for sd, bid, ask, spot in snaps
                if bid is not None and ask is not None
            )
            if len(points) < 2:
                rep.drop("snapshots_without_quotes")
                continue

            out.append(PricePath(
                position_id=f"ledger:{eid}", symbol=ticker, strategy=strat,
                entry_date=entry_date[:10], entry_price=abs(float(credit)),
                capital_at_risk=float(car),
                is_credit=strat in _CREDIT_STRATEGIES,
                points=points, actual_exit_date=exit_date[:10],
                actual_pnl_frac=float(pnl_usd) / float(car),
                corpus="B",
            ))
            rep.loaded += 1
    finally:
        con.close()
    return out, rep
