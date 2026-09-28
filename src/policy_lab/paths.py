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
from typing import Any, Dict, List, Optional, Tuple

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


def corpus_a_fingerprint(db_path: str, strategy: str) -> Dict[str, Any]:
    """Read-only snapshot of what Corpus A could offer `strategy` right now.

    `data/candidates.db` is written continuously by a live scheduler while
    the lab reads it (closed Bull Put count observed moving 98,003 ->
    105,392 -> 113,442 -> 115,381 within a single session), so a run's
    manifest must record what the corpus looked like AT READ TIME, not just
    how many rows the loader happened to keep.

    `max_ts` is `MAX(candidates.ts)` — the newest row the scheduler has
    written, independent of strategy or status. `terminal_count` is the
    number of CLOSED `candidate_positions` joined to `candidates` for this
    strategy — the same join `load_corpus_a` uses, but counting every
    terminal row the corpus could offer, not just the ones that survived
    the loader's own invariants (missing marks, gaps, unsizable rows, ...).
    A fingerprint built from the loaded count alone would call two runs
    "the same" when the corpus grew but the loader happened to drop the
    same number of new rows it kept — this counts the corpus itself.
    """
    con = _ro(db_path)
    try:
        max_ts = con.execute("SELECT MAX(ts) FROM candidates").fetchone()[0]
        terminal_count = con.execute(
            """SELECT COUNT(*)
                 FROM candidate_positions p
                 JOIN candidates c
                   ON c.scan_id = p.scan_id AND c.board = p.board
                  AND c.contract_key = p.contract_key
                WHERE c.strategy_name = ? AND p.status = 'CLOSED'""",
            (strategy,)).fetchone()[0]
    finally:
        con.close()
    return {
        "strategy": strategy,
        "max_ts": max_ts,
        "terminal_count": int(terminal_count),
    }


def corpus_b_fingerprint(ledger_path: str, archive_path: str,
                         strategy: str) -> Dict[str, Any]:
    """Read-only snapshot of what Corpus B could offer `strategy` right now.

    Mirrors `corpus_a_fingerprint`. `max_ts` has no single column here: the
    ledger (`trades`) and the archive (`chain_snapshots`) are separate
    databases with separate write cadences, so this reports both — the
    newest closed trade's `exit_date` and the newest archived
    `chain_snapshots.snap_date` — rather than collapsing them into one
    number that would hide which side moved. `terminal_count` is the count
    of CLOSED `trades` for this strategy, independent of how many of those
    survived `load_corpus_b`'s two-leg snapshot join.
    """
    con = _ro(ledger_path)
    try:
        try:
            con.execute("ATTACH ? AS ca", (f"file:{archive_path}?mode=ro",))
        except sqlite3.OperationalError:
            con.close()
            con = _ro(ledger_path)
            con.execute(f"ATTACH 'file:{archive_path}?mode=ro' AS ca")
        max_snap_date = con.execute(
            "SELECT MAX(snap_date) FROM ca.chain_snapshots").fetchone()[0]
        max_exit_date = con.execute(
            "SELECT MAX(exit_date) FROM trades WHERE status = 'CLOSED'"
        ).fetchone()[0]
        terminal_count = con.execute(
            """SELECT COUNT(*) FROM trades
                WHERE status = 'CLOSED' AND strategy_name = ?""",
            (strategy,)).fetchone()[0]
    finally:
        con.close()
    return {
        "strategy": strategy,
        "max_ts": {"snap_date": max_snap_date, "exit_date": max_exit_date},
        "terminal_count": int(terminal_count),
    }


# Strategies whose opening price lives in `net_credit`, not `entry_price`.
# `net_credit` is NULL for every closed Long Call/Long Put/Short Put row
# (single-leg structures record their price in `entry_price` instead); the
# three spread strategies below are the only ones where `net_credit` is
# populated (163/163, 135/135, 148/148 respectively).
_SPREAD_STRATEGIES = {"Bull Put", "Bear Call", "Iron Condor"}


def load_corpus_b(ledger_path: str, archive_path: str, strategy: str,
                  max_gap_days: int = 5) -> Tuple[List[PricePath], LoadReport]:
    """Live ledger trades joined to archived chain snapshots of their short leg.

    Small — only trades with two or more distinct archived snapshot dates of
    their short leg between entry and exit survive — but every point carries
    a real two-sided quote, so this is the corpus that checks a cost
    assumption instead of believing it. Its role is validation: a policy
    winning in Corpus A and reversing here is not promoted.

    The short-leg strike to join on is `trades.short_put_strike` ONLY for
    Iron Condor (the only strategy where that column is populated); every
    other strategy — including Bull Put, whose 163 closed rows are 100% NULL
    in `short_put_strike` — carries its short leg in the generic `strike`
    column instead. `COALESCE(short_put_strike, strike)` picks the right one
    for every strategy, since `short_put_strike` is NULL exactly where
    `strike` is the column to use. Joining on `short_put_strike` alone
    returns zero rows for every non-Iron-Condor strategy.

    The structure's opening price is `net_credit` for the three spread
    strategies (`_SPREAD_STRATEGIES`) and `entry_price` for the three
    single-leg ones (Long Call, Long Put, Short Put), where `net_credit` is
    100% NULL. Requiring `net_credit` unconditionally silently drops every
    single-leg trade. `spread_width` is likewise NULL for all single-leg
    rows, so it is only required (alongside `net_credit`) for spreads.

    `chain_snapshots` stores one row per (symbol, expiration, strike,
    snap_date, TYPE) and 345,064 of 456,343 strike-days carry both a call and
    a put row at the same strike. The join MUST filter on option type or it
    silently interleaves the wrong leg's quotes with the right one's and
    inflates the snapshot count. The type to filter on is `'put'` for Iron
    Condor (its short leg, via `short_put_strike`, is the put) and the
    ledger's own `trades.type` column otherwise (verified non-null on every
    closed row). Snapshots are additionally deduplicated to one point per
    `snap_date` as a second guard against a strike/date pair ever returning
    more than one row.

    For the three spread strategies (`_SPREAD_STRATEGIES`), `entry_price` is
    `net_credit` — the TWO-LEG spread's price — so the path must be built
    from both legs, not the short leg alone. The short leg alone trades at a
    different (larger) magnitude than the net credit, which used to trip
    `stop_mult` on the first interior point of nearly every path. The long
    leg's strike is `trades.long_strike` (or `long_put_strike` for Iron
    Condor, mirroring the short leg's `COALESCE(short_put_strike, strike)`).
    Both legs' snapshots are joined on the SAME `snap_date` (same symbol,
    expiration, option type); a date missing on either leg is dropped and
    counted as `leg_snapshot_missing` rather than pairing a stale leg with a
    fresh one. Closing a credit spread means buying back the short leg
    (paying its ask) and selling the long leg (taking its bid), so
    `ask = short_ask - long_bid` (expensive way to close) and
    `bid = short_bid - long_ask` (cheap way); `mid = short_mid - long_mid`.
    Single-leg strategies (Long Call, Long Put, Short Put) are unchanged —
    they genuinely are one leg and their `entry_price` comes from
    `trades.entry_price`, never `net_credit`.

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
                      spread_width, entry_price, type,
                      COALESCE(short_put_strike, strike),
                      COALESCE(long_put_strike, long_strike)
                 FROM trades
                WHERE status = 'CLOSED' AND strategy_name = ?
                  AND exit_date IS NOT NULL""", (strategy,)).fetchall()

        for (eid, entry_date, ticker, expiration, strat, exit_date, pnl_usd,
             car, credit, width, entry_price, leg_type,
             short_leg_strike, long_leg_strike) in rows:
            if car is None or car <= 0:
                rep.drop("no_capital_at_risk")
                continue

            is_spread = strat in _SPREAD_STRATEGIES
            if is_spread:
                price = credit
                if price is None or width is None:
                    rep.drop("missing_structure_fields")
                    continue
            else:
                price = entry_price
                if price is None:
                    rep.drop("missing_structure_fields")
                    continue

            opt_type = "put" if strat == "Iron Condor" else leg_type
            if (opt_type is None or short_leg_strike is None
                    or expiration is None or entry_date is None):
                rep.drop("missing_structure_fields")
                continue
            if is_spread and long_leg_strike is None:
                rep.drop("missing_structure_fields")
                continue
            if pnl_usd is None:
                rep.drop("no_pnl_recorded")
                continue

            def _snap_by_date(strike: float) -> Dict[str, Tuple]:
                """One (bid, ask, spot) per snap_date for one leg's strike.

                Keeps the first row seen for a date (rows arrive ordered by
                snap_date), so a strike/date pair can never contribute more
                than one observation. A row with a NULL bid or ask never
                becomes a usable snapshot for that date.
                """
                snaps = con.execute(
                    """SELECT snap_date, bid, ask, spot
                         FROM ca.chain_snapshots
                        WHERE symbol = ? AND expiration = ? AND strike = ?
                          AND type = ?
                          AND snap_date >= ? AND snap_date <= ?
                        ORDER BY snap_date""",
                    (ticker, expiration[:10], strike, opt_type,
                     entry_date[:10], exit_date[:10])).fetchall()
                out_by_date: Dict[str, Tuple] = {}
                for sd, bid, ask, spot in snaps:
                    if sd in out_by_date:
                        continue
                    if bid is None or ask is None:
                        continue
                    out_by_date[sd] = (float(bid), float(ask), spot)
                return out_by_date

            short_by_date = _snap_by_date(float(short_leg_strike))

            if is_spread:
                long_by_date = _snap_by_date(float(long_leg_strike))

                # Both legs must carry a real quote on the SAME snap_date —
                # the two-leg join `load_corpus_b`'s docstring describes. A
                # date present on only one leg is dropped and counted rather
                # than pairing a stale leg with a fresh one.
                common_dates = []
                for sd in sorted(set(short_by_date) | set(long_by_date)):
                    if sd in short_by_date and sd in long_by_date:
                        common_dates.append(sd)
                    else:
                        rep.drop("leg_snapshot_missing")

                if len(common_dates) < 2:
                    rep.drop("fewer_than_two_snapshots")
                    continue
                if not _gap_ok(common_dates, max_gap_days):
                    rep.drop("gap_too_large")
                    continue

                exp_day = date.fromisoformat(expiration[:10])
                points = tuple(
                    PathPoint(
                        date=sd,
                        bid=short_by_date[sd][0] - long_by_date[sd][1],
                        ask=short_by_date[sd][1] - long_by_date[sd][0],
                        mid=((short_by_date[sd][0] + short_by_date[sd][1]) / 2.0
                             - (long_by_date[sd][0] + long_by_date[sd][1]) / 2.0),
                        spot=(float(short_by_date[sd][2])
                              if short_by_date[sd][2] is not None else None),
                        dte=(exp_day - date.fromisoformat(sd[:10])).days,
                        spread_imputed=False,
                    )
                    for sd in common_dates
                )
            else:
                dedup_dates = sorted(short_by_date)
                if len(dedup_dates) < 2:
                    rep.drop("fewer_than_two_snapshots")
                    continue
                if not _gap_ok(dedup_dates, max_gap_days):
                    rep.drop("gap_too_large")
                    continue

                exp_day = date.fromisoformat(expiration[:10])
                points = tuple(
                    PathPoint(date=sd, bid=short_by_date[sd][0],
                              ask=short_by_date[sd][1],
                              mid=(short_by_date[sd][0] + short_by_date[sd][1]) / 2.0,
                              spot=(float(short_by_date[sd][2])
                                    if short_by_date[sd][2] is not None else None),
                              dte=(exp_day - date.fromisoformat(sd[:10])).days)
                    for sd in dedup_dates
                )

            if len(points) < 2:
                rep.drop("snapshots_without_quotes")
                continue

            out.append(PricePath(
                position_id=f"ledger:{eid}", symbol=ticker, strategy=strat,
                entry_date=entry_date[:10], entry_price=abs(float(price)),
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
