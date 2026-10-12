"""Scorecard for the operator's own long-option picks.

The ledger has tagged who opened a trade since schema v25 (`trades.origin`),
and nothing read the tag. This module does: it measures the trades the
operator logged by hand against what the board offered on the same days.

It observes. Nothing gates, sizes or enters on anything here, and the ledger
is opened read-only.

DEFINITIONS - fixed on 2026-10-11, before a single manual pick existed.

  pick         a ledger row with origin='manual' and strategy_name
               'Long Call' or 'Long Put'. NULL origin predates the column and
               is NOT a pick.
  session k    the k-th trading session after the entry date, k in 1, 2, 3,
               counted on the stock's own daily closes.
  direction    sign * (stock return - SPY return), entry-day close to
               session-k close; sign is +1 for a call and -1 for a put.
               Answers "right name, right way?" free of option pricing.
  option       bid at session k / recorded entry price - 1. The entry price
               is the fill that crossed the spread when the ledger has one,
               otherwise the logged entry price. A zero bid is a total loss;
               a missing quote is not recorded and is never counted as zero.
  board        every gate_passed=1 Long Call/Long Put in data/candidates.db
               on a day that has a pick, minus the picked contracts; one
               observation per (contract, day), bought at the recorded ask.
               Its option column exists only while the simulated position
               stayed open, so sessions 2 and 3 undercount early exits.
  cluster      (ticker, ISO week of entry). Statistics are over cluster
               means; the interval is a bootstrap over clusters.
  verdict      none below MIN_CLUSTERS clusters: the numbers are shown with
               "too few to judge" and no interval.
"""
from __future__ import annotations

import logging
import os
import random
import sqlite3
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from . import candidate_marks as cm
from . import candidate_record as cr

log = logging.getLogger(__name__)

SESSIONS = (1, 2, 3)
MIN_CLUSTERS = 30
BENCHMARK = "SPY"
# Three sessions never span more than a week of calendar days, holidays included.
MARK_WINDOW_DAYS = 7
_BOOTSTRAP_DRAWS = 2000
_STRATEGIES = ("Long Call", "Long Put")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pick_marks (
  entry_id  INTEGER NOT NULL,
  mark_date TEXT NOT NULL,
  bid REAL, ask REAL,
  PRIMARY KEY (entry_id, mark_date)
);
"""

Cluster = Tuple[str, int, int]
Closes = Dict[str, float]


def _ledger_path(ledger_path: Optional[str]) -> str:
    if ledger_path:
        return ledger_path
    from .check_pnl import DB_PATH
    return DB_PATH


def _connect(db_path: Optional[str]) -> sqlite3.Connection:
    conn = cm.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


# ── Definitions ──────────────────────────────────────────────────────────────

def session_date(dates: Iterable[str], entry_date: str, k: int) -> Optional[str]:
    """The k-th session after `entry_date`, or None if it has not happened."""
    ordered = sorted(dates)
    if entry_date not in ordered:
        return None
    i = ordered.index(entry_date) + k
    return ordered[i] if i < len(ordered) else None


def direction_return(closes: Closes, benchmark: Closes, entry_date: str,
                     k: int, opt_type: str) -> Optional[float]:
    target = session_date(closes.keys(), entry_date, k)
    if target is None:
        return None
    try:
        stock = closes[target] / closes[entry_date] - 1.0
        bench = benchmark[target] / benchmark[entry_date] - 1.0
    except (KeyError, ZeroDivisionError, TypeError):
        return None
    sign = -1.0 if str(opt_type).lower() == "put" else 1.0
    return sign * (stock - bench)


def option_return(entry_price: Optional[float],
                  bid: Optional[float]) -> Optional[float]:
    if bid is None or entry_price is None or entry_price <= 0:
        return None
    return bid / entry_price - 1.0


def cluster_key(ticker: str, entry_date: str) -> Cluster:
    iso = date.fromisoformat(entry_date[:10]).isocalendar()
    return (ticker, iso[0], iso[1])


def summarize(observations: List[Tuple[Cluster, float]]) -> Dict[str, Any]:
    """Cluster-mean statistics; an interval only at MIN_CLUSTERS or more."""
    by_cluster: Dict[Cluster, List[float]] = {}
    for key, value in observations:
        by_cluster.setdefault(key, []).append(value)
    means = [sum(v) / len(v) for _, v in sorted(by_cluster.items())]
    n = len(means)
    out: Dict[str, Any] = {"n_obs": len(observations), "n_clusters": n,
                           "mean": None, "hit_rate": None, "ci": None}
    if not n:
        return out
    out["mean"] = sum(means) / n
    out["hit_rate"] = sum(1 for m in means if m > 0) / n
    if n >= MIN_CLUSTERS:
        rng = random.Random(0)
        draws = sorted(sum(rng.choices(means, k=n)) / n
                       for _ in range(_BOOTSTRAP_DRAWS))
        out["ci"] = (draws[int(0.025 * _BOOTSTRAP_DRAWS)],
                     draws[int(0.975 * _BOOTSTRAP_DRAWS) - 1])
    return out


# ── The ledger side ──────────────────────────────────────────────────────────

def manual_picks(ledger_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = _ledger_path(ledger_path)
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)")}
        if "origin" not in cols:
            return []
        rows = conn.execute(
            "SELECT entry_id, date, ticker, expiration, strike, type, "
            "entry_price, entry_price_cross FROM trades "
            "WHERE origin = 'manual' AND strategy_name IN (?, ?) "
            "ORDER BY entry_id", _STRATEGIES).fetchall()
    finally:
        conn.close()
    picks = []
    for r in rows:
        pick = dict(r)
        pick["date"] = str(pick["date"])[:10]
        pick["expiration"] = str(pick["expiration"])[:10]
        pick["type"] = str(pick["type"]).lower()
        cross = pick.pop("entry_price_cross")
        pick["entry_basis"] = cross if cross is not None else pick["entry_price"]
        picks.append(pick)
    return picks


@cr._safe(default=0)
def record_pick_marks(*, ledger_path: Optional[str] = None,
                      db_path: Optional[str] = None,
                      today: Optional[str] = None, fetch=None) -> int:
    """Quote every manual pick still inside its three-session window.

    The candidate marker stops quoting a contract once its simulated exit
    fires, which would drop exactly the picks that moved most. These marks
    run to the end of the window regardless.
    """
    if today is None:
        today = date.today().isoformat()
    fetch = fetch or cm._default_fetch
    earliest = (date.fromisoformat(today)
                - timedelta(days=MARK_WINDOW_DAYS)).isoformat()
    due = [p for p in manual_picks(ledger_path)
           if earliest <= p["date"] <= today]
    if not due:
        return 0
    chains: Dict[Tuple[str, str], Dict[Any, Any]] = {}
    written = 0
    conn = _connect(db_path)
    try:
        for p in due:
            pair = (p["ticker"], p["expiration"])
            if pair not in chains:
                chains[pair] = cm._fetch_chain(fetch, *pair)
            quote = chains[pair].get((float(p["strike"]), p["type"]))
            if not quote or quote[0] is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO pick_marks VALUES (?,?,?,?)",
                (p["entry_id"], today, quote[0], quote[1]))
            written += 1
        conn.commit()
    finally:
        conn.close()
    return written


# ── The scorecard ────────────────────────────────────────────────────────────

def _default_closes(ticker: str) -> Closes:
    from .paper_manager import _get_yf_and_session
    yf, session = _get_yf_and_session()
    hist = yf.Ticker(ticker, session=session).history(period="6mo",
                                                      interval="1d")
    return {ts.strftime("%Y-%m-%d"): float(close)
            for ts, close in hist["Close"].dropna().items()}


def _board_rows(conn: sqlite3.Connection, days: List[str],
                picked: set) -> List[Dict[str, Any]]:
    """One row per (contract, day) the board passed and the operator did not take."""
    # A ts range per day, not substr(ts): the range is what the
    # (strategy_name, ts) index can serve on a multi-gigabyte table. The day
    # is the scan timestamp's own (UTC) date.
    rows: List[tuple] = []
    for day in days:
        rows.extend(conn.execute(
            "SELECT contract_key, symbol, expiration, strike, opt_type, "
            "substr(ts, 1, 10) AS day, ask FROM candidates "
            "WHERE gate_passed = 1 AND strategy_name IN (?, ?) "
            "AND ts >= ? AND ts < ? ORDER BY ts",
            (*_STRATEGIES, day, day + "~")).fetchall())
    seen = set()
    out = []
    for key, symbol, expiration, strike, opt_type, day, ask in rows:
        opt_type = str(opt_type).lower()
        if (key, day) in seen:
            continue
        seen.add((key, day))
        if (symbol, str(expiration)[:10], opt_type, float(strike), day) in picked:
            continue
        out.append({"key": key, "ticker": symbol, "type": opt_type,
                    "date": day, "entry_basis": ask})
    return out


def _score(rows: List[Dict[str, Any]], closes: Callable[[str], Closes],
           bid_at: Callable[[Dict[str, Any], str], Optional[float]]
           ) -> Dict[str, Dict[int, Dict[str, Any]]]:
    direction: Dict[int, list] = {k: [] for k in SESSIONS}
    option: Dict[int, list] = {k: [] for k in SESSIONS}
    bench = closes(BENCHMARK)
    for row in rows:
        series = closes(row["ticker"])
        cluster = cluster_key(row["ticker"], row["date"])
        for k in SESSIONS:
            d = direction_return(series, bench, row["date"], k, row["type"])
            if d is not None:
                direction[k].append((cluster, d))
            target = session_date(series.keys(), row["date"], k)
            if target is None:
                continue
            o = option_return(row["entry_basis"], bid_at(row, target))
            if o is not None:
                option[k].append((cluster, o))
    return {"direction": {k: summarize(direction[k]) for k in SESSIONS},
            "option": {k: summarize(option[k]) for k in SESSIONS}}


def build_scorecard(*, ledger_path: Optional[str] = None,
                    db_path: Optional[str] = None,
                    closes_fn: Optional[Callable[[str], Closes]] = None,
                    include_board: bool = True) -> Dict[str, Any]:
    picks = manual_picks(ledger_path)
    if not picks:
        return {"n_picks": 0, "picks": None, "board": None}

    fetch_closes = closes_fn or _default_closes
    cache: Dict[str, Closes] = {}

    def closes(ticker: str) -> Closes:
        if ticker not in cache:
            try:
                cache[ticker] = fetch_closes(ticker) or {}
            except Exception:
                log.warning("no closes for %s", ticker, exc_info=True)
                cache[ticker] = {}
        return cache[ticker]

    conn = _connect(db_path)
    try:
        pick_marks = {(e, d): bid for e, d, bid in conn.execute(
            "SELECT entry_id, mark_date, bid FROM pick_marks")}
        days = sorted({p["date"] for p in picks})
        picked = {(p["ticker"], p["expiration"], p["type"],
                   float(p["strike"]), p["date"]) for p in picks}
        board = _board_rows(conn, days, picked) if include_board else []
        keys = sorted({b["key"] for b in board})
        cand_marks: Dict[Tuple[str, str], Optional[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            cand_marks.update({(k, d): bid for k, d, bid in conn.execute(
                "SELECT contract_key, mark_date, bid FROM candidate_marks "
                f"WHERE contract_key IN ({','.join('?' * len(chunk))})",
                chunk)})
    finally:
        conn.close()

    return {
        "n_picks": len(picks),
        "picks": _score(picks, closes,
                        lambda p, d: pick_marks.get((p["entry_id"], d))),
        "board": _score(board, closes,
                        lambda b, d: cand_marks.get((b["key"], d)))
        if include_board else None,
    }


# ── Rendering ────────────────────────────────────────────────────────────────

def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _cell(stat: Dict[str, Any]) -> str:
    from . import formatting as fmt
    if not stat["n_clusters"]:
        return "not recorded"
    text = (f"{fmt.style_sign(_pct(stat['mean']), stat['mean'])}"
            f"  {stat['hit_rate'] * 100:.0f}% right"
            f"  n={stat['n_clusters']}")
    if stat["ci"] is not None:
        text += f"  CI [{_pct(stat['ci'][0])}, {_pct(stat['ci'][1])}]"
    return text


def render(card: Dict[str, Any]) -> List[str]:
    from . import formatting as fmt
    lines = [fmt.format_header("OWN PICKS - manual long options, 1-3 sessions")]
    if not card["n_picks"]:
        lines.append("  No manual long-option picks in the ledger yet. Log one "
                     "from a scan menu (or a bare ticker) and it is scored here.")
        return lines
    for title, side in (("Your picks", card["picks"]),
                        ("Board, same days, not picked", card["board"])):
        if side is None:
            continue
        lines.append(f"  {title}")
        for label, group in (("direction vs SPY", side["direction"]),
                             ("option, entry to bid", side["option"])):
            for k in SESSIONS:
                lines.append(f"    {label:<21} +{k}  {_cell(group[k])}")
    clusters = card["picks"]["direction"][1]["n_clusters"]
    if clusters < MIN_CLUSTERS:
        lines.append(f"  {clusters} of {MIN_CLUSTERS} clusters: too few to "
                     "judge. No interval, no verdict.")
    lines.append("  A cluster is one ticker in one week. Coin-flip base rate "
                 "for direction is 48-52%.")
    return lines


def panel_lines(*, ledger_path: Optional[str] = None,
                db_path: Optional[str] = None,
                closes_fn: Optional[Callable[[str], Closes]] = None
                ) -> List[str]:
    """The portfolio view's short form: the picks alone, no board fetch."""
    card = build_scorecard(ledger_path=ledger_path, db_path=db_path,
                           closes_fn=closes_fn, include_board=False)
    lines = render(card)
    if card["n_picks"]:
        lines.append("  Against the board: python -m src.own_picks")
    return lines


def main() -> None:
    for line in render(build_scorecard()):
        print(line)


if __name__ == "__main__":
    main()
