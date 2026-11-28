"""Read archived single-leg quotes for an educational expiry comparison."""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from src.paths import repo_path

ARCHIVE_DB = "data/chain_archive.db"
EQUITY_DB = "data/equity_ohlcv.db"
CONTRACT_MULTIPLIER = 100
_CONTRACT_ID = re.compile(r"([A-Z]{1,6})\s*(\d{6})([CP])(\d{8})")


@dataclass(frozen=True)
class ReplayQuote:
    symbol: str
    snapshot_date: str
    contract: str
    option_type: str
    expiration: str
    days_to_expiration: int
    strike: float
    bid: float
    ask: float
    source: Optional[str]
    stored_timestamp: Optional[str]
    last_trade_at: Optional[str]
    recorded_stock_price: Optional[float]
    quote_at: Optional[str] = None
    fetched_at: Optional[str] = None


@dataclass(frozen=True)
class ExpiryOutcome:
    max_loss: float
    break_even: Optional[float]
    bid_ask_gap: float
    pnl_at_ask: float
    pnl_at_midpoint: float


@dataclass(frozen=True)
class ReplayScenario:
    quotes: tuple[ReplayQuote, ...]
    expiration_price: float
    fee: float
    earnings: Optional[ArchivedEarnings] = None


@dataclass(frozen=True)
class ArchivedEarnings:
    day: str
    observed_at: str
    source: str


@dataclass(frozen=True)
class ArchiveDay:
    day: str
    market_session: bool
    rows: int
    expiries: int
    status: str


@dataclass(frozen=True)
class ReplayMark:
    day: str
    expiration: str
    status: str
    stock_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass(frozen=True)
class MarkOutcome:
    midpoint_change: float
    spread_cost: float
    fees: float
    net: float


def _connect(chain_db: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(repo_path(chain_db)).resolve().as_uri() + "?mode=ro", uri=True)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _date(value: Any) -> Optional[dt.date]:
    try:
        day = dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return day if day.isoformat() == value else None


def archive_version(path: str) -> tuple[tuple[int, int], ...]:
    """Include SQLite WAL changes when invalidating a cached read."""
    resolved = repo_path(path)
    changes = []
    for filename in (resolved, resolved + "-wal"):
        try:
            stat = Path(filename).stat()
            changes.append((stat.st_mtime_ns, stat.st_size))
        except OSError:
            changes.append((0, 0))
    return tuple(changes)


def replay_sessions(symbol: str, *, chain_db: str = ARCHIVE_DB) -> list[str]:
    """Return up to sixty recorded sessions. An unavailable archive raises a SQLite error."""
    with closing(_connect(chain_db)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT snap_date FROM chain_snapshots WHERE symbol = ? ORDER BY snap_date DESC LIMIT 60",
            (symbol.upper().strip(),),
        ).fetchall()
    return [day for (day,) in rows if _date(day) is not None]


def load_replay_quotes(
    symbol: str, snapshot_date: str, option_type: str, *, chain_db: str = ARCHIVE_DB,
) -> list[ReplayQuote]:
    """Load two-sided, standard-looking contracts from the exact recorded session."""
    symbol = symbol.upper().strip()
    session = _date(snapshot_date)
    root = symbol.replace(".", "").replace("-", "")
    if session is None or option_type not in ("call", "put") or not re.fullmatch(r"[A-Z]{1,6}", root):
        return []

    with closing(_connect(chain_db)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chain_snapshots)")}
        provenance = ", ".join(
            f"q.{name}" if name in columns else "NULL" for name in ("source", "snapshot_ts", "last_trade_time", "spot")
        )
        times = "NULL, NULL"
        join = ""
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='archive_quote_times'").fetchone():
            times = "t.quote_at, t.fetched_at"
            join = "LEFT JOIN archive_quote_times t ON q.contract=t.contract AND q.snap_date=t.snap_date "
        rows = conn.execute(
            f"SELECT q.contract, q.expiration, q.strike, q.bid, q.ask, {provenance}, {times} "
            f"FROM chain_snapshots q {join}WHERE q.symbol=? AND q.snap_date=? AND LOWER(q.type)=? "
            "ORDER BY q.expiration, q.strike, q.contract",
            (symbol, snapshot_date, option_type),
        ).fetchall()

    quotes: list[ReplayQuote] = []
    for contract, expires, strike_raw, bid_raw, ask_raw, source, stored_at, trade_at, spot_raw, quote_at, fetched_at in rows:
        expiry = _date(expires)
        identity = _CONTRACT_ID.fullmatch(str(contract))
        strike, bid, ask = _number(strike_raw), _number(bid_raw), _number(ask_raw)
        if expiry is None or expiry < session or identity is None or strike is None or strike <= 0:
            continue
        if bid is None or ask is None or not 0 < bid <= ask:
            continue
        expected = root, expiry.strftime("%y%m%d"), option_type[0].upper()
        if identity.groups()[:3] != expected or int(identity.group(4)) / 1000 != strike:
            continue
        spot = _number(spot_raw)
        quotes.append(ReplayQuote(
            symbol, snapshot_date, contract, option_type, expires, (expiry - session).days, strike, bid, ask,
            source or None, stored_at or None, trade_at or None, spot if spot is not None and spot > 0 else None,
            quote_at or None, fetched_at or None,
        ))
    return quotes


def expiry_outcome(quote: ReplayQuote, expiration_price: float, fee: float) -> ExpiryOutcome:
    """Cash-equivalent payoff of one hypothetical long contract, at its own expiration."""
    if not math.isfinite(expiration_price) or expiration_price < 0 or not math.isfinite(fee) or fee < 0:
        raise ValueError("The expiration price and fee must be finite and non-negative")
    is_call = quote.option_type == "call"
    intrinsic = max(0.0, expiration_price - quote.strike if is_call else quote.strike - expiration_price)
    cost_at_ask = quote.ask * CONTRACT_MULTIPLIER + fee
    cost_at_midpoint = (quote.bid + quote.ask) / 2 * CONTRACT_MULTIPLIER + fee
    break_even = quote.strike + (cost_at_ask / CONTRACT_MULTIPLIER) * (1 if is_call else -1)
    return ExpiryOutcome(
        max_loss=round(cost_at_ask, 2),
        break_even=round(break_even, 4) if break_even >= 0 else None,
        bid_ask_gap=round((quote.ask - quote.bid) * CONTRACT_MULTIPLIER, 2),
        pnl_at_ask=round(intrinsic * CONTRACT_MULTIPLIER - cost_at_ask, 2),
        pnl_at_midpoint=round(intrinsic * CONTRACT_MULTIPLIER - cost_at_midpoint, 2),
    )


def replay_record(
    quotes: Sequence[ReplayQuote], expiration_price: float, fee: float, *, earnings: Optional[ArchivedEarnings] = None,
) -> dict[str, Any]:
    """Freeze inputs and dollar results under a versioned comparison schema."""
    return {
        "schema_version": 1,
        "model": "Single long option, cash-equivalent payoff at its own expiration",
        "entry_price_assumptions": ["archived ask", "archived midpoint (sensitivity)"],
        "contract_multiplier_assumed": CONTRACT_MULTIPLIER,
        "scenario_stock_price_usd": expiration_price,
        "simulated_entry_fee_usd_per_contract": fee,
        "archived_earnings": asdict(earnings) if earnings is not None else None,
        "comparisons": [
            {"quote": asdict(quote), "outcome_usd": asdict(expiry_outcome(quote, expiration_price, fee))}
            for quote in quotes
        ],
    }


def read_replay_record(payload: bytes) -> ReplayScenario:
    """Restore the original inputs and ignore uploaded outcome values."""
    if len(payload) > 64 * 1024:
        raise ValueError("Replay files must be 64 KB or smaller.")
    try:
        record = json.loads(payload)
    except (ValueError, RecursionError) as exc:
        raise ValueError("This file does not contain a readable replay.") from exc
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("This replay version is unsupported.")
    comparisons = record.get("comparisons")
    if record.get("contract_multiplier_assumed") != CONTRACT_MULTIPLIER or not isinstance(comparisons, list) or len(comparisons) not in (2, 3):
        raise ValueError("Select a replay with two or three standard contracts.")

    quotes = []
    for comparison in comparisons:
        try:
            quote = ReplayQuote(**comparison["quote"])
            root = quote.symbol.replace(".", "").replace("-", "")
            expiry, session = _date(quote.expiration), _date(quote.snapshot_date)
            identity = _CONTRACT_ID.fullmatch(quote.contract)
            if expiry is None or session is None or identity is None or quote.option_type not in ("call", "put"):
                raise ValueError
            if quote.days_to_expiration != (expiry - session).days or quote.days_to_expiration < 0:
                raise ValueError
            if identity.groups()[:3] != (root, expiry.strftime("%y%m%d"), quote.option_type[0].upper()):
                raise ValueError
            if not 0 < quote.strike == int(identity.group(4)) / 1000 or not 0 < quote.bid <= quote.ask <= 1_000_000:
                raise ValueError
            if quote.recorded_stock_price is not None and not 0 < quote.recorded_stock_price <= 1_000_000:
                raise ValueError
            for value in (quote.source, quote.stored_timestamp, quote.last_trade_at, quote.quote_at, quote.fetched_at):
                if value is not None and (not isinstance(value, str) or len(value) > 128 or not value.isprintable()):
                    raise ValueError
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError("The replay contains invalid contract inputs.") from exc
        quotes.append(quote)

    identity_keys = {(q.symbol, q.snapshot_date, q.option_type, q.strike, q.source) for q in quotes}
    if len(identity_keys) != 1 or len({q.expiration for q in quotes}) != len(quotes):
        raise ValueError("The contracts must use distinct expirations from one comparison.")
    price = _number(record.get("scenario_stock_price_usd"))
    fee = _number(record.get("simulated_entry_fee_usd_per_contract"))
    if price is None or fee is None or not 0 <= price <= 1_000_000 or not 0 <= fee <= 100_000:
        raise ValueError("The saved scenario price or fee is outside the supported range.")
    earnings = None
    if record.get("archived_earnings") is not None:
        try:
            earnings = ArchivedEarnings(**record["archived_earnings"])
        except TypeError as exc:
            raise ValueError("The replay contains an invalid earnings record.") from exc
        if not _valid_earnings(earnings, quotes[0].snapshot_date):
            raise ValueError("The earnings record does not match the replay session.")
    return ReplayScenario(tuple(sorted(quotes, key=lambda q: q.expiration)), price, fee, earnings)


def _valid_earnings(event: ArchivedEarnings, session: str) -> bool:
    try:
        observed = dt.datetime.fromisoformat(event.observed_at.replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return False
    return (
        _date(event.day) is not None and event.day >= session and observed.tzinfo is not None
        and observed.astimezone(ZoneInfo("America/New_York")).date().isoformat() == session
        and isinstance(event.source, str) and 0 < len(event.source) <= 128 and event.source.isprintable()
    )


def load_archived_earnings(symbol: str, session: str, *, chain_db: str = ARCHIVE_DB) -> Optional[ArchivedEarnings]:
    with closing(_connect(chain_db)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='archive_earnings'").fetchone():
            return None
        saved = conn.execute(
            "SELECT event_date,observed_at,source FROM archive_earnings WHERE symbol=? AND snap_date=?", (symbol, session),
        ).fetchone()
    if not saved:
        return None
    event = ArchivedEarnings(*saved)
    return event if _valid_earnings(event, session) else None


def archive_days(
    symbol: str, start: str, end: str, *, chain_db: str = ARCHIVE_DB, equity_db: str = EQUITY_DB,
) -> list[ArchiveDay]:
    """Join recorded capture attempts and a reference exchange-session calendar."""
    params = (symbol, start, end)
    with closing(_connect(chain_db)) as conn:
        observed = {day: (rows, exps) for day, rows, exps in conn.execute(
            "SELECT snap_date,COUNT(*),COUNT(DISTINCT expiration) FROM chain_snapshots "
            "WHERE symbol=? AND snap_date BETWEEN ? AND ? GROUP BY snap_date", params,
        )}
        attempts = {}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='archive_attempts'").fetchone():
            attempts = dict(conn.execute(
                "SELECT snap_date,status FROM archive_attempts WHERE symbol=? AND snap_date BETWEEN ? AND ?", params,
            ))
    calendar = set()
    try:
        with closing(_connect(equity_db)) as conn:
            calendar = {day for day, in conn.execute(
                "SELECT date FROM ohlcv WHERE ticker='SPY' AND date BETWEEN ? AND ?", (start, end),
            )}
    except sqlite3.OperationalError:
        # Existing archives can be read without a local reference calendar.
        pass
    status_labels = {"no_rows": "No rows returned", "error": "Capture error", "feed_down": "Feed unavailable"}
    days = []
    for day in sorted(calendar | observed.keys() | attempts.keys()):
        rows, expiries = observed.get(day, (0, 0))
        status = "Captured" if rows else status_labels.get(attempts.get(day, ""), "No capture recorded")
        days.append(ArchiveDay(day, day in calendar, rows, expiries, status))
    return days


def load_replay_timeline(
    quotes: Sequence[ReplayQuote], *, chain_db: str = ARCHIVE_DB, equity_db: str = EQUITY_DB,
) -> list[ReplayMark]:
    """Load exact identities on dates shared by every selected expiration."""
    first = quotes[0]
    end = min(q.expiration for q in quotes)
    days = archive_days(first.symbol, first.snapshot_date, end, chain_db=chain_db, equity_db=equity_db)
    with closing(_connect(chain_db)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(chain_snapshots)")}
        source = "source" if "source" in columns else "NULL"
        spot = "spot" if "spot" in columns else "NULL"
        placeholders = ",".join("?" for _ in quotes)
        rows = conn.execute(
            f"SELECT snap_date,contract,bid,ask,{source},{spot} FROM chain_snapshots "
            f"WHERE symbol=? AND snap_date BETWEEN ? AND ? AND contract IN ({placeholders})",
            (first.symbol, first.snapshot_date, end, *(q.contract for q in quotes)),
        )
        found = {(day, contract): (bid, ask, provider, stock) for day, contract, bid, ask, provider, stock in rows}

    marks = []
    for day in sorted({d.day for d in days} | {first.snapshot_date}):
        for quote in quotes:
            if day == quote.snapshot_date:
                marks.append(ReplayMark(day, quote.expiration, "Entry snapshot", quote.recorded_stock_price, quote.bid, quote.ask))
                continue
            values = found.get((day, quote.contract))
            if values is None:
                marks.append(ReplayMark(day, quote.expiration, "No archived quote"))
                continue
            bid, ask, provider, stock = values
            if quote.source is None or provider != quote.source:
                marks.append(ReplayMark(day, quote.expiration, "Provider unverified or changed"))
                continue
            bid, ask, stock = _number(bid), _number(ask), _number(stock)
            stock = stock if stock and stock > 0 else None
            if bid is None or ask is None or not 0 < bid <= ask:
                marks.append(ReplayMark(day, quote.expiration, "No two-sided quote", stock))
            else:
                marks.append(ReplayMark(day, quote.expiration, "Archived quote", stock, bid, ask))
    return marks


def mark_outcome(entry: ReplayQuote, exit_mark: ReplayMark, fee: float) -> Optional[MarkOutcome]:
    """Decompose a simulated ask-to-bid exit using the actually archived quote."""
    if exit_mark.bid is None or exit_mark.ask is None:
        return None
    entry_mid = (entry.ask + entry.bid) / 2
    exit_mid = (exit_mark.ask + exit_mark.bid) / 2
    dollars = CONTRACT_MULTIPLIER
    return MarkOutcome(
        round((exit_mid - entry_mid) * dollars, 2),
        round(((entry.ask - entry_mid) + (exit_mid - exit_mark.bid)) * dollars, 2),
        round(2 * fee, 2),
        round((exit_mark.bid - entry.ask) * dollars - 2 * fee, 2),
    )
