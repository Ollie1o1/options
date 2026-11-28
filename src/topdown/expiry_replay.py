"""Read archived single-leg quotes for an educational expiry comparison."""
from __future__ import annotations

import datetime as dt
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from src.paths import repo_path

ARCHIVE_DB = "data/chain_archive.db"
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


@dataclass(frozen=True)
class ExpiryOutcome:
    max_loss: float
    break_even: Optional[float]
    bid_ask_gap: float
    pnl_at_ask: float
    pnl_at_midpoint: float


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
            name if name in columns else "NULL" for name in ("source", "snapshot_ts", "last_trade_time", "spot")
        )
        rows = conn.execute(
            f"SELECT contract, expiration, strike, bid, ask, {provenance} FROM chain_snapshots "
            "WHERE symbol = ? AND snap_date = ? AND LOWER(type) = ? ORDER BY expiration, strike, contract",
            (symbol, snapshot_date, option_type),
        ).fetchall()

    quotes: list[ReplayQuote] = []
    for contract, expires, strike_raw, bid_raw, ask_raw, source, stored_at, trade_at, spot_raw in rows:
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


def replay_record(quotes: Sequence[ReplayQuote], expiration_price: float, fee: float) -> dict[str, Any]:
    """Freeze inputs and dollar results under a versioned comparison schema."""
    return {
        "schema_version": 1,
        "model": "Single long option, cash-equivalent payoff at its own expiration",
        "entry_price_assumptions": ["archived ask", "archived midpoint (sensitivity)"],
        "contract_multiplier_assumed": CONTRACT_MULTIPLIER,
        "scenario_stock_price_usd": expiration_price,
        "simulated_entry_fee_usd_per_contract": fee,
        "comparisons": [
            {"quote": asdict(quote), "outcome_usd": asdict(expiry_outcome(quote, expiration_price, fee))}
            for quote in quotes
        ],
    }
