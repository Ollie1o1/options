"""One collection path for the research CLI and existing maintenance scheduler."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.breakout.data import DEFAULT_DB
from src.breakout.refresh import refresh_universe
from src.paths import repo_path
from .coverage import snapshot, write_report


def collect(*, config_path: str = repo_path("configs/discovery_research_v1.json"),
            prices_path: str = DEFAULT_DB, news_path: str = repo_path("data/news_archive.db"),
            analysts_path: str = repo_path("data/estimates_archive.db"),
            output_dir: str = repo_path("reports/discovery"), update: bool = True,
            now: Optional[datetime] = None) -> dict:
    spec = json.loads(Path(config_path).read_text())
    symbols = list(spec["universe"])
    # Keep collecting existing symbols instead of silently shrinking their history.
    if Path(prices_path).exists():
        try:
            with closing(sqlite3.connect(Path(prices_path).resolve().as_uri() + "?mode=ro", uri=True)) as con:
                symbols = list(dict.fromkeys(symbols + [r[0] for r in con.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")]))
        except sqlite3.Error:
            # The coverage reader reports the actual schema/read error as
            # unknown, including when an existing file has no price table.
            pass
    result = refresh_universe(prices_path, symbols, now=now) if update else None
    coverage = snapshot(prices_path, news_path, analysts_path, symbols, spec["evaluation"]["start"], now)
    if result is not None:
        coverage["collection"] = result
    write_report(coverage, output_dir)
    return {"complete": coverage["prices"]["status"] == "current" and (result is None or result["complete"]),
            "coverage": coverage, "collection": result}


def due(state: dict, now: datetime) -> bool:
    from zoneinfo import ZoneInfo
    from src.data_quality import holiday_calendar_covers, is_trading_day
    et = ZoneInfo("America/New_York")
    local = now.replace(tzinfo=et) if now.tzinfo is None else now.astimezone(et)
    return (local.hour >= 18 and holiday_calendar_covers(local.year) and is_trading_day(local) and
            state.get("last_discovery_prices") != local.date().isoformat())
