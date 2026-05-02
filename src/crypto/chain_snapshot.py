"""Persistent chain snapshots for crypto options.

Every time the user runs a live Deribit fetch, we save a copy of the chain
to disk keyed by (date, currency). Over weeks/months this accumulates into
a real historical chain dataset that the backtester can replay against
without paying for a third-party data feed.

Storage layout:
    data/crypto_snapshots/<YYYY-MM-DD>/<CURRENCY>.parquet

Parquet is used because option chains are wide (10+ columns × 800+ rows
per snapshot), and parquet keeps the disk footprint small while preserving
dtypes. Falls back to compressed JSON if pyarrow/fastparquet aren't
available — never silently fails.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SNAPSHOT_ROOT = _PROJECT_ROOT / "data" / "crypto_snapshots"


def _today_str() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _path_for(date_str: str, currency: str, ext: str = "parquet") -> Path:
    return _SNAPSHOT_ROOT / date_str / f"{currency.upper()}.{ext}"


def save_snapshot(chain: pd.DataFrame, currency: str, date_str: Optional[str] = None) -> Optional[Path]:
    """Save `chain` to disk under data/crypto_snapshots/<date>/<currency>.parquet.

    Idempotent — overwrites any existing snapshot for the same (date, currency).
    Returns the path written, or None on failure.
    """
    if chain is None or chain.empty:
        return None
    date_str = date_str or _today_str()
    target_dir = _SNAPSHOT_ROOT / date_str
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    # Convert date columns to string for cross-engine compatibility.
    df = chain.copy()
    if "expiration" in df.columns:
        df["expiration"] = df["expiration"].astype(str)
    df["snapshot_date"] = date_str

    # Try parquet first (compact + typed).
    parquet_path = _path_for(date_str, currency, "parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError, Exception):
        pass

    # Fallback: gzip-compressed JSON (always works).
    json_path = _path_for(date_str, currency, "json.gz")
    try:
        with gzip.open(json_path, "wt") as f:
            json.dump(df.to_dict("records"), f, default=str)
        return json_path
    except (OSError, TypeError):
        return None


def load_snapshot(date_str: str, currency: str) -> Optional[pd.DataFrame]:
    """Load a single snapshot. Returns None if not present or unreadable."""
    parquet_path = _path_for(date_str, currency, "parquet")
    json_path = _path_for(date_str, currency, "json.gz")

    if parquet_path.is_file():
        try:
            df = pd.read_parquet(parquet_path)
            if "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
            return df
        except (ImportError, ValueError, OSError):
            pass

    if json_path.is_file():
        try:
            with gzip.open(json_path, "rt") as f:
                rows = json.load(f)
            df = pd.DataFrame(rows)
            if not df.empty and "expiration" in df.columns:
                df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
            return df
        except (OSError, ValueError):
            pass

    return None


def list_snapshots(currency: Optional[str] = None) -> List[str]:
    """Return the sorted list of dates for which snapshots exist."""
    if not _SNAPSHOT_ROOT.is_dir():
        return []
    dates = []
    for entry in _SNAPSHOT_ROOT.iterdir():
        if not entry.is_dir():
            continue
        if currency is None:
            dates.append(entry.name)
            continue
        if (entry / f"{currency.upper()}.parquet").is_file() or (entry / f"{currency.upper()}.json.gz").is_file():
            dates.append(entry.name)
    return sorted(dates)


def snapshot_count(currency: Optional[str] = None) -> int:
    return len(list_snapshots(currency))
