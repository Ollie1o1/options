"""OI snapshot persistence — track open interest changes between runs."""

import json
import logging
import os
import tempfile

import pandas as pd

_OI_SNAPSHOT_PATH = ".oi_snapshot.json"

logger = logging.getLogger(__name__)


def load_oi_snapshot() -> dict:
    """Load previous OI snapshot {symbol_strike_expiry_type: oi}."""
    try:
        with open(_OI_SNAPSHOT_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(
            "OI snapshot at %s is unreadable (%s); starting from empty — "
            "OI delta tracking will be zero until next successful save.",
            _OI_SNAPSHOT_PATH, e,
        )
        return {}


def save_oi_snapshot(df_picks: pd.DataFrame) -> None:
    """Save current OI values keyed by symbol+strike+expiry+type.

    Uses temp-file + os.replace for POSIX-atomic writes so a crash mid-write
    cannot truncate the existing snapshot.
    """
    if df_picks.empty:
        return
    snapshot = {}
    for _, row in df_picks.iterrows():
        key = f"{row.get('symbol','')}_{row.get('strike','')}_{row.get('expiration','')}_{row.get('type','')}"
        oi_val = row.get("openInterest", 0)
        if pd.isna(oi_val):
            oi_val = 0
        snapshot[key] = int(oi_val)

    tmp_path = None
    try:
        d = os.path.dirname(os.path.abspath(_OI_SNAPSHOT_PATH)) or "."
        with tempfile.NamedTemporaryFile(
            "w", dir=d, delete=False, suffix=".tmp", prefix=".oi_snapshot."
        ) as tf:
            json.dump(snapshot, tf)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_path = tf.name
        os.replace(tmp_path, _OI_SNAPSHOT_PATH)
    except OSError as e:
        logger.warning(
            "Failed to save OI snapshot to %s: %s — OI delta tracking will "
            "miss this run.", _OI_SNAPSHOT_PATH, e,
        )
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
