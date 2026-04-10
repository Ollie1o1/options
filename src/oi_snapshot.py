"""OI snapshot persistence — track open interest changes between runs."""

import json

import pandas as pd

_OI_SNAPSHOT_PATH = ".oi_snapshot.json"


def load_oi_snapshot() -> dict:
    """Load previous OI snapshot {symbol_strike_expiry_type: oi}."""
    try:
        with open(_OI_SNAPSHOT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_oi_snapshot(df_picks: pd.DataFrame) -> None:
    """Save current OI values keyed by symbol+strike+expiry+type."""
    if df_picks.empty:
        return
    snapshot = {}
    for _, row in df_picks.iterrows():
        key = f"{row.get('symbol','')}_{row.get('strike','')}_{row.get('expiration','')}_{row.get('type','')}"
        oi_val = row.get("openInterest", 0)
        if pd.isna(oi_val):
            oi_val = 0
        snapshot[key] = int(oi_val)
    try:
        with open(_OI_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f)
    except Exception:
        pass
