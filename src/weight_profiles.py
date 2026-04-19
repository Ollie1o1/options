"""Load, list, and validate weight profiles for the screener.

A weight profile is a JSON file under ``configs/weights/<name>.json`` containing
a ``composite_weights`` dict. Profiles are merged into ``config.json`` at scan
time via ``run_scan(custom_weights=...)``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

PROFILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "weights",
)

KNOWN_KEYS = {
    "pop", "em_realism", "iv_mispricing", "rr", "momentum", "iv_rank",
    "liquidity", "catalyst", "theta", "ev", "trader_pref", "iv_edge",
    "skew_align", "gamma_theta", "pcr", "gex", "oi_change", "sentiment",
    "option_rvol", "vrp", "gamma_pin", "max_pain", "iv_velocity",
    "gamma_magnitude", "vega_risk", "term_structure", "spread",
}


def _resolve(name_or_path: str) -> str:
    if os.path.sep in name_or_path or name_or_path.endswith(".json"):
        return name_or_path
    return os.path.join(PROFILES_DIR, f"{name_or_path}.json")


def load_weight_profile(name_or_path: str) -> Tuple[str, Dict[str, float]]:
    """Return ``(profile_id, weights_dict)``.

    ``profile_id`` is the filename stem (e.g. ``baseline``) and is what gets
    stored on each logged trade so profiles can be compared later.
    """
    path = _resolve(name_or_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Weight profile not found: {path}. "
            f"Available: {', '.join(list_profiles()) or '(none)'}"
        )
    with open(path, "r") as f:
        raw = json.load(f)

    weights = raw.get("composite_weights", raw)
    if not isinstance(weights, dict):
        raise ValueError(f"{path}: expected dict under 'composite_weights'")

    cleaned: Dict[str, float] = {}
    for k, v in weights.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"{path}: weight '{k}' must be numeric, got {type(v).__name__}")
        if k not in KNOWN_KEYS:
            logger.warning("Weight profile %s: unknown key '%s' (will be passed through)", path, k)
        cleaned[k] = float(v)

    profile_id = os.path.splitext(os.path.basename(path))[0]
    return profile_id, cleaned


def list_profiles() -> List[str]:
    if not os.path.isdir(PROFILES_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(PROFILES_DIR)
        if f.endswith(".json")
    )
