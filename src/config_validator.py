"""Config validation — importable independently of config_ai.py."""

import logging

_logger = logging.getLogger(__name__)


def validate_core_config(cfg: dict) -> list:
    """Validate core screener config.json. Returns list of warning strings (empty = OK)."""
    warnings = []

    # Composite weights should sum to ~1.0
    cw = cfg.get("composite_weights", {})
    if cw:
        total = sum(cw.values())
        if not (0.85 <= total <= 1.15):
            warnings.append(f"composite_weights sum to {total:.3f} (expected ~1.0)")

    # Filters sanity
    f = cfg.get("filters", {})
    min_dte = f.get("min_days_to_expiration", 7)
    max_dte = f.get("max_days_to_expiration", 45)
    if min_dte >= max_dte:
        warnings.append(f"min_days_to_expiration ({min_dte}) >= max_days_to_expiration ({max_dte})")

    d_min = f.get("delta_min", 0.15)
    d_max = f.get("delta_max", 0.35)
    if d_min >= d_max:
        warnings.append(f"delta_min ({d_min}) >= delta_max ({d_max})")

    # Exit rules
    er = cfg.get("exit_rules", {})
    tp = er.get("take_profit", 0.50)
    sl = er.get("stop_loss", -0.25)
    if tp <= 0:
        warnings.append(f"exit_rules.take_profit must be > 0, got {tp}")
    if sl >= 0:
        warnings.append(f"exit_rules.stop_loss must be < 0, got {sl}")

    return warnings


def validate_ai_config(cfg: dict) -> None:
    """Raise ValueError with a clear message if any config value is out of range."""
    aw = cfg.get("ai_weight", 0)
    tw = cfg.get("technical_weight", 0)
    if not (0.0 <= aw <= 1.0):
        raise ValueError(f"ai_weight must be in [0.0, 1.0], got {aw}")
    if not (0.0 <= tw <= 1.0):
        raise ValueError(f"technical_weight must be in [0.0, 1.0], got {tw}")
    if aw + tw > 1.05:
        raise ValueError(f"ai_weight + technical_weight must be <= 1.05, got {aw + tw}")
    bs = cfg.get("batch_size", 1)
    if bs < 1:
        raise ValueError(f"batch_size must be >= 1, got {bs}")
    mt = cfg.get("max_tokens", 256)
    if mt < 256:
        raise ValueError(f"max_tokens must be >= 256, got {mt}")
    temp = cfg.get("temperature", 0)
    if not (0.0 <= temp <= 2.0):
        raise ValueError(f"temperature must be in [0.0, 2.0], got {temp}")
    fields = cfg.get("fields_to_include", [])
    if not isinstance(fields, list) or len(fields) == 0:
        raise ValueError(f"fields_to_include must be a non-empty list, got: {fields!r}")
    pf = cfg.get("divergence_penalty_factor", 0.15)
    if not (0.0 <= pf <= 0.5):
        raise ValueError(f"divergence_penalty_factor must be in [0.0, 0.5], got {pf}")
    bf = cfg.get("divergence_boost_factor", 0.10)
    if not (0.0 <= bf <= 0.5):
        raise ValueError(f"divergence_boost_factor must be in [0.0, 0.5], got {bf}")
