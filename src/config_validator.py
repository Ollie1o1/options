"""AI config validation — importable independently of config_ai.py."""


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
