"""AI scorer configuration."""

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    "provider": "openrouter",
    "model": "arcee-ai/trinity-large-preview:free",
    "fallback_model": "openrouter/hunter-alpha",                       # used after 2 failed retries
    "second_fallback_model": "nvidia/nemotron-3-super-120b-a12b:free", # used after 3 failed retries
    "third_fallback_model": "meta-llama/llama-3.3-70b-instruct:free",  # used after 4 failed retries
    "api_key_env": "OPENROUTER_ARCEE_KEY",

    # ── Per-model API key overrides ───────────────────────────────────────────
    # Maps model id → env var name for models that use a different key.
    # Models NOT listed here fall back to "api_key_env".
    "model_key_map": {
        "arcee-ai/trinity-large-preview:free":  "OPENROUTER_ARCEE_KEY",
        "openrouter/hunter-alpha":              "OPENROUTER_HUNTER_KEY",
        "nvidia/nemotron-3-super-120b-a12b:free": "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct:free": "OPENROUTER_API_KEY",
        "mistralai/mistral-7b-instruct:free":   "OPENROUTER_API_KEY",
    },

    # ── Scoring Weights ───────────────────────────────────────────────────────
    "ai_weight": 0.30,
    "technical_weight": 0.70,
    "divergence_penalty_factor": 0.15,
    "divergence_boost_factor":   0.10,

    # ── Dynamic weight multipliers by VIX regime ──────────────────────────────
    # final ai_weight = base_ai_weight * regime_mult * confidence_mult * liquidity_adj
    "regime_weight_multipliers": {
        "low":    0.80,   # quant signals dominate; catalyst risk low
        "normal": 1.00,
        "high":   1.30,   # catalyst/news awareness worth more
    },

    # ── Feature flags ─────────────────────────────────────────────────────────
    "cache_enabled": True,          # same-day SQLite score cache
    "confidence_enabled": True,     # AI returns ai_confidence (0-10)
    "two_pass_enabled": True,       # ticker-level context pass before contract scoring
    "news_enabled": True,           # inject top-3 news headlines into prompt

    # ── API Call Settings ─────────────────────────────────────────────────────
    "batch_size": 3,     # smaller batches = shorter responses = less truncation risk
    "max_tokens": 4096,  # enough room for 3 candidates with full reasoning
    "temperature": 0.1,
    "timeout": 60,

    # ── Narrative thresholds for context enrichment ───────────────────────────
    "narrative_thresholds": {
        "iv_rank_high": 0.70,       # above this = "expensive"
        "iv_rank_low": 0.30,        # below this = "cheap"
        "iv_vs_hv_rich": 0.05,      # IV > HV by this much = seller edge
        "iv_vs_hv_cheap": -0.05,    # IV < HV by this much = buyer edge
        "pop_strong": 0.65,
        "pop_weak": 0.45,
        "rr_good": 1.5,
        "rr_poor": 0.75,
        "rvol_unusual": 1.5,
        "theta_decay_high": 0.05,   # theta/premium ratio
        "spread_wide": 0.15,
        "divergence_flag_threshold": 0.20,
    },

    # ── Polygon.io enrichment settings ───────────────────────────────────────────
    "polygon": {
        "enabled": True,
        "news_limit": 10,
        "unusual_flow_min_premium": 25_000,
        "news_max_age_hours": 48,
    },

    # ── Fields sent to AI (keep only what the narrative does NOT already cover) ──
    "fields_to_include": [
        "symbol", "type", "strike", "expiration", "underlying",
        "premium", "iv_rank", "delta",
        "prob_profit", "pop_sim", "ev_per_contract", "rr_ratio",
        "be_dist_pct", "annualized_return",
        "Earnings Play", "Trend_Aligned",
        "macro_warning", "sr_warning", "decay_warning", "gamma_ramp",
    ],
}


def resolve_api_key_env(model_id: str, config: dict) -> str:
    """Return the env-var name holding the API key for *model_id*.

    Priority:
    1. Explicit ``model_key_map`` entry in *config*.
    2. Prefix-based lookup (arcee-ai/, openrouter/, anthropic/).
    3. Fallback to ``config["api_key_env"]``.
    """
    explicit = config.get("model_key_map", {})
    if model_id in explicit:
        return explicit[model_id]
    prefix_map = {
        "arcee-ai/": "OPENROUTER_ARCEE_KEY",
        "openrouter/": "OPENROUTER_HUNTER_KEY",
        "anthropic/": "ANTHROPIC_API_KEY",
    }
    for prefix, env_var in prefix_map.items():
        if model_id.startswith(prefix):
            return env_var
    return config.get("api_key_env", "OPENROUTER_API_KEY")


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

validate_ai_config(AI_CONFIG)
