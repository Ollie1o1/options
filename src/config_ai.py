"""AI scorer configuration."""

import logging

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    "provider": "openrouter",
    # Key chain (2026-07-27): three current keys — Ling 3.0 primary, Nemotron 3 Ultra
    # first fallback, Laguna S 2.1 second. All previous keys (z-ai, gpt-oss, elephant,
    # arcee, stepfun, hunter) were retired and their env vars removed from .env.
    # Only three models in the chain, so the 4th attempt re-tries the last one.
    # Session-level circuit breaker in AIScorer still blacklists any model after
    # 2 consecutive hard-timeouts.
    "model": "inclusionai/ling-3.0-flash:free",                            # attempt 1 (PRIMARY)
    "fallback_model": "nvidia/nemotron-3-ultra-550b-a55b:free",            # attempt 2
    "second_fallback_model": "poolside/laguna-s-2.1:free",                 # attempt 3
    "third_fallback_model": None,                                          # unused — no 4th key
    "api_key_env": "OPENROUTER_API_KEY",

    # ── Per-model API key overrides ───────────────────────────────────────────
    # Maps model id → env var name for models that use a different key.
    # Models NOT listed here fall back to "api_key_env".
    "model_key_map": {
        "inclusionai/ling-3.0-flash:free":         "OPENROUTER_LING_KEY",
        "nvidia/nemotron-3-ultra-550b-a55b:free":  "OPENROUTER_NVIDIA_KEY",
        "poolside/laguna-s-2.1:free":              "OPENROUTER_POOLSIDE_KEY",
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
    "thematic_analysis_enabled": False,   # opt-in: AI Pass 0 sector sentiment (one extra API call per session)

    # ── Sector ETF label map (used by analyze_thematic_sentiment) ─────────────
    "etf_to_sector": {
        "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
        "XLY": "Consumer Discretionary", "XLP": "Consumer Staples",
        "XLV": "Health Care", "XLI": "Industrials", "XLB": "Materials",
        "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services",
    },

    # ── API Call Settings ─────────────────────────────────────────────────────
    "batch_size": 3,     # smaller batches = shorter responses = less truncation risk
    "max_tokens": 2048,  # enough room for 3 candidates at ~600 tokens each with margin
    "temperature": 0.1,
    "timeout": 30,       # raised from 12 (2026-07-27): measured nemotron-3-ultra at ~20s
                         # on a real scoring prompt, so a 12s cap killed the 1st fallback on
                         # every call and blacklisted it after 2 attempts. Primary (ling-3.0-flash)
                         # answers in ~2s, so this only affects the slow fallbacks.

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
        "option_rvol", "iv_skew_rank", "vrp_mean", "vrp_regime",
        "predicted_iv_crush", "crush_confidence", "max_gamma_strike", "gamma_pin_dist_pct",
        "momentum_confluence", "risk_flag_count", "max_pain_strike", "max_pain_dist_pct", "hv_parkinson",
        "iv_surface_residual",
    ],
}


logger = logging.getLogger(__name__)
_warned_models: set = set()


def resolve_api_key_env(model_id: str, config: dict) -> str:
    """Return the env-var name holding the API key for *model_id*.

    Priority:
    1. Explicit ``model_key_map`` entry in *config*.
    2. Prefix-based lookup (inclusionai/, nvidia/, poolside/, anthropic/).
    3. Fallback to ``config["api_key_env"]``.
    """
    explicit = config.get("model_key_map", {})
    if model_id in explicit:
        return explicit[model_id]
    prefix_map = {
        "inclusionai/": "OPENROUTER_LING_KEY",
        "nvidia/":      "OPENROUTER_NVIDIA_KEY",
        "poolside/":    "OPENROUTER_POOLSIDE_KEY",
        "anthropic/":   "ANTHROPIC_API_KEY",
    }
    for prefix, env_var in prefix_map.items():
        if model_id.startswith(prefix):
            return env_var
    fallback_env = config.get("api_key_env", "OPENROUTER_API_KEY")
    if model_id not in _warned_models:
        _warned_models.add(model_id)
        logger.warning(
            "Model '%s' not in model_key_map and no prefix match — "
            "falling back to default key env '%s'. "
            "Add an explicit entry to model_key_map to silence this warning.",
            model_id, fallback_env,
        )
    return fallback_env


