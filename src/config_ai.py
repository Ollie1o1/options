"""AI scorer configuration."""

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    "provider": "openrouter",
    "model": "nvidia/nemotron-3-super-120b-a12b:free",
    "fallback_model": "meta-llama/llama-3.3-70b-instruct:free",       # used after 2 failed retries
    "second_fallback_model": "mistralai/mistral-7b-instruct:free",    # used after 3 failed retries (supports system prompts)
    "api_key_env": "OPENROUTER_API_KEY",

    # ── Scoring Weights ───────────────────────────────────────────────────────
    "ai_weight": 0.30,
    "technical_weight": 0.70,

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

    # ── Fields sent to AI (keep only what the narrative does NOT already cover) ──
    "fields_to_include": [
        "symbol", "type", "strike", "expiration", "underlying",
        "premium", "iv_rank", "delta",
        "prob_profit", "ev_per_contract", "rr_ratio",
        "Earnings Play", "Trend_Aligned",
        "macro_warning", "sr_warning", "decay_warning",
    ],
}
