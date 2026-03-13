"""AI scorer configuration."""

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    "provider": "openrouter",
    "model": "meta-llama/llama-3.3-70b-instruct:free",
    "fallback_model": "google/gemma-3-12b-it:free",   # used after 2 failed retries
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
    "batch_size": 5,
    "max_tokens": 2048,
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

    # ── Fields sent to AI ─────────────────────────────────────────────────────
    "fields_to_include": [
        "symbol", "type", "strike", "expiration", "underlying",
        "premium", "impliedVolatility", "iv_rank", "iv_percentile",
        "hv_30d", "iv_vs_hv",
        "delta", "theta",
        "prob_profit", "pop_sim", "ev_per_contract", "rr_ratio",
        "quality_score", "score_drivers",
        "earnings_date", "Earnings Play",
        "sentiment_tag", "Trend_Aligned", "Moneyness",
        "spread_pct", "volume", "openInterest", "rvol",
        "rsi_14", "ret_5d", "short_interest",
        "macro_warning", "sr_warning", "decay_warning", "event_flag",
    ],
}
