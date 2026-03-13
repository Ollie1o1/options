"""AI scorer configuration.

Adjust these settings to tune AI scoring behaviour, swap providers,
or change the weighting between technical and AI scores.
"""

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    # "openrouter" uses the OpenAI-compatible API at openrouter.ai (supports
    # free models).  Switch to "anthropic" if you have an Anthropic key.
    "provider": "openrouter",
    "model": "meta-llama/llama-3.3-70b-instruct:free",
    "api_key_env": "OPENROUTER_API_KEY",   # name of the env var holding the key

    # ── Scoring Weights ───────────────────────────────────────────────────────
    # final_score = technical_weight * quality_score
    #             + ai_weight        * (ai_score / 100)
    "ai_weight": 0.30,
    "technical_weight": 0.70,

    # ── API Call Settings ─────────────────────────────────────────────────────
    "batch_size": 5,      # candidates per API request
    "max_tokens": 2048,
    "temperature": 0.1,
    "timeout": 60,

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
