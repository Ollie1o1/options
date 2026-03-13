"""AI scorer configuration.

Adjust these settings to tune AI scoring behaviour, swap providers,
or change the weighting between technical and AI scores.
"""

AI_CONFIG: dict = {
    # ── API Provider ──────────────────────────────────────────────────────────
    # "anthropic" is the default.  To use OpenAI instead, swap the values of
    # "provider", "model", and "api_key_env" then update ai_scorer.py's
    # _get_client() to instantiate openai.OpenAI(api_key=...) and adjust the
    # messages.create call to the Chat Completions format.
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",          # Model used for scoring
    "api_key_env": "ANTHROPIC_API_KEY",    # Name of the env var holding the key

    # ── Scoring Weights ───────────────────────────────────────────────────────
    # final_score = technical_weight * quality_score
    #             + ai_weight        * (ai_score / 100)
    # Both weights should sum to 1.0 for a [0, 1] final_score.
    "ai_weight": 0.30,           # How much the AI score contributes
    "technical_weight": 0.70,    # How much the screener quality_score contributes

    # ── API Call Settings ─────────────────────────────────────────────────────
    "batch_size": 5,      # Candidates per API request (reduces cost vs 1 call each)
    "max_tokens": 2048,   # Upper bound on response length
    "temperature": 0.1,   # Low = consistent, deterministic scoring
    "timeout": 60,        # Seconds before a request is considered timed-out

    # ── Fields sent to AI ─────────────────────────────────────────────────────
    # Only columns present in the picks DataFrame are forwarded.
    # Trim this list to cut token usage; expand it for richer context.
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
