"""Integration tests for AIScorer.

These tests use real AIScorer objects but mock out the API layer so no
network calls are made.  They verify the scoring pipeline end-to-end:
candidate extraction → batch scoring → result DataFrame assembly.
"""

from __future__ import annotations

import pandas as pd
from unittest.mock import patch


def _make_picks(n: int = 3) -> pd.DataFrame:
    """Build a minimal screener-picks DataFrame with required columns."""
    rows = []
    for i in range(n):
        rows.append({
            "symbol": ["AAPL", "MSFT", "TSLA"][i % 3],
            "type": "call" if i % 2 == 0 else "put",
            "strike": 150.0 + i * 5,
            "expiration": "2026-06-20",
            "underlying": 155.0,
            "premium": 3.50 + i * 0.5,
            "iv_rank": 0.60,
            "delta": 0.40,
            "prob_profit": 0.55,
            "pop_sim": 0.57,
            "ev_per_contract": 120.0,
            "rr_ratio": 1.8,
            "be_dist_pct": 3.2,
            "annualized_return": 0.35,
            "Earnings Play": "NO",
            "Trend_Aligned": True,
            "macro_warning": None,
            "sr_warning": None,
            "decay_warning": False,
            "gamma_ramp": False,
            "quality_score": 0.70 + i * 0.02,
        })
    return pd.DataFrame(rows)


def _fake_score_result(batch):
    """Return a plausible score list for a batch of candidates."""
    return [
        {
            "id": c["_id"],
            "ai_score": 72.0,
            "reasoning": "solid IV rank, good PoP",
            "flags": ["IV_HIGH"],
            "catalyst_risk": "low",
            "iv_justified": True,
            "ai_confidence": 7.5,
        }
        for c in batch
    ]


class TestAIScorerIntegration:

    def test_score_candidates_returns_correct_columns(self):
        from src.ai_scorer import AIScorer

        picks = _make_picks(3)
        scorer = AIScorer(config={"cache_enabled": False})

        with patch.object(scorer, "_score_batch", side_effect=_fake_score_result):
            result = scorer.score_candidates(picks)

        required = {"ai_score", "ai_reasoning", "catalyst_risk", "ai_confidence"}
        assert required.issubset(set(result.columns)), (
            f"Missing columns: {required - set(result.columns)}"
        )
        assert len(result) == len(picks)

    def test_score_candidates_scores_in_range(self):
        from src.ai_scorer import AIScorer

        picks = _make_picks(5)
        scorer = AIScorer(config={"cache_enabled": False})

        with patch.object(scorer, "_score_batch", side_effect=_fake_score_result):
            result = scorer.score_candidates(picks)

        assert result["ai_score"].between(0, 100).all(), (
            f"Scores out of range: {result['ai_score'].tolist()}"
        )

    def test_narrative_context_not_empty(self):
        from src.ai_scorer import _enrich_candidate_context
        from src.config_ai import AI_CONFIG

        candidate = {
            "iv_rank": 0.75,
            "iv_vs_hv": 0.08,
            "prob_profit": 0.62,
            "rr_ratio": 2.1,
            "theta": -0.05,
            "premium": 3.5,
            "spread_pct": 0.10,
        }
        ctx = _enrich_candidate_context(candidate, AI_CONFIG)
        assert ctx and ctx != "no-context", f"Expected non-empty context, got: {ctx!r}"
        assert "IV:" in ctx, f"Expected 'IV:' in context, got: {ctx!r}"

    def test_fallback_on_api_error(self):
        from src.ai_scorer import AIScorer

        picks = _make_picks(2)
        scorer = AIScorer(config={"cache_enabled": False})

        def _always_fail(batch, model=None):
            raise RuntimeError("simulated API failure")

        with patch.object(scorer, "_score_batch", side_effect=_always_fail):
            result = scorer.score_candidates(picks)

        # Should return neutral defaults rather than raising
        assert len(result) == len(picks)
        assert (result["ai_score"] == 50.0).all(), (
            f"Expected neutral score 50.0 on failure, got: {result['ai_score'].tolist()}"
        )
