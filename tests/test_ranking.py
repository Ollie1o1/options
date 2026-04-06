"""Tests for src/ranking.py combine_scores and ai_rank._rank_without_ai."""
import sys
import os
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking import combine_scores


def _make_picks(**overrides):
    """Return a minimal picks DataFrame row."""
    row = {
        "symbol": "AAPL",
        "type": "call",
        "strike": 150.0,
        "expiration": "2026-06-20",
        "premium": 3.50,
        "quality_score": 0.60,
        "impliedVolatility": 0.30,
        "pop_sim": 0.55,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _make_ai_df(ai_score=60.0, ai_confidence=7.0, **overrides):
    """Return a minimal AI scores DataFrame."""
    row = {
        "ai_score": ai_score,
        "ai_reasoning": "Test reasoning",
        "ai_flags": "",
        "catalyst_risk": "medium",
        "iv_justified": True,
        "ai_confidence": ai_confidence,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_basic_combination():
    """quality_score=0.8, ai_score=70 -> final_score in [0, 1]."""
    picks = _make_picks(quality_score=0.8)
    ai_df = _make_ai_df(ai_score=70.0)
    ranked = combine_scores(picks, ai_df, vix_regime="normal")
    assert "final_score" in ranked.columns
    fs = float(ranked["final_score"].iloc[0])
    assert 0.0 <= fs <= 1.0


def test_no_divergence_when_ai_skipped():
    """_rank_without_ai rows should all have divergence_flag=False."""
    from ai_rank import _rank_without_ai
    picks = pd.concat([
        _make_picks(quality_score=0.4, symbol="AAPL"),
        _make_picks(quality_score=0.8, symbol="TSLA"),
    ], ignore_index=True)
    ranked = _rank_without_ai(picks)
    assert "divergence_flag" in ranked.columns
    assert ranked["divergence_flag"].fillna(False).astype(bool).any() == False


def test_high_vix_increases_ai_weight():
    """vix_regime='high' should produce higher ai_weight_used than 'low'."""
    picks = _make_picks(quality_score=0.5)
    ai_df = _make_ai_df(ai_score=80.0, ai_confidence=9.0)
    ranked_high = combine_scores(picks, ai_df, vix_regime="high")
    ranked_low = combine_scores(picks, ai_df, vix_regime="low")
    aw_high = float(ranked_high["ai_weight_used"].iloc[0])
    aw_low = float(ranked_low["ai_weight_used"].iloc[0])
    assert aw_high > aw_low


def test_divergence_detected_correctly():
    """ai_score=90, quality_score=0.40 -> divergence_flag=True and direction AI>TECH."""
    picks = _make_picks(quality_score=0.40)
    ai_df = _make_ai_df(ai_score=90.0)
    ranked = combine_scores(picks, ai_df, vix_regime="normal")
    assert bool(ranked["divergence_flag"].iloc[0]) is True
    assert ranked["divergence_direction"].iloc[0] == "AI>TECH"


def test_catalyst_risk_preserved():
    """catalyst_risk column should appear in output."""
    picks = _make_picks()
    ai_df = _make_ai_df(catalyst_risk="high")
    ranked = combine_scores(picks, ai_df)
    assert "catalyst_risk" in ranked.columns
    assert ranked["catalyst_risk"].iloc[0] == "high"


def test_rank_ordering():
    """3 rows with different quality_score should get rank 1, 2, 3."""
    picks = pd.concat([
        _make_picks(quality_score=0.9, symbol="A"),
        _make_picks(quality_score=0.5, symbol="B"),
        _make_picks(quality_score=0.2, symbol="C"),
    ], ignore_index=True)
    ai_df = pd.concat([
        _make_ai_df(ai_score=50),
        _make_ai_df(ai_score=50),
        _make_ai_df(ai_score=50),
    ], ignore_index=True)
    ranked = combine_scores(picks, ai_df, vix_regime="normal")
    ranks = list(ranked["rank"].values)
    assert sorted(ranks) == [1, 2, 3]
    # Highest quality score should be rank 1
    top = ranked[ranked["rank"] == 1].iloc[0]
    assert top["symbol"] == "A"
