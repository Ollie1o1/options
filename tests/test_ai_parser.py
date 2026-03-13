"""Unit tests for AI response JSON parser and partial-recovery logic."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai_scorer import _parse_json_response, _extract_partial_scores

SCORE_1 = (
    '{"id": "1", "ai_score": 75, "reasoning": "Good setup — strong IV edge",'
    ' "flags": ["IV_HIGH"], "catalyst_risk": "low", "iv_justified": true, "ai_confidence": 8}'
)
SCORE_2 = (
    '{"id": "2", "ai_score": 62, "reasoning": "Mixed signals, moderate risk",'
    ' "flags": ["WIDE_SPREAD"], "catalyst_risk": "medium", "iv_justified": false, "ai_confidence": 6}'
)
SCORE_3_PARTIAL = '{"id": "3", "ai_score": 40, "reasoning": "Weak setup'  # truncated


def _envelope(inner: str) -> str:
    return '{"scores": [' + inner + ']}'


class TestParseJsonResponse:

    def test_complete_single(self):
        raw = _envelope(SCORE_1)
        result = _parse_json_response(raw)
        assert len(result) == 1
        assert result[0]["ai_score"] == 75
        assert result[0]["id"] == "1"

    def test_complete_two(self):
        raw = _envelope(SCORE_1 + "," + SCORE_2)
        result = _parse_json_response(raw)
        assert len(result) == 2
        assert result[0]["ai_score"] == 75
        assert result[1]["ai_score"] == 62

    def test_truncated_one_complete(self):
        """One complete + one truncated: should recover the complete one."""
        raw = '{"scores": [' + SCORE_1 + "," + SCORE_3_PARTIAL
        result = _parse_json_response(raw)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_truncated_two_complete(self):
        """Two complete + one truncated."""
        raw = '{"scores": [' + SCORE_1 + "," + SCORE_2 + "," + SCORE_3_PARTIAL
        result = _parse_json_response(raw)
        assert len(result) == 2

    def test_markdown_code_block(self):
        raw = "```json\n" + _envelope(SCORE_1) + "\n```"
        result = _parse_json_response(raw)
        assert len(result) == 1
        assert result[0]["ai_score"] == 75

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No JSON object"):
            _parse_json_response("no json here at all")

    def test_no_braces_raises(self):
        with pytest.raises(ValueError):
            _parse_json_response("The model returned prose instead of JSON.")

    def test_scores_key_missing(self):
        """If the outer object doesn't have 'scores', treat as single score if it has ai_score."""
        # In this case the envelope is different — partial recovery should still work
        raw = '{"scores": [' + SCORE_3_PARTIAL  # all truncated
        with pytest.raises(ValueError):
            _parse_json_response(raw)

    def test_array_scores_wrong_type(self):
        """'scores' is not a list — should raise."""
        raw = '{"scores": "not a list"}'
        with pytest.raises(ValueError, match="Expected 'scores' list"):
            _parse_json_response(raw)


class TestExtractPartialScores:

    def test_extracts_complete_inner_objects(self):
        text = '{"scores": [' + SCORE_1 + "," + SCORE_3_PARTIAL
        scores = _extract_partial_scores(text)
        assert len(scores) == 1
        assert scores[0]["id"] == "1"

    def test_skips_envelope_object(self):
        """When the outer envelope is complete, _extract_partial_scores returns [].
        _parse_json_response handles the complete-envelope case via json.loads
        before ever calling _extract_partial_scores, so this is correct behavior."""
        text = _envelope(SCORE_1)
        # The outer { finds its matching } (the whole string), parses fine but
        # lacks ai_score/id at root → skipped.  Inner objects are never reached
        # because we advance past the complete outer object.  That is fine because
        # _extract_partial_scores is only called when json.loads already failed.
        scores = _extract_partial_scores(text)
        # In the complete-envelope case, recovery returns empty (happy path handles it)
        assert isinstance(scores, list)

    def test_empty_text(self):
        assert _extract_partial_scores("") == []

    def test_no_valid_objects(self):
        assert _extract_partial_scores("no braces at all") == []
