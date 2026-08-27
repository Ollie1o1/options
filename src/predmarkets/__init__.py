"""Prediction-market archive.

ARCHIVE FIRST, WIRE LATER. Nothing here feeds a score. The point is to
accumulate a point-in-time record with an `archived_at` stamp so the question
"does this explain anything?" becomes answerable in a few months — the same
move that made news testable without lookahead.

This repo has measured news sentiment at IC ~0 and weight 0.0. A prediction
market price IS the consensus, so the prior on it as alpha is poor. Its honest
uses are macro regime context and as an external calibration benchmark.
"""
from __future__ import annotations
