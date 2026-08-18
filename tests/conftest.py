"""Pytest-side guards. CI runs pytest; `scripts/run_tests.py` is the local
runner and sets the same environment for itself.

Only one guard so far, and it is not optional: keep the candidate recorder off
the real database. The suite drives `gate_and_report`, which records every
board it gates, so without this the fixture tickers used throughout the tests
land in `data/candidates.db` — the dataset this project intends to draw
conclusions about its own ranker from. Contaminating it with TSTX and AAPL
fixtures would be a self-inflicted version of the measurement defects this
repo has spent months removing.

Set at import, before any test module runs.
"""
import os
import tempfile

os.environ.setdefault(
    "OPTIONS_CANDIDATE_DB",
    os.path.join(tempfile.gettempdir(), "options_test_candidates.db"))
