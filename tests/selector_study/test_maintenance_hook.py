"""The paper selector study runs from maintenance only after the day's archive completes."""
from __future__ import annotations

import os
from datetime import datetime

from src import maintenance as m


def _kw(tmp_path, study, archive_rows):
    now = datetime(2026, 6, 4, 16, 0)  # Thu afternoon
    return dict(db_path=str(tmp_path / "t.db"), phase1_start="2026-05-27",
                state_path=str(tmp_path / "state.json"), now=now, now_fn=lambda: now,
                runner=lambda cmd: 0, checkpoint_fn=lambda **k: None, track_record_fn=lambda **k: None,
                chain_archive_fn=lambda: archive_rows.pop(0), walk_forward_fn=lambda **k: None,
                watch_fn=lambda db: None, online_fn=lambda: True, estimates_archive_fn=lambda: {},
                discovery_prices_fn=lambda: {}, lens_snapshot_fn=lambda **k: {},
                selector_study_fn=study)


def test_study_waits_for_a_complete_archive_and_retries_a_partial_capture(tmp_path):
    calls, results = [], [{"captures": [{"run_status": "partial"}]},
                          {"captures": [{"run_status": "complete", "pairs_new": 3}]}]

    def study(session):
        calls.append(session)
        return results.pop(0)

    archive = [{"rows": 0, "failed": ["SPY"], "symbols": 2, "complete": False},
               {"rows": 9, "complete": True}, {"rows": 9, "complete": True}, {"rows": 9, "complete": True}]
    kw = _kw(tmp_path, study, archive)
    m.run_startup_maintenance(**kw)
    assert calls == []                                     # archive incomplete: study not due
    out = m.run_startup_maintenance(**kw)
    assert calls == ["2026-06-04"] and "selector-study:partial(will retry)" in out["ran"]
    out = m.run_startup_maintenance(**kw)
    assert calls == ["2026-06-04"] * 2 and "selector-study:3pairs" in out["ran"]
    assert m.load_state(kw["state_path"])["last_selector_study"] == "2026-06-04"
    m.run_startup_maintenance(**kw)
    assert len(calls) == 2                                 # stamped: not re-run today


def test_suite_never_writes_the_real_study_database():
    assert os.environ.get("OPTIONS_SELECTOR_STUDY_DB", "").endswith("options_test_selector_study.db")
