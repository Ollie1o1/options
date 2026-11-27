"""One call per step of the documented workflow: capture -> mark -> readout."""
from __future__ import annotations

import datetime as dt
import os
from typing import Any, Dict, Optional, Sequence

from src.selector_study import capture, marks, readout
from src.selector_study.db import connect, default_settings


def _et_today() -> str:
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def run(*, study_db: Optional[str] = None, chain_db: str = "data/chain_archive.db",
        ohlcv_db: str = "data/equity_ohlcv.db", sessions: Optional[Sequence[str]] = None,
        today: Optional[str] = None, earnings: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Capture the given (default: every archived) session, mark all entries, read out.

    `earnings` applies only to a session equal to `today`: the upcoming-earnings map is
    current information and would be look-ahead for an older session."""
    today = today or _et_today()
    conn = connect(study_db)
    try:
        todo = list(sessions) if sessions else capture.archive_sessions(chain_db)
        captured = [capture.capture_session(conn, s, chain_db=chain_db,
                                            earnings=earnings if s == today else None)
                    for s in todo if s <= today]
        marked = marks.mark_entries(conn, chain_db=chain_db, ohlcv_db=ohlcv_db, today=today)
        report = readout.build(conn, today=today)
    finally:
        conn.close()
    return {"captures": captured, "marks": marked, "readout": report}


def demo(directory: str, *, sessions: int = 85, seed: int = 7) -> Dict[str, Any]:
    """Build a synthetic archive in `directory` and run the full workflow over it."""
    from src.selector_study import synthetic

    data = synthetic.build(directory, sessions=sessions, seed=seed)
    study_db = os.path.join(directory, "selector_study.db")
    if os.path.exists(study_db):
        os.remove(study_db)
    settings = default_settings()
    settings["symbols"] = data["symbols"]
    connect(study_db, settings=settings, synthetic=True).close()
    out = run(study_db=study_db, chain_db=data["chain_db"], ohlcv_db=data["ohlcv_db"],
              today=data["sessions"][-1])
    out["synthetic"] = {k: v for k, v in data.items() if k != "sessions"}
    return out
