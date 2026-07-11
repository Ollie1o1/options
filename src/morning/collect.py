"""Gather the morning-briefing sidecar. Every panel is individually fail-safe:
one dead fetch degrades to a placeholder panel, never a dead page."""
import threading
import time
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _EASTERN = None

SCHEMA_VERSION = 1
PANEL_IDS = ("health", "market", "vol", "macro_events", "signals",
             "portfolio", "gate", "notes")

# Static caveats that must appear on every page: the honesty footer.
_STANDING_NOTES = [
    "All signal panels are display-only; nothing here feeds scoring or execution.",
    "Real money is OFF until the forward-cohort gate (n>=50) fires.",
    "Quotes are 15+ min delayed free data; use for planning, not live execution.",
]


def _now_eastern(now=None):
    if now is not None:
        return now
    if _EASTERN is not None:
        return datetime.now(_EASTERN)
    return datetime.now()


def session_phase(now=None) -> str:
    """'pre-market' | 'open' | 'closed' by US/Eastern wall clock."""
    ts = _now_eastern(now)
    if ts.isoweekday() > 5:
        return "closed"
    hhmm = ts.hour * 100 + ts.minute
    if hhmm < 400:
        return "closed"
    if hhmm < 930:
        return "pre-market"
    return "open" if hhmm < 1600 else "closed"


def _safe(panel_id, fn, panels, failures, default=None):
    try:
        panels[panel_id] = fn()
    except Exception as exc:
        panels[panel_id] = default
        failures.append(f"{panel_id}: {type(exc).__name__}: {exc}")


def _desk_notes(panels, failures):
    notes = list(_STANDING_NOTES)
    for f in failures:
        notes.append(f"Panel unavailable this morning — {f}")
    return notes


def _fetch_health_report():
    from src.maintenance import load_state, DEFAULT_STATE_PATH
    from src.maintenance_health import compute_health
    return compute_health(load_state(DEFAULT_STATE_PATH), datetime.now())


def _panel_health(_report_fn=None) -> dict:
    report = (_report_fn or _fetch_health_report)()
    return {
        "worst": report.worst,
        "jobs": [{"name": j.name, "cadence": j.cadence, "last_run": j.last_run,
                  "stale_days": j.business_days_stale, "severity": j.severity}
                 for j in report.jobs],
    }


def _panel_gate(_evidence_fn=None) -> dict:
    from src.evidence import load_model_evidence, GATE_TARGET_N
    ev = dict((_evidence_fn or load_model_evidence)())
    ev["target_n"] = GATE_TARGET_N
    return ev


_TAPE_SYMBOLS = ("SPY", "QQQ", "IWM")


def _index_rows_from_closes(closes_by_sym) -> dict:
    """{sym: {last, chg_1d_pct, chg_5d_pct, closes}} from raw close series."""
    out = {}
    for sym, closes in closes_by_sym.items():
        vals = [float(c) for c in (closes or []) if c is not None]
        if len(vals) < 2:
            continue
        last = vals[-1]
        row = {"sym": sym, "last": last,
               "chg_1d_pct": (last / vals[-2] - 1.0) * 100.0,
               "chg_5d_pct": (last / vals[-6] - 1.0) * 100.0 if len(vals) >= 6 else None,
               "closes": vals[-30:]}
        out[sym] = row
    return out


def _fetch_index_rows():
    # fetch_index_directions only returns {price, verdict, ...}; sparklines and
    # day changes need the raw closes, so reuse the same guarded fetcher.
    from src.regime_dashboard import _safe_hist
    closes_by_sym = {}
    for sym in _TAPE_SYMBOLS:
        series = _safe_hist(sym, "2mo")
        if series is not None and len(series) >= 2:
            closes_by_sym[sym] = series.tolist()
    return _index_rows_from_closes(closes_by_sym)


def _fetch_rates():
    from src.macro_rates import fetch_rates_snapshot
    snap = fetch_rates_snapshot()
    return {"t10y": getattr(snap, "dgs10", None),
            "t3m": getattr(snap, "dgs3mo", None)}


def _panel_market(_regime_fn=None, _dirs_fn=None, _rates_fn=None) -> dict:
    if _regime_fn is None:
        from src.regime_dashboard import fetch_market_regime
        _regime_fn = fetch_market_regime
    regime = _regime_fn()

    indexes = []
    try:
        dirs = (_dirs_fn or _fetch_index_rows)() or {}
        for sym in _TAPE_SYMBOLS:
            info = dirs.get(sym)
            if not isinstance(info, dict):
                continue
            indexes.append({
                "sym": sym,
                "last": info.get("last"),
                "chg_1d_pct": info.get("chg_1d_pct"),
                "chg_5d_pct": info.get("chg_5d_pct"),
                "closes": list(info.get("closes") or [])[-30:],
            })
    except Exception:
        indexes = []

    rates = {"t10y": None, "t3m": None, "slope": None}
    try:
        raw = (_rates_fn or _fetch_rates)()
        rates["t10y"], rates["t3m"] = raw.get("t10y"), raw.get("t3m")
        if rates["t10y"] is not None and rates["t3m"] is not None:
            rates["slope"] = rates["t10y"] - rates["t3m"]
    except Exception:
        pass

    return {"regime": regime, "indexes": indexes, "rates": rates}


def _default_fetchers():
    # Each entry is (panel_id, zero-arg callable); grown panel by panel.
    return [("health", _panel_health), ("market", _panel_market),
            ("gate", _panel_gate)]


def build(now=None, slow=True, budget_s=20.0, _fetchers=None) -> dict:
    ts = _now_eastern(now)
    date_s = ts.strftime("%Y-%m-%d")
    meta = {
        "schema": SCHEMA_VERSION,
        "date": date_s,
        "generated_at": ts.strftime("%Y-%m-%d %H:%M %Z").strip(),
        "session": session_phase(ts),
        "sidecar": f"{date_s}.json",
        "title": f"Morning Briefing — {date_s}",
    }
    fetchers = _fetchers if _fetchers is not None else _default_fetchers()
    panels = {pid: None for pid in PANEL_IDS}
    failures = []

    if slow and fetchers:
        threads = []
        for pid, fn in fetchers:
            t = threading.Thread(target=_safe, args=(pid, fn, panels, failures),
                                 daemon=True)
            t.start()
            threads.append(t)
        deadline = time.monotonic() + budget_s
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))
        for (pid, _), t in zip(fetchers, threads):
            if t.is_alive() and panels.get(pid) is None:
                failures.append(f"{pid}: TimeoutError: budget {budget_s:.0f}s exceeded")
    else:
        for pid, fn in fetchers:
            _safe(pid, fn, panels, failures)

    panels["notes"] = _desk_notes(panels, failures)
    return {"meta": meta, "panels": panels, "failures": failures}
