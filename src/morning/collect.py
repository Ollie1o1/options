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
             "portfolio", "gate", "movers", "notes")

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


# One honest line on the crypto vol sleeve; static because the briefing must
# not fetch Deribit every morning just to restate a validated finding.
_CRYPTO_NOTE = ("Crypto: BTC short-vol carry is the one net-validated edge "
                "(carry, not timing); equity VRP is currently the opposite sign.")


def _fetch_vol_rows():
    from src.vol_intel.engine import build_rows
    return build_rows()


def _panel_vol(_rows_fn=None) -> dict:
    movers, vrp_rows = (_rows_fn or _fetch_vol_rows)()
    ranked = sorted((m for m in movers if m.get("d_iv") is not None),
                    key=lambda m: abs(m["d_iv"]), reverse=True)
    n_cov = len({r["symbol"] for r in vrp_rows})
    return {"movers": ranked[:8], "vrp": list(vrp_rows)[:8],
            "n_cov": n_cov, "crypto_note": _CRYPTO_NOTE}


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


def _fetch_calendar():
    from src.worldnews.panel import next_events
    return [{"date": e.get("date"), "name": e.get("name") or "?"}
            for e in next_events(limit=6)]


def _fetch_pulse():
    from src.worldnews import panel, scoring, sources
    items = sources.fetch_all()
    if not items:
        return None, []
    agg = scoring.aggregate(items)
    headlines = [i.get("title", "") for i in items[:5] if i.get("title")]
    return panel.pulse_line(agg), headlines


def _briefing_symbols(_personal_fn=None, _config_fn=None) -> list:
    """Single names for the watchlist-earnings panel.

    The personal watchlist wins when it has anything in it. It is a file the
    user builds from the screener menu and it starts out absent, so on its own
    the panel was permanently empty; the fallback is the configured watchlists
    in their declared order. Index and sector ETFs are dropped from the
    fallback: they have no earnings date, and each would spend one of a
    handful of per-symbol lookups on a guaranteed blank.
    """
    if _personal_fn is None:
        from src.watchlist import load_watchlist
        _personal_fn = load_watchlist
    personal = list(_personal_fn() or [])
    if personal:
        return personal
    lists = (_config_fn or _fetch_config_watchlists)() or {}
    etfs = set(_TAPE_SYMBOLS) | {str(s).upper() for s in lists.get("sector_etfs") or []}
    out = []
    for name, symbols in lists.items():
        if name == "sector_etfs" or not isinstance(symbols, list):
            continue
        for sym in symbols:
            sym = str(sym).upper()
            if sym not in etfs and sym not in out:
                out.append(sym)
    return out


def _fetch_config_watchlists() -> dict:
    import json as _json
    with open("config.json") as f:
        return _json.load(f).get("watchlists") or {}


def _fetch_watchlist_earnings(limit=10, horizon_days=7):
    import json as _json
    from datetime import timedelta
    from src.earnings_provider import next_earnings_date, resolve_api_key
    try:
        with open("config.json") as f:
            cfg = _json.load(f)
    except Exception:
        cfg = None
    key = resolve_api_key(cfg)
    today = _now_eastern().date()
    out = []
    for sym in _briefing_symbols()[:limit]:
        try:
            dt = next_earnings_date(sym, api_key=key)
        except Exception:
            continue
        if dt is None:
            continue
        d = dt.date()
        if today <= d <= today + timedelta(days=horizon_days):
            out.append({"sym": sym, "date": d.strftime("%Y-%m-%d")})
    return out


_TIME_EXIT_DTE = 21  # mirror of the 21-DTE time-exit rule in exit enforcement

# Columns worth carrying into the sidecar when present (the paper DB is
# SELECT *, so be defensive about which exist).
_POSITION_COLS = ("ticker", "strategy", "type", "strike", "expiration", "dte",
                  "entry_price", "current_price", "pnl_usd", "pnl_pct",
                  "pnl_percent", "delta", "vega")


def _fetch_positions():
    from src.portfolio_risk import RiskAggregator
    df = RiskAggregator().get_open_positions_with_greeks()
    if df is None or getattr(df, "empty", True):
        return []
    df = df.copy()
    if "dte" not in df.columns and "T_years" in df.columns:
        df["dte"] = df["T_years"] * 365.0
    keep = [c for c in _POSITION_COLS if c in df.columns]
    rows = df[keep].head(20).to_dict("records")
    # sqlite/numpy scalars -> plain python so the sidecar stays JSON-clean
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            try:
                clean[k] = v.item()  # numpy scalar
            except AttributeError:
                clean[k] = v
        out.append(clean)
    return out


def _fetch_net_greeks():
    from src.portfolio_risk import RiskAggregator
    g = RiskAggregator().get_portfolio_greeks() or {}
    g.pop("positions_df", None)  # DataFrame is not sidecar material
    return {k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in g.items() if isinstance(v, (int, float, str))}


def _fetch_guard_lines():
    from src.formatting import _strip_ansi
    from src.portfolio_guard import format_guard_lines
    return [_strip_ansi(l) for l in format_guard_lines(_fetch_positions(),
                                                       "Portfolio")]


def _panel_portfolio(_positions_fn=None, _greeks_fn=None, _guard_fn=None) -> dict:
    positions = (_positions_fn or _fetch_positions)()
    exits_due = []
    for p in positions:
        try:
            dte = float(p.get("dte"))
        except (TypeError, ValueError):
            continue
        if dte <= _TIME_EXIT_DTE:
            exits_due.append(f"{p.get('ticker', '?')}: {dte:.0f} DTE <= "
                             f"{_TIME_EXIT_DTE} — time-exit window")
    greeks, guard = {}, []
    try:
        greeks = (_greeks_fn or _fetch_net_greeks)()
    except Exception:
        pass
    try:
        guard = list((_guard_fn or _fetch_guard_lines)())
    except Exception:
        pass
    return {"positions": positions, "net_greeks": greeks, "guard": guard,
            "exits_due": exits_due, "n_open": len(positions)}


def _fetch_uoa():
    from src.uoa import uoa_report
    return uoa_report()


def _fetch_insider(limit=6):
    from src.pick_context import insider_summary
    from src.watchlist import load_watchlist
    out = []
    # Personal watchlist only, deliberately NOT _briefing_symbols(): a cold
    # insider lookup is up to ten EDGAR fetches per name (measured 20s for six
    # on 2026-10-11), which is the whole build budget and would take the UOA
    # and outlook halves of this panel down with it every morning.
    for sym in load_watchlist()[:limit]:
        try:
            s = insider_summary(sym, fetch=True)
        except Exception:
            continue
        if not s:
            continue
        if isinstance(s, dict):
            label = s.get("label") or "?"
            if label in ("NONE", "?"):
                continue
            text = f"{label} (score {s.get('score', 0):.1f})"
        else:
            text = str(s)
        out.append({"sym": sym, "summary": text})
    return out


def _fetch_outlook():
    from src.outlook.display import load_outlook_cache, compute_outlook_cache
    cache = load_outlook_cache()
    if not cache:
        compute_outlook_cache()
        cache = load_outlook_cache()
    return cache


def _panel_signals(_uoa_fn=None, _insider_fn=None, _outlook_fn=None) -> dict:
    out = {"uoa": [], "insider": [],
           "outlook": {"top": [], "bottom": [], "as_of": None}}
    try:
        rep = (_uoa_fn or _fetch_uoa)() or {}
        for r in list(rep.get("rows") or [])[:8]:
            out["uoa"].append({"symbol": r.get("symbol"),
                               "score": r.get("score"),
                               "net_call_share": r.get("net_call_share"),
                               "n_unusual": len(r.get("unusual") or [])})
    except Exception:
        pass
    try:
        out["insider"] = (_insider_fn or _fetch_insider)()
    except Exception:
        pass
    try:
        cache = (_outlook_fn or _fetch_outlook)() or {}
        rows = list(cache.get("rows") or [])
        out["outlook"] = {"top": rows[:3],
                          "bottom": rows[-3:] if len(rows) > 3 else [],
                          "as_of": cache.get("as_of")}
    except Exception:
        pass
    return out


def _panel_macro_events(_calendar_fn=None, _pulse_fn=None, _earnings_fn=None) -> dict:
    out = {"calendar": [], "pulse": None, "headlines": [], "earnings": []}
    try:
        out["calendar"] = (_calendar_fn or _fetch_calendar)()
    except Exception:
        pass
    try:
        pulse, headlines = (_pulse_fn or _fetch_pulse)()
        out["pulse"], out["headlines"] = pulse, headlines
    except Exception:
        pass
    try:
        out["earnings"] = (_earnings_fn or _fetch_watchlist_earnings)()
    except Exception:
        pass
    return out


# A shock is the event the 2026-10-10 post-shock study measured: a close-to-close
# move of at least 15% on at least 5x the prior 20 sessions' mean volume. The
# thresholds are that study's, so the base rate printed beside the panel
# describes exactly the rows it flags; change one and the note stops being true.
_SHOCK_MOVE = 0.15
_SHOCK_VOL_MULT = 5.0
_SHOCK_LOOKBACK = 20      # sessions in the volume mean and in the "20d high"
_SHOCK_LISTED_FOR = 5     # sessions a shock stays on the panel
_MOVERS_MAX_ROWS = 10
_SCAN_WINDOW_DAYS = 7     # symbols the scans reached this recently are priced

# Static for the same reason as _CRYPTO_NOTE: it restates a finished measurement.
_SHOCK_BASE_RATE = (
    "Base rate: 558 up-shocks of this size (S&P 1500, 2019-2026) — 51% were "
    "higher 20 sessions later; mean return against SPY and the stock's own "
    "drift -0.4% (95% CI -2.2 to +1.2). A mover is not a signal.")


def _fetch_recent_scan(db_path=None) -> dict:
    """Symbols the scans reached lately, and what passed the gate on the latest
    scan day. Read-only: the briefing must never create or migrate this DB."""
    import sqlite3
    from datetime import timedelta
    from src.candidate_record import _resolve_db_path
    conn = sqlite3.connect(f"file:{_resolve_db_path(db_path)}?mode=ro", uri=True,
                           timeout=5.0)
    try:
        latest = conn.execute("SELECT MAX(ts) FROM candidates").fetchone()[0]
        if not latest:
            return {"scan_date": None, "symbols": [], "passing": {}}
        scan_date = latest[:10]
        since = (datetime.strptime(scan_date, "%Y-%m-%d")
                 - timedelta(days=_SCAN_WINDOW_DAYS)).strftime("%Y-%m-%d")
        symbols = [r[0] for r in conn.execute(
            "SELECT DISTINCT symbol FROM candidates WHERE ts >= ? AND symbol <> '' "
            "ORDER BY symbol", (since,))]
        # One contract appears on several boards; count it once. Rows with a
        # blank strategy_name are the single-leg boards' copies of the same
        # contracts, so they are skipped rather than shown as unnamed.
        passing = {}
        for sym, strat, n in conn.execute(
                "SELECT symbol, strategy_name, COUNT(DISTINCT contract_key) "
                "FROM candidates WHERE substr(ts, 1, 10) = ? AND gate_passed = 1 "
                "AND strategy_name <> '' GROUP BY symbol, strategy_name "
                "ORDER BY symbol, strategy_name", (scan_date,)):
            passing.setdefault(sym, []).append({"strategy": strat, "n": n})
        return {"scan_date": scan_date, "symbols": symbols, "passing": passing}
    finally:
        conn.close()


def _fetch_mover_bars(symbols) -> dict:
    """{sym: [(close, volume), ...]} oldest first, one batched download."""
    import warnings
    import yfinance as yf
    symbols = list(symbols)
    if not symbols:
        return {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(symbols, period="3mo", auto_adjust=True, progress=False,
                           threads=True, group_by="ticker")
    out = {}
    for sym in symbols:
        try:
            df = data[sym][["Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(df):
            out[sym] = [(float(c), float(v)) for c, v in
                        zip(df["Close"].tolist(), df["Volume"].tolist())]
    return out


def _mover_rows(bars_by_sym, passing) -> list:
    """Names with a recent shock or a new 20-day closing high, shocks first."""
    rows = []
    for sym, bars in bars_by_sym.items():
        if len(bars) < _SHOCK_LOOKBACK + 2:
            continue
        closes = [b[0] for b in bars]
        vols = [b[1] for b in bars]
        last = len(bars) - 1

        def vol_mult(i):
            base = sum(vols[i - _SHOCK_LOOKBACK:i]) / _SHOCK_LOOKBACK
            return vols[i] / base if base > 0 else None

        shock = None
        for i in range(last, max(last - _SHOCK_LISTED_FOR, _SHOCK_LOOKBACK), -1):
            move, mult = closes[i] / closes[i - 1] - 1.0, vol_mult(i)
            if abs(move) >= _SHOCK_MOVE and mult is not None and mult >= _SHOCK_VOL_MULT:
                shock = {"days_ago": last - i, "move_pct": move * 100.0, "vol_mult": mult}
                break
        is_high = closes[last] > max(closes[last - _SHOCK_LOOKBACK:last])
        if shock is None and not is_high:
            continue
        rows.append({
            "sym": sym, "last": closes[last],
            "chg_1d_pct": (closes[last] / closes[last - 1] - 1.0) * 100.0,
            "chg_20d_pct": (closes[last] / closes[last - _SHOCK_LOOKBACK] - 1.0) * 100.0,
            "vol_mult": vol_mult(last),
            "flag": "shock" if shock else "20d high",
            "shock": shock,
            "passing": list(passing.get(sym) or []),
        })
    rows.sort(key=lambda r: (
        (0, r["shock"]["days_ago"], -abs(r["shock"]["move_pct"])) if r["shock"]
        else (1, -r["chg_20d_pct"], 0.0), r["sym"]))
    return rows[:_MOVERS_MAX_ROWS]


def _panel_movers(_scan_fn=None, _bars_fn=None) -> dict:
    scan = (_scan_fn or _fetch_recent_scan)()
    symbols = list(scan.get("symbols") or [])
    bars = (_bars_fn or _fetch_mover_bars)(symbols)
    return {"rows": _mover_rows(bars, scan.get("passing") or {}),
            "scan_date": scan.get("scan_date"),
            "n_symbols": len(symbols), "n_priced": len(bars),
            "base_rate": _SHOCK_BASE_RATE}


def _default_fetchers():
    # Each entry is (panel_id, zero-arg callable); grown panel by panel.
    return [("health", _panel_health), ("market", _panel_market),
            ("vol", _panel_vol), ("macro_events", _panel_macro_events),
            ("signals", _panel_signals), ("portfolio", _panel_portfolio),
            ("gate", _panel_gate), ("movers", _panel_movers)]


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
