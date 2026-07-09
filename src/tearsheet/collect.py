"""Gather the three data tiers into a TearsheetData dict.

The ONLY module in this package that performs IO. Everything it returns is plain
JSON: no NaN, no numpy scalars, no pandas objects, no datetimes.
"""
import math
from datetime import datetime

_ZONE_IDS = ("decision", "vol", "name", "narrative", "context")
_SLOW_IDS = ("earnings", "insider", "news")


def _jsonable(v):
    """NaN/inf -> None; numpy scalars -> python. Keeps the sidecar valid JSON."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    item = getattr(v, "item", None)
    if callable(item):
        try:
            v = item()
        except Exception:
            pass
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (int, float, str)):
        return v
    return str(v)


def _f(row, key, default=None):
    return _jsonable(row.get(key, default))


def _safe(panel_id, fn, panels, default):
    """Run `fn`; on failure mark the panel unavailable with the reason.

    A panel that fails must LOOK failed. Silently dropping it would turn
    absence-of-data into absence-of-signal.
    """
    try:
        out = fn()
        panels.setdefault(panel_id, {"status": "ok", "reason": ""})
        return out
    except Exception as exc:  # noqa: BLE001 - a bad panel must not kill the page
        panels[panel_id] = {"status": "unavailable",
                            "reason": "{}: {}".format(type(exc).__name__, exc)}
        return default


def _stress(row, spot):
    """stress_test wants a book of trade dicts, not a pick row. Synthesize one."""
    from src import stress_test
    trade = {"ticker": row.get("symbol"), "strike": float(row["strike"]),
             "expiration": row["expiration"], "option_type": row.get("type", "call"),
             "entry_price": float(row["premium"]), "quantity": 1,
             "entry_iv": float(row.get("impliedVolatility") or 0.25),
             "strategy_name": row.get("strategy_name", "long_call")}
    df = stress_test.run_stress_test([trade], {row.get("symbol"): float(spot)})
    if df is None or df.empty:
        raise ValueError("stress grid unavailable")
    moves = sorted(df["stock_move"].unique())
    ivs = sorted(df["iv_shock"].unique(), reverse=True)
    rows, worst, worst_txt = [], float("inf"), "n/a"
    for iv in ivs:
        pnls = []
        for m in moves:
            sub = df[(df["stock_move"] == m) & (df["iv_shock"] == iv)]
            p = float(sub["total_pnl_usd"].iloc[0]) if not sub.empty else 0.0
            pnls.append(p)
            if p < worst:
                worst = p
                worst_txt = "${:,.0f} @ {:+.0%} spot / {:+.0f}pp IV".format(
                    p, m, iv * 100)
        rows.append({"iv": float(iv), "pnls": pnls})
    return {"moves": [float(m) for m in moves], "rows": rows, "worst": worst_txt}


def _vol_cone(ticker):
    from src.vol_analytics import compute_vol_cone
    cone = compute_vol_cone(ticker)
    if not cone:
        raise ValueError("vol cone unavailable")
    return [{"window": int(w), "p25": float(d["p25"]), "median": float(d["median"]),
             "p75": float(d["p75"]), "current": float(d["current"]),
             "pctile": float(d["pctile"])} for w, d in sorted(cone.items())]


def _evidence():
    from src.evidence import load_model_evidence
    ev = load_model_evidence()
    return {k: _jsonable(ev.get(k)) for k in
            ("pooled_ic", "p_value", "n_oos", "cohort_n", "gate_decision", "as_of")}


def _context(row):
    """Signals with no demonstrated out-of-sample edge. Each names its own failure."""
    out = [{"label": "Quality score",
            "value": "{:.2f}".format(float(row.get("quality_score") or 0)),
            "badge": "IC +0.03", "badge_kind": "bad"}]
    if row.get("sentiment_tag"):
        out.append({"label": "Sentiment", "value": str(row["sentiment_tag"]),
                    "badge": "zero variance", "badge_kind": "bad"})
    if row.get("seasonal_win_rate") is not None:
        out.append({"label": "Seasonality",
                    "value": "{:.0%} win".format(float(row["seasonal_win_rate"])),
                    "badge": "not OOS", "badge_kind": "warn"})
    return out


def _name_block(row, sym, spot, allow_network):
    """Price history + levels. With allow_network=False the chart degrades to
    empty rather than reaching for yfinance."""
    from src import pick_context
    closes = []
    if allow_network:
        import yfinance as yf
        hist = yf.Ticker(sym).history(period="6mo", interval="1d")
        if not hist.empty:
            closes = [float(c) for c in hist["Close"].tolist()]
    levels = {}
    if closes:
        from src.levels import support_resistance_levels
        levels = support_resistance_levels(closes, current=spot)
    uoa = None
    try:
        flow = pick_context.flow_summary(sym)
        uoa = str(flow) if flow else None
    except Exception:
        uoa = None
    return {"closes": closes[-130:], "supports": levels.get("supports", []),
            "resistances": levels.get("resistances", []), "rsi": _f(row, "rsi_14"),
            "ret_5d": _f(row, "ret_5d"), "pcr": _f(row, "pcr"),
            "oi_change": _f(row, "oi_change"), "max_pain": _f(row, "max_pain"),
            "uoa": uoa}


def _narrative_block(row, ctx):
    """Reuses src/pick_context.py, which already caches these lookups."""
    from src.trade_analysis import generate_trade_thesis
    from src import pick_context
    import pandas as pd

    thesis = generate_trade_thesis(pd.Series(row))

    try:
        from src.leverage_selector import leverage_vehicle_line
        vehicle = leverage_vehicle_line(row, ctx.get("config"))
    except Exception:
        vehicle = None

    dte = int(round(float(row.get("T_years") or 0) * 365))
    try:
        stats = pick_context.analog_stats(
            row.get("strategy_name", "long_call"), dte, row.get("delta"))
        history = str(stats) if stats else None
    except Exception:
        history = None

    try:
        fit = list(pick_context.open_book(row.get("symbol")) or [])
    except Exception:
        fit = []

    return {"thesis": thesis, "vehicle": vehicle, "portfolio_fit": fit,
            "history": history}


def build(row: dict, ctx: dict, slow: bool = True) -> dict:
    """Assemble a TearsheetData from a pick row plus scan context.

    `slow=False` forbids every network call (tests, offline previews). The three
    networked panels are then marked `not_fetched` with reason `disabled`, never
    silently omitted.
    """
    panels = {}
    sym = row.get("symbol")
    spot = float(ctx.get("spot") or row.get("underlying") or 0.0)
    dte = int(round(float(row.get("T_years") or 0) * 365))

    net = _f(row, "ev_per_contract")
    gross = _f(row, "ev_gross_per_contract")
    cost = _f(row, "ev_cost_per_contract")

    # The waterfall must sum to net EV, so the last term absorbs rounding.
    waterfall = []
    if gross is not None and cost is not None and net is not None:
        half = -abs(cost) / 2.0
        waterfall = [["Gross edge", gross], ["Entry spread", half],
                     ["Exit spread", net - gross - half]]

    stress = _safe("decision", lambda: _stress(row, spot), panels,
                   {"moves": [], "rows": [], "worst": "n/a"})

    iv, hv = _f(row, "impliedVolatility"), _f(row, "hv_30d")
    vol = _safe("vol", lambda: {
        "iv": iv, "hv": hv,
        "vrp": (iv - hv) if (iv is not None and hv is not None) else None,
        "iv_rank": _f(row, "iv_percentile_30"),
        "svi_residual": _f(row, "iv_surface_residual"),
        "cone": _vol_cone(sym) if slow else [],
        "term": [[dte, iv]],
        "skew_vp": (_f(row, "iv_skew") or 0) * 100.0,
        "skew_rank": _f(row, "iv_skew_rank"),
        "expected_move": _f(row, "expected_move"),
        "required_move": _f(row, "required_move"),
    }, panels, {})

    name = _safe("name", lambda: _name_block(row, sym, spot, slow), panels, {})
    narrative = _safe("narrative", lambda: _narrative_block(row, ctx), panels, {})
    evidence = _safe("evidence", _evidence, panels,
                     {"pooled_ic": None, "p_value": None, "n_oos": 0,
                      "cohort_n": 0, "gate_decision": "UNKNOWN", "as_of": None})
    panels.setdefault("context", {"status": "ok", "reason": ""})

    if slow:
        slow_vals, slow_panels = gather_slow(sym)
    else:
        slow_vals = {}
        slow_panels = {k: {"status": "not_fetched", "reason": "disabled"}
                       for k in _SLOW_IDS}
    panels.update(slow_panels)
    for key in _SLOW_IDS:
        narrative[key] = slow_vals.get(key)

    base = "{}_{:g}{}_{}".format(sym, float(row["strike"]),
                                 str(row.get("type", "c"))[0].upper(),
                                 str(row["expiration"]).replace("-", ""))
    return {
        "meta": {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "ticker": sym, "strike": _f(row, "strike"),
                 "opt_type": row.get("type", "call"), "expiration": row.get("expiration"),
                 "dte": dte, "mode": ctx.get("mode"), "rank": ctx.get("rank"),
                 "n_picks": ctx.get("n_picks"), "spot": spot,
                 "rfr": _jsonable(ctx.get("rfr")), "vix": _jsonable(ctx.get("vix")),
                 "vix_regime": ctx.get("vix_regime", "unknown"),
                 "config_sha": ctx.get("config_sha", "unknown"),
                 "sidecar": base + ".json"},
        "verdict": {"decision": "", "reason": "", "net_ev": net,
                    "gross_ev": gross, "cost": cost},
        "stats": {"pop": _f(row, "prob_profit"), "max_loss": _f(row, "max_loss"),
                  "breakeven": _f(row, "breakeven")},
        "cost_waterfall": waterfall,
        "greeks": {k: _f(row, k) for k in ("delta", "gamma", "vega", "theta")},
        "liquidity": {"spread_pct": _f(row, "spread_pct"), "oi": _f(row, "openInterest"),
                      "volume": _f(row, "volume"),
                      "quote_freshness": row.get("quote_freshness", "unknown")},
        "stress": stress, "vol": vol, "name": name, "narrative": narrative,
        "evidence": evidence, "context": _context(row), "panels": panels,
    }
