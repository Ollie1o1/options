"""Gather the three data tiers into a TearsheetData dict.

The ONLY module in this package that performs IO. Everything it returns is plain
JSON: no NaN, no numpy scalars, no pandas objects, no datetimes.
"""
import math
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime

_ZONE_IDS = ("decision", "vol", "name", "narrative", "context")
_SLOW_IDS = ("earnings", "insider", "news")


def _jsonable(v):
    """NaN/inf -> None; numpy scalars -> python; dicts/lists recurse.

    Recursion matters: without it a dict returned by e.g. `cluster_score` gets
    str()'d into "{'n_buyers': 0, ...}" and that literal lands on the page.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
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


_NO_EDGE_TERMS = ("seasonal", "sentiment", "breakout")


def _greeks_full(row):
    """First order, second order, and dollar-space exposures.

    The row already carries rho/vanna/charm/vega_dollar/gamma_theta_ratio; the
    page was showing four of them.
    """
    return {
        "first": {"delta": _f(row, "delta"), "gamma": _f(row, "gamma"),
                  "vega": _f(row, "vega"), "theta": _f(row, "theta")},
        "second": {"rho": _f(row, "rho"), "vanna": _f(row, "vanna"),
                   "charm": _f(row, "charm")},
        "dollar": {"vega_dollar": _f(row, "vega_dollar"),
                   "gamma_theta_ratio": _f(row, "gamma_theta_ratio"),
                   "theta_burn_rate": _f(row, "Theta_Burn_Rate"),
                   "abs_delta": _f(row, "abs_delta")},
    }


def _quote(row):
    """Everything needed to actually price and place the order."""
    bid, ask = _f(row, "bid"), _f(row, "ask")
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
    return {"premium": _f(row, "premium"), "bid": bid, "ask": ask, "mid": mid,
            "spread_pct": _f(row, "spread_pct"), "volume": _f(row, "volume"),
            "oi": _f(row, "openInterest"), "oi_change": _f(row, "oi_change"),
            "liquidity_flag": row.get("liquidity_flag"),
            "spread_flag": row.get("spread_flag"),
            "quote_freshness": row.get("quote_freshness"),
            "iv_confidence": row.get("iv_confidence"),
            "iv_surface_confidence": _f(row, "iv_surface_confidence"),
            "prob_touch": _f(row, "prob_touch"), "rr_ratio": _f(row, "rr_ratio"),
            "annualized_return": _f(row, "annualized_return"),
            "be_dist_pct": _f(row, "be_dist_pct"),
            "max_pain_dist_pct": _f(row, "max_pain_dist_pct"),
            "gamma_pin_dist_pct": _f(row, "gamma_pin_dist_pct"),
            "strategy_name": row.get("strategy_name")}


def _ticket(row, config):
    """Limit price, profit target, stop, and the execution guidance line."""
    from src.trade_analysis import calculate_entry_exit_levels, generate_execution_guidance
    import pandas as pd
    series = pd.Series(row)
    lv = calculate_entry_exit_levels(series, config or {})
    try:
        guidance = generate_execution_guidance(series)
    except Exception:
        guidance = None
    return {k: _jsonable(lv.get(k)) for k in
            ("entry_price", "profit_target", "stop_loss", "breakeven",
             "max_loss", "potential_profit", "risk_reward_ratio")} | {
        "guidance": _jsonable(guidance)}


def _term_from_siblings(row, ctx):
    """(dte, median IV) per expiry across the scan's other picks on this name.

    A single pick yields one point, and a one-point curve cannot be drawn. Rather
    than emit a blank chart under a heading, we return whatever we have and let
    the renderer say why it is not a curve.
    """
    sym = row.get("symbol")
    rows = [r for r in (ctx.get("sibling_rows") or []) if r.get("symbol") == sym]
    if not rows:
        rows = [row]
    buckets = {}
    for r in rows:
        try:
            dte = int(round(float(r.get("T_years") or 0) * 365))
            iv = float(r.get("impliedVolatility"))
        except (TypeError, ValueError):
            continue
        if not (0.01 < iv < 5.0) or dte <= 0:
            continue
        buckets.setdefault(dte, []).append(iv)
    curve = [[d, sorted(v)[len(v) // 2]] for d, v in sorted(buckets.items())]
    return curve


def _thesis_caveat(thesis):
    """Name the no-edge signals a thesis leans on, if any.

    The live thesis read 'Strong seasonality (100% win rate)' — seasonality sits
    in this page's own no-demonstrated-edge zone. Say so next to the sentence.
    """
    if not thesis:
        return None
    low = str(thesis).lower()
    hits = [t for t in _NO_EDGE_TERMS if t in low]
    if not hits:
        return None
    return "rests on {} — see zone V, no demonstrated edge".format(
        " and ".join(hits))


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
    try:
        uoa = _jsonable(pick_context.flow_summary(sym)) or None
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
        # Feed it the rate the scan already resolved. Its default rate_fetcher is
        # a LIVE get_risk_free_rate() call — a hidden network hop inside what is
        # documented as a tier-2 local panel.
        rfr = ctx.get("rfr")
        fetcher = (lambda: float(rfr)) if rfr else None
        vehicle = leverage_vehicle_line(row, ctx.get("config"), rate_fetcher=fetcher)
    except Exception:
        vehicle = None

    dte = int(round(float(row.get("T_years") or 0) * 365))
    try:
        history = _jsonable(pick_context.analog_stats(
            row.get("strategy_name", "long_call"), dte, row.get("delta"))) or None
    except Exception:
        history = None

    try:
        fit = [str(x) for x in (pick_context.open_book(row.get("symbol")) or [])]
    except Exception:
        fit = []

    return {"thesis": thesis, "thesis_caveat": _thesis_caveat(thesis),
            "vehicle": vehicle, "portfolio_fit": fit, "history": history}


def _earnings(sym):
    """`next_earnings_date(symbol, ...)` — Finnhub with a yfinance fallback."""
    from src.earnings_provider import next_earnings_date
    return next_earnings_date(sym)


def _insider(sym):
    """`pick_context.insider_summary` wraps edgar + parse_form4 + cluster_score
    behind a disk cache. Do NOT call src.insider.edgar directly: it has no
    ticker-level entry point, and pick_context already owns the caching."""
    from src.pick_context import insider_summary
    return insider_summary(sym)


def _news(sym):
    """worldnews has no per-ticker pulse; `fetch_google_news` takes topics."""
    from src.worldnews.sources import fetch_google_news
    items = fetch_google_news([sym]) or []
    return "; ".join(str(i.get("title", ""))[:80] for i in items[:3]) or None


def gather_slow(sym, budget_s: float = 2.5, _fns=None):
    """Run the networked panels concurrently under one wall-clock budget.

    Returns (values, panel_status). A panel that exceeds the budget is
    `not_fetched`; a panel that raises is `unavailable`. Those are different
    facts and the page says which.

    The pool is NOT used as a context manager: `__exit__` joins running threads,
    which would silently blow the budget while looking correct.
    """
    fns = _fns if _fns is not None else {
        "earnings": lambda: _earnings(sym),
        "insider": lambda: _insider(sym),
        "news": lambda: _news(sym),
    }
    values, panels = {}, {}
    pool = ThreadPoolExecutor(max_workers=max(1, len(fns)))
    try:
        futures = {pool.submit(fn): name for name, fn in fns.items()}
        wait(list(futures), timeout=budget_s)
        for fut, name in futures.items():
            if not fut.done():
                fut.cancel()
                panels[name] = {"status": "not_fetched",
                                "reason": "exceeded {}s budget".format(budget_s)}
                continue
            try:
                values[name] = _jsonable(fut.result(timeout=0))
                panels[name] = {"status": "ok", "reason": ""}
            except Exception as exc:  # noqa: BLE001
                panels[name] = {"status": "unavailable",
                                "reason": "{}: {}".format(type(exc).__name__, exc)}
    finally:
        pool.shutdown(wait=False)
    return values, panels


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
        "term": _term_from_siblings(row, ctx),
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
    events = {key: slow_vals.get(key) for key in _SLOW_IDS}

    greeks_full = _safe("greeks", lambda: _greeks_full(row), panels, {})
    quote = _safe("quote", lambda: _quote(row), panels, {})
    ticket = _safe("ticket", lambda: _ticket(row, ctx.get("config")), panels, {})

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
        "greeks_full": greeks_full, "quote": quote, "ticket": ticket,
        "events": events,
    }
