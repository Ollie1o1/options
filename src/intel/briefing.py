"""Per-ticker Intel Briefing orchestration: fetch → signals → verdict → render.

Price-based signals come from a single resilient close-history fetch so the
briefing still works even when the heavier options/news fetch fails. Each fetch
is best-effort: a missing source degrades to an n/a panel, never a crash.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.intel import playbook, signals as S, ui, verdict as Vd
from src.intel.reliability import load_or_compute_reliability


@dataclass
class Briefing:
    symbol: str
    name: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, S.Signal] = field(default_factory=dict)
    verdict: Optional[Vd.Verdict] = None
    primary_action: str = ""
    secondary_action: str = ""
    market_line: str = ""
    ok: bool = True
    error: str = ""


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def gather(symbol: str) -> Briefing:
    """Collect everything for one symbol and compute the verdict + action."""
    from src.regime_dashboard import _safe_hist, fetch_market_regime
    from src.levels import support_resistance_levels, bounce_stats, rsi as _rsi

    sym = symbol.upper().strip()
    closes_s = _safe_hist(sym, "3y")
    if closes_s is None or len(closes_s) < 30:
        return Briefing(sym, ok=False, error="no price history available")
    closes = [float(x) for x in closes_s.tolist() if x is not None]

    price = closes[-1]
    ma50 = sum(closes[-50:]) / min(len(closes), 50)
    ma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
    ret_5d = price / closes[-6] - 1.0 if len(closes) >= 6 else None
    ret_20d = price / closes[-21] - 1.0 if len(closes) >= 21 else None
    low_60 = min(closes[-60:]) if len(closes) >= 60 else min(closes)

    levels = support_resistance_levels(closes)
    supports = levels.get("supports", [])
    resists = levels.get("resistances", [])
    nearest_support = supports[0] if supports else None
    nearest_resist = resists[0] if resists else None
    pct_to_support = nearest_support["pct"] if nearest_support else None

    bounce = bounce_stats(closes, horizons=(10,)).get("by_horizon", {}).get(10, {})
    rsi_val = _rsi(closes)

    # ── Best-effort options / news / earnings context ──
    ctx: Dict[str, Any] = {}
    _opt = _safe(lambda: _fetch_context(sym))
    if _opt:
        ctx = _opt

    iv_rank = ctx.get("iv_rank")
    term_spread = ctx.get("term_structure_spread")
    earnings_date = ctx.get("earnings_date")
    news_data = ctx.get("news_data")

    days_to_earnings = None
    if isinstance(earnings_date, _dt.datetime):
        days_to_earnings = (earnings_date.date() - _dt.date.today()).days

    news_sentiment = None
    raises = cuts = 0
    headlines: List[str] = []
    if news_data is not None:
        news_sentiment = getattr(news_data, "aggregate_sentiment", None)
        headlines = list(getattr(news_data, "top_headlines", []) or [])[:3]
        for ch in getattr(news_data, "analyst_changes", []) or []:
            act = (getattr(ch, "action", "") or "").lower()
            if act in ("upgrade", "initiate"):
                raises += 1
            elif act == "downgrade":
                cuts += 1

    state = {
        "price": price, "ma50": ma50, "ma200": ma200,
        "ret_5d": ret_5d, "ret_20d": ret_20d,
        "bounce_rate": bounce.get("bounce_rate"), "bounce_n": bounce.get("n", 0),
        "rsi": rsi_val, "pct_to_support": pct_to_support,
        "news_sentiment": news_sentiment,
        "analyst_raises": raises, "analyst_cuts": cuts,
        "iv_rank": iv_rank, "skew": None,
        "days_to_earnings": days_to_earnings,
    }
    sigs = S.build_signals(state)
    reliability = load_or_compute_reliability()
    vdict = Vd.decide(sigs, reliability)

    # ── Market context line (best-effort) ──
    market_line = ""
    reg = _safe(fetch_market_regime, {}) or {}
    if reg.get("posture"):
        vix = reg.get("vix")
        market_line = f"{reg['posture'].replace('_', ' ').title()}" + (
            f" · VIX {vix:.0f}" if vix else "")

    # ── Playbook state + selection ──
    pb = _build_playbook_state(
        vdict, sigs, state, closes, ma50, ma200, low_60,
        nearest_support, nearest_resist, supports, term_spread, reg)
    primary, secondary = playbook.select(pb)

    b = Briefing(sym, name=ctx.get("name", sym), state=state, signals=sigs,
                 verdict=vdict, primary_action=primary or "",
                 secondary_action=secondary or "", market_line=market_line)
    b.state["_headlines"] = headlines
    b.state["_nearest_support"] = nearest_support
    b.state["_nearest_resist"] = nearest_resist
    b.state["_bounce"] = bounce
    b.state["_term_spread"] = term_spread
    return b


def _fetch_context(sym: str) -> Dict[str, Any]:
    """Best-effort news + earnings via the reliable yfinance/free path.

    (The yahooquery chain path that also yields IV rank / term structure is
    optional and not always installed; when absent those panels are simply
    omitted rather than blocking the briefing.)
    """
    import yfinance as yf
    from src.data_fetching import get_next_earnings_date
    from src.news_fetcher import fetch_news_and_events

    ctx: Dict[str, Any] = {}
    tkr = _safe(lambda: yf.Ticker(sym))
    if tkr is not None:
        ctx["earnings_date"] = _safe(lambda: get_next_earnings_date(tkr))
        ctx["news_data"] = _safe(lambda: fetch_news_and_events(sym, tkr))
    # IV rank / term structure: only if the heavier chain provider is present.
    iv = _safe(lambda: _maybe_iv_context(sym))
    if iv:
        ctx.update(iv)
    return ctx


def _maybe_iv_context(sym: str) -> Dict[str, Any]:
    """IV rank + term structure if the yahooquery provider is installed; else {}."""
    try:
        from src.data_fetching import fetch_options_yahooquery
    except Exception:
        return {}
    res = fetch_options_yahooquery(sym, max_expiries=6)
    c = (res or {}).get("context", {}) or {}
    return {"iv_rank": c.get("iv_rank"),
            "term_structure_spread": c.get("term_structure_spread")}


def _build_playbook_state(vdict, sigs, state, closes, ma50, ma200, low_60,
                          nearest_support, nearest_resist, supports,
                          term_spread, reg) -> playbook.PlaybookState:
    price = state["price"]
    ret_5d = state.get("ret_5d") or 0.0
    ret_20d = state.get("ret_20d") or 0.0
    near_resist = bool(nearest_resist and (nearest_resist["pct"] <= 0.03))
    new_60d_low = price <= low_60 * 1.005
    reclaimed_50d = (len(closes) >= 2 and closes[-2] < ma50 <= price)
    death_cross = (ma200 is not None and ma50 < ma200)
    parabolic = (ret_20d != 0 and ret_5d > 0 and ret_5d > 0.6 * ret_20d and ret_5d > 0.06)
    support_200d = next((s for s in supports if "200d" in s["label"]), None)
    near_200d = (ma200 is not None and abs(price / ma200 - 1.0) < 0.025)

    def money(x):
        return f"${x:,.2f}" if x is not None else "n/a"

    fmt = {
        "ma50": money(ma50), "ma200": money(ma200),
        "nearest_support": money(nearest_support["level"]) if nearest_support else "n/a",
        "resist": money(nearest_resist["level"]) if nearest_resist else "n/a",
        "support_50d": money(ma50),
    }
    if support_200d:
        fmt["support_200d"] = money(support_200d["level"])

    top_driver = vdict.drivers[0].text if vdict.drivers else ""

    return playbook.PlaybookState(
        verdict=vdict.call,
        trend=sigs["trend"].value, momentum=sigs["momentum"].value,
        rsi=state.get("rsi"),
        bounce_rate=state.get("bounce_rate"), bounce_n=state.get("bounce_n", 0),
        support_dist=abs(nearest_support["pct"]) if nearest_support else None,
        below_200d=(ma200 is not None and price < ma200),
        death_cross=death_cross, reclaimed_50d=reclaimed_50d,
        new_60d_low=new_60d_low, parabolic=parabolic, near_resistance=near_resist,
        near_200d=near_200d,
        iv_rank=state.get("iv_rank"), skew=state.get("skew"),
        term_backwardated=(term_spread is not None and term_spread < 0),
        days_to_earnings=state.get("days_to_earnings"),
        analyst_raises=state.get("analyst_raises", 0),
        analyst_cuts=state.get("analyst_cuts", 0),
        news_sentiment=state.get("news_sentiment"),
        price_down_5d=(ret_5d <= -0.03),
        regime=(reg.get("posture", "") if reg else ""),
        top_driver=top_driver, fmt=fmt,
    )


def render(b: Briefing, width: int = 64) -> List[str]:
    """Render a Briefing into printable lines."""
    if not b.ok:
        return [ui.color(f"  Could not build briefing for {b.symbol}: {b.error}",
                         ui._C.RED if ui._HAS else "")]
    st = b.state
    sigs = b.signals
    v = b.verdict
    body: List[str] = []

    if b.market_line:
        body.append(ui.row("MARKET", b.market_line, color_value=ui._C.DIM if ui._HAS else ""))
        body.append("\x00")

    # PRICE
    price = st["price"]
    r5 = st.get("ret_5d")
    spark = ui.SPARK([float(x) for x in []])  # filled below if history present
    r5s = f"{r5:+.1%}/5d" if r5 is not None else ""
    rsi_v = st.get("rsi")
    price_line = f"${price:,.2f}   {r5s}"
    if rsi_v is not None:
        price_line += f"   RSI {rsi_v:.0f}"
    body.append(ui.row("PRICE", price_line, color_value=ui.direction_color(r5 or 0)))

    # TREND / LEVELS / BOUNCE
    body.append(ui.row("TREND", f"{sigs['trend'].label}  {sigs['trend'].detail}",
                       color_value=ui.direction_color(sigs['trend'].value)))
    ns, nr = st.get("_nearest_support"), st.get("_nearest_resist")
    lvl = []
    if ns:
        lvl.append(f"support ${ns['level']:,.2f} ({ns['pct']:+.1%})")
    if nr:
        lvl.append(f"resist ${nr['level']:,.2f} ({nr['pct']:+.1%})")
    body.append(ui.row("LEVELS", "  ·  ".join(lvl) if lvl else "n/a"))
    bnc = st.get("_bounce", {})
    if bnc.get("bounce_rate") is not None:
        tag = "ok" if bnc.get("n", 0) >= 10 else "thin"
        body.append(ui.row("BOUNCE",
                           f"{bnc['bounce_rate']:.0%} higher in 10d after similar drops  [n={bnc.get('n')}, {tag}]"))

    # OPTIONS
    ivr = st.get("iv_rank")
    if ivr is not None:
        rich = "rich" if ivr > 0.7 else ("cheap" if ivr < 0.3 else "mid")
        ts = st.get("_term_spread")
        ts_txt = ""
        if ts is not None:
            ts_txt = " · term backwardated" if ts < 0 else " · term contango"
        body.append(ui.row("OPTIONS", f"IV rank {ivr:.0%} ({rich}){ts_txt}"))

    # NEWS / ANALYST
    hl = st.get("_headlines") or []
    nsent = st.get("news_sentiment")
    if nsent is not None or hl:
        s_txt = f"sentiment {nsent:+.2f}" if nsent is not None else "n/a"
        head = f'  "{hl[0][:24]}…"' if hl else ""
        body.append(ui.row("NEWS", f"{s_txt}{head}"))
    if st.get("analyst_raises") or st.get("analyst_cuts"):
        body.append(ui.row("ANALYST",
                           f"{st.get('analyst_raises',0)} raises / {st.get('analyst_cuts',0)} cuts (30d)"))

    # EARNINGS
    dte = st.get("days_to_earnings")
    if dte is not None:
        warn = "  ⚠ event risk" if 0 <= dte <= 10 else ""
        when = f"in {dte} days" if dte >= 0 else f"{-dte} days ago"
        body.append(ui.row("EARNINGS", f"{when}{warn}",
                           color_value=(ui._C.RED if (ui._HAS and 0 <= dte <= 10) else "")))

    body.append("\x00")

    # VERDICT
    vc = ui.verdict_color(v.call)
    conf = f"(confidence: {v.confidence})"
    body.append(ui.row("VERDICT", ui.color(f"{v.call}  {conf}", vc, True)))
    for d in v.drivers[:4]:
        gl = ui.color(d.glyph, {"+": ui._C.GREEN, "-": ui._C.RED}.get(d.glyph, ui._C.DIM)) if ui._HAS else d.glyph
        tag = ui.color(f"[{d.tag}]", ui._C.DIM) if ui._HAS else f"[{d.tag}]"
        body.append(f"  {gl} {d.text[:24]:<24} {tag}")
    if v.note:
        body.append(ui.color(f"  ! {v.note}", ui._C.YELLOW) if ui._HAS else f"  ! {v.note}")

    body.append("\x00")

    # WHAT TO DO
    body.append(ui.color("WHAT TO DO", ui._C.BRIGHT_CYAN, True) if ui._HAS else "WHAT TO DO")
    for ln in ui.wrap(b.primary_action, width - 4, indent="  "):
        body.append(ln)
    if b.secondary_action:
        body.append("")
        body.append(ui.color("  Also:", ui._C.DIM) if ui._HAS else "  Also:")
        for ln in ui.wrap(b.secondary_action, width - 4, indent="  "):
            body.append(ui.color(ln, ui._C.DIM) if ui._HAS else ln)

    title_r = f"{b.symbol} · {b.name}" if b.name and b.name != b.symbol else b.symbol
    return ui.box("INTEL BRIEFING", title_r, body, width)


def print_briefing(symbol: str, width: int = 64) -> None:
    b = gather(symbol)
    print()
    for line in render(b, width):
        print(line)
