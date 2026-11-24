"""Explain scan observations without inventing a forecast.

Pure, JSON-ready evidence shared by terminal cards and tearsheets. No fetching,
ledger reads, score changes, or confidence inferred from counting indicators.
Related observations name their channel so they cannot masquerade as independent
confirmation. Missing history remains missing, rather than becoming neutral.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Mapping, Optional

SCHEMA = 1
VALIDATION = "Descriptive; predictive value not established by this card."


def _text(value: Any, default: str = "") -> str:
    result = str(value).strip() if value is not None else ""
    return default if result.lower() in ("", "nan", "nat", "none", "<na>") else result


def _field(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def history_observations(history: Any) -> dict:
    """Summarize dated daily closes; never bridge holes in the lookback.

    Endpoints retain their session dates. A provider's latest daily candle may
    still be in progress, so this deliberately does not call it a settled close.
    """
    if history is None or getattr(history, "empty", True) or "Close" not in history:
        return {}
    close = history["Close"]
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        return {}
    values = [number(v) for v in close.tolist()]
    last = values[-1] if values else None
    if last is None or last <= 0:
        return {}
    out: dict[str, Any] = {"last": last, "as_of": str(close.index[-1])[:10],
                           "source": "Yahoo Finance daily price history", "returns": {}}
    for sessions in (1, 5, 20):
        window = values[-sessions - 1:]
        valid = [v for v in window if v is not None and v > 0]
        if len(window) == sessions + 1 and len(valid) == len(window):
            out["returns"][str(sessions)] = {
                "value": last / valid[0] - 1.0,
                "start": str(close.index[-sessions - 1])[:10], "end": out["as_of"],
            }
    return out


def relative_observations(stock: Any, sector: Any, sessions: int = 5) -> dict:
    """Compare the same dated observations, independently of legacy scorers."""
    if stock is None or sector is None or "Close" not in stock or "Close" not in sector:
        return {}
    if any(not h.index.is_monotonic_increasing or h.index.has_duplicates for h in (stock, sector)):
        return {}
    common = stock.index.intersection(sector.index)[-sessions - 1:]
    if len(common) < 2:
        return {}
    a, b = [number(v) for v in stock.loc[common, "Close"]], [number(v) for v in sector.loc[common, "Close"]]
    if any(v is None or v <= 0 for v in a + b):
        return {}
    valid_a, valid_b = [v for v in a if v is not None], [v for v in b if v is not None]
    return {"ticker_return": valid_a[-1] / valid_a[0] - 1, "sector_return": valid_b[-1] / valid_b[0] - 1,
            "window_start": str(common[0])[:10], "window_end": str(common[-1])[:10],
            "sessions": len(common) - 1, "aligned_window": True}


def _direction(row: Mapping, mode: str) -> int:
    strategy = _text(row.get("strategy_name")).replace("_", " ").lower()
    known = {"long call": 1, "short put": 1, "bull put": 1,
             "long put": -1, "short call": -1, "bear call": -1,
             "iron condor": 0}
    if strategy in known:
        return known[strategy]
    opt = _text(row.get("type")).lower()
    if mode == "Premium Selling":
        return 1 if opt == "put" else 0
    if mode in ("Credit Spreads", "Iron Condor"):
        return 0
    return {"call": 1, "put": -1}.get(opt, 0)


def build(row: Mapping, ctx: Optional[Mapping] = None) -> dict:
    """Use only supplied observations. Never reconstruct an original gate reason.

    ``ticker_contexts`` and ``market_observations`` belong to THIS scan. Dates
    are observations' dates, never the report generation time. Older sidecars
    can omit either, in which case the information gap is printed explicitly.
    """
    ctx = ctx or {}
    symbol = _text(row.get("symbol"), "UNKNOWN")
    mode = str(ctx.get("mode") or "scan")
    ticker = (ctx.get("ticker_contexts") or {}).get(symbol) or {}
    market = ctx.get("market_observations") or {}
    direction = _direction(row, mode)
    observations: list[dict[str, Any]] = []
    unknowns: list[str] = []
    next_checks: list[dict[str, str]] = []

    def add(key, channel, label, fact, implication, source, as_of=None,
            horizon="not supplied", sign=0):
        alignment = ("supports" if sign == direction else "conflicts") if sign and direction else "context"
        try:
            age_days = (date.fromisoformat(str(ctx.get("as_of"))[:10])
                        - date.fromisoformat(str(as_of)[:10])).days
        except ValueError:
            age_days = None
        if age_days is not None and age_days < 0:
            unknowns.append(f"{label} observation is dated after this scan and was excluded.")
            return
        if age_days is not None and age_days > 7:
            alignment = "context"
            unknowns.append(f"{label} observation is {age_days} calendar days old; refresh before interpreting direction.")
        observations.append({"id": key, "channel": channel, "label": label,
                             "fact": fact, "implication": implication, "source": source,
                             "as_of": as_of, "horizon": horizon, "alignment": alignment,
                             "validation": VALIDATION})

    def check(tool, question):
        item = {"tool": tool, "question": question}
        if item not in next_checks:
            next_checks.append(item)

    # Market direction and option-implied uncertainty are different channels.
    spy = market.get("spy") or {}
    spy5 = (spy.get("returns") or {}).get("5") or {}
    ret = number(spy5.get("value"))
    if ret is not None:
        add("market_momentum", "market_price", "Market", f"SPY {ret:+.1%} over 5 sessions",
            "Describes the broad-market backdrop; it does not forecast this stock.",
            spy.get("source") or "Yahoo Finance daily price history", spy.get("as_of"),
            f"{spy5.get('start')} to {spy5.get('end')}", (1 if ret > 0 else -1) if ret else 0)
    else:
        unknowns.append("Dated broad-market momentum is unavailable.")

    vix = market.get("vix") or {}
    level = number(vix.get("last"))
    if level is None:
        level = number(ctx.get("vix"))
    if level is not None and level > 0:
        parts = [f"VIX daily observation {level:.2f}" if vix.get("as_of") else f"VIX scan reading {level:.2f}"]
        for n in (1, 5):
            value = number(((vix.get("returns") or {}).get(str(n)) or {}).get("value"))
            if value is not None:
                parts.append(f"{n}-session change {value:+.1%}")
        add("vix", "market_options", "Volatility", "; ".join(parts),
            "VIX describes S&P 500 implied volatility, not an up/down forecast. "
            "Compare stock-specific IV and event risk before interpreting an option move.",
            vix.get("source") or "scan VIX reading", vix.get("as_of"),
            "30-day implied volatility; changes use daily observations")
    if not (vix.get("returns") or {}).get("5"):
        unknowns.append("A dated 5-session VIX trend is unavailable.")
    if market.get("cache_stale"):
        unknowns.append("Market observations exceeded the scan cache lifetime; refresh before relying on them.")

    # Stock momentum uses its own timestamp, never the option's last-trade time.
    price = ticker.get("price_observations") or {}
    stock5 = (price.get("returns") or {}).get("5") or {}
    ret = number(stock5.get("value"))
    if ret is None:
        ret = number(row.get("ret_5d"))
    if ret is not None:
        add("stock_momentum", "stock_price", "Stock", f"{symbol} {ret:+.1%} over 5 sessions",
            ("Recent stock direction is context; no directional exposure is inferred." if not direction or not ret
             else "Recent price direction agrees with the candidate's exposure." if ret * direction > 0
             else "Recent price direction may challenge the candidate's exposure; investigate the catalyst."),
            price.get("source") or "scan price history", price.get("as_of"),
            f"{stock5['start']} to {stock5['end']}" if stock5 else "5 sessions; dates unavailable",
            (1 if ret > 0 else -1) if ret else 0)
    else:
        unknowns.append("Stock momentum is unavailable.")

    # Old sector returns used differing endpoints. Only compare aligned windows.
    sector_context = ticker.get("sector_perf") or {}
    sector = sector_context.get("aligned_returns") or {}
    sector_ret, stock_ret = number(sector.get("sector_return")), number(sector.get("ticker_return"))
    if sector_ret is not None and stock_ret is not None and sector.get("aligned_window") is True:
        excess = stock_ret - sector_ret
        add("sector_relative", "relative_price", "Sector", f"{symbol} minus {sector_context.get('sector_etf', 'sector')} "
            f"{excess * 100:+.1f} percentage points",
            "Relative performance describes leadership; it does not establish causation or a lead-lag relationship.",
            "Yahoo Finance aligned daily price histories", sector.get("window_end"),
            f"{sector.get('window_start')} to {sector.get('window_end')}")
    else:
        unknowns.append("Stock/sector returns lack a verified matching window; relative leadership is not inferred.")
        check("TICKER / sector research", "Compare the stock, sector and relevant peers on matching dates.")

    iv, hv = number(row.get("impliedVolatility")), number(row.get("hv_30d"))
    quote_date = _text(row.get("quote_as_of")) or None
    source = _text(row.get("quote_source"), "scan option chain (source unspecified)")
    if iv is not None and hv is not None and iv > 0 and hv > 0:
        gap = (iv - hv) * 100
        add("iv_hv", "stock_options", "Options", f"IV {iv:.1%}; trailing 30-session HV {hv:.1%} "
            f"(gap {gap:+.1f} vol points)",
            "Implied and trailing realized volatility cover different windows. This gap alone does not prove mispricing or direction.",
            source, quote_date, "contract expiry vs trailing 30 sessions")
    else:
        unknowns.append("Comparable option IV and trailing realized volatility are unavailable.")
    freshness = _text(row.get("quote_freshness"), "unknown")
    if freshness != "fresh":
        unknowns.append(f"Option data freshness: {freshness}; verify bid/ask before interpreting a premium change.")
    if not quote_date:
        unknowns.append("The option observation timestamp is unavailable.")
    else:
        unknowns.append("Option timestamps may date the last trade rather than the current bid/ask.")

    net = number(row.get("ev_per_contract"))
    if net is not None:
        add("model_ev", "valuation_model", "Model", f"Estimated net EV ${net:+.0f}/contract after modeled costs",
            "This depends on the volatility forecast and fill assumptions; positive EV is not verified future profit.",
            "scan pricing and transaction-cost model", None, "model assumptions and contract horizon")
        if net <= 0:
            check("STRUCTURE / Probability Lab", "Does a different expression fit the view after costs and volatility assumptions?")

    # An event date must be known and inside the contract window, not an old
    # earnings flag inherited from a prior scan.
    earnings = _text(ticker.get("earnings_date")) or _text(row.get("earnings_date"))
    as_of = str(ctx.get("as_of") or "")[:10]
    expiry = _text(row.get("expiration"))[:10]
    try:
        event_day = str(earnings)[:10]
        event_inside = date.fromisoformat(as_of) <= date.fromisoformat(event_day) <= date.fromisoformat(expiry)
    except (TypeError, ValueError):
        event_inside = False
    if event_inside:
        add("earnings", "company_event", "Event", f"Earnings {event_day} falls before this contract expires",
            "Direction and option profit can diverge when event volatility changes.",
            "scan earnings calendar", None, f"through expiry {expiry}")
        check("TICKER / Probability Lab", "Compare the event's implied move with prior event moves and IV changes.")
    elif not earnings:
        unknowns.append("The earnings date is unavailable; event exposure has not been ruled out.")
    elif not event_inside:
        unknowns.append("The supplied earnings date could not establish an event inside this window; verify the next release.")

    # Reuse the scan's cached news, including a genuine zero sentiment. An empty
    # fetch's default 0.0 must never turn into a neutral sentiment observation.
    news = ticker.get("news_data")
    items = _field(news, "items", [])
    items = items if isinstance(items, (list, tuple)) else []
    sentiment = number(_field(news, "aggregate_sentiment"))
    if items and sentiment is not None:
        publishers = sorted({_text(_field(item, "source"), "unspecified") for item in items})
        published = sorted({_text(_field(item, "published"))[:10] for item in items
                            if _text(_field(item, "published"))})
        add("news_tone", "news", "News tone", f"Headline sentiment {sentiment:+.2f} across {len(items)} retrieved stories",
            "This measures coverage tone, not a validated return forecast. Syndicated stories can overstate independent agreement.",
            ", ".join(publishers), published[-1] if published else None,
            "cached scan coverage; fetched " + _text(_field(news, "fetched_at"), "time unknown"))
    else:
        unknowns.append("No usable cached news sample is available; sentiment is unknown.")
    changes = _field(news, "analyst_changes", [])
    changes = changes if isinstance(changes, (list, tuple)) else []
    if changes:
        upgrades = sum(_text(_field(c, "action")).lower() == "upgrade" for c in changes)
        downgrades = sum(_text(_field(c, "action")).lower() == "downgrade" for c in changes)
        dates = sorted({_text(_field(c, "date"))[:10] for c in changes if _text(_field(c, "date"))})
        add("analyst_changes", "analysts", "Analysts", f"{upgrades} upgrades / {downgrades} downgrades in {len(changes)} cached changes",
            "Research opinion can lag price; independent predictive value has not been established.",
            "scan analyst-change feed", dates[-1] if dates else None, "cached feed window (not independently verified)")

    unknowns.append("Peer lead-lag, sentiment's predictive value and company-specific drivers have not been validated by this card.")
    if any(o["as_of"] is None for o in observations):
        unknowns.append("Some observations lack source timestamps; they cannot establish what changed since a previous scan.")
    check("INTEL / research desk", "Read dated news and filings for the catalyst; distinguish new information from repeated coverage.")
    check("Scenario analysis", "Compare price, IV and time-decay scenarios at the intended holding horizon.")
    from .discovery_research.channels import driver_profile
    drivers = driver_profile(symbol, _text(sector_context.get("sector_etf")))

    support = [o["id"] for o in observations if o["alignment"] == "supports"]
    conflict = [o["id"] for o in observations if o["alignment"] == "conflicts"]
    summary = ("Directional observations disagree; inspect the conflicting channels." if support and conflict
               else "Recent price observations align with this exposure; check option pricing and events separately." if support
               else "Recent price observations run against this exposure; investigate the catalyst." if conflict
               else "These are observations to investigate; no directional forecast is established.")
    rank = number(ctx.get("rank"))
    if rank is None:
        rank = number(row.get("_rank"))
    position = f", displayed position {int(rank)}" if rank is not None and rank > 0 else ""
    bucket = _text(row.get("price_bucket"))
    group = f", {bucket} premium group" if bucket else ""
    return {"schema": SCHEMA, "symbol": symbol, "summary": summary,
            "selection": f"Shown in {mode}{position}{group}. Display order is not a validated forecast; eligibility is explained in the scan gate report.",
            "observations": observations, "supporting": support, "conflicting": conflict,
            "unknowns": unknowns, "next_checks": next_checks,
            "validation": VALIDATION,
            "business_drivers": drivers,
            "dependence_note": "Price trend, momentum and relative strength share price inputs; agreement is not independent confirmation."}


def lines(data: Mapping, compact: bool = False) -> list:
    """Concise opening card; the saved sidecar carries every source and caveat."""
    out = [("Why shown", data["selection"]), ("Research", data["summary"])]
    observations = data.get("observations") or []
    for title, alignment in (("Supports", "supports"), ("Challenges", "conflicts")):
        items = [o for o in observations if o["alignment"] == alignment]
        if items:
            out.append((title, "; ".join(f"{o['fact']} ({o.get('as_of') or 'date unknown'})" for o in items)))
    context_ids = {"vix", "iv_hv", "earnings"} if compact else {"vix", "iv_hv", "earnings", "news_tone", "analyst_changes", "sector_relative"}
    for obs in observations:
        if obs["id"] not in context_ids:
            continue
        caveat = (" Uncertainty, not direction." if obs["id"] == "vix" else
                  " This gap alone does not establish mispricing." if obs["id"] == "iv_hv" else
                  " Coverage/opinion context, not a validated forecast." if obs["channel"] in ("news", "analysts") else "")
        out.append((obs["label"], f"{obs['fact']} ({obs.get('as_of') or 'date unknown'}).{caveat}"))
    unknowns = list(data.get("unknowns") or [])
    # Keep quote/observation age in front of less urgent coverage gaps. The full
    # list remains on the HTML card; terminal output should not bury the facts.
    urgent = ("freshness", "calendar days old", "after this scan", "earnings", "last trade")
    unknowns.sort(key=lambda s: 0 if any(word in s.lower() for word in urgent) else 1)
    if unknowns:
        limit = 2 if compact else 3
        text = " ".join(unknowns[:limit])
        if len(unknowns) > limit:
            text += f" {len(unknowns) - limit} additional coverage/source limits in the tearsheet."
        out.append(("Unknowns", text))
    drivers = data.get("business_drivers") or {}
    if drivers and not compact:
        out.append(("Drivers", "Investigate: " + drivers["drivers"] + "."))
    checks = data.get("next_checks") or []
    if checks:
        out.append(("Investigate", f"{checks[0]['tool']}: {checks[0]['question']}"))
    out.append(("Evidence", data["validation"]))
    return out
