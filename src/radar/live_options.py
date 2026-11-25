"""Opt-in live 60-120 DTE wide ETF spread scanner (Test O1 geometry) & 1-click price updater."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.radar.schema import Opportunity
from src.structure.execution_plan import build_walk_away_ladder

DEFAULT_LIQUID_ETFS: Tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "XLE", "XLV", "XLP", "XLK", "XLF", "SMH", "GLD", "TLT",
)


def refresh_local_prices(db_path: str = "data/equity_ohlcv.db") -> Dict[str, Any]:
    """Update all symbols currently stored in `data/equity_ohlcv.db` via the audited Yahoo refresher."""
    import sqlite3
    from src.breakout.refresh import refresh_universe
    from src.paths import repo_path

    full = repo_path(db_path)
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        syms = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker").fetchall()]
    finally:
        con.close()

    if not syms:
        return {"updated": 0, "failed": 0, "symbols": 0}
    res = refresh_universe(syms, db_path=full)
    try:
        from src.cache_release import release_yfinance_caches

        release_yfinance_caches()
    except Exception:
        pass
    return {
        "symbols": len(syms),
        "updated": len(res.get("updated") or []),
        "current": len(res.get("current") or []),
        "failed": len(res.get("failed") or []),
    }


def _default_fetch_etf_chain(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch spot + nearest 60-120 DTE expiration put/call chain via yfinance."""
    import yfinance as yf

    tkr = yf.Ticker(symbol)
    exps = list(tkr.options or ())
    if not exps:
        return None
    today = dt.date.today()
    scored_exps: List[Tuple[float, int, str]] = []
    for e in exps:
        try:
            dte = (dt.date.fromisoformat(e) - today).days
        except ValueError:
            continue
        if 60 <= dte <= 120:
            scored_exps.append((abs(dte - 90.0), dte, e))
    if not scored_exps:
        return None
    scored_exps.sort()
    _, chosen_dte, chosen_exp = scored_exps[0]

    hist = tkr.history(period="5d")
    if hist is None or hist.empty:
        return None
    spot = float(hist["Close"].iloc[-1])
    ch = tkr.option_chain(chosen_exp)
    puts = ch.puts.to_dict("records") if ch.puts is not None else []
    calls = ch.calls.to_dict("records") if ch.calls is not None else []
    return {
        "symbol": symbol,
        "spot": spot,
        "expiration": chosen_exp,
        "dte": chosen_dte,
        "puts": puts,
        "calls": calls,
    }


def scan_live_etf_wide_spreads(
    symbols: Sequence[str] = DEFAULT_LIQUID_ETFS,
    *,
    side: str = "bull_put",
    max_friction_to_credit: float = 0.25,
    fetch_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[Opportunity], List[str]]:
    """Scan tight-spread ETFs for Test O1 (`60-120 DTE, 5%-wide, <=25% friction`) credit spreads.

    Returns `(opportunities, warnings)`. Always computes exact Walk-Away Limit Price so the user
    knows the exact price to post (and never crosses market orders).
    """
    fetcher = fetch_fn or _default_fetch_etf_chain
    as_of = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opps: List[Opportunity] = []
    warnings: List[str] = []
    is_put = str(side).lower() != "bear_call"

    for sym in symbols:
        try:
            data = fetcher(sym.upper())
        except Exception as exc:
            warnings.append(f"{sym}: {exc}")
            continue
        if not data:
            continue
        spot = float(data.get("spot") or 0.0)
        exp = str(data.get("expiration") or "")
        dte = int(data.get("dte") or 90)
        rows = data.get("puts" if is_put else "calls") or []
        if spot <= 0 or not rows:
            continue

        # Clean valid quotes (bid > 0, ask >= bid)
        valid = []
        for r in rows:
            try:
                k = float(r["strike"])
                b = float(r["bid"])
                a = float(r["ask"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(k) and math.isfinite(b) and math.isfinite(a) and b > 0 and a >= b:
                valid.append({"strike": k, "bid": b, "ask": a, "mid": 0.5 * (b + a)})
        if len(valid) < 2:
            continue

        # Target short strike ~6% OTM (~0.25 delta at 90 DTE) and long strike 5% of spot wider
        target_short = spot * (0.94 if is_put else 1.06)
        short_leg = min(valid, key=lambda x: abs(x["strike"] - target_short))
        k_short = short_leg["strike"]

        if is_put:
            long_cands = [x for x in valid if x["strike"] < k_short]
            target_long = k_short - 0.05 * spot
        else:
            long_cands = [x for x in valid if x["strike"] > k_short]
            target_long = k_short + 0.05 * spot
        if not long_cands:
            continue
        long_leg = min(long_cands, key=lambda x: abs(x["strike"] - target_long))
        k_long = long_leg["strike"]
        width = abs(k_short - k_long)
        if width <= 0:
            continue

        cr_mid = short_leg["mid"] - long_leg["mid"]
        cr_cross = short_leg["bid"] - long_leg["ask"]
        if cr_mid <= 0 or cr_cross <= 0 or cr_mid >= width:
            continue

        rt_spread = (short_leg["ask"] - short_leg["bid"]) + (long_leg["ask"] - long_leg["bid"])
        fric_ratio = rt_spread / cr_cross
        if fric_ratio > max_friction_to_credit:
            continue

        ladder = build_walk_away_ladder(
            structure="Wide Credit Spread",
            is_credit=True,
            p_mid=cr_mid,
            p_cross=cr_cross,
            ev_view_gross_per_share=cr_mid,
            max_friction_to_credit=max_friction_to_credit,
        )
        max_risk_usd = (width - ladder.p_walk) * 100.0
        strat_label = "Bull Put" if is_put else "Bear Call"
        opps.append(
            Opportunity(
                ticker=sym.upper(),
                asset_class="option",
                playbook_id="etf_wide_spread",
                playbook_label=f"60-120d Wide {strat_label}",
                evidence_tier="LEAD",
                direction="LONG" if is_put else "SHORT",
                spot=round(spot, 2),
                chg_1d_pct=None,
                chg_5d_pct=None,
                horizon_label=f"{dte}d",
                setup_summary=(
                    f"{strat_label} {k_short:g}/{k_long:g} exp {exp} · "
                    f"Mid ${cr_mid:.2f} · Min Limit ${ladder.p_walk:.2f} · Risk ${max_risk_usd:,.0f}"
                ),
                badges=("LIVE_CHAIN", f"DTE:{dte}", f"LIMIT>={ladder.p_walk:.2f}", "TP50_21DTE"),
                risk_flags=("LIMIT_ONLY_NO_MKT",),
                friction_label=f"{fric_ratio*100:.1f}% cr",
                friction_value=round(fric_ratio, 4),
                confluence_count=1,
                as_of=as_of,
                extra={
                    "expiration": exp,
                    "dte": dte,
                    "short_strike": k_short,
                    "long_strike": k_long,
                    "credit_mid": round(cr_mid, 2),
                    "credit_cross": round(cr_cross, 2),
                    "walk_away_limit": ladder.p_walk,
                    "max_risk_usd": round(max_risk_usd, 2),
                },
            )
        )

    if fetch_fn is None:
        try:
            from src.cache_release import release_yfinance_caches

            release_yfinance_caches()
        except Exception:
            pass

    opps.sort(key=lambda o: (o.friction_value, o.ticker))
    return opps, warnings
