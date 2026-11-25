"""Opt-in live 60-120 DTE wide ETF spread scanner (Test O1 geometry) & 1-click price updater."""
from __future__ import annotations

import datetime as dt
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.radar.schema import Opportunity
from src.structure.execution_plan import build_walk_away_ladder

DEFAULT_LIQUID_ETFS: Tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "XLE", "XLV", "XLP", "XLK", "XLF", "SMH", "GLD", "TLT",
)


def _detect_settled_yahoo_session(default_through: Optional[str]) -> Optional[str]:
    """Return latest trading date where Yahoo Finance has a finite SPY Close & Adj Close.

    After 18:00 ET Yahoo occasionally emits today's row with `Close = NaN` / `Adj Close = NaN`
    before EOD settlement completes. Falling back one trading day avoids rejecting the entire
    universe without ever fabricating a price.
    """
    if default_through is None:
        return None
    try:
        import yfinance as yf

        hist = yf.Ticker("SPY").history(period="7d", auto_adjust=False, actions=True, repair=False, timeout=15)
        if hist is None or hist.empty:
            return default_through
        settled_dates = []
        for idx, rec in hist.iterrows():
            d_str = idx.strftime("%Y-%m-%d")
            if d_str > default_through:
                continue
            c = float(rec.get("Close", math.nan))
            ac = float(rec.get("Adj Close", math.nan))
            if math.isfinite(c) and math.isfinite(ac) and c > 0 and ac > 0:
                settled_dates.append(d_str)
        if settled_dates:
            return max(settled_dates)
    except Exception:
        pass
    return default_through


def refresh_local_prices(
    db_path: str = "data/equity_ohlcv.db",
    *,
    expand_universe: bool = False,
    extra_symbols: Optional[Sequence[str]] = None,
    through: Optional[str] = None,
) -> Dict[str, Any]:
    """Update all symbols in `data/equity_ohlcv.db` (and optionally expand to ~250+ liquid stocks/ETFs)."""
    import sqlite3
    from src.breakout.refresh import completed_session, refresh_universe
    from src.paths import repo_path

    full = repo_path(db_path)
    existing_max: Dict[str, str] = {}
    if os.path.exists(full):
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            for tk, mx in con.execute("SELECT ticker, MAX(date) FROM ohlcv GROUP BY ticker").fetchall():
                existing_max[str(tk).upper()] = str(mx or "")
        finally:
            con.close()

    sym_set = set(existing_max.keys())
    if expand_universe:
        from src.equity_playbooks.universe import build_expanded_universe

        sym_set.update(build_expanded_universe())
    if extra_symbols:
        sym_set.update(str(s).upper().strip() for s in extra_symbols if s)
    syms = sorted(sym_set)

    if not syms:
        return {"updated": 0, "failed": 0, "symbols": 0, "current": 0, "expanded": expand_universe}

    eff_through = through or _detect_settled_yahoo_session(completed_session())

    # Skip symbols already at/beyond eff_through so we never trigger spurious errors
    already_current = [
        s for s in syms if eff_through and existing_max.get(s, "") >= eff_through
    ]
    to_fetch = [s for s in syms if s not in set(already_current)]

    updated_n = 0
    current_n = len(already_current)
    failed_n = 0
    inserted_bars = 0

    # Refresh in chunks of 25 so an isolated delisted symbol cannot starve 300 healthy ones,
    # while a genuine network outage (all 25 fail) still halts cleanly.
    chunk_size = 25
    for i in range(0, len(to_fetch), chunk_size):
        chunk = to_fetch[i : i + chunk_size]
        res = refresh_universe(full, chunk, through=eff_through, pause=0.15)
        raw_sym = res.get("symbols")
        sym_map: Dict[str, Any] = raw_sym if isinstance(raw_sym, dict) else {}
        inserted_bars += int(res.get("inserted") or 0)
        chunk_ok = 0
        for v in sym_map.values():
            if not isinstance(v, dict):
                continue
            st = v.get("status")
            if st in ("updated", "quality_review"):
                chunk_ok += 1
                if (v.get("inserted") or 0) > 0:
                    updated_n += 1
                else:
                    current_n += 1
            else:
                failed_n += 1
        if chunk and chunk_ok == 0:
            # Whole chunk failed — stop early rather than hammering an unreachable feed
            failed_n += len(to_fetch) - (i + len(chunk))
            break

    try:
        from src.cache_release import release_yfinance_caches

        release_yfinance_caches()
    except Exception:
        pass
    return {
        "symbols": len(syms),
        "updated": updated_n,
        "current": current_n,
        "failed": failed_n,
        "inserted_bars": inserted_bars,
        "through": eff_through,
        "expanded": expand_universe,
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

    sym_list = [s.upper() for s in symbols]
    fetched: Dict[str, Optional[Dict[str, Any]]] = {}
    if fetch_fn is None and len(sym_list) > 1:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_one(tk: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
            try:
                return tk, fetcher(tk), None
            except Exception as exc:
                return tk, None, f"{tk}: {exc}"

        with ThreadPoolExecutor(max_workers=min(4, len(sym_list))) as ex:
            for tk, data, err in ex.map(_fetch_one, sym_list):
                if err:
                    warnings.append(err)
                else:
                    fetched[tk] = data
    else:
        for tk in sym_list:
            try:
                fetched[tk] = fetcher(tk)
            except Exception as exc:
                warnings.append(f"{tk}: {exc}")

    for sym in sym_list:
        data = fetched.get(sym)
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
