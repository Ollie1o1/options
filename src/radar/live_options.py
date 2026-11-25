"""Opt-in live 60-120 DTE wide ETF spread scanner (Test O1 geometry) & 1-click price updater."""
from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass, field
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
    """Spot + the O1 expiration's (nearest 90 DTE in [60, 120]) put/call chain via yfinance."""
    import yfinance as yf

    from src.radar.wide_spread_rule import choose_expiration

    tkr = yf.Ticker(symbol)
    exps = list(tkr.options or ())
    if not exps:
        return None
    today = dt.date.today()
    exp_dtes: Dict[str, float] = {}
    for e in exps:
        try:
            exp_dtes[e] = float((dt.date.fromisoformat(e) - today).days)
        except ValueError:
            continue
    chosen_exp = choose_expiration(exp_dtes)
    if chosen_exp is None:
        return {"symbol": symbol, "spot": 0.0, "expiration": "", "dte": 0, "puts": [], "calls": [],
                "no_expiry": True}
    chosen_dte = int(exp_dtes[chosen_exp])

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


SKIP_PLAIN = {
    "no_expiry": "no expiration listed 60-120 DTE",
    "no_short_delta": "no strike within 0.08 of the 0.25-delta target",
    "no_long_leg": "no quoted strike for the 5%-of-spot long leg",
    "zero_width": "long leg collapsed onto the short strike",
    "non_pos_credit": "crossed credit not positive (or not below width)",
    "friction_gt_25pct": "round-trip spread above 25% of the credit",
    "no_chain": "chain unavailable",
}


@dataclass
class LiveScanResult:
    opportunities: List[Opportunity]
    warnings: List[str]                       # fetch errors
    skipped: Dict[str, str] = field(default_factory=dict)   # ticker -> plain-English reason
    rate_source: str = ""


def _rows_to_quotes(rows: Sequence[Dict[str, Any]], *, spot: float, dte: float, opt_type: str,
                    r: float) -> List[Any]:
    """LegQuotes: use a supplied `delta` column when the source has one, else compute it."""
    from src.radar.wide_spread_rule import LegQuote, quotes_from_rows

    if rows and all("delta" in row for row in rows):
        out = []
        for row in rows:
            try:
                k = float(row["strike"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append(LegQuote(strike=k, bid=_num(row.get("bid")), ask=_num(row.get("ask")),
                                delta=_num(row.get("delta")), iv=_num(row.get("iv")),
                                open_interest=_num(row.get("openInterest")), volume=_num(row.get("volume"))))
        return sorted(out, key=lambda q: q.strike)
    return quotes_from_rows(rows, spot=spot, dte=dte, opt_type=opt_type, r=r)


def quote_coverage(quotes: Sequence[Any], *, spot: float, is_put: bool) -> Tuple[int, int]:
    """(strikes within 20% OTM of spot on the traded side, how many have a two-sided quote)."""
    if spot <= 0:
        return 0, 0
    lo, hi = (0.80 * spot, spot) if is_put else (spot, 1.20 * spot)
    near = [q for q in quotes if lo <= float(q.strike) <= hi]
    return len(near), sum(1 for q in near if q.two_sided)


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def scan_live_etf_wide_spreads_detailed(
    symbols: Sequence[str] = DEFAULT_LIQUID_ETFS,
    *,
    side: str = "bull_put",
    max_friction_to_credit: float = 0.25,
    fetch_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    risk_free: Optional[float] = None,
) -> LiveScanResult:
    """Live chains through the Test O1 rule (src.radar.wide_spread_rule.select_legs).

    Every admitted spread carries an exact Walk-Away Limit Price (never cross the
    touch); every symbol that produced nothing says why.
    """
    from src.radar.wide_spread_rule import risk_free_rate, select_legs

    fetcher = fetch_fn or _default_fetch_etf_chain
    as_of = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opps: List[Opportunity] = []
    warnings: List[str] = []
    skipped: Dict[str, str] = {}
    is_put = str(side).lower() != "bear_call"
    if risk_free is None:
        r, r_src = risk_free_rate()
    else:
        r, r_src = float(risk_free), f"given {float(risk_free):.2%}"

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
        if sym not in fetched:
            continue
        data = fetched.get(sym)
        if not data:
            skipped[sym] = SKIP_PLAIN["no_chain"]
            continue
        if data.get("no_expiry"):
            skipped[sym] = SKIP_PLAIN["no_expiry"]
            continue
        spot = float(data.get("spot") or 0.0)
        exp = str(data.get("expiration") or "")
        dte = int(data.get("dte") or 90)
        rows = data.get("puts" if is_put else "calls") or []
        if spot <= 0 or not rows:
            skipped[sym] = SKIP_PLAIN["no_chain"]
            continue

        quotes = _rows_to_quotes(rows, spot=spot, dte=dte, opt_type="put" if is_put else "call", r=r)
        legs, reason = select_legs(quotes, spot=spot, side=side, expiration=exp, dte=dte,
                                   max_friction_to_credit=max_friction_to_credit)
        if legs is None:
            msg = SKIP_PLAIN.get(reason, reason)
            n_near, n_two = quote_coverage(quotes, spot=spot, is_put=is_put)
            if reason in ("no_short_delta", "no_long_leg") and n_near and n_two < 0.5 * n_near:
                # A data condition, not a market one: Yahoo often serves 0/0
                # quotes for most strikes for a while after the open.
                msg = (f"Yahoo quotes thin — only {n_two} of {n_near} strikes near the money have a "
                       f"live bid/ask; retry later ({msg})")
            skipped[sym] = msg
            continue

        k_short, k_long = legs.short.strike, legs.long.strike
        cr_mid, cr_cross, width, fric_ratio = legs.credit_mid, legs.credit_cross, legs.width, legs.friction_ratio
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
        short_delta = float(legs.short.delta or 0.0)
        flags = ["LIMIT_ONLY_NO_MKT"]
        if sym != "SPY":
            # The LEAD verdict was measured on SPY 2010-2023 only.
            flags.append("TESTED_ON_SPY_ONLY")
        if abs(legs.width_pct - 0.05) > 0.01:
            # Nearest quoted strike to the 5% wing was far from it (thin or coarse chain).
            flags.append(f"WING_{legs.width_pct * 100:.1f}PCT_NOT_5")
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
                    f"{strat_label} {k_short:g}/{k_long:g} ({legs.width_pct * 100:.1f}% wide) exp {exp} · "
                    f"short Δ{short_delta:+.2f} · "
                    f"Mid ${cr_mid:.2f} · Min Limit ${ladder.p_walk:.2f} · Risk ${max_risk_usd:,.0f}"
                ),
                badges=("LIVE_CHAIN", f"DTE:{dte}", f"Δ{short_delta:+.2f}", f"LIMIT>={ladder.p_walk:.2f}", "TP50_21DTE"),
                risk_flags=tuple(flags),
                friction_label=f"{fric_ratio*100:.1f}% cr",
                friction_value=round(fric_ratio, 4),
                confluence_count=1,
                as_of=as_of,
                extra={
                    "expiration": exp,
                    "dte": dte,
                    "short_strike": k_short,
                    "long_strike": k_long,
                    "short_delta": round(short_delta, 4),
                    "short_iv": None if legs.short.iv is None else round(float(legs.short.iv), 4),
                    "net_delta": None if legs.net_delta is None else round(float(legs.net_delta), 4),
                    "width": round(width, 4),
                    "width_pct": round(legs.width_pct, 4),
                    "credit_mid": round(cr_mid, 2),
                    "credit_cross": round(cr_cross, 2),
                    "walk_away_limit": ladder.p_walk,
                    "max_risk_usd": round(max_risk_usd, 2),
                    "short_oi": legs.short.open_interest,
                    "long_oi": legs.long.open_interest,
                    "rule": "O1: nearest 90 DTE in [60,120], short nearest 0.25 delta (+/-0.08), long 5% of spot OTM",
                    "delta_source": "mid-implied IV, Black-Scholes" if legs.short.iv is not None else "chain",
                    "rate_source": r_src,
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
    return LiveScanResult(opportunities=opps, warnings=warnings, skipped=skipped, rate_source=r_src)


def scan_live_etf_wide_spreads(
    symbols: Sequence[str] = DEFAULT_LIQUID_ETFS,
    *,
    side: str = "bull_put",
    max_friction_to_credit: float = 0.25,
    fetch_fn: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    risk_free: Optional[float] = None,
) -> Tuple[List[Opportunity], List[str]]:
    """`(opportunities, fetch warnings)` — see scan_live_etf_wide_spreads_detailed for skip reasons."""
    res = scan_live_etf_wide_spreads_detailed(
        symbols, side=side, max_friction_to_credit=max_friction_to_credit,
        fetch_fn=fetch_fn, risk_free=risk_free,
    )
    return res.opportunities, res.warnings
