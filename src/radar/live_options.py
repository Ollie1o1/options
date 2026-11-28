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

# Measured institutional OPRA `cbbo-1m` 15:45 ET liquidity map for Test O1 60-120 DTE spreads
# (docs/RESULT_H4_INTRADAY_OPTIONS_EXEC_20261127.md, 24 monthly sessions 2024-2025).
OPRA_O1_LIQUIDITY_TIER: Dict[str, Dict[str, Any]] = {
    "SPY": {"tier": "TIER_1", "med_fric_pct": 2.3, "med_depth": 100, "pass_rate": 1.00},
    "QQQ": {"tier": "TIER_1", "med_fric_pct": 3.8, "med_depth": 42, "pass_rate": 1.00},
    "GLD": {"tier": "TIER_1", "med_fric_pct": 4.9, "med_depth": 252, "pass_rate": 1.00},
    "IWM": {"tier": "TIER_1", "med_fric_pct": 5.0, "med_depth": 67, "pass_rate": 1.00},
    "TLT": {"tier": "TIER_1", "med_fric_pct": 7.2, "med_depth": 92, "pass_rate": 1.00},
    "XLE": {"tier": "TIER_2_WIDE", "med_fric_pct": 12.9, "med_depth": 37, "pass_rate": 0.83},
}


def options_execution_window_guidance(now_et: Optional[dt.datetime] = None) -> Dict[str, Any]:
    """Return H4 OPRA intraday options execution timing guidance (America/New_York).

    Backed by docs/RESULT_H4_INTRADAY_OPTIONS_EXEC_20261127.md:
      * 09:30-10:30 ET: spreads are 2.2x-4.0x wider than midday/close (avoid submitting limits).
      * 11:30-13:30 ET & 15:15-15:50 ET: peak institutional OPRA liquidity and book depth.
      * 13:55-14:15 ET: check FOMC / Fed Minutes calendar before routing orders (spread blowouts).
    """
    from zoneinfo import ZoneInfo

    t = now_et
    if t is None:
        t = dt.datetime.now(ZoneInfo("America/New_York"))
    elif t.tzinfo is not None:
        t = t.astimezone(ZoneInfo("America/New_York"))
    mins = t.hour * 60 + t.minute
    if t.weekday() >= 5 or mins < 9 * 60 + 30 or mins >= 16 * 60:
        return {
            "window": "AFTER_HOURS",
            "flag": "EXEC_AFTER_HOURS_INDICATIVE",
            "note": "Market closed — submit limits during 11:30-13:30 or 15:15-15:50 ET (H4 OPRA).",
        }
    if mins < 10 * 60 + 45:
        return {
            "window": "OPEN_WIDE",
            "flag": "EXEC_OPEN_WIDE_WAIT_1130",
            "note": "Early session (H4 OPRA: 09:45 spreads 2.2x wider than 12:00/15:45 ET) — wait for 11:30+ ET.",
        }
    if 13 * 60 + 55 <= mins <= 14 * 60 + 15:
        return {
            "window": "FED_1400_WINDOW",
            "flag": "EXEC_1400_CHECK_FOMC",
            "note": "14:00 ET window — verify no FOMC decision/Minutes before routing options orders (H4 OPRA).",
        }
    if (11 * 60 + 30 <= mins <= 13 * 60 + 30) or (15 * 60 + 15 <= mins <= 15 * 60 + 50):
        return {
            "window": "OPTIMAL_OPRA_WINDOW",
            "flag": None,
            "note": "Peak OPRA liquidity window (H4: tightest spreads & deepest books).",
        }
    return {
        "window": "NORMAL_RTH",
        "flag": None,
        "note": "Normal RTH session; tightest OPRA windows are 11:30-13:30 and 15:15-15:50 ET (H4).",
    }


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


def _payload_from_provider_rows(
    symbol: str,
    rows: Sequence[Dict[str, Any]],
    *,
    spot: Optional[float],
    source_label: str,
) -> Optional[Dict[str, Any]]:
    """Build `_default_fetch_etf_chain` payload for the O1 [60, 120] DTE window from CBOE/Alpaca rows."""
    from src.radar.wide_spread_rule import DTE_HI, DTE_LO, TARGET_DTE

    today = dt.date.today()
    by_exp: Dict[str, List[Dict[str, Any]]] = {}
    eff_spot = spot
    for r in rows or []:
        exp = str(r.get("expiration") or "")[:10]
        if len(exp) != 10:
            continue
        by_exp.setdefault(exp, []).append(r)
        if eff_spot is None:
            sp = _num(r.get("spot"))
            if sp is not None and sp > 0:
                eff_spot = sp
    if not by_exp or eff_spot is None or eff_spot <= 0:
        return None

    cands: List[Tuple[float, str, int]] = []
    for e in by_exp:
        try:
            dte = (dt.date.fromisoformat(e) - today).days
        except ValueError:
            continue
        if DTE_LO <= dte <= DTE_HI:
            cands.append((abs(float(dte) - TARGET_DTE), e, dte))
    if not cands:
        return None
    cands.sort()

    for _dist, exp, dte in cands[:3]:
        puts: List[Dict[str, Any]] = []
        calls: List[Dict[str, Any]] = []
        for r in by_exp[exp]:
            rec = {
                "strike": r.get("strike"),
                "bid": r.get("bid"),
                "ask": r.get("ask"),
                "delta": r.get("delta"),
                "iv": r.get("iv"),
                "openInterest": r.get("open_interest"),
                "volume": r.get("volume"),
            }
            if str(r.get("type") or "").lower().startswith("p"):
                puts.append(rec)
            else:
                calls.append(rec)
        two_sided = sum(
            1
            for r in puts
            if _num(r.get("bid")) and _num(r.get("ask"))
            and float(r.get("bid") or 0) > 0
            and float(r.get("ask") or 0) >= float(r.get("bid") or 0)
            and 0.80 * eff_spot <= float(r.get("strike") or 0) <= eff_spot
        )
        if two_sided >= 3:
            return {
                "symbol": symbol,
                "spot": eff_spot,
                "expiration": exp,
                "dte": dte,
                "puts": puts,
                "calls": calls,
                "source": source_label,
            }
    return None


def _default_fetch_etf_chain(symbol: str) -> Optional[Dict[str, Any]]:
    """Spot + O1 [60, 120] DTE chain via 3-tier router: CBOE -> Alpaca indicative -> yfinance.

    Eliminates Yahoo's post-open 0/0 quote gap while preserving provenance (`source`).
    """
    from src.radar.wide_spread_rule import DTE_HI, DTE_LO, TARGET_DTE, choose_expiration

    # 1. Free CBOE delayed chain (full exchange Greeks + bid/ask sizes in 1 request)
    try:
        from src.cboe_client import fetch_chain

        cboe_rows = fetch_chain(symbol, timeout=10)
        if cboe_rows:
            payload = _payload_from_provider_rows(symbol, cboe_rows, spot=None, source_label="cboe")
            if payload is not None:
                return payload
    except Exception:
        pass

    # 2. Alpaca indicative snapshots (when credentials configured; avoids Yahoo 0/0 open)
    spot_hint: Optional[float] = None
    try:
        from src.alpaca_data import fetch_option_chain_snapshots, has_credentials

        if has_credentials():
            import yfinance as yf

            hist0 = yf.Ticker(symbol).history(period="5d")
            if hist0 is not None and not hist0.empty:
                spot_hint = float(hist0["Close"].dropna().iloc[-1])
            apca_rows = fetch_option_chain_snapshots(
                symbol, feed="indicative", min_dte=DTE_LO, max_dte=DTE_HI, spot=spot_hint
            )
            if apca_rows:
                payload = _payload_from_provider_rows(
                    symbol, apca_rows, spot=spot_hint, source_label="alpaca_indicative"
                )
                if payload is not None:
                    return payload
    except Exception:
        pass

    # 3. Yahoo Finance fallback
    import yfinance as yf

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

    if spot_hint is not None and spot_hint > 0:
        spot = spot_hint
    else:
        hist = tkr.history(period="5d")
        if hist is None or hist.empty:
            return None
        spot = float(hist["Close"].dropna().iloc[-1])

    # Candidate expirations in [60, 120] sorted by proximity to 90 DTE
    candidates = sorted(
        [(abs(d - TARGET_DTE), e, int(d)) for e, d in exp_dtes.items() if DTE_LO <= d <= DTE_HI]
    )
    best_payload: Optional[Dict[str, Any]] = None
    best_two_sided = -1
    for _dist, exp, dte in candidates[:3]:
        try:
            ch = tkr.option_chain(exp)
        except Exception:
            continue
        puts = ch.puts.to_dict("records") if ch.puts is not None else []
        calls = ch.calls.to_dict("records") if ch.calls is not None else []
        two_sided = sum(
            1
            for r in puts
            if _num(r.get("bid")) and _num(r.get("ask"))
            and float(r.get("bid") or 0) > 0
            and float(r.get("ask") or 0) >= float(r.get("bid") or 0)
            and 0.80 * spot <= float(r.get("strike") or 0) <= spot
        )
        payload = {
            "symbol": symbol,
            "spot": spot,
            "expiration": exp,
            "dte": dte,
            "puts": puts,
            "calls": calls,
            "source": "yfinance",
        }
        if two_sided > best_two_sided:
            best_two_sided = two_sided
            best_payload = payload
        # Primary expiry has plenty of two-sided quotes — stop immediately
        if two_sided >= 5:
            break
    return best_payload


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
        flags = ["LIMIT_ONLY_NO_MKT", "TAXABLE_ONLY"]
        exec_win = options_execution_window_guidance()
        if exec_win.get("flag"):
            flags.append(str(exec_win["flag"]))
        opra_tier = OPRA_O1_LIQUIDITY_TIER.get(sym)
        if opra_tier and opra_tier.get("tier") == "TIER_2_WIDE":
            flags.append("OPRA_WIDE_TIER2")
        if sym == "GLD" and is_put:
            # Study 3A (2024-2026 OPRA OOS): GLD bull puts lagged linear gold delta by -4.33% [-10.9%, -0.01%].
            flags.append("GLD_OOS_LAGGED_DELTA")
        if sym != "SPY":
            # The LEAD verdict was measured on SPY 2010-2023 only (see Study 3A for 2024-2026 OOS).
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
                badges=("LIVE_CHAIN", f"DTE:{dte}", f"Δ{short_delta:+.2f}", f"LIMIT>={ladder.p_walk:.2f}", "TP50_21DTE", "TAXABLE_ONLY"),
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
                    "quote_source": str(data.get("source") or "yfinance"),
                    "rate_source": r_src,
                    "execution_window": exec_win["window"],
                    "execution_note": exec_win["note"],
                    "opra_liquidity_tier": opra_tier,
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
