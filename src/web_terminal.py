"""Zero-reload Bloomberg Terminal × Amber CRT local Web Cockpit (`src/web_terminal.py`).

Served via `opt web` (automatically uses FastAPI/uvicorn if installed, or built-in stdlib
`ThreadingHTTPServer` with ZERO extra dependencies) and also mounted on `src/api.py` at `/desk`.
100% self-contained HTML + CSS + Vanilla JS + crisp inline SVG charts (zero external CDN
dependencies, works fully offline, live ETF spreads, live price & alpha refresh, interactive
Payoff + SVI Sandbox, Wealthsimple CSV drag-and-drop, CRA TFSA safety & 10s Post-Mortem tags).
"""
from __future__ import annotations

import json
import math
import urllib.parse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


def _clean_json(v: Any) -> Any:
    """Recursively convert numpy scalars, NaN/inf, and Timestamps to JSON-safe primitives."""
    try:
        import numpy as np

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        if isinstance(v, np.ndarray):
            return [_clean_json(x) for x in v.tolist()]
    except ImportError:
        pass
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, dict):
        return {str(k): _clean_json(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_json(x) for x in v]
    if hasattr(v, "isoformat") and callable(v.isoformat):
        return v.isoformat()
    return v


def build_desk_overview(
    *,
    refresh: bool = False,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assemble full local cockpit JSON payload for `/desk` (100% offline-safe)."""
    from src.alpha.edgar_events import load_cached_edgar_events
    from src.alpha.formatters import format_macro_text
    from src.alpha.macro_plumbing import load_macro_plumbing
    from src.alpha.options_for_shares import scan_option_signals_for_shares
    from src.cli.guide import plain_english_action_plan
    from src.cli.morning_ritual import build_morning_ritual
    from src.equity_playbooks.rebalance import compute_tfsa_rebalance
    from src.occ_archive import scan_occ_uoa
    from src.profile import estimate_cash_yield_annual, load_profile, rebalance_due_status
    from src.radar.calendar_spreads import scan_etf_calendar_spreads
    from src.tfsa_tax import compute_tfsa_safety_report
    from src.tui.state import DeskDataProvider

    prov = provider or DeskDataProvider()
    snap = prov.get_radar(refresh=refresh)
    prof = load_profile()
    _y_ann, y_lbl = estimate_cash_yield_annual()

    sector_board: List[Dict[str, Any]] = []
    spot_lookup: Dict[str, float] = {}
    try:
        hub = prov.get_hub(refresh=refresh)
        sector_board = hub.board.to_dict().get("rows", [])
        spot_lookup = {
            sym: float(cs[-1])
            for sym, (_ds, cs, _vs) in (hub.ohlcv or {}).items()
            if len(cs) > 0
        }
    except Exception:
        pass

    skew_signals: List[Dict[str, Any]] = []
    try:
        skew_signals = [
            s.to_dict()
            for s in scan_option_signals_for_shares(chain_db=prov.chain_db)[:20]
        ]
    except Exception:
        pass

    macro_text = ""
    try:
        macro_text = format_macro_text(load_macro_plumbing(), rich=False)
    except Exception:
        pass

    edgar_events: List[Dict[str, Any]] = []
    try:
        edgar_events = [e.to_dict() for e in load_cached_edgar_events(limit=30)]
    except Exception:
        pass

    occ_uoa: List[Dict[str, Any]] = []
    try:
        occ_uoa = [u.to_dict() for u in scan_occ_uoa(limit=15)]
    except Exception:
        pass

    calendar_spreads: List[Dict[str, Any]] = []
    try:
        calendar_spreads = [c.to_dict() for c in scan_etf_calendar_spreads(chain_db=prov.chain_db)[:12]]
    except Exception:
        pass

    ghost_race: Dict[str, Any] = {}
    try:
        race = prov.get_ghost_race()
        if race is not None:
            ghost_race = race.to_dict()
    except Exception:
        pass

    book: Dict[str, Any] = {}
    try:
        book = prov.get_book()
    except Exception:
        pass

    rebalance: Dict[str, Any] = {}
    try:
        plan = compute_tfsa_rebalance(
            snap.core_allocation,
            account_usd=prof.tfsa_usd,
            current_shares=prof.core_holdings,
            spot_lookup=spot_lookup,
            fractional=prof.fractional_shares,
            drift_threshold_pct=prof.drift_threshold_pct,
        )
        due, due_msg = rebalance_due_status(
            prof,
            max_drift_pct=plan.max_drift_pct,
            rebalance_recommended=plan.rebalance_recommended,
        )
        rebalance = {
            **plan.to_dict(),
            "due": due,
            "due_message": due_msg,
            "cash_etf": prof.cash_etf,
        }
    except Exception:
        pass

    tfsa_safety: Dict[str, Any] = {}
    try:
        tfsa_safety = compute_tfsa_safety_report(
            profile=prof,
            core_allocation=snap.core_allocation,
            equity_picks=book.get("equity_picks") or [],
        ).to_dict()
    except Exception:
        pass

    morning_ritual: Dict[str, Any] = {}
    try:
        morning_ritual = build_morning_ritual(provider=prov, refresh=False)
    except Exception:
        pass

    health_info: Dict[str, Any] = {}
    try:
        health_info = prov.get_health()
    except Exception:
        pass

    playbooks: List[Dict[str, Any]] = []
    try:
        playbooks = [asdict(p) for p in prov.get_playbooks()]
    except Exception:
        pass

    return _clean_json(
        {
            "snapshot": snap.to_dict(),
            "action_plan": plain_english_action_plan(snap),
            "morning_ritual": morning_ritual,
            "profile": prof.to_dict(),
            "cash_yield_label": y_lbl,
            "tfsa_safety": tfsa_safety,
            "sector_board": sector_board,
            "alpha": {
                "skew_signals": skew_signals,
                "macro_text": macro_text,
                "edgar_events": edgar_events,
                "occ_uoa": occ_uoa,
                "calendar_spreads": calendar_spreads,
            },
            "ghost_race": ghost_race,
            "book": book,
            "rebalance": rebalance,
            "health": health_info,
            "playbooks": playbooks,
        }
    )


def build_desk_card(
    ticker: str,
    *,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assemble Evidence Card + deterministic Pre-Mortem Devil's Advocate for one ticker."""
    from src.alpha.macro_plumbing import load_macro_plumbing
    from src.alpha.options_for_shares import scan_option_signals_for_shares
    from src.alpha.premortem import build_premortem
    from src.profile import load_profile
    from src.tui.state import DeskDataProvider

    tk = ticker.upper().strip()
    prov = provider or DeskDataProvider()
    dos = prov.get_dossier(tk, fetch_chain=False)
    if dos.spot is None:
        dos = prov.get_dossier(tk, fetch_chain=True, allow_live_spot=True)
    hub = prov.get_hub()
    card = hub.card(tk, dossier=dos, chain_db=prov.chain_db)
    sk_rows = scan_option_signals_for_shares(chain_db=prov.chain_db, symbols=(tk,))
    mac = load_macro_plumbing()
    prof = load_profile()
    pm = build_premortem(
        tk,
        card.dossier,
        skew_signal=sk_rows[0] if sk_rows else None,
        macro=mac,
        account_type=prof.account_type,
    )
    return _clean_json(
        {
            "ticker": tk,
            "verdict": card.verdict,
            "verdict_style": card.verdict_style,
            "reasons": list(card.reasons),
            "warnings": list(card.warnings),
            "dossier": card.dossier.to_dict(),
            "premortem": pm.to_dict(),
        }
    )


def build_desk_live_options(*, side: str = "bull_put") -> Dict[str, Any]:
    """Run live 60-120d wide ETF spread scan (Test O1 rule) with full skip reasons & warnings."""
    from src.radar.live_options import scan_live_etf_wide_spreads_detailed

    res = scan_live_etf_wide_spreads_detailed(side=side)
    return _clean_json(
        {
            "side": side,
            "count": len(res.opportunities),
            "opportunities": [o.to_dict() for o in res.opportunities],
            "skipped": dict(res.skipped),
            "warnings": list(res.warnings),
            "rate_source": res.rate_source,
        }
    )


def run_desk_price_update() -> Dict[str, Any]:
    """Refresh daily OHLCV prices + FRED rates and rebuild local radar cache."""
    from src.tui.state import DeskDataProvider

    prov = DeskDataProvider()
    res = prov.update_prices(expand_universe=False)
    prov.get_radar(refresh=True)
    return _clean_json({"ok": True, "result": res})


def run_desk_alpha_refresh(*, chain_limit: int = 15) -> Dict[str, Any]:
    """Refresh live Macro Plumbing (4-pt VIX/FRED/COT/Polymarket), SEC EDGAR, Chains & OCC OI."""
    from src.alpha.edgar_events import refresh_edgar_events
    from src.alpha.macro_plumbing import refresh_macro_plumbing
    from src.alpha.options_for_shares import EXPANDED_CHAIN_UNIVERSE
    from src.chain_archive import archive_symbols
    from src.occ_archive import archive_occ_daily

    out: Dict[str, Any] = {}
    try:
        mac = refresh_macro_plumbing()
        out["macro"] = {"vix_curve": mac.vix_curve, "term": mac.vix_term_label, "errors": mac.errors}
    except Exception as exc:
        out["macro"] = {"error": str(exc)}
    try:
        ed = refresh_edgar_events()
        out["edgar"] = {"saved": ed.get("saved", 0), "errors": ed.get("errors", [])}
    except Exception as exc:
        out["edgar"] = {"error": str(exc)}
    try:
        st: Dict[str, Any] = {}
        written = archive_symbols(list(EXPANDED_CHAIN_UNIVERSE[: max(1, int(chain_limit))]), stats=st)
        out["chains"] = {"rows": written, **st}
    except Exception as exc:
        out["chains"] = {"error": str(exc)}
    try:
        out["occ"] = archive_occ_daily()
    except Exception as exc:
        out["occ"] = {"error": str(exc)}
    return _clean_json({"ok": True, **out})


def build_desk_express(
    *,
    ticker: str = "SPY",
    direction: str = "BULLISH",
    target_pct: float = 8.0,
    days: int = 30,
    risk_usd: float = 500.0,
    account_type: str = "tfsa",
    owns_100: bool = False,
    allow_live: bool = True,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute interactive Payoff + SVI Probability Sandbox payload for one ticker/view."""
    import numpy as np
    from src.cli.guide import plain_english_vehicle_verdict
    from src.structure.chain_data import load_chain_for_horizon
    from src.structure.express import _build_lognormal_density, compare_vehicles
    from src.structure.types import DetailedView
    from src.tui.state import DeskDataProvider

    tk = str(ticker or "SPY").upper().strip() or "SPY"
    prov = provider or DeskDataProvider()
    dos = prov.get_dossier(tk, fetch_chain=False)
    if dos.spot is None and allow_live:
        dos = prov.get_dossier(tk, fetch_chain=True, allow_live_spot=True)
    spot = float(dos.spot) if dos.spot and dos.spot > 0 else 100.0
    iv = (
        (dos.atm_iv / 100.0)
        if dos.atm_iv is not None
        else ((dos.rv_20_ann / 100.0) if dos.rv_20_ann else 0.25)
    )

    dir_u = str(direction or "BULLISH").upper()
    if dir_u not in ("BULLISH", "BEARISH", "NEUTRAL"):
        dir_u = "BULLISH"
    t_move = float(target_pct) / 100.0
    if dir_u == "BEARISH" and t_move > 0:
        t_move = -t_move
    elif dir_u == "NEUTRAL":
        t_move = 0.0

    view = DetailedView(
        symbol=tk,
        direction=dir_u,
        target_move_pct=t_move,
        horizon_days=max(5, min(365, int(days))),
        risk_budget_usd=max(50.0, float(risk_usd)),
        account_type=str(account_type or "tfsa").lower(),
        owns_100_shares=bool(owns_100),
    )
    vehicles = compare_vehicles(
        view,
        spot=spot,
        iv=iv,
        chain_db=prov.chain_db,
        allow_live=(allow_live and dos.atm_iv is None),
    )

    # Build 41-point price & probability density curve for SVG rendering
    sl = load_chain_for_horizon(
        tk,
        target_dte=view.horizon_days,
        spot=spot,
        chain_db=prov.chain_db,
        allow_live=(allow_live and dos.atm_iv is None),
        build_rnd=True,
    )
    dens = sl.rnd_density if (sl is not None and sl.rnd_density is not None) else _build_lognormal_density(spot, iv, view.horizon_days / 365.0)
    rnd_source = sl.rnd_source if (sl is not None and sl.rnd_density is not None) else "lognormal"

    lo_px = round(spot * 0.78, 2)
    hi_px = round(spot * 1.24, 2)
    grid = np.linspace(lo_px, hi_px, 41)
    pdf_vals = np.interp(grid, dens.K, dens.pdf, left=0.0, right=0.0)
    max_pdf = float(np.max(pdf_vals)) if len(pdf_vals) else 1.0
    pdf_norm = [round(float(v) / max(max_pdf, 1e-12), 4) for v in pdf_vals]

    # Simple share payoff + per-vehicle horizon P&L curves across grid
    shares_qty = max(0.01, view.risk_budget_usd / max(spot, 1e-6))
    sgn = -1.0 if dir_u == "BEARISH" else 1.0
    share_pnl = [round(sgn * (float(px) - spot) * shares_qty, 2) for px in grid]

    vehicle_payoffs: List[List[float]] = []
    for v in vehicles:
        vn = str(v.vehicle)
        max_r = max(1.0, float(v.capital_at_risk_usd or view.risk_budget_usd))
        max_p = max(1.0, float(v.max_profit_usd or max_r))
        dd = float(v.delta_dollars or 0.0)
        th_tot = float(v.theta_per_day_usd or 0.0) * view.horizon_days
        if vn == "Shares":
            vehicle_payoffs.append(list(share_pnl))
        elif "Credit Spread" in vn:
            # Defined-risk credit spread payoff shape anchored to max profit & max loss
            curve_v = []
            for px in grid:
                pct = (float(px) - spot) / max(spot, 1e-6)
                if dir_u == "BEARISH":
                    # Bear call: profits if price stays below +5%, loses above
                    if pct <= 0.04:
                        val = max_p
                    elif pct >= 0.09:
                        val = -max_r
                    else:
                        val = max_p - ((pct - 0.04) / 0.05) * (max_p + max_r)
                else:
                    # Bull put: profits if price stays above -5%, loses below
                    if pct >= -0.04:
                        val = max_p
                    elif pct <= -0.09:
                        val = -max_r
                    else:
                        val = max_p - ((-0.04 - pct) / 0.05) * (max_p + max_r)
                curve_v.append(round(val, 2))
            vehicle_payoffs.append(curve_v)
        elif "Covered Call" in vn:
            # Long 100 shares capped at ~+6% + collected credit
            cap_px = spot * 1.06
            credit_est = max(0.0, max_p - 100.0 * (cap_px - spot))
            vehicle_payoffs.append([
                round(100.0 * (min(float(px), cap_px) - spot) + credit_est, 2) for px in grid
            ])
        elif "Debit Vertical" in vn:
            curve_v = []
            for px in grid:
                pct = (float(px) - spot) / max(spot, 1e-6)
                dir_pct = -pct if dir_u == "BEARISH" else pct
                if dir_pct <= 0.0:
                    val = -max_r
                elif dir_pct >= 0.08:
                    val = max_p
                else:
                    val = -max_r + (dir_pct / 0.08) * (max_p + max_r)
                curve_v.append(round(val, 2))
            vehicle_payoffs.append(curve_v)
        else:
            # Convex long option / Deep-ITM LEAPS approximation floored at -max_risk
            curve_v = []
            for px in grid:
                pct = (float(px) - spot) / max(spot, 1e-6)
                lin = dd * pct + 0.35 * abs(dd) * (pct * pct) * 4.0 + th_tot
                curve_v.append(round(max(-max_r, lin), 2))
            vehicle_payoffs.append(curve_v)

    return _clean_json(
        {
            "ticker": tk,
            "spot": round(spot, 2),
            "iv_pct": round(iv * 100.0, 1),
            "rnd_source": rnd_source,
            "view": asdict(view),
            "verdict": plain_english_vehicle_verdict(vehicles, view),
            "vehicles": [asdict(v) for v in vehicles],
            "curve": {
                "prices": [round(float(x), 2) for x in grid],
                "pdf_norm": pdf_norm,
                "share_pnl": share_pnl,
                "vehicle_payoffs": vehicle_payoffs,
                "target_price": round(spot * (1.0 + t_move), 2),
            },
        }
    )


def run_desk_live_quotes(tickers: List[str]) -> Dict[str, Any]:
    from src.tui.state import DeskDataProvider

    return _clean_json({"quotes": DeskDataProvider().get_live_quotes(tickers)})


def run_desk_log_pick(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.own_picks import log_equity_pick

    tk = str(payload.get("ticker") or "").upper().strip()
    px = float(payload.get("price") or 0)
    sh = float(payload.get("shares") or 1)
    pb = str(payload.get("playbook_id") or "discretionary").strip() or "discretionary"
    stop = float(payload["stop_price"]) if payload.get("stop_price") not in (None, "") else None
    tgt = float(payload["target_price"]) if payload.get("target_price") not in (None, "") else None
    note = str(payload.get("notes") or "")
    pid = log_equity_pick(tk, px, shares=sh, stop_price=stop, target_price=tgt, playbook_id=pb, notes=note)
    return _clean_json({"ok": True, "pick_id": pid, "ticker": tk})


def run_desk_close_pick(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.own_picks import close_equity_pick

    pid = int(payload["pick_id"])
    exit_px = float(payload["exit_price"])
    tag = str(payload.get("close_tag") or "").upper().strip() or None
    notes = str(payload.get("notes") or "").strip() or None
    ok = close_equity_pick(pid, exit_px, notes=notes, close_tag=tag)
    return _clean_json({"ok": bool(ok), "pick_id": pid, "exit_price": exit_px, "close_tag": tag or ""})


def run_desk_import_ws_csv(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.profile import import_wealthsimple_csv_text

    csv_text = str(payload.get("csv_text") or "")
    filename = str(payload.get("filename") or "wealthsimple_upload.csv")
    if not csv_text.strip():
        raise ValueError("CSV text is empty")
    return _clean_json({"ok": True, **import_wealthsimple_csv_text(csv_text, filename=filename)})


def run_desk_update_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save TFSA balance, cash ETF, risk budget, and CRA room straight from Web Cockpit."""
    from src.profile import load_profile, save_profile

    prof = load_profile()
    if payload.get("tfsa_usd") not in (None, ""):
        prof.tfsa_usd = max(100.0, float(payload["tfsa_usd"]))
    if payload.get("risk_budget_usd") not in (None, ""):
        prof.risk_budget_usd = max(25.0, float(payload["risk_budget_usd"]))
    if payload.get("cash_etf"):
        prof.cash_etf = str(payload["cash_etf"]).upper().strip()
    if payload.get("account_type") in ("tfsa", "taxable"):
        prof.account_type = str(payload["account_type"]).lower()
    if payload.get("tfsa_room_cad") not in (None, ""):
        prof.tfsa_room_cad = max(0.0, float(payload["tfsa_room_cad"]))
    if payload.get("tfsa_contributed_ytd_cad") not in (None, ""):
        prof.tfsa_contributed_ytd_cad = max(0.0, float(payload["tfsa_contributed_ytd_cad"]))
    save_profile(prof)
    return _clean_json({"ok": True, "profile": prof.to_dict()})


def run_desk_apply_rebalance() -> Dict[str, Any]:
    """Compute current VM_OUTLOOK_TOP3 target shares and persist into data/tfsa_profile.json."""
    from src.equity_playbooks.rebalance import compute_tfsa_rebalance
    from src.profile import apply_rebalance_to_profile, load_profile
    from src.tui.state import DeskDataProvider

    prov = DeskDataProvider()
    snap = prov.get_radar(refresh=False)
    prof = load_profile()
    plan = compute_tfsa_rebalance(
        snap.core_allocation,
        account_usd=prof.tfsa_usd,
        current_shares=prof.core_holdings,
        fractional=prof.fractional_shares,
        drift_threshold_pct=prof.drift_threshold_pct,
    )
    target_map = {o.ticker: o.target_shares for o in plan.orders if o.action != "RESERVE"}
    updated = apply_rebalance_to_profile(prof, target_map)
    return _clean_json({"ok": True, "last_rebalance_date": updated.last_rebalance_date, "core_holdings": updated.core_holdings})


def build_desk_options_scan(
    *,
    strategy: str = "all",
    query: str = "",
    live: bool = False,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """Multi-strategy options scan (archive-first or opt-in live Yahoo) for Web Cockpit `[8] OPTIONS HUB`."""
    from src.tui.options_engine import describe_cal_pop_for_row, filter_option_rows
    from src.tui.state import DeskDataProvider

    prov = provider or DeskDataProvider()
    rows, note = prov.scan_options(strategy=strategy, live=live)
    filtered = filter_option_rows(rows, query)
    serialized: List[Dict[str, Any]] = []
    for r in filtered[:80]:
        d = r.to_dict()
        d["cal_pop_note"] = describe_cal_pop_for_row(r)
        serialized.append(d)
    return _clean_json({"count": len(serialized), "total": len(rows), "note": note, "rows": serialized})


def build_desk_options_chain(
    *,
    ticker: str = "SPY",
    expiration: Optional[str] = None,
    band_pct: float = 0.12,
    live: bool = False,
    provider: Optional[Any] = None,
) -> Dict[str, Any]:
    """T-shaped Call | Strike | Put Chain Matrix + GEX + IV Smile + Term Structure."""
    from src.tui.state import DeskDataProvider

    prov = provider or DeskDataProvider()
    mat = prov.get_chain_matrix(
        ticker,
        expiration=expiration or None,
        band_pct=band_pct,
        allow_live=live,
    )
    if mat is None:
        return _clean_json({"found": False, "ticker": ticker.upper(), "error": f"No chain found for {ticker.upper()}"})
    return _clean_json({"found": True, "chain": asdict(mat)})


def run_desk_options_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Price 1-4 custom option legs with live Greeks, Walk-Away Ladder, Payoff curves & 2D Risk Surface."""
    from src.tui.options_engine import OptionLeg, analyze_custom_legs

    tk = str(payload.get("ticker") or "SPY").upper().strip()
    spot = float(payload.get("spot") or 100.0)
    days_elapsed = int(payload.get("days_elapsed") or 0)
    iv_shock_pct = float(payload.get("iv_shock_pct") or 0.0)
    raw_legs = payload.get("legs") or []
    legs: List[OptionLeg] = []
    for lg in raw_legs[:4]:
        legs.append(
            OptionLeg(
                opt_type=str(lg.get("opt_type") or "put").lower(),
                side=str(lg.get("side") or "BUY").upper(),
                strike=float(lg["strike"]),
                expiration=str(lg.get("expiration") or ""),
                dte=max(1, int(lg.get("dte") or 30)),
                bid=float(lg.get("bid") or 0.0),
                ask=float(lg.get("ask") or 0.0),
                mid=float(lg.get("mid") or 0.05),
                iv=float(lg.get("iv") or 0.25),
                delta=float(lg.get("delta") or 0.0),
                theta=float(lg.get("theta") or 0.0),
                open_interest=int(lg.get("open_interest") or 0),
                volume=int(lg.get("volume") or 0),
                qty=max(1, int(lg.get("qty") or 1)),
            )
        )
    if not legs:
        raise ValueError("At least one option leg is required")
    an = analyze_custom_legs(tk, spot, legs, days_elapsed=days_elapsed, iv_shock_pct=iv_shock_pct)
    return _clean_json({"ok": True, "analysis": asdict(an)})


def run_desk_paper_log_option(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Log a paper option or spread straight into `paper_trades.db` from Web Cockpit."""
    from src.tui.state import DeskDataProvider

    prov = DeskDataProvider()
    ok = prov.log_paper_option(
        ticker=str(payload["ticker"]).upper().strip(),
        strategy_name=str(payload["strategy_name"]),
        expiration=str(payload["expiration"])[:10],
        strike=float(payload["strike"]),
        opt_type=str(payload.get("opt_type") or "put").lower(),
        entry_price=float(payload["entry_price"]),
        quantity=float(payload.get("quantity") or 1.0),
        long_strike=float(payload["long_strike"]) if payload.get("long_strike") not in (None, "") else None,
        max_loss_usd=float(payload["max_loss_usd"]) if payload.get("max_loss_usd") not in (None, "") else None,
        entry_iv=float(payload["entry_iv"]) if payload.get("entry_iv") not in (None, "") else None,
        entry_delta=float(payload["entry_delta"]) if payload.get("entry_delta") not in (None, "") else None,
    )
    return _clean_json({"ok": bool(ok)})


def serve_web_terminal(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Zero-dependency stdlib ThreadingHTTPServer serving `/desk` and all live JSON endpoints."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = urllib.parse.parse_qs(parsed.query)
            if path in ("/", "/desk"):
                self._send(200, TERMINAL_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/health":
                self._send(200, b'{"status":"ok"}', "application/json")
                return
            if path == "/desk/api/overview":
                ref = qs.get("refresh", ["0"])[0] in ("1", "true", "yes")
                try:
                    data = build_desk_overview(refresh=ref)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path == "/desk/api/live-options":
                side = qs.get("side", ["bull_put"])[0]
                try:
                    data = build_desk_live_options(side=side)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path == "/desk/api/quotes":
                raw_t = qs.get("tickers", [""])[0]
                syms = [x.strip().upper() for x in raw_t.split(",") if x.strip()]
                try:
                    data = run_desk_live_quotes(syms)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path == "/desk/api/express":
                try:
                    data = build_desk_express(
                        ticker=qs.get("ticker", ["SPY"])[0],
                        direction=qs.get("dir", ["BULLISH"])[0],
                        target_pct=float(qs.get("target", ["8"])[0]),
                        days=int(qs.get("days", ["30"])[0]),
                        risk_usd=float(qs.get("risk", ["500"])[0]),
                        account_type=qs.get("account", ["tfsa"])[0],
                        owns_100=qs.get("owns", ["0"])[0] in ("1", "true", "yes"),
                        allow_live=qs.get("live", ["1"])[0] not in ("0", "false"),
                    )
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path == "/desk/api/options/scan":
                try:
                    data = build_desk_options_scan(
                        strategy=qs.get("strategy", ["all"])[0],
                        query=qs.get("q", [""])[0],
                        live=qs.get("live", ["0"])[0] in ("1", "true", "yes"),
                    )
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path == "/desk/api/options/chain":
                try:
                    data = build_desk_options_chain(
                        ticker=qs.get("ticker", ["SPY"])[0],
                        expiration=qs.get("exp", [""])[0] or None,
                        band_pct=float(qs.get("band", ["0.12"])[0]),
                        live=qs.get("live", ["0"])[0] in ("1", "true", "yes"),
                    )
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            if path.startswith("/desk/api/card/"):
                tk = urllib.parse.unquote(path.split("/desk/api/card/", 1)[1]).strip()
                try:
                    data = build_desk_card(tk)
                    self._send(200, json.dumps(data).encode("utf-8"), "application/json")
                except Exception as exc:
                    self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path == "/desk/api/update-prices":
                    data = run_desk_price_update()
                elif path == "/desk/api/alpha/refresh":
                    body = self._read_json()
                    data = run_desk_alpha_refresh(chain_limit=int(body.get("chain_limit", 15)))
                elif path == "/desk/api/log-pick":
                    data = run_desk_log_pick(self._read_json())
                elif path == "/desk/api/close-pick":
                    data = run_desk_close_pick(self._read_json())
                elif path == "/desk/api/import-ws-csv":
                    data = run_desk_import_ws_csv(self._read_json())
                elif path == "/desk/api/profile":
                    data = run_desk_update_profile(self._read_json())
                elif path == "/desk/api/rebalance-apply":
                    data = run_desk_apply_rebalance()
                elif path == "/desk/api/options/analyze":
                    data = run_desk_options_analyze(self._read_json())
                elif path == "/desk/api/options/paper-log":
                    data = run_desk_paper_log_option(self._read_json())
                else:
                    self._send(404, b'{"error":"not found"}', "application/json")
                    return
                self._send(200, json.dumps(data).encode("utf-8"), "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

    srv = ThreadingHTTPServer((host, int(port)), _Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


TERMINAL_HTML = """<!doctype html>
<html lang="en" data-theme="amber_crt">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OPTIONS DESK // LIVE COCKPIT</title>
<style>
:root, [data-theme="amber_crt"] {
  --bg: #110d07; --surface: #1b150c; --panel: #241b0f; --border: #4a371b;
  --ink: #f2d7a7; --ink-strong: #ffeed2; --muted: #a88d65;
  --primary: #ffb03b; --accent: #e69628;
  --good: #5ec98d; --warn: #d6a452; --bad: #e06c75; --info: #82aad2;
}
[data-theme="quant_desk"] {
  --bg: #0c1016; --surface: #131922; --panel: #1a222e; --border: #2a3545;
  --ink: #cbd5e1; --ink-strong: #f8fafc; --muted: #7c8b9e;
  --primary: #82aad2; --accent: #58a6ff;
  --good: #5ec98d; --warn: #d6a452; --bad: #e06c75; --info: #82aad2;
}
[data-theme="paper"] {
  --bg: #faf9f6; --surface: #f3f1ec; --panel: #ebe7de; --border: #d5cfc3;
  --ink: #23282e; --ink-strong: #111518; --muted: #787266;
  --primary: #9a6a15; --accent: #8a5a0e;
  --good: #1f6f43; --warn: #8a6516; --bad: #9d2c33; --info: #2d5f8b;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-size: 13px; line-height: 1.45;
}
.topbar {
  position: sticky; top: 0; z-index: 20;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  background: var(--panel); border-bottom: 1px solid var(--border);
  padding: 8px 16px;
}
.brand { color: var(--primary); font-weight: 800; letter-spacing: 0.08em; }
.pill {
  padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 11px;
  border: 1px solid var(--border); background: var(--surface);
}
.pill.good { color: var(--good); border-color: var(--good); }
.pill.bad  { color: var(--bad);  border-color: var(--bad); }
.pill.warn { color: var(--warn); border-color: var(--warn); }
.pill.info { color: var(--info); border-color: var(--info); }
.cmdbar { margin-left: auto; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.cmdbar input, .inp {
  background: var(--bg); color: var(--ink-strong); border: 1px solid var(--border);
  padding: 4px 8px; font-family: inherit; font-size: 12px; border-radius: 3px;
}
.cmdbar input { border-color: var(--primary); width: 200px; }
.btn {
  background: var(--surface); color: var(--primary); border: 1px solid var(--border);
  padding: 4px 9px; font-family: inherit; font-size: 11px; cursor: pointer; border-radius: 3px;
}
.btn:hover, .btn.active { background: var(--primary); color: var(--bg); font-weight: 700; }
.btn.good-btn { color: var(--good); border-color: var(--good); }
.btn.good-btn:hover { background: var(--good); color: var(--bg); }
.nav {
  display: flex; gap: 6px; padding: 8px 16px; background: var(--surface);
  border-bottom: 1px solid var(--border); flex-wrap: wrap; align-items: center;
}
.toast {
  margin-left: auto; font-size: 11px; color: var(--info); padding: 2px 8px;
}
.wrap { padding: 14px 16px; max-width: 1520px; margin: 0 auto; }
.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }
.c12 { grid-column: span 12; } .c8 { grid-column: span 8; } .c7 { grid-column: span 7; }
.c6  { grid-column: span 6; }  .c5 { grid-column: span 5; } .c4 { grid-column: span 4; }
@media (max-width: 960px) { .c8,.c7,.c6,.c5,.c4 { grid-column: span 12; } }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 12px 14px;
}
.card h3 {
  margin: 0 0 8px 0; font-size: 12px; letter-spacing: 0.08em;
  color: var(--primary); text-transform: uppercase;
  border-bottom: 1px solid var(--border); padding-bottom: 5px;
  display: flex; justify-content: space-between; align-items: center; gap: 8px;
}
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 5px 7px; border-bottom: 1px solid var(--border); text-align: left; font-variant-numeric: tabular-nums; }
th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; }
tr.clickable { cursor: pointer; }
tr.clickable:hover { background: var(--panel); }
.num { text-align: right; }
.good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); } .muted { color: var(--muted); }
.led-row { display: flex; flex-wrap: wrap; gap: 8px; }
.led { padding: 3px 8px; border-radius: 3px; border: 1px solid var(--border); background: var(--panel); font-size: 11px; }
.led .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
.dot.OK { background: var(--good); } .dot.WARN { background: var(--warn); } .dot.STALE { background: var(--bad); }
.tab { display: none; } .tab.active { display: block; }
svg { width: 100%; height: auto; display: block; }
.bar-wrap { height: 7px; background: var(--panel); border-radius: 3px; overflow: hidden; margin-top: 3px; }
.bar-fill { height: 100%; background: var(--primary); }
.mono-pre { white-space: pre-wrap; font-size: 12px; color: var(--ink); margin: 0; }
.dropzone {
  border: 1px dashed var(--primary); border-radius: 4px; padding: 12px;
  text-align: center; cursor: pointer; background: var(--panel); margin-top: 8px;
}
.dropzone.over { background: var(--surface); border-style: solid; }
.ritual-step { border-left: 3px solid var(--primary); padding: 6px 10px; margin: 8px 0; background: var(--panel); border-radius: 3px; }
.form-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
.modal-bg {
  position: fixed; inset: 0; background: rgba(0,0,0,0.65); z-index: 50;
  display: none; align-items: center; justify-content: center;
}
.modal-bg.open { display: flex; }
.modal-box { background: var(--surface); border: 1px solid var(--primary); border-radius: 5px; padding: 16px; width: 380px; max-width: 94vw; }
</style>
</head>
<body>
<div class="topbar">
  <span class="brand">◆ OPTIONS DESK</span>
  <span id="tb-spy" class="pill">SPY —</span>
  <span id="tb-vix" class="pill">VIX —</span>
  <span id="tb-regime" class="pill">—</span>
  <span id="tb-macro" class="muted">Next: —</span>
  <div class="cmdbar">
    <input id="cmd-in" placeholder="[:] NVDA · NVDA WB · SKEW · REB"/>
    <button class="btn" onclick="runCmd()">GO</button>
    <button class="btn good-btn" onclick="scanLiveEtfs('bull_put')">LIVE ETFs</button>
    <button class="btn" onclick="updatePricesLive()">UPD PRICES</button>
    <button class="btn" onclick="refreshAlphaLive()">UPD ALPHA</button>
    <button class="btn" id="btn-live-tape" onclick="toggleLiveTape()">LIVE TAPE: OFF</button>
    <button class="btn" onclick="cycleTheme()" title="Shortcut: t">THEME</button>
    <button class="btn" onclick="loadAll(true)">REFRESH</button>
  </div>
</div>

<div class="nav">
  <button class="btn" data-tab="ritual" onclick="showTab('ritual')">[0] 60s RITUAL</button>
  <button class="btn active" data-tab="today" onclick="showTab('today')">[1] TODAY &amp; TFSA</button>
  <button class="btn" data-tab="sectors" onclick="showTab('sectors')">[2] SECTORS &amp; RRG</button>
  <button class="btn" data-tab="alpha" onclick="showTab('alpha')">[3] ALPHA LABS</button>
  <button class="btn" data-tab="book" onclick="showTab('book')">[4] GHOST RACE &amp; BOOK</button>
  <button class="btn" data-tab="lab" onclick="showTab('lab')">[5] MISSION LAB</button>
  <button class="btn" data-tab="card" onclick="showTab('card')">[6] EVIDENCE CARD</button>
  <button class="btn" data-tab="sandbox" onclick="showTab('sandbox'); runSandbox()">[7] PAYOFF SANDBOX</button>
  <button class="btn" data-tab="options" onclick="showTab('options'); loadOptionsScan(false)">[8] OPTIONS HUB</button>
  <span id="status-toast" class="toast">Ready</span>
</div>

<div class="wrap">
  <!-- [0] 60-SECOND MORNING RITUAL -->
  <div id="tab-ritual" class="tab">
    <div class="card">
      <h3>60-Second Morning Ritual <span id="r-grade" class="pill good">—</span></h3>
      <div id="r-steps">Loading morning checklist…</div>
    </div>
  </div>

  <!-- [1] TODAY -->
  <div id="tab-today" class="tab active">
    <div class="grid">
      <div class="card c7">
        <h3>Morning Desk Dispatch <span id="t-asof" class="muted"></span></h3>
        <div id="t-plan">Loading…</div>
      </div>
      <div class="card c5">
        <h3>Core TFSA Sleeve (PROVEN · Wealthsimple USD) <span id="t-sgov" class="good"></span></h3>
        <div id="t-core"></div>
      </div>
      <div class="card c12">
        <h3>
          <span>Live 60-120d Wide ETF Spreads (Test O1 · Limit Only · Taxable)</span>
          <span>
            <button class="btn good-btn" onclick="scanLiveEtfs('bull_put')">Scan Bull Puts</button>
            <button class="btn" onclick="scanLiveEtfs('bear_call')">Scan Bear Calls</button>
          </span>
        </h3>
        <div id="t-live-meta" class="muted" style="margin-bottom:6px">Click "Scan Bull Puts" or "LIVE ETFs" above to pull fresh Yahoo option chains live.</div>
        <table id="t-live-etfs"><thead><tr><th>Ticker</th><th>Exp / DTE</th><th>Strikes</th><th class="num">Short Δ</th><th class="num">Mid</th><th class="num">Min Limit</th><th class="num">Spread%</th><th class="num">Max Risk</th><th>Flags</th></tr></thead><tbody><tr><td colspan="9" class="muted">Not scanned yet — click Scan Bull Puts.</td></tr></tbody></table>
        <div id="t-live-skipped" class="muted" style="margin-top:6px;font-size:11px"></div>
      </div>
      <div class="card c12">
        <h3>Unified Opportunity Radar <span class="muted">Click any ticker for Evidence Card</span></h3>
        <table id="t-radar"><thead><tr><th>Tier</th><th>Ticker</th><th>Strategy</th><th>Dir</th><th class="num">Match</th><th class="num">Cost</th><th>Flags</th><th>Setup</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [2] SECTORS RRG -->
  <div id="tab-sectors" class="tab">
    <div class="grid">
      <div class="card c6">
        <h3>4-Quadrant Sector Rotation Graph (RRG) <span class="muted">RS 3m vs Acceleration</span></h3>
        <div id="s-rrg"></div>
      </div>
      <div class="card c6">
        <h3>16-ETF Outlook Ranking (PROVEN Order)</h3>
        <table id="s-board"><thead><tr><th>#</th><th>ETF</th><th>Sector</th><th>Call</th><th class="num">Core%</th><th class="num">RS 1m</th><th class="num">RS 3m</th><th class="num">&gt;200d</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [3] ALPHA LABS -->
  <div id="tab-alpha" class="tab">
    <div class="grid">
      <div class="card c6">
        <h3>Option-Implied Signals → Trade SHARES <span class="pill warn">WATCHLIST</span></h3>
        <table id="a-skew"><thead><tr><th>Ticker</th><th class="num">Spot</th><th class="num">CW IV Spread</th><th class="num">Put Smirk</th><th class="num">Net GEX ($M)</th><th>Read</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c6">
        <h3>Macro Plumbing &amp; Alt-Data <span class="pill info">CONTEXT</span></h3>
        <div id="a-macro">Loading…</div>
      </div>
      <div class="card c6">
        <h3>OCC Market-Wide Option Volume / OI Surges <span class="pill warn">WATCHLIST</span></h3>
        <table id="a-occ"><thead><tr><th>Ticker</th><th class="num">Vol</th><th class="num">OI</th><th class="num">Vol/OI</th><th class="num">1d OI Δ%</th><th>Read</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c6">
        <h3>Taxable ETF Calendar Spreads (Front IV &gt; Back IV) <span class="pill warn">TAXABLE_ONLY</span></h3>
        <table id="a-cal"><thead><tr><th>ETF</th><th>Front/Back</th><th class="num">Strike</th><th class="num">IV Diff</th><th class="num">Mid Debit</th><th class="num">Fric%</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c12">
        <h3>SEC EDGAR Events (Form 10 Spinoffs · 13D Activists · 8-K Guidance · 10-K Supply Chain · Buybacks) <span class="pill warn">WATCHLIST</span></h3>
        <table id="a-edgar"><thead><tr><th>Type</th><th>Ticker</th><th>Filed</th><th>Headline / Metric</th><th>Source</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [4] GHOST RACE & BOOK -->
  <div id="tab-book" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>Ghost Benchmark Race (Normalized Base = 100) <span id="b-race-meta" class="muted"></span></h3>
        <div id="b-race-svg"></div>
      </div>
      <div class="card c7">
        <h3>Open &amp; Closed Stock/ETF Picks (data/own_picks.db)</h3>
        <div class="form-row">
          <input id="lg-tk" class="inp" style="width:76px" placeholder="Ticker"/>
          <input id="lg-px" class="inp" style="width:82px" placeholder="Price"/>
          <input id="lg-sh" class="inp" style="width:68px" value="10" placeholder="Shares"/>
          <input id="lg-st" class="inp" style="width:80px" placeholder="Stop"/>
          <input id="lg-tg" class="inp" style="width:80px" placeholder="Target"/>
          <select id="lg-pb" class="inp">
            <option value="equity_pullback">equity_pullback</option>
            <option value="vol_managed_outlook">vol_managed_outlook</option>
            <option value="discretionary">discretionary</option>
          </select>
          <button class="btn good-btn" onclick="submitLogPick()">+ Log Pick</button>
        </div>
        <table id="b-picks"><thead><tr><th>ID</th><th>Date</th><th>Ticker</th><th>Playbook</th><th class="num">Entry</th><th class="num">Now</th><th class="num">P&amp;L%</th><th>Status / Action</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card c5">
        <h3>CRA TFSA Safety, 15% Tax-Drag &amp; Wealthsimple Sync</h3>
        <div id="b-safety"></div>
        <div class="form-row" style="margin-top:6px">
          <input id="pf-tfsa" class="inp" style="width:82px" placeholder="TFSA $"/>
          <input id="pf-cash" class="inp" style="width:74px" placeholder="Cash ETF"/>
          <input id="pf-room" class="inp" style="width:86px" placeholder="Room CAD"/>
          <input id="pf-ytd" class="inp" style="width:82px" placeholder="YTD CAD"/>
          <button class="btn" onclick="saveProfileWeb()">Save Profile</button>
        </div>
        <div id="ws-drop" class="dropzone" onclick="document.getElementById('ws-file').click()">
          Drop Wealthsimple Activity CSV here (or click to browse) — idempotent zero-typing sync
          <input type="file" id="ws-file" accept=".csv" style="display:none" onchange="handleWsFile(this.files[0])"/>
        </div>
        <div id="b-rebal" style="margin-top:10px"></div>
      </div>
    </div>
  </div>

  <!-- [5] MISSION LAB -->
  <div id="tab-lab" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>Data Pipeline Heartbeat LEDs</h3>
        <div id="l-leds" class="led-row"></div>
      </div>
      <div class="card c6">
        <h3>10-Second Post-Mortem Mistake Cost Analytics</h3>
        <table id="l-mistakes"><thead><tr><th>Tag</th><th class="num">Trades</th><th class="num">Win%</th><th class="num">Avg Ret</th><th class="num">Net $</th></tr></thead><tbody></tbody></table>
        <h3 style="margin-top:14px">Preregistered Experiment Clocks</h3>
        <div id="l-clocks"></div>
      </div>
      <div class="card c6">
        <h3>Playbook Evidence Registry (configs/playbooks.json)</h3>
        <table id="l-playbooks"><thead><tr><th>Tier</th><th>Playbook</th><th>Accounts</th><th>Horizon</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- [6] EVIDENCE CARD -->
  <div id="tab-card" class="tab">
    <div class="card">
      <h3 id="c-title">Evidence Card <span class="muted">Type any ticker in the command bar above</span></h3>
      <div id="c-body" class="muted">Select a ticker from Radar, Sectors, or type NVDA in the top command bar.</div>
    </div>
  </div>

  <!-- [7] INTERACTIVE PAYOFF + SVI SANDBOX -->
  <div id="tab-sandbox" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>Interactive Payoff &amp; SVI Probability Sandbox <span id="sb-meta" class="muted"></span></h3>
        <div class="form-row">
          <label>Ticker <input id="sb-tk" class="inp" style="width:76px" value="SPY"/></label>
          <label>Dir <select id="sb-dir" class="inp"><option value="BULLISH">BULLISH</option><option value="BEARISH">BEARISH</option><option value="NEUTRAL">NEUTRAL</option></select></label>
          <label>Target <input id="sb-tgt" type="range" min="-25" max="25" value="8" oninput="document.getElementById('sb-tgt-v').textContent=(this.value>=0?'+':'')+this.value+'%'" onchange="runSandbox()"/><span id="sb-tgt-v">+8%</span></label>
          <label>Days <input id="sb-days" type="range" min="7" max="180" value="30" oninput="document.getElementById('sb-days-v').textContent=this.value+'d'" onchange="runSandbox()"/><span id="sb-days-v">30d</span></label>
          <label>Risk $ <input id="sb-risk" class="inp" style="width:74px" value="500"/></label>
          <label>Acct <select id="sb-acct" class="inp" onchange="runSandbox()"><option value="tfsa">TFSA</option><option value="taxable">Taxable</option></select></label>
          <label><input id="sb-owns" type="checkbox" onchange="runSandbox()"/> Own 100sh</label>
          <button class="btn good-btn" onclick="runSandbox()">Run Sandbox</button>
        </div>
        <div id="sb-verdict" style="margin:6px 0;font-weight:700"></div>
      </div>
      <div class="card c6">
        <h3>SVI / Market Probability Curve &amp; Share Payoff</h3>
        <div id="sb-svg"></div>
      </div>
      <div class="card c6">
        <h3>Vehicle Comparison &amp; Walk-Away Ticket <button class="btn" onclick="copyTicket()">Copy Ticket</button></h3>
        <table id="sb-table"><thead><tr><th>Vehicle</th><th>TFSA</th><th class="num">Risk</th><th class="num">Fric%</th><th class="num">Your Edge</th><th class="num">Win%</th></tr></thead><tbody></tbody></table>
        <pre id="sb-ticket" class="mono-pre" style="margin-top:8px;padding:8px;background:var(--panel);border-radius:3px"></pre>
      </div>
    </div>
  </div>

  <!-- [8] OPTIONS HUB (Multi-Strategy Scanner + T-Shaped Chain Matrix + 2D Risk Surface) -->
  <div id="tab-options" class="tab">
    <div class="grid">
      <div class="card c12">
        <h3>
          <span>Multi-Strategy Options Scanner &amp; T-Shaped Chain Matrix</span>
          <span>
            <button class="btn active" id="oh-btn-scan" onclick="showOptSub('scan')">Scanner</button>
            <button class="btn" id="oh-btn-chain" onclick="showOptSub('chain')">Chain Matrix</button>
          </span>
        </h3>
        <div id="oh-pane-scan">
          <div class="form-row">
            <select id="oh-strat" class="inp" onchange="loadOptionsScan(false)">
              <option value="all">All Strategies</option>
              <option value="bull_put">Bull Put [PROMISING · TAX]</option>
              <option value="long_call">Long Call [TFSA OK]</option>
              <option value="long_put">Long Put [TFSA OK]</option>
              <option value="calendar">Calendar [WATCHLIST · TAX]</option>
              <option value="bear_call">Bear Call [CAUTION · TAX]</option>
              <option value="iron_condor">Iron Condor [CAUTION · TAX]</option>
              <option value="short_put">Short Put [TAXABLE]</option>
            </select>
            <input id="oh-q" class="inp" style="width:240px" placeholder="Filter: bull_put tfsa clean fric<12% dte>=30" oninput="renderOptScanRows()"/>
            <button class="btn" onclick="document.getElementById('oh-q').value='bull_put clean';renderOptScanRows()">BullPut Clean</button>
            <button class="btn" onclick="document.getElementById('oh-q').value='tfsa';renderOptScanRows()">TFSA OK</button>
            <button class="btn" onclick="document.getElementById('oh-q').value='fric<12%';renderOptScanRows()">Fric&lt;12%</button>
            <button class="btn" onclick="loadOptionsScan(false)">Archive Scan</button>
            <button class="btn good-btn" onclick="loadOptionsScan(true)">Live Yahoo Scan</button>
            <span id="oh-scan-note" class="muted"></span>
          </div>
          <div class="grid">
            <div class="c7" style="overflow-x:auto">
              <table id="oh-scan-tbl"><thead><tr><th>Badge</th><th>Ticker</th><th>Strategy</th><th>Exp</th><th>Strikes</th><th class="num">Mid</th><th class="num">Ref Price</th><th class="num">Fric%</th><th class="num">Risk$</th><th>TFSA</th></tr></thead><tbody></tbody></table>
            </div>
            <div class="c5">
              <div id="oh-scan-insp" class="card" style="background:var(--panel)"><span class="muted">Click any candidate row to inspect its 2D Price×IV Risk Heatmap and Greeks, or jump to its Chain Matrix.</span></div>
            </div>
          </div>
        </div>

        <div id="oh-pane-chain" style="display:none">
          <div class="form-row">
            <input id="oh-ch-tk" class="inp" style="width:82px" value="SPY" placeholder="Ticker"/>
            <select id="oh-ch-exp" class="inp" onchange="loadWebChain(false)"><option value="">Default (~35d)</option></select>
            <select id="oh-ch-band" class="inp" onchange="loadWebChain(false)">
              <option value="0.12">±12%</option><option value="0.20">±20%</option><option value="0.0">All</option>
            </select>
            <button class="btn" onclick="loadWebChain(false)">Load Archive</button>
            <button class="btn good-btn" onclick="loadWebChain(true)">Live Yahoo</button>
            <button class="btn" onclick="clearWebLegs()">Clear Legs</button>
            <span id="oh-ch-meta" class="muted"></span>
          </div>
          <div class="grid">
            <div class="c7" style="max-height:520px;overflow-y:auto">
              <table id="oh-ch-tbl"><thead><tr><th>Stage C</th><th class="num">C.OI</th><th class="num">C.IV</th><th class="num">C.Δ</th><th class="num">C.Bid/Ask</th><th style="text-align:center">STRIKE</th><th class="num">P.Bid/Ask</th><th class="num">P.Δ</th><th class="num">P.IV</th><th class="num">P.OI</th><th>Stage P</th></tr></thead><tbody></tbody></table>
            </div>
            <div class="c5">
              <div class="form-row">
                <label>T+<span id="oh-t-v">0</span>d <input id="oh-t-sl" type="range" min="0" max="60" value="0" oninput="document.getElementById('oh-t-v').textContent=this.value" onchange="analyzeWebLegs()"/></label>
                <label>IV <span id="oh-iv-v">0</span>% <input id="oh-iv-sl" type="range" min="-30" max="30" value="0" oninput="document.getElementById('oh-iv-v').textContent=(this.value>=0?'+':'')+this.value" onchange="analyzeWebLegs()"/></label>
              </div>
              <div id="oh-ch-insp" class="card" style="background:var(--panel)"><span class="muted">Load a ticker to view IV Smile, Dealer GEX walls, or click +B/+S on strikes to build a live multi-leg payoff &amp; 2D Risk Surface.</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Post-Mortem Close Pick Modal -->
<div id="close-modal" class="modal-bg">
  <div class="modal-box">
    <h3 style="margin-top:0;color:var(--primary)">Close Pick &amp; 10-Second Post-Mortem</h3>
    <div id="cm-label" style="margin-bottom:8px"></div>
    <div class="form-row"><label>Exit Price <input id="cm-px" class="inp" style="width:110px"/></label></div>
    <div class="form-row">
      <label>Post-Mortem Tag
        <select id="cm-tag" class="inp">
          <option value="PLAN_WORKED">1. Plan Worked</option>
          <option value="BAD_FILL">2. Bad Fill / Slippage</option>
          <option value="THESIS_WRONG">3. Thesis Wrong</option>
          <option value="REGIME_SHIFT">4. Regime Shift</option>
          <option value="BROKE_RULES">5. Broke My Rules</option>
          <option value="TIME_EXIT">6. Time Exit</option>
        </select>
      </label>
    </div>
    <div class="form-row"><input id="cm-note" class="inp" style="width:100%" placeholder="Short note (optional)"/></div>
    <div class="form-row" style="justify-content:flex-end">
      <button class="btn" onclick="hideCloseModal()">Cancel</button>
      <button class="btn good-btn" onclick="confirmClosePick()">Close Pick</button>
    </div>
  </div>
</div>

<script>
const THEMES = ['amber_crt', 'quant_desk', 'paper'];
let STATE = null;
let LIVE_TAPE_TIMER = null;
let CLOSING_PICK_ID = null;
let LAST_TICKET_TEXT = '';

function setToast(msg) {
  const el = document.getElementById('status-toast');
  if (el) el.textContent = msg;
}
function cycleTheme() {
  const cur = document.documentElement.getAttribute('data-theme') || 'amber_crt';
  const nxt = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
  document.documentElement.setAttribute('data-theme', nxt);
  try { localStorage.setItem('desk-bloomberg-theme', nxt); } catch {}
}
(function initTheme() {
  try {
    const saved = localStorage.getItem('desk-bloomberg-theme');
    if (saved && THEMES.includes(saved)) document.documentElement.setAttribute('data-theme', saved);
  } catch {}
})();

function showTab(id) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav .btn').forEach(el => el.classList.toggle('active', el.dataset.tab === id));
  const pane = document.getElementById('tab-' + id);
  if (pane) pane.classList.add('active');
}

document.addEventListener('keydown', e => {
  if (document.activeElement && ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) {
    if (e.key === 'Enter' && document.activeElement.id === 'cmd-in') runCmd();
    if (e.key === 'Escape') document.activeElement.blur();
    return;
  }
  const map = {'0':'ritual','1':'today','2':'sectors','3':'alpha','4':'book','5':'lab','6':'card','7':'sandbox','8':'options'};
  if (map[e.key]) { showTab(map[e.key]); if (e.key === '7') runSandbox(); if (e.key === '8') loadOptionsScan(false); }
  else if (e.key === ':' || e.key === '/') { e.preventDefault(); document.getElementById('cmd-in').focus(); }
  else if (e.key === 't' || e.key === 'T') cycleTheme();
  else if (e.key === 'r') loadAll(true);
});

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

async function runCmd() {
  const raw = document.getElementById('cmd-in').value.trim();
  if (!raw) return;
  const parts = raw.split(/\\s+/);
  const cmd = parts[0].toUpperCase();
  const arg = (parts[1] || '').toUpperCase();
  if (['RITUAL','MORNING'].includes(cmd)) { showTab('ritual'); return; }
  if (['TODAY','SECTORS','ALPHA','BOOK','LAB'].includes(cmd)) { showTab(cmd.toLowerCase()); return; }
  if (['OPTIONS','OPT','SCAN'].includes(cmd)) {
    showTab('options'); showOptSub('scan');
    if (arg) document.getElementById('oh-strat').value = arg.toLowerCase();
    await loadOptionsScan(false); return;
  }
  if (cmd === 'CHAIN') {
    if (arg) document.getElementById('oh-ch-tk').value = arg;
    showTab('options'); showOptSub('chain'); await loadWebChain(false); return;
  }
  if (arg === 'CHAIN') {
    document.getElementById('oh-ch-tk').value = cmd;
    showTab('options'); showOptSub('chain'); await loadWebChain(false); return;
  }
  if (['SKEW','GEX','EDGAR','MACRO','OCC'].includes(cmd)) { showTab('alpha'); return; }
  if (['REB','REBAL'].includes(cmd)) { showTab('book'); return; }
  if (arg === 'WB' || arg === 'SANDBOX') {
    document.getElementById('sb-tk').value = cmd;
    showTab('sandbox');
    await runSandbox();
    return;
  }
  await openCard(cmd);
}

async function scanLiveEtfs(side = 'bull_put') {
  setToast('Scanning live 60-120d ETF spreads…');
  showTab('today');
  try {
    const resp = await fetch('/desk/api/live-options?side=' + encodeURIComponent(side));
    const d = await resp.json();
    const opps = d.opportunities || [];
    const skipped = d.skipped || {};
    const warns = d.warnings || [];
    document.getElementById('t-live-meta').textContent =
      `${opps.length} live ${side.replace('_',' ')} spread(s) passed <=25% cost · r = ${d.rate_source || 'cache'}` +
      (warns.length ? ' · warnings: ' + warns.slice(0,2).join('; ') : '');
    document.querySelector('#t-live-etfs tbody').innerHTML = opps.map(o => {
      const ex = o.extra || {};
      return `<tr class="clickable" onclick="document.getElementById('sb-tk').value='${esc(o.ticker)}';showTab('sandbox');runSandbox()">
        <td><b>${esc(o.ticker)}</b></td>
        <td>${esc(ex.expiration || '')} (${ex.dte || ''}d)</td>
        <td>${ex.short_strike ?? '—'} / ${ex.long_strike ?? '—'}</td>
        <td class="num">${ex.short_delta != null ? Number(ex.short_delta).toFixed(2) : '—'}</td>
        <td class="num good">$${Number(ex.credit_mid || 0).toFixed(2)}</td>
        <td class="num"><b>$${Number(ex.walk_away_limit || 0).toFixed(2)}</b></td>
        <td class="num">${esc(o.friction_label || '')}</td>
        <td class="num">$${Number(ex.max_risk_usd || 0).toLocaleString()}</td>
        <td class="warn">${esc((o.risk_flags || []).join(', '))}</td></tr>`;
    }).join('') || '<tr><td colspan="9" class="muted">0 live ETF spreads cleared <=25% spread cost right now.</td></tr>';
    const skipPairs = Object.entries(skipped).map(([k,v]) => `<b>${esc(k)}</b>: ${esc(v)}`);
    document.getElementById('t-live-skipped').innerHTML = skipPairs.length ? 'Not shown: ' + skipPairs.join(' · ') : '';
    setToast(`Live ETFs: ${opps.length} found`);
  } catch (err) {
    setToast('Live ETF error: ' + err);
  }
}

async function updatePricesLive() {
  setToast('Updating daily OHLCV prices & FRED…');
  try {
    const resp = await fetch('/desk/api/update-prices', { method: 'POST' });
    const d = await resp.json();
    const r = d.result || {};
    setToast(`Prices updated (${r.updated || 0} new / ${r.symbols || 0} syms)`);
    await loadAll(true);
  } catch (err) {
    setToast('Price update failed: ' + err);
  }
}

async function refreshAlphaLive() {
  setToast('Refreshing Macro VIX, SEC EDGAR, Chains & OCC OI…');
  try {
    const resp = await fetch('/desk/api/alpha/refresh', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({chain_limit: 15})
    });
    const d = await resp.json();
    setToast(`Alpha refreshed: chains ${d.chains?.archived ?? 0}, EDGAR ${d.edgar?.saved ?? 0}, OCC ${d.occ?.saved ?? 0}`);
    await loadAll(true);
  } catch (err) {
    setToast('Alpha refresh failed: ' + err);
  }
}

function toggleLiveTape() {
  const btn = document.getElementById('btn-live-tape');
  if (LIVE_TAPE_TIMER) {
    clearInterval(LIVE_TAPE_TIMER);
    LIVE_TAPE_TIMER = null;
    btn.textContent = 'LIVE TAPE: OFF';
    btn.classList.remove('active');
    setToast('Live tape stopped');
    return;
  }
  btn.textContent = 'LIVE TAPE: ON';
  btn.classList.add('active');
  pollLiveQuotes();
  LIVE_TAPE_TIMER = setInterval(pollLiveQuotes, 60000);
}

async function pollLiveQuotes() {
  const picks = (STATE?.book?.equity_picks || []).filter(p => p.status === 'OPEN').map(p => p.ticker);
  const syms = Array.from(new Set(['SPY', 'QQQ', ...picks]));
  try {
    const resp = await fetch('/desk/api/quotes?tickers=' + encodeURIComponent(syms.join(',')));
    const d = await resp.json();
    const q = d.quotes || {};
    if (q.SPY) {
      const chg = q.SPY.chg_1d_pct != null ? ` (${q.SPY.chg_1d_pct>=0?'+':''}${q.SPY.chg_1d_pct.toFixed(1)}%)` : '';
      document.getElementById('tb-spy').textContent = `SPY $${q.SPY.price.toFixed(2)}${chg}`;
    }
    setToast('Live quotes polled @ ' + new Date().toLocaleTimeString());
  } catch {}
}

async function runSandbox() {
  const tk = document.getElementById('sb-tk').value.trim().toUpperCase() || 'SPY';
  const dir = document.getElementById('sb-dir').value;
  const tgt = document.getElementById('sb-tgt').value;
  const days = document.getElementById('sb-days').value;
  const risk = document.getElementById('sb-risk').value || '500';
  const acct = document.getElementById('sb-acct').value;
  const owns = document.getElementById('sb-owns').checked ? '1' : '0';
  setToast(`Running sandbox for ${tk}…`);
  try {
    const url = `/desk/api/express?ticker=${encodeURIComponent(tk)}&dir=${dir}&target=${tgt}&days=${days}&risk=${risk}&account=${acct}&owns=${owns}`;
    const resp = await fetch(url);
    const d = await resp.json();
    document.getElementById('sb-meta').textContent = `${d.ticker} $${Number(d.spot||0).toFixed(2)} · IV ${d.iv_pct}% · RND: ${d.rnd_source}`;
    document.getElementById('sb-verdict').textContent = d.verdict || '';
    document.getElementById('sb-svg').innerHTML = renderSandboxSVG(d);
    const rows = d.vehicles || [];
    document.querySelector('#sb-table tbody').innerHTML = rows.map((v, i) => `<tr class="clickable" onclick="showTicket(${i})">
      <td>${v.recommended ? '★ ' : ''}<b>${esc(v.vehicle)}</b><div class="muted" style="font-size:11px">${esc(v.description)}</div></td>
      <td class="${v.tfsa_legal?'good':'warn'}">${v.tfsa_legal?'OK':'TAX'}</td>
      <td class="num">$${Number(v.capital_at_risk_usd||0).toLocaleString()}</td>
      <td class="num">${Number(v.friction_pct_of_reward||0).toFixed(1)}%</td>
      <td class="num ${v.ev_view_net_usd>=0?'good':'bad'}">$${Number(v.ev_view_net_usd||0).toFixed(0)}</td>
      <td class="num">${(Number(v.pop_view||0)*100).toFixed(0)}%</td></tr>`).join('');
    window._SB_DATA = d;
    window._SB_ROWS = rows;
    const recIdx = Math.max(0, rows.findIndex(v => v.recommended));
    showTicket(recIdx);
    setToast(`Sandbox ready: ${d.ticker}`);
  } catch (err) {
    setToast('Sandbox error: ' + err);
  }
}

function showTicket(idx) {
  const v = (window._SB_ROWS || [])[idx];
  if (!v) return;
  if (window._SB_DATA) {
    document.getElementById('sb-svg').innerHTML = renderSandboxSVG(window._SB_DATA, idx);
  }
  const lines = [
    `ORDER TICKET // ${v.vehicle} (${v.tfsa_legal ? 'TFSA LEGAL' : 'TAXABLE ONLY'})`,
    `Plan: ${v.description}`,
    `Max Risk: $${Number(v.capital_at_risk_usd||0).toLocaleString()} · Friction: ${Number(v.friction_pct_of_reward||0).toFixed(1)}%`,
    ...(v.walk_away_steps && v.walk_away_steps.length ? v.walk_away_steps : [v.walk_away_label || '']),
    `Delta $: $${Number(v.delta_dollars||0).toFixed(0)} · Theta/day: $${Number(v.theta_per_day_usd||0).toFixed(2)}`,
    ...(v.caution ? ['CAUTION: ' + v.caution] : []),
  ];
  LAST_TICKET_TEXT = lines.join('\\n');
  document.getElementById('sb-ticket').textContent = LAST_TICKET_TEXT;
}

function copyTicket() {
  if (!LAST_TICKET_TEXT) return;
  navigator.clipboard?.writeText(LAST_TICKET_TEXT).then(() => setToast('Order ticket copied!')).catch(() => setToast('Copy blocked by browser'));
}

function renderSandboxSVG(d, vehIdx = 0) {
  const c = d.curve || {};
  const px = c.prices || [], pdf = c.pdf_norm || [];
  const vpAll = c.vehicle_payoffs || [];
  const pnl = (vpAll[vehIdx] && vpAll[vehIdx].length === px.length) ? vpAll[vehIdx] : (c.share_pnl || []);
  const vehName = ((d.vehicles || [])[vehIdx] || {}).vehicle || 'Shares';
  if (px.length < 2) return '<div class="muted">No curve data.</div>';
  const W = 540, H = 220, pad = 34;
  const loX = px[0], hiX = px[px.length - 1];
  const minP = Math.min(-10, ...pnl), maxP = Math.max(10, ...pnl);
  const sx = x => pad + ((x - loX) / Math.max(hiX - loX, 1e-6)) * (W - 2 * pad);
  const syPdf = p => H - pad - p * (H - 2 * pad) * 0.85;
  const syPnl = v => H - pad - ((v - minP) / Math.max(maxP - minP, 1e-6)) * (H - 2 * pad);
  const areaPts = `${sx(loX).toFixed(1)},${H-pad} ` + px.map((x,i) => `${sx(x).toFixed(1)},${syPdf(pdf[i]).toFixed(1)}`).join(' ') + ` ${sx(hiX).toFixed(1)},${H-pad}`;
  const pnlPts = px.map((x,i) => `${sx(x).toFixed(1)},${syPnl(pnl[i]).toFixed(1)}`).join(' ');
  const xSpot = sx(d.spot).toFixed(1);
  const xTgt = sx(c.target_price || d.spot).toFixed(1);
  const y0 = syPnl(0).toFixed(1);
  return `<svg viewBox="0 0 ${W} ${H}">
    <polygon points="${areaPts}" fill="var(--info)" opacity="0.20"/>
    <line x1="${pad}" y1="${y0}" x2="${W-pad}" y2="${y0}" stroke="var(--border)" stroke-dasharray="4,3"/>
    <line x1="${xSpot}" y1="${pad}" x2="${xSpot}" y2="${H-pad}" stroke="var(--muted)" stroke-dasharray="3,3"/>
    <line x1="${xTgt}" y1="${pad}" x2="${xTgt}" y2="${H-pad}" stroke="var(--good)" stroke-width="1.6"/>
    <polyline points="${pnlPts}" fill="none" stroke="var(--primary)" stroke-width="2.4"/>
    <text x="${xSpot}" y="${pad-8}" text-anchor="middle" fill="var(--muted)" font-size="10">Spot $${d.spot}</text>
    <text x="${xTgt}" y="${H-10}" text-anchor="middle" fill="var(--good)" font-size="10">Target $${c.target_price}</text>
    <text x="${pad}" y="14" fill="var(--info)" font-size="10">■ Market Density (${esc(d.rnd_source)})</text>
    <text x="${pad+185}" y="14" fill="var(--primary)" font-size="10">━ ${esc(vehName)} P&amp;L ($${Math.round(minP)} to $${Math.round(maxP)})</text>
  </svg>`;
}

async function saveProfileWeb() {
  const body = {
    tfsa_usd: document.getElementById('pf-tfsa').value,
    cash_etf: document.getElementById('pf-cash').value,
    tfsa_room_cad: document.getElementById('pf-room').value,
    tfsa_contributed_ytd_cad: document.getElementById('pf-ytd').value,
  };
  const resp = await fetch('/desk/api/profile', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const d = await resp.json();
  if (d.ok) { setToast('Saved TFSA profile!'); await loadAll(false); }
  else setToast('Profile error: ' + (d.error || ''));
}

async function applyRebalanceWeb() {
  const resp = await fetch('/desk/api/rebalance-apply', { method: 'POST' });
  const d = await resp.json();
  if (d.ok) { setToast(`Rebalance saved (${d.last_rebalance_date})`); await loadAll(false); }
  else setToast('Rebalance save error: ' + (d.error || ''));
}

async function submitLogPick() {
  const body = {
    ticker: document.getElementById('lg-tk').value.trim().toUpperCase(),
    price: Number(document.getElementById('lg-px').value),
    shares: Number(document.getElementById('lg-sh').value || 1),
    stop_price: document.getElementById('lg-st').value || null,
    target_price: document.getElementById('lg-tg').value || null,
    playbook_id: document.getElementById('lg-pb').value,
  };
  if (!body.ticker || !(body.price > 0)) { setToast('Enter valid ticker & price'); return; }
  const resp = await fetch('/desk/api/log-pick', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const d = await resp.json();
  if (d.ok) {
    setToast(`Logged pick #${d.pick_id} (${body.ticker})`);
    document.getElementById('lg-tk').value = '';
    document.getElementById('lg-px').value = '';
    await loadAll(false);
  } else setToast('Log error: ' + (d.error || 'failed'));
}

function openCloseModal(pid, ticker, curPx) {
  CLOSING_PICK_ID = pid;
  document.getElementById('cm-label').textContent = `Closing Pick #${pid} — ${ticker}`;
  document.getElementById('cm-px').value = curPx != null ? Number(curPx).toFixed(2) : '';
  document.getElementById('cm-note').value = '';
  document.getElementById('close-modal').classList.add('open');
}
function hideCloseModal() { document.getElementById('close-modal').classList.remove('open'); }
async function confirmClosePick() {
  const px = Number(document.getElementById('cm-px').value);
  if (!(px > 0) || !CLOSING_PICK_ID) { setToast('Enter valid exit price'); return; }
  const body = {
    pick_id: CLOSING_PICK_ID,
    exit_price: px,
    close_tag: document.getElementById('cm-tag').value,
    notes: document.getElementById('cm-note').value.trim(),
  };
  const resp = await fetch('/desk/api/close-pick', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const d = await resp.json();
  hideCloseModal();
  if (d.ok) { setToast(`Closed #${d.pick_id} [${d.close_tag}]`); await loadAll(false); }
  else setToast('Close failed: ' + (d.error || ''));
}

async function handleWsFile(file) {
  if (!file) return;
  const text = await file.text();
  setToast(`Importing ${file.name}…`);
  const resp = await fetch('/desk/api/import-ws-csv', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ filename: file.name, csv_text: text }),
  });
  const d = await resp.json();
  if (d.ok) {
    setToast(`Imported ${file.name}: ${d.core_updates} core, ${d.picks_logged} picks (${d.skipped_duplicates} dupes skipped)`);
    await loadAll(false);
  } else setToast('CSV import error: ' + (d.error || ''));
}
(function wireDrop() {
  const dz = document.getElementById('ws-drop');
  if (!dz) return;
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleWsFile(e.dataTransfer.files[0]);
  });
})();

async function openCard(ticker) {
  showTab('card');
  document.getElementById('c-title').innerHTML = `EVIDENCE CARD // ${esc(ticker)}`;
  document.getElementById('c-body').innerHTML = 'Loading evidence card…';
  try {
    const resp = await fetch('/desk/api/card/' + encodeURIComponent(ticker));
    const d = await resp.json();
    const dos = d.dossier || {};
    const prem = d.premortem || {};
    const risks = (prem.kill_switches || []).map(x => `<li class="warn">${esc(x)}</li>`).join('');
    const flags = (dos.risk_flags || []).map(f => `<span class="pill bad">${esc(f)}</span>`).join(' ');
    document.getElementById('c-body').innerHTML = `
      <div class="grid">
        <div class="card c6">
          <h3>Verdict &amp; Structure <button class="btn" onclick="document.getElementById('sb-tk').value='${esc(ticker)}';showTab('sandbox');runSandbox()">Open in Sandbox</button></h3>
          <p><span class="pill ${d.verdict_style === 'green' ? 'good' : d.verdict_style === 'red' ? 'bad' : 'warn'}">${esc(d.verdict)}</span>
          <b>${esc(ticker)}</b> $${Number(dos.spot || 0).toFixed(2)} ${flags}</p>
          <ul>${(d.reasons || []).map(r => `<li>${esc(r)}</li>`).join('')}${(d.warnings || []).map(w => `<li class="warn">⚠ ${esc(w)}</li>`).join('')}</ul>
          <p class="muted">Trend vs 200d: ${dos.dist_sma200_pct ?? '—'}% · 12-1m: ${dos.mom_12_1_pct ?? '—'}% · RSI(2): ${dos.rsi_2 ?? '—'} · ATM IV: ${dos.atm_iv ?? '—'}%</p>
        </div>
        <div class="card c6">
          <h3>Pre-Mortem Devil's Advocate</h3>
          <p class="muted">${esc(prem.summary || 'Deterministic pre-trade risk check.')}</p>
          <ul>${risks || '<li class="good">No immediate deterministic red flags.</li>'}</ul>
        </div>
      </div>`;
  } catch (err) {
    document.getElementById('c-body').innerHTML = `<span class="bad">Failed to load card: ${esc(err)}</span>`;
  }
}

function renderRRG(rows) {
  const pts = (rows || []).filter(r => r.rs_63d_pct != null && r.rs_21d_pct != null).map(r => ({
    etf: r.etf, x: Number(r.rs_63d_pct), y: Number(r.rs_21d_pct) - Number(r.rs_63d_pct) / 3.0,
    core: Number(r.weight || 0) > 0, avoid: r.posture === 'AVOID'
  }));
  if (!pts.length) return '<div class="muted">No sector RS data.</div>';
  const W = 540, H = 310, pad = 34;
  const maxX = Math.max(5, ...pts.map(p => Math.abs(p.x))) * 1.15;
  const maxY = Math.max(3, ...pts.map(p => Math.abs(p.y))) * 1.15;
  const sx = x => pad + ((x + maxX) / (2 * maxX)) * (W - 2 * pad);
  const sy = y => pad + ((maxY - y) / (2 * maxY)) * (H - 2 * pad);
  const mx = sx(0), my = sy(0);
  const dots = pts.map(p => {
    const cx = sx(p.x).toFixed(1), cy = sy(p.y).toFixed(1);
    const col = p.core ? 'var(--good)' : (p.avoid ? 'var(--bad)' : 'var(--primary)');
    const r = p.core ? 6 : 4;
    return `<g style="cursor:pointer" onclick="openCard('${esc(p.etf)}')">
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" stroke="var(--bg)" stroke-width="1.5"/>
      <text x="${Number(cx)+8}" y="${Number(cy)+4}" fill="${col}" font-size="10" font-weight="${p.core?800:500}">${p.core?'★':''}${esc(p.etf)}</text>
    </g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}">
    <rect x="${mx}" y="${pad}" width="${W-pad-mx}" height="${my-pad}" fill="var(--good)" opacity="0.06"/>
    <rect x="${pad}" y="${my}" width="${mx-pad}" height="${H-pad-my}" fill="var(--bad)" opacity="0.06"/>
    <line x1="${pad}" y1="${my}" x2="${W-pad}" y2="${my}" stroke="var(--border)" stroke-dasharray="4,3"/>
    <line x1="${mx}" y1="${pad}" x2="${mx}" y2="${H-pad}" stroke="var(--border)" stroke-dasharray="4,3"/>
    <text x="${W-pad-4}" y="${pad+14}" text-anchor="end" fill="var(--good)" font-size="10">LEADING ↗</text>
    <text x="${pad+4}" y="${pad+14}" fill="var(--warn)" font-size="10">↖ IMPROVING</text>
    <text x="${pad+4}" y="${H-pad-6}" fill="var(--bad)" font-size="10">↙ LAGGING</text>
    <text x="${W-pad-4}" y="${H-pad-6}" text-anchor="end" fill="var(--muted)" font-size="10">WEAKENING ↘</text>
    ${dots}
  </svg>`;
}

function renderGhostSVG(race) {
  const spy = race?.spy || [], vm = race?.vm_outlook_sgov || [], real = race?.real_book || [];
  if (spy.length < 2) return '<div class="muted">No OHLCV history available for Ghost Race.</div>';
  const W = 920, H = 210, pad = 34;
  const all = [...spy, ...vm, ...real];
  const lo = Math.min(...all) - 1, hi = Math.max(...all) + 1;
  const sx = i => pad + (i / Math.max(spy.length - 1, 1)) * (W - 2 * pad);
  const sy = v => pad + ((hi - v) / Math.max(hi - lo, 1e-6)) * (H - 2 * pad);
  const poly = arr => arr.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
  const y100 = sy(100).toFixed(1);
  return `<svg viewBox="0 0 ${W} ${H}">
    <line x1="${pad}" y1="${y100}" x2="${W-pad}" y2="${y100}" stroke="var(--border)" stroke-dasharray="4,4"/>
    <polyline points="${poly(spy)}" fill="none" stroke="var(--muted)" stroke-width="1.6" stroke-dasharray="4,3"/>
    <polyline points="${poly(vm)}" fill="none" stroke="var(--primary)" stroke-width="2.4"/>
    ${real.length >= 2 ? `<polyline points="${poly(real)}" fill="none" stroke="var(--good)" stroke-width="2.2"/>` : ''}
    <text x="${pad}" y="18" fill="var(--primary)" font-size="11" font-weight="700">━ VM_OUTLOOK+SGOV ${race.vm_ret_pct >= 0 ? '+' : ''}${race.vm_ret_pct}% (+${race.vm_cash_only_boost_pct}% T-bill)</text>
    <text x="${pad+310}" y="18" fill="var(--muted)" font-size="11">┄ SPY ${race.spy_ret_pct >= 0 ? '+' : ''}${race.spy_ret_pct}%</text>
    <text x="${pad+460}" y="18" fill="var(--good)" font-size="11">━ Real Book ${race.real_ret_pct == null ? '— (log picks to race)' : (race.real_ret_pct >= 0 ? '+' : '') + race.real_ret_pct + '%'}</text>
  </svg>`;
}

async function loadAll(refresh = false) {
  const url = '/desk/api/overview' + (refresh ? '?refresh=1' : '');
  const resp = await fetch(url);
  STATE = await resp.json();
  const s = STATE.snapshot || {};
  document.getElementById('tb-spy').textContent = `SPY $${Number(s.spy_last||0).toFixed(2)} (${s.spy_chg_1d_pct>=0?'+':''}${Number(s.spy_chg_1d_pct||0).toFixed(1)}%)`;
  document.getElementById('tb-vix').textContent = `VIX ${s.vix!=null?Number(s.vix).toFixed(1):'—'} ${s.vix_term||''}`;
  const reg = document.getElementById('tb-regime');
  reg.textContent = s.regime_posture || '—';
  reg.className = 'pill ' + (s.regime_posture === 'RISK_ON' ? 'good' : s.regime_posture === 'RISK_OFF' ? 'bad' : 'warn');
  document.getElementById('tb-macro').textContent = 'Next: ' + (s.next_macro_event || '—');
  document.getElementById('t-asof').textContent = (s.generated_at || '').slice(0, 16);

  // [0] Morning Ritual
  const mr = STATE.morning_ritual || {};
  document.getElementById('r-grade').textContent = mr.status_grade || 'READY';
  document.getElementById('r-steps').innerHTML = (mr.steps || []).map(st =>
    `<div class="ritual-step"><b>${st.step}. ${esc(st.title)}</b> <span class="pill info">${esc(st.status)}</span><div>${esc(st.headline)}</div><div class="muted">${esc(st.detail)}</div></div>`
  ).join('');

  // [1] Today plan & Core
  document.getElementById('t-plan').innerHTML = (STATE.action_plan || []).map(x => `<p style="margin:6px 0">${esc(x)}</p>`).join('');
  document.getElementById('t-sgov').textContent = `${STATE.profile?.cash_etf || 'SGOV'} @ ${STATE.cash_yield_label || ''}`;
  document.getElementById('t-core').innerHTML = (s.core_allocation || []).map(c => {
    const pct = (Number(c.target_weight||0)*100).toFixed(1);
    const tk = c.ticker === 'CASH' ? (STATE.profile?.cash_etf || 'SGOV') + ' (Cash)' : c.ticker;
    return `<div style="margin-bottom:8px"><b>${esc(tk)}</b> <span style="float:right">${pct}%</span><div class="bar-wrap"><div class="bar-fill" style="width:${pct}%"></div></div></div>`;
  }).join('');

  document.querySelector('#t-radar tbody').innerHTML = (s.opportunities || []).map(o => `<tr class="clickable" onclick="openCard('${esc(o.ticker)}')">
    <td><span class="pill ${o.evidence_tier==='VALIDATED'?'good':o.evidence_tier==='LEAD'?'info':'warn'}">${esc(o.evidence_tier)}</span></td>
    <td><b>${esc(o.ticker)}</b></td><td>${esc(o.playbook_label)}</td>
    <td class="${o.direction==='LONG'?'good':o.direction==='SHORT'?'bad':'muted'}">${esc(o.direction)}</td>
    <td class="num">${o.confluence_count}</td><td class="num">${esc(o.friction_label)}</td>
    <td class="warn">${esc((o.risk_flags||[]).join(', ')||'—')}</td><td>${esc(o.setup_summary)}</td></tr>`).join('');

  // [2] Sectors + RRG
  const rows = STATE.sector_board || [];
  document.getElementById('s-rrg').innerHTML = renderRRG(rows);
  document.querySelector('#s-board tbody').innerHTML = rows.map(r => `<tr class="clickable" onclick="openCard('${esc(r.etf)}')">
    <td>${r.rank}</td><td><b>${esc(r.etf)}</b></td><td>${esc(r.name)}</td>
    <td class="${r.posture==='FAVOR'?'good':r.posture==='AVOID'?'bad':'muted'}">${esc(r.posture)}</td>
    <td class="num">${(Number(r.weight||0)*100).toFixed(0)}%</td>
    <td class="num ${r.rs_21d_pct>=0?'good':'bad'}">${r.rs_21d_pct!=null?Number(r.rs_21d_pct).toFixed(1)+'%':'—'}</td>
    <td class="num ${r.rs_63d_pct>=0?'good':'bad'}">${r.rs_63d_pct!=null?Number(r.rs_63d_pct).toFixed(1)+'%':'—'}</td>
    <td class="num">${r.pct_above_200d!=null?Number(r.pct_above_200d).toFixed(0)+'%':'—'}</td></tr>`).join('');

  // [3] Alpha Labs
  const alpha = STATE.alpha || {};
  document.querySelector('#a-skew tbody').innerHTML = (alpha.skew_signals || []).slice(0, 12).map(x => `<tr class="clickable" onclick="openCard('${esc(x.ticker)}')">
    <td><b>${esc(x.ticker)}</b></td><td class="num">$${Number(x.spot||0).toFixed(2)}</td>
    <td class="num ${x.cw_iv_spread_vp>=0?'good':'bad'}">${x.cw_iv_spread_vp!=null?x.cw_iv_spread_vp.toFixed(2)+'vp':'—'}</td>
    <td class="num">${x.put_smirk_vp!=null?x.put_smirk_vp.toFixed(2)+'vp':'—'}</td>
    <td class="num ${x.net_gex_m>=0?'good':'bad'}">${x.net_gex_m!=null?x.net_gex_m.toFixed(1):'—'}</td>
    <td>${esc(x.signal_label)}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">Click UPD ALPHA above to archive live option chains.</td></tr>';
  document.getElementById('a-macro').innerHTML = `<pre class="mono-pre">${esc(alpha.macro_text || 'Click UPD ALPHA to populate macro plumbing.')}</pre>`;
  document.querySelector('#a-occ tbody').innerHTML = (alpha.occ_uoa || []).slice(0, 12).map(u => `<tr class="clickable" onclick="openCard('${esc(u.symbol)}')">
    <td><b>${esc(u.symbol)}</b></td><td class="num">${Number(u.total_vol||0).toLocaleString()}</td>
    <td class="num">${Number(u.total_oi||0).toLocaleString()}</td>
    <td class="num">${u.vol_to_oi!=null?u.vol_to_oi.toFixed(2)+'x':'—'}</td>
    <td class="num">${u.oi_delta_pct!=null?(u.oi_delta_pct>=0?'+':'')+u.oi_delta_pct.toFixed(1)+'%':'—'}</td>
    <td>${esc(u.signal_label)}</td></tr>`).join('') || '<tr><td colspan="6" class="muted">Click UPD ALPHA or run `opt alpha archive-occ`.</td></tr>';
  document.querySelector('#a-cal tbody').innerHTML = (alpha.calendar_spreads || []).slice(0, 10).map(c => `<tr class="clickable" onclick="openCard('${esc(c.ticker)}')">
    <td><b>${esc(c.ticker)}</b></td><td>${c.front_dte}d / ${c.back_dte}d</td>
    <td class="num">$${c.strike} ${esc(c.opt_type[0].toUpperCase())}</td>
    <td class="num good">+${Number(c.iv_diff_vp).toFixed(1)}vp</td>
    <td class="num">$${Number(c.net_debit_mid).toFixed(2)}</td>
    <td class="num">${(Number(c.friction_pct)*100).toFixed(1)}%</td></tr>`).join('') || '<tr><td colspan="6" class="muted">No front-rich ETF calendars right now.</td></tr>';
  document.querySelector('#a-edgar tbody').innerHTML = (alpha.edgar_events || []).slice(0, 20).map(e => `<tr class="clickable" onclick="openCard('${esc(e.ticker)}')">
    <td><span class="pill warn">${esc(e.event_type)}</span></td><td><b>${esc(e.ticker)}</b></td>
    <td>${esc(e.filed_date)}</td><td>${esc(e.summary)}</td><td class="muted">${esc(e.source)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">Click UPD ALPHA to fetch SEC EDGAR filings.</td></tr>';

  // [4] Book, CRA Safety & Ghost Race
  const race = STATE.ghost_race || {};
  document.getElementById('b-race-meta').textContent = `${(race.dates||[])[0]||''} .. ${(race.dates||[]).slice(-1)[0]||''}`;
  document.getElementById('b-race-svg').innerHTML = renderGhostSVG(race);
  document.querySelector('#b-picks tbody').innerHTML = (STATE.book?.equity_picks || []).map(p => `<tr>
    <td>#${p.pick_id}</td><td>${esc((p.date||'').slice(0,10))}</td><td><b>${esc(p.ticker)}</b></td>
    <td>${esc(p.playbook_id)}</td><td class="num">$${Number(p.entry_price||0).toFixed(2)}</td>
    <td class="num">${p.current_price!=null?'$'+Number(p.current_price).toFixed(2):'—'}</td>
    <td class="num ${(p.unrealized_pct||0)>=0?'good':'bad'}">${p.unrealized_pct!=null?Number(p.unrealized_pct).toFixed(1)+'%':'—'}</td>
    <td>${p.status === 'OPEN'
      ? `<button class="btn" onclick="openCloseModal(${p.pick_id},'${esc(p.ticker)}',${p.current_price ?? p.entry_price})">Close</button> <span class="warn">${esc(p.exit_signal||'')}</span>`
      : `<span class="muted">CLOSED ${p.close_tag ? '['+esc(p.close_tag)+']' : ''}</span>`}</td></tr>`).join('') || '<tr><td colspan="8" class="muted">No stock/ETF picks logged yet.</td></tr>';

  const sf = STATE.tfsa_safety || {};
  const craCls = sf.cra_status === 'SAFE_INVESTOR' ? 'good' : (sf.cra_status === 'WATCH_CADENCE' ? 'warn' : 'bad');
  document.getElementById('b-safety').innerHTML =
    `<p><span class="pill ${craCls}">${esc(sf.cra_status || 'SAFE_INVESTOR')}</span> ` +
    `<b>${sf.trades_ytd ?? 0}</b> YTD trades · 15% US Div Drag: <b>$${Number(sf.total_tax_drag_usd_yr||0).toFixed(2)}/yr</b> (${Number(sf.weighted_tax_drag_bps||0).toFixed(0)} bps)` +
    (sf.room_remaining_cad != null ? ` · TFSA Room Left: <b>$${Number(sf.room_remaining_cad).toLocaleString()} CAD</b>` : '') +
    `</p>` + (sf.cra_notes || []).map(n => `<div class="muted" style="font-size:11px;margin-bottom:3px">• ${esc(n)}</div>`).join('');

  const pf = STATE.profile || {};
  document.getElementById('pf-tfsa').value = pf.tfsa_usd ?? '';
  document.getElementById('pf-cash').value = pf.cash_etf ?? 'SGOV';
  document.getElementById('pf-room').value = pf.tfsa_room_cad ?? '';
  document.getElementById('pf-ytd').value = pf.tfsa_contributed_ytd_cad ?? '';

  const reb = STATE.rebalance || {};
  document.getElementById('b-rebal').innerHTML = `<p><b>${esc(reb.due_message||'')}</b> <button class="btn good-btn" style="float:right" onclick="applyRebalanceWeb()">Apply &amp; Save Rebalance</button><br><span class="muted">Account $${Number(reb.account_usd||0).toLocaleString()} USD · ${esc(STATE.profile?.cash_etf||'SGOV')} reserve $${Number(reb.estimated_cash_after_usd||0).toLocaleString()}</span></p>` +
    `<table><thead><tr><th>Action</th><th>ETF</th><th class="num">Order Sh</th><th class="num">Delta $</th></tr></thead><tbody>` +
    (reb.orders||[]).map(o => `<tr><td class="${o.action==='BUY'?'good':o.action==='SELL'?'bad':'muted'}">${esc(o.action)}</td><td><b>${esc(o.ticker)}</b></td><td class="num">${o.delta_shares||0}</td><td class="num">$${Number(o.delta_usd||0).toFixed(0)}</td></tr>`).join('') + '</tbody></table>';

  // [5] Lab LEDs, Mistakes & Playbooks
  const pipes = STATE.health?.pipes || {};
  document.getElementById('l-leds').innerHTML = Object.entries(pipes).map(([k, v]) =>
    `<div class="led"><span class="dot ${esc(v.status)}"></span><b>${esc(k.toUpperCase())}</b> <span class="muted">${esc(v.detail)}</span></div>`).join('');
  const mrows = STATE.book?.mistake_costs?.by_tag || [];
  document.querySelector('#l-mistakes tbody').innerHTML = mrows.map(m => `<tr>
    <td><b>${esc(m.tag)}</b></td><td class="num">${m.count}</td>
    <td class="num">${Number(m.win_rate_pct||0).toFixed(0)}%</td>
    <td class="num ${m.avg_ret_pct>=0?'good':'bad'}">${m.avg_ret_pct>=0?'+':''}${Number(m.avg_ret_pct||0).toFixed(1)}%</td>
    <td class="num ${m.pnl_usd>=0?'good':'bad'}">$${Number(m.pnl_usd||0).toFixed(0)}</td></tr>`).join('') || '<tr><td colspan="5" class="muted">Close picks with Post-Mortem tags to measure mistake costs.</td></tr>';
  document.getElementById('l-clocks').innerHTML = (STATE.health?.readouts || []).map(r => `<p style="margin:6px 0">⏱ ${esc(r)}</p>`).join('');
  document.querySelector('#l-playbooks tbody').innerHTML = (STATE.playbooks || []).map(p => `<tr>
    <td><span class="pill ${p.evidence_tier==='VALIDATED'?'good':p.evidence_tier==='LEAD'?'info':'warn'}">${esc(p.evidence_tier)}</span></td>
    <td><b>${esc(p.id)}</b></td><td>${esc((p.accounts||[]).join(', '))}</td><td>${esc(p.horizon_label)}</td></tr>`).join('');
}

// ── [8] OPTIONS HUB (Scanner + T-Shaped Chain Matrix + 2D Heatmap) ──
let OH_SCAN_ROWS = [];
let OH_CHAIN = null;
let OH_LEGS = [];

function showOptSub(sub) {
  document.getElementById('oh-pane-scan').style.display = sub === 'scan' ? 'block' : 'none';
  document.getElementById('oh-pane-chain').style.display = sub === 'chain' ? 'block' : 'none';
  document.getElementById('oh-btn-scan').classList.toggle('active', sub === 'scan');
  document.getElementById('oh-btn-chain').classList.toggle('active', sub === 'chain');
  if (sub === 'chain' && !OH_CHAIN) loadWebChain(false);
}

async function loadOptionsScan(live = false) {
  const strat = document.getElementById('oh-strat').value || 'all';
  setToast(live ? 'Running live Yahoo options scan…' : 'Scanning archived chains…');
  try {
    const resp = await fetch(`/desk/api/options/scan?strategy=${encodeURIComponent(strat)}&live=${live ? 1 : 0}`);
    const d = await resp.json();
    OH_SCAN_ROWS = d.rows || [];
    document.getElementById('oh-scan-note').textContent = d.note || '';
    renderOptScanRows();
    setToast(`Options scan: ${OH_SCAN_ROWS.length} candidates`);
  } catch (err) {
    setToast('Options scan error: ' + err);
  }
}

function renderOptScanRows() {
  const q = (document.getElementById('oh-q').value || '').trim().toLowerCase();
  const toks = q ? q.split(/\\s+/) : [];
  const filtered = OH_SCAN_ROWS.filter(r => toks.every(t => {
    if (['bull_put','bear_call','long_call','long_put','iron_condor','calendar','short_put'].includes(t)) return r.strategy_key === t;
    if (t === 'tfsa') return !!r.tfsa_legal;
    if (t === 'taxable') return !r.tfsa_legal;
    if (t === 'clean') return !(r.risk_flags||[]).some(f => f.startsWith('EARN_') || f === 'WIDE_SPREAD' || f === 'HIST_WEAK_EDGE');
    if (t.startsWith('fric<')) { const lim = parseFloat(t.slice(5)) / (t.includes('%') ? 100 : 1); return r.friction_pct <= lim; }
    if (t.startsWith('dte>=')) return r.dte >= parseInt(t.slice(5), 10);
    if (t.startsWith('risk<')) return r.max_risk_usd <= parseFloat(t.slice(5));
    return `${r.ticker} ${r.strategy} ${r.strikes_label} ${r.expiration} ${(r.risk_flags||[]).join(' ')}`.toLowerCase().includes(t);
  }));
  window._OH_VIEW_ROWS = filtered;
  document.querySelector('#oh-scan-tbl tbody').innerHTML = filtered.slice(0, 60).map((r, i) => `<tr class="clickable" onclick="inspectOptRow(${i})">
    <td><span class="pill ${r.evidence_badge==='PROMISING'?'good':r.evidence_badge==='CAUTION'?'bad':'warn'}">${esc(r.evidence_badge)}</span></td>
    <td><b>${esc(r.ticker)}</b></td><td>${esc(r.strategy)}</td><td>${esc((r.expiration||'').slice(5))} (${r.dte}d)</td>
    <td>${esc(r.strikes_label)}</td>
    <td class="num ${r.is_credit?'good':''}">${r.is_credit?'+':'-'}$${Number(r.mid_price||0).toFixed(2)}</td>
    <td class="num">$${Number(r.walk_away_limit||0).toFixed(2)}</td>
    <td class="num ${r.friction_pct<=0.12?'good':r.friction_pct<=0.22?'warn':'bad'}">${(Number(r.friction_pct||0)*100).toFixed(1)}%</td>
    <td class="num">$${Number(r.max_risk_usd||0).toLocaleString()}</td>
    <td class="${r.tfsa_legal?'good':'warn'}">${r.tfsa_legal?'OK':'TAX'}</td></tr>`).join('') || '<tr><td colspan="10" class="muted">No candidates matched.</td></tr>';
  if (filtered.length) inspectOptRow(0);
}

function renderHeatmapHTML(an) {
  const ps = an.grid_price_shocks || [], ivs = an.grid_iv_shocks || [], grid = an.grid_pnl || [];
  if (!grid.length) return '';
  const flat = grid.flat().map(Math.abs);
  const mx = Math.max(1, ...flat);
  let h = '<table><thead><tr><th>IV\\Px</th>' + ps.map(p => `<th class="num">${p>=0?'+':''}${Math.round(p*100)}%</th>`).join('') + '</tr></thead><tbody>';
  ivs.forEach((iv, ri) => {
    h += `<tr><td class="muted">${iv>=0?'+':''}${Math.round(iv*100)}%</td>`;
    (grid[ri] || []).forEach(v => {
      const f = v / mx;
      const cls = f > 0.08 ? 'good' : (f < -0.08 ? 'bad' : 'muted');
      const bg = f > 0.08 ? `rgba(94,201,141,${Math.min(0.28, f*0.28).toFixed(2)})` : (f < -0.08 ? `rgba(224,108,117,${Math.min(0.28, -f*0.28).toFixed(2)})` : 'transparent');
      h += `<td class="num ${cls}" style="background:${bg}">${v>=0?'+':''}${Math.round(v)}</td>`;
    });
    h += '</tr>';
  });
  return h + '</tbody></table>';
}

async function inspectOptRow(idx) {
  const r = (window._OH_VIEW_ROWS || [])[idx];
  if (!r) return;
  const box = document.getElementById('oh-scan-insp');
  let heatHtml = '', ladHtml = '', greekHtml = '';
  if (r.legs && r.legs.length) {
    try {
      const resp = await fetch('/desk/api/options/analyze', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ ticker: r.ticker, spot: r.spot, legs: r.legs, days_elapsed: 0, iv_shock_pct: 0 }),
      });
      const d = await resp.json();
      if (d.ok && d.analysis) {
        const an = d.analysis;
        greekHtml = `<p class="muted">Δ$ ${an.delta_dollars} · Θ/d $${an.theta_per_day_usd} · Vega $${an.vega_per_1pct_usd} · Mkt PoP ${Math.round(an.pop_mkt*100)}% · BE ${(an.breakevens||[]).map(b=>'$'+b).join(', ')}</p>`;
        heatHtml = `<h4 style="margin:8px 0 4px;color:var(--primary)">2D Price × IV Risk Surface ($)</h4>` + renderHeatmapHTML(an);
        ladHtml = `<pre class="mono-pre" style="margin-top:8px;font-size:11px">${esc(an.ladder.step1_label)}\\n${esc(an.ladder.step2_label)}\\n${esc(an.ladder.abort_label)}</pre>`;
      }
    } catch {}
  }
  const calHtml = r.cal_pop_note ? `<p class="good" style="font-size:12px">${esc(r.cal_pop_note)}</p>` : '';
  box.innerHTML = `
    <div class="form-row" style="justify-content:space-between">
      <b>${esc(r.ticker)} ${esc(r.strategy)} ${esc(r.strikes_label)}</b>
      <span>
        <button class="btn" onclick="openWebChainFromScan('${esc(r.ticker)}','${esc((r.expiration||'').slice(0,10))}')">Open Chain</button>
        <button class="btn good-btn" onclick="paperLogWebScan(${idx})">Paper Log</button>
      </span>
    </div>
    <p class="muted">Spot $${r.spot} · Exp ${esc(r.expiration)} (${r.dte}d) · ${r.tfsa_legal?'<span class="good">TFSA OK</span>':'<span class="warn">TAXABLE ONLY</span>'} · Flags: ${esc((r.risk_flags||[]).join(', ')||'Clean')}</p>
    ${calHtml}${greekHtml}${heatHtml}${ladHtml}`;
}

function openWebChainFromScan(tk, exp) {
  document.getElementById('oh-ch-tk').value = tk;
  showOptSub('chain');
  loadWebChain(false, exp);
}

async function paperLogWebScan(idx) {
  const r = (window._OH_VIEW_ROWS || [])[idx];
  if (!r) return;
  const legs = r.legs || [];
  const primary = legs.find(l => l.side === 'SELL') || legs[0] || { strike: r.spot, opt_type: 'put' };
  const wing = legs.find(l => l !== primary) || null;
  const resp = await fetch('/desk/api/options/paper-log', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      ticker: r.ticker, strategy_name: r.strategy, expiration: (r.expiration||'').slice(0,10),
      strike: primary.strike, long_strike: wing ? wing.strike : null, opt_type: primary.opt_type,
      entry_price: r.mid_price, quantity: 1, max_loss_usd: r.max_risk_usd,
      entry_iv: r.iv_pct ? r.iv_pct/100 : null, entry_delta: r.short_delta,
    }),
  });
  const d = await resp.json();
  setToast(d.ok ? `Paper-logged ${r.ticker} ${r.strategy} into paper_trades.db` : ('Paper log err: ' + (d.error||'')));
}

async function loadWebChain(live = false, forceExp = null) {
  const tk = (document.getElementById('oh-ch-tk').value || 'SPY').trim().toUpperCase();
  const exp = forceExp != null ? forceExp : (document.getElementById('oh-ch-exp').value || '');
  const band = document.getElementById('oh-ch-band').value || '0.12';
  setToast(`Loading ${tk} ${live?'live':'archived'} chain…`);
  try {
    const url = `/desk/api/options/chain?ticker=${encodeURIComponent(tk)}&exp=${encodeURIComponent(exp)}&band=${band}&live=${live?1:0}`;
    const resp = await fetch(url);
    const d = await resp.json();
    if (!d.found) {
      document.getElementById('oh-ch-meta').textContent = d.error || 'Chain unavailable';
      document.querySelector('#oh-ch-tbl tbody').innerHTML = '<tr><td colspan="11" class="muted">No archived chain — click Live Yahoo.</td></tr>';
      return;
    }
    OH_CHAIN = d.chain;
    OH_LEGS = [];
    const expSel = document.getElementById('oh-ch-exp');
    expSel.innerHTML = (OH_CHAIN.available_expirations || []).map(([e, dt]) =>
      `<option value="${esc(e)}" ${e === OH_CHAIN.expiration ? 'selected' : ''}>${esc(e)} (${dt}d)</option>`
    ).join('');
    document.getElementById('oh-ch-meta').textContent =
      `${OH_CHAIN.ticker} $${OH_CHAIN.spot} · ATM IV ${OH_CHAIN.atm_iv_pct ?? '—'}% · 1σ ±${OH_CHAIN.expected_move_pct ?? '—'}% · Net GEX $${OH_CHAIN.net_gex_m ?? '—'}M [${OH_CHAIN.source}]`;
    document.querySelector('#oh-ch-tbl tbody').innerHTML = (OH_CHAIN.rows || []).map((row, ri) => {
      const c = row.call, p = row.put;
      const tags = (row.tags || []).join('·');
      const isSpot = (row.tags || []).includes('SPOT');
      return `<tr style="${isSpot ? 'background:var(--panel);font-weight:700' : ''}">
        <td>${c ? `<button class="btn" onclick="stageWebLeg(${ri},'call','BUY')">+B</button><button class="btn" onclick="stageWebLeg(${ri},'call','SELL')">-S</button>` : ''}</td>
        <td class="num">${c ? c.open_interest.toLocaleString() : '—'}</td>
        <td class="num">${c && c.iv!=null ? Math.round(c.iv*100)+'%' : '—'}</td>
        <td class="num">${c && c.delta!=null ? c.delta.toFixed(2) : '—'}</td>
        <td class="num">${c ? c.bid.toFixed(2)+'/'+c.ask.toFixed(2) : '—'}</td>
        <td style="text-align:center;color:${isSpot?'var(--primary)':'var(--ink-strong)'}"><b>${row.strike}</b> ${tags ? `<span class="muted" style="font-size:10px">${esc(tags)}</span>` : ''}</td>
        <td class="num">${p ? p.bid.toFixed(2)+'/'+p.ask.toFixed(2) : '—'}</td>
        <td class="num">${p && p.delta!=null ? p.delta.toFixed(2) : '—'}</td>
        <td class="num">${p && p.iv!=null ? Math.round(p.iv*100)+'%' : '—'}</td>
        <td class="num">${p ? p.open_interest.toLocaleString() : '—'}</td>
        <td>${p ? `<button class="btn" onclick="stageWebLeg(${ri},'put','BUY')">+B</button><button class="btn" onclick="stageWebLeg(${ri},'put','SELL')">-S</button>` : ''}</td></tr>`;
    }).join('');
    renderWebChainDefaultInspector();
    setToast(`Loaded ${OH_CHAIN.ticker} (${(OH_CHAIN.rows||[]).length} strikes)`);
  } catch (err) {
    setToast('Chain error: ' + err);
  }
}

function renderWebChainDefaultInspector() {
  if (!OH_CHAIN) return;
  const gex = (OH_CHAIN.gex_by_strike || []).slice(0, 8);
  const mxG = Math.max(0.1, ...gex.map(g => Math.abs(g.gex_m || 0)));
  const gexHtml = gex.map(g => {
    const pct = Math.min(100, Math.round(Math.abs(g.gex_m) / mxG * 100));
    const col = g.gex_m >= 0 ? 'var(--good)' : 'var(--bad)';
    return `<div style="margin:3px 0;font-size:11px">$${g.strike} <span style="float:right;color:${col}">${g.gex_m>=0?'+':''}$${g.gex_m}M</span><div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${col}"></div></div></div>`;
  }).join('') || '<div class="muted">GEX profile unavailable.</div>';
  const term = (OH_CHAIN.term_points || []).map(([d, iv]) => `${d}d: <b>${iv}%</b>`).join(' · ');
  document.getElementById('oh-ch-insp').innerHTML = `
    <b>${esc(OH_CHAIN.ticker)} Dealer GEX &amp; Term Structure</b>
    <p class="muted" style="font-size:11px">Flip: $${OH_CHAIN.gamma_flip ?? '—'} · Call Wall: $${OH_CHAIN.call_wall ?? '—'} · Put Wall: $${OH_CHAIN.put_wall ?? '—'}</p>
    <p style="font-size:11px">Term IV: ${term || '—'}</p>
    ${gexHtml}
    <p class="muted" style="margin-top:8px;font-size:11px">Click +B / -S on any strike row to stage 1–4 legs and simulate live T+Nd &amp; IV-shocked 2D Risk Surface.</p>`;
}

function clearWebLegs() { OH_LEGS = []; renderWebChainDefaultInspector(); setToast('Cleared staged legs'); }

function stageWebLeg(rowIdx, optType, side) {
  if (!OH_CHAIN || OH_LEGS.length >= 4) return;
  const row = OH_CHAIN.rows[rowIdx];
  const c = optType === 'call' ? row.call : row.put;
  if (!c) return;
  OH_LEGS.push({
    opt_type: optType, side, strike: c.strike, expiration: OH_CHAIN.expiration, dte: OH_CHAIN.dte,
    bid: c.bid, ask: c.ask, mid: c.mid, iv: c.iv || (OH_CHAIN.atm_iv_pct ? OH_CHAIN.atm_iv_pct/100 : 0.25),
    delta: c.delta || 0, theta: c.theta || 0, open_interest: c.open_interest || 0, volume: c.volume || 0, qty: 1,
  });
  analyzeWebLegs();
}

async function analyzeWebLegs() {
  if (!OH_CHAIN || !OH_LEGS.length) { renderWebChainDefaultInspector(); return; }
  const days = parseInt(document.getElementById('oh-t-sl').value || '0', 10);
  const ivShk = parseFloat(document.getElementById('oh-iv-sl').value || '0');
  const resp = await fetch('/desk/api/options/analyze', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ ticker: OH_CHAIN.ticker, spot: OH_CHAIN.spot, legs: OH_LEGS, days_elapsed: days, iv_shock_pct: ivShk }),
  });
  const d = await resp.json();
  if (!d.ok || !d.analysis) return;
  const an = d.analysis;
  const legLines = OH_LEGS.map(l => `<span class="pill ${l.side==='BUY'?'good':'bad'}">${l.side} ${l.strike}${l.opt_type[0].toUpperCase()}</span>`).join(' ');
  document.getElementById('oh-ch-insp').innerHTML = `
    <div><b>${esc(an.structure_name)}</b> ${an.tfsa_legal?'<span class="pill good">TFSA OK</span>':'<span class="pill warn">TAXABLE</span>'}</div>
    <div style="margin:6px 0">${legLines}</div>
    <p class="muted" style="font-size:12px">Net ${an.is_credit?'Credit':'Debit'} <b>$${an.net_mid}</b> · Fric ${(an.friction_pct*100).toFixed(1)}% · Risk $${Number(an.max_risk_usd).toLocaleString()} · BE ${(an.breakevens||[]).map(b=>'$'+b).join(', ')||'—'}<br>
    Δ$ ${an.delta_dollars} · Θ/d $${an.theta_per_day_usd} · Vega $${an.vega_per_1pct_usd} · Mkt PoP ${Math.round(an.pop_mkt*100)}%</p>
    ${renderHeatmapHTML(an)}
    <pre class="mono-pre" style="margin-top:6px;font-size:11px">${esc(an.ladder.abort_label)}</pre>`;
}
loadAll(false);
</script>
</body>
</html>
"""
