"""Route a view into the structures whose measured breakeven it can clear.

Five filters, each recording WHY it rejected something - the user learns the
logic instead of obeying a number.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from .execution_plan import build_walk_away_ladder
from .types import (BEARISH_STRUCTURES, BULLISH_STRUCTURES, CREDIT_STRUCTURES,
                    DEBIT_STRUCTURES, LEG_COUNT, NEUTRAL_STRUCTURES,
                    DetailedView, Expression, Rejection, VehicleComparison, View)
from .view import implied_hit
from ..execution_costs import FALLBACK_COMMISSION_PER_CONTRACT
from src.paths import repo_path

DEFAULT_COMMISSION = FALLBACK_COMMISSION_PER_CONTRACT
DEFAULT_SLIPPAGE = 0.05


def load_costs(config_path: str = "config.json") -> Tuple[float, float]:
    """Read the ONE source of truth for execution costs."""
    try:
        with open(repo_path(config_path)) as f:
            pt = (json.load(f).get("paper_trading") or {})
    except (OSError, ValueError):
        return DEFAULT_COMMISSION, DEFAULT_SLIPPAGE
    return (float(pt.get("commission_per_contract", DEFAULT_COMMISSION)),
            float(pt.get("slippage_per_share", DEFAULT_SLIPPAGE)))


def round_trip_cost(legs: int, commission: float = DEFAULT_COMMISSION,
                    slippage: float = DEFAULT_SLIPPAGE) -> float:
    """Open + close, both legs, slippage per share x100 shares per contract."""
    return (slippage * 100.0 * legs * 2.0) + (commission * legs * 2.0)


def _allowed_for(direction: str):
    if direction == "BULLISH":
        return set(BULLISH_STRUCTURES)
    if direction == "BEARISH":
        return set(BEARISH_STRUCTURES)
    return set(NEUTRAL_STRUCTURES)


def express(view: View, table: Dict, capital_usd: float,
            candidates: Dict[str, dict], max_cost_drag_pct: float = 25.0,
            commission: Optional[float] = None,
            slippage: Optional[float] = None
            ) -> Tuple[List[Expression], List[Rejection]]:
    commission = DEFAULT_COMMISSION if commission is None else commission
    slippage = DEFAULT_SLIPPAGE if slippage is None else slippage

    out: List[Expression] = []
    rejections: List[Rejection] = []
    allowed = _allowed_for(view.direction)
    view_hit = implied_hit(view)

    for name, margin in sorted(table.items()):
        # 1. direction
        if name not in allowed:
            rejections.append(Rejection(name, "wrong direction for a {} view"
                                        .format(view.direction)))
            continue
        # 2. bench / evidence
        if margin.state == "BENCHED":
            rejections.append(Rejection(
                name, "BENCHED (margin {:+.1f} pts)".format(margin.margin * 100)))
            continue
        if margin.state == "UNPROVEN":
            rejections.append(Rejection(
                name, "UNPROVEN (n={}, {}W/{}L - not enough evidence)"
                .format(margin.n, margin.wins, margin.losses)))
            continue
        # 3. confidence gate - debit structures only.
        # A debit structure wins only if the underlying moves far enough, fast
        # enough, so the view must supply that accuracy directly. A credit
        # structure wins on direction OR theta OR vol contraction, so its
        # realized hit rate already prices in that redundancy - gating it on
        # directional confidence would double-count direction.
        if name in DEBIT_STRUCTURES and view_hit < margin.breakeven_hit:
            rejections.append(Rejection(
                name, "needs {:.1%} to break even; this view implies only "
                "{:.1%}".format(margin.breakeven_hit, view_hit)))
            continue
        if name in CREDIT_STRUCTURES and margin.margin <= 0:
            rejections.append(Rejection(name, "margin not positive"))
            continue

        cand = candidates.get(name)
        if not cand:
            rejections.append(Rejection(name, "no candidate contract found"))
            continue
        cap_req = float(cand.get("capital_required") or 0.0)
        max_profit = float(cand.get("max_profit") or 0.0)
        if cap_req <= 0:
            rejections.append(Rejection(name, "unknown capital requirement"))
            continue
        # 4. affordability
        if cap_req > capital_usd:
            rejections.append(Rejection(
                name, "needs ${:.0f}, you have ${:.0f}".format(
                    cap_req, capital_usd)))
            continue
        # 5. cost drag
        legs = LEG_COUNT.get(name, 2)
        cost = round_trip_cost(legs, commission, slippage)
        drag = 100.0 * cost / max_profit if max_profit > 0 else float("inf")
        if drag > max_cost_drag_pct:
            rejections.append(Rejection(
                name, "cost drag {:.0f}% of max profit (limit {:.0f}%)".format(
                    drag, max_cost_drag_pct)))
            continue

        warning = ""
        if name in DEBIT_STRUCTURES and margin.margin < 0:
            warning = ("your view is OVERRIDING this structure's record: it "
                       "has cleared its breakeven only {:.1f}% vs the {:.1f}% "
                       "it needs".format(margin.realized_hit * 100,
                                         margin.breakeven_hit * 100))

        out.append(Expression(
            strategy=name, margin=margin.margin,
            breakeven_hit=margin.breakeven_hit,
            realized_hit=margin.realized_hit, capital_required=cap_req,
            cost_drag_pct=drag, legs=legs, warning=warning))

    out.sort(key=lambda e: (-e.margin, e.cost_drag_pct))
    return out, rejections


def _build_lognormal_density(spot: float, sigma: float, t_years: float, r: float = 0.045):
    """Build a clean Probability Lab `Density` on [0.40 S, 2.50 S] when a full SVI chain isn't supplied."""
    import numpy as np
    from src.probability_lab.rnd import Density

    k_grid = np.linspace(0.40 * spot, 2.50 * spot, 500)
    sd = max(sigma * np.sqrt(max(t_years, 1.0 / 365.0)), 1e-4)
    mu = np.log(spot) + (r - 0.5 * sigma * sigma) * max(t_years, 1.0 / 365.0)
    pdf = np.exp(-0.5 * ((np.log(k_grid) - mu) / sd) ** 2) / (k_grid * sd * np.sqrt(2.0 * np.pi))
    area = float(np.trapezoid(pdf, k_grid))
    return Density(k_grid, pdf / max(area, 1e-12))


def _bs_px_delta(spot: float, strike: float, t_years: float, r: float, sigma: float, opt_type: str) -> Tuple[float, float, float]:
    """Return (price, delta, theta_per_day) under Black-Scholes."""
    import math
    from scipy.stats import norm

    t = max(t_years, 1.0 / 365.0)
    sig = max(sigma, 1e-4)
    sqt = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sig * sig) * t) / (sig * sqt)
    d2 = d1 - sig * sqt
    disc = math.exp(-r * t)
    pdf1 = float(norm.pdf(d1))
    if opt_type == "call":
        px = spot * float(norm.cdf(d1)) - strike * disc * float(norm.cdf(d2))
        delta = float(norm.cdf(d1))
        theta_yr = -(spot * pdf1 * sig) / (2.0 * sqt) - r * strike * disc * float(norm.cdf(d2))
    else:
        px = strike * disc * float(norm.cdf(-d2)) - spot * float(norm.cdf(-d1))
        delta = float(norm.cdf(d1)) - 1.0
        theta_yr = -(spot * pdf1 * sig) / (2.0 * sqt) + r * strike * disc * float(norm.cdf(-d2))
    return (max(0.01, px), delta, theta_yr / 365.0)


def compare_vehicles(
    view: DetailedView,
    *,
    spot: float,
    iv: float = 0.25,
    r: float = 0.045,
    option_half_spread_pct: float = 0.04,
    market_density=None,
    chain_db: Optional[str] = None,
    allow_live: bool = False,
    chain_slices: Optional[Dict[str, Any]] = None,
) -> List[VehicleComparison]:
    """Compare Shares, Deep-ITM LEAPS, Debit Vertical, Wide 60-120d Credit Spread,
    Covered Call (if 100 sh owned), and Single Option side-by-side under Probability Lab RND.

    When `chain_db`, `allow_live=True`, or preloaded `chain_slices` are provided,
    structures are priced from **real listed option quotes** (real bid/ask spreads,
    real IVs, and Breeden-Litzenberger SVI RND) with graceful fallback to Black-Scholes
    when a specific tenor/strike is absent from the chain.
    """
    import numpy as np
    from src.probability_lab.view import apply_view
    from .chain_data import load_chain_for_horizon

    def _get_slice(key: str, target_dte: int, min_dte: int, max_dte: int):
        if chain_slices is not None and key in chain_slices:
            return chain_slices[key]
        if not chain_db and not allow_live:
            return None
        return load_chain_for_horizon(
            view.symbol,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
            spot=spot,
            chain_db=chain_db or "data/chain_archive.db",
            allow_live=allow_live,
            build_rnd=True,
        )

    h_days = max(int(view.horizon_days), 5)
    t_h = h_days / 365.0
    sl_horizon = _get_slice("horizon", h_days, max(7, h_days - 20), h_days + 30)
    eff_iv = (
        float(sl_horizon.atm_iv)
        if (sl_horizon is not None and sl_horizon.atm_iv is not None and 0.03 <= sl_horizon.atm_iv <= 3.5)
        else float(iv)
    )
    eff_half_pct = (
        float(sl_horizon.median_half_spread_pct)
        if (sl_horizon is not None and sl_horizon.median_half_spread_pct is not None and 0.002 <= sl_horizon.median_half_spread_pct <= 0.35)
        else float(option_half_spread_pct)
    )

    if market_density is not None:
        q_mkt = market_density
    elif sl_horizon is not None and sl_horizon.rnd_density is not None:
        q_mkt = sl_horizon.rnd_density
        t_h = max(sl_horizon.dte, 5) / 365.0
    else:
        q_mkt = _build_lognormal_density(spot, eff_iv, t_h, r)
    q_view = apply_view(q_mkt, drift_pct=view.target_move_pct, vol_mult=view.vol_mult)

    d_upper = str(view.direction).upper()
    is_tfsa = str(view.account_type).lower() == "tfsa"
    results: List[VehicleComparison] = []

    def _round_strike(x: float) -> float:
        step = 1.0 if spot < 50 else (2.5 if spot < 150 else 5.0)
        return round(x / step) * step

    # 1. Shares (Bullish only in TFSA; also short-able in taxable)
    if d_upper == "BULLISH" or (d_upper == "BEARISH" and not is_tfsa):
        sign = 1.0 if d_upper == "BULLISH" else -1.0
        stop_dist = max(spot * float(view.stop_loss_pct), 0.01)
        stop_px = spot - sign * stop_dist
        shares = max(1, int(view.risk_budget_usd // stop_dist))
        cash_outlay = shares * spot
        car = shares * stop_dist
        fric = cash_outlay * 0.0005  # 5 bps round-trip ($0 Wealthsimple commission)

        def _sh_payoff(K: np.ndarray) -> np.ndarray:
            raw = sign * (K - spot)
            clipped = np.maximum(raw, -stop_dist)
            return shares * clipped

        ev_mkt = q_mkt.expected_payoff(_sh_payoff) - fric
        ev_view = q_view.expected_payoff(_sh_payoff) - fric
        pop = q_view.prob_payoff_exceeds(_sh_payoff, fric)
        tgt_gain = shares * abs(spot * view.target_move_pct)
        rr = tgt_gain / max(car, 1.0)
        results.append(
            VehicleComparison(
                vehicle="Shares",
                description=f"{'Buy' if sign > 0 else 'Short'} {shares} sh @ ${spot:.2f} (stop ${stop_px:.2f})",
                legs=0,
                tfsa_legal=(sign > 0),
                capital_at_risk_usd=round(car, 2),
                cash_outlay_usd=round(cash_outlay, 2),
                max_profit_usd=round(max(tgt_gain, car), 2),
                friction_usd=round(fric, 2),
                friction_pct_of_reward=round(100.0 * fric / max(tgt_gain, car, 1.0), 2),
                ev_mkt_net_usd=round(ev_mkt, 2),
                ev_view_net_usd=round(ev_view, 2),
                pop_view=round(pop, 3),
                rr_ratio=round(rr, 2),
                theta_per_day_usd=0.0,
                delta_dollars=round(sign * shares * spot, 2),
                walk_away_label=f"Limit @ ${spot:.2f} (0.05% spread, $0 comm)",
            )
        )

    # 2. Deep-ITM LEAPS (180-365 DTE, ~0.80 delta)
    if d_upper in ("BULLISH", "BEARISH"):
        ot = "call" if d_upper == "BULLISH" else "put"
        sl_leaps = _get_slice("leaps", 270, 150, 420)
        target_k_leaps = spot * (0.85 if ot == "call" else 1.15)
        real_leaps = sl_leaps.match_contract(ot, target_k_leaps, max_rel_dist=0.10) if sl_leaps is not None else None
        if real_leaps is not None and sl_leaps is not None:
            k_leaps = real_leaps.strike
            t_leaps = max(sl_leaps.dte, 90) / 365.0
            px = real_leaps.mid
            half = max(0.02, real_leaps.half_spread)
            p_cross = real_leaps.ask
            bs_px, bs_dl, bs_th = _bs_px_delta(spot, k_leaps, t_leaps, r, real_leaps.iv or eff_iv, ot)
            dl = real_leaps.delta if real_leaps.delta is not None else bs_dl
            th = real_leaps.theta if real_leaps.theta is not None else bs_th
            leaps_dte_lbl = f"{sl_leaps.dte}d"
            leaps_src_note = f" [{sl_leaps.source}]"
        else:
            k_leaps = _round_strike(target_k_leaps)
            t_leaps = 270.0 / 365.0
            px, dl, th = _bs_px_delta(spot, k_leaps, t_leaps, r, eff_iv, ot)
            half = max(0.05, px * eff_half_pct)
            p_cross = px + half
            leaps_dte_lbl = "270d"
            leaps_src_note = ""
        car = p_cross * 100.0
        fric = 2.0 * half * 100.0
        t_rem = max(t_leaps - t_h, 30.0 / 365.0)

        def _leaps_payoff(K: np.ndarray) -> np.ndarray:
            intr = np.maximum(K - k_leaps, 0.0) if ot == "call" else np.maximum(k_leaps - K, 0.0)
            time_val = max(0.0, px - (max(spot - k_leaps, 0.0) if ot == "call" else max(k_leaps - spot, 0.0)))
            rem_tv = time_val * np.sqrt(t_rem / t_leaps) * np.exp(-0.5 * ((K - k_leaps) / (0.35 * spot)) ** 2)
            return 100.0 * ((intr + rem_tv) - px)

        ev_mkt = q_mkt.expected_payoff(_leaps_payoff) - fric
        ev_view = q_view.expected_payoff(_leaps_payoff) - fric
        pop = q_view.prob_payoff_exceeds(_leaps_payoff, fric)
        ladder = build_walk_away_ladder(
            structure="Deep-ITM LEAPS",
            is_credit=False,
            p_mid=px,
            p_cross=p_cross,
            ev_view_gross_per_share=px + (ev_view + fric) / 100.0,
            exit_half_spread_per_share=half,
        )
        results.append(
            VehicleComparison(
                vehicle="Deep-ITM LEAPS",
                description=f"Buy {leaps_dte_lbl} {k_leaps:g}{ot[0].upper()} (~{abs(dl):.2f}Δ) @ ${px:.2f} mid{leaps_src_note}",
                legs=1,
                tfsa_legal=True,
                capital_at_risk_usd=round(car, 2),
                cash_outlay_usd=round(car, 2),
                max_profit_usd=round(car * 2.0, 2),
                friction_usd=round(fric, 2),
                friction_pct_of_reward=round(100.0 * fric / max(car, 1.0), 2),
                ev_mkt_net_usd=round(ev_mkt, 2),
                ev_view_net_usd=round(ev_view, 2),
                pop_view=round(pop, 3),
                rr_ratio=round(max(0.0, spot * abs(view.target_move_pct) * abs(dl) * 100.0) / max(car, 1.0), 2),
                theta_per_day_usd=round(th * 100.0, 2),
                delta_dollars=round(dl * 100.0 * spot, 2),
                walk_away_label=ladder.abort_label,
                caution="Over budget" if car > view.risk_budget_usd * 3 else "",
            )
        )

    # 3. Debit Vertical (ATM to +6% / -6%, horizon DTE)
    if d_upper in ("BULLISH", "BEARISH"):
        ot = "call" if d_upper == "BULLISH" else "put"
        target_kl = spot
        target_ks = spot * (1.06 if ot == "call" else 0.94)
        real_l = sl_horizon.match_contract(ot, target_kl, max_rel_dist=0.05) if sl_horizon is not None else None
        real_s = sl_horizon.match_contract(ot, target_ks, max_rel_dist=0.06) if sl_horizon is not None else None
        if (
            real_l is not None
            and real_s is not None
            and sl_horizon is not None
            and ((ot == "call" and real_s.strike > real_l.strike) or (ot == "put" and real_s.strike < real_l.strike))
            and (real_l.mid - real_s.mid) > 0.05
        ):
            k_long, k_short = real_l.strike, real_s.strike
            width = abs(k_short - k_long)
            debit_mid = real_l.mid - real_s.mid
            p_cross = min(width - 0.02, real_l.ask - real_s.bid)
            half = max(0.02, p_cross - debit_mid)
            _, dl_l_bs, th_l_bs = _bs_px_delta(spot, k_long, t_h, r, real_l.iv or eff_iv, ot)
            _, dl_s_bs, th_s_bs = _bs_px_delta(spot, k_short, t_h, r, real_s.iv or eff_iv, ot)
            dl_l = real_l.delta if real_l.delta is not None else dl_l_bs
            dl_s = real_s.delta if real_s.delta is not None else dl_s_bs
            th_l = real_l.theta if real_l.theta is not None else th_l_bs
            th_s = real_s.theta if real_s.theta is not None else th_s_bs
            dv_dte_lbl = f"{sl_horizon.dte}d"
        else:
            k_long = _round_strike(target_kl)
            k_short = _round_strike(target_ks)
            if abs(k_short - k_long) < 1.0:
                k_short = k_long + (5.0 if ot == "call" else -5.0)
            width = abs(k_short - k_long)
            pl, dl_l, th_l = _bs_px_delta(spot, k_long, t_h, r, eff_iv, ot)
            ps, dl_s, th_s = _bs_px_delta(spot, k_short, t_h, r, eff_iv, ot)
            debit_mid = max(0.05, pl - ps)
            half = max(0.04, (pl + ps) * eff_half_pct)
            p_cross = min(width - 0.05, debit_mid + half)
            dv_dte_lbl = f"{h_days}d"

        car = p_cross * 100.0
        max_prof = max(1.0, (width - p_cross) * 100.0)
        fric = 2.0 * half * 100.0

        def _dv_payoff(K: np.ndarray) -> np.ndarray:
            if ot == "call":
                val = np.clip(K - k_long, 0.0, width)
            else:
                val = np.clip(k_long - K, 0.0, width)
            return 100.0 * (val - debit_mid)

        ev_mkt = q_mkt.expected_payoff(_dv_payoff) - fric
        ev_view = q_view.expected_payoff(_dv_payoff) - fric
        pop = q_view.prob_payoff_exceeds(_dv_payoff, fric)
        ladder = build_walk_away_ladder(
            structure="Debit Vertical",
            is_credit=False,
            p_mid=debit_mid,
            p_cross=p_cross,
            ev_view_gross_per_share=debit_mid + (ev_view + fric) / 100.0,
            exit_half_spread_per_share=half,
        )
        results.append(
            VehicleComparison(
                vehicle="Debit Vertical",
                description=f"{dv_dte_lbl} {k_long:g}/{k_short:g} {ot.title()} Debit @ ${debit_mid:.2f}",
                legs=2,
                tfsa_legal=True,
                capital_at_risk_usd=round(car, 2),
                cash_outlay_usd=round(car, 2),
                max_profit_usd=round(max_prof, 2),
                friction_usd=round(fric, 2),
                friction_pct_of_reward=round(100.0 * fric / max_prof, 2),
                ev_mkt_net_usd=round(ev_mkt, 2),
                ev_view_net_usd=round(ev_view, 2),
                pop_view=round(pop, 3),
                rr_ratio=round(max_prof / max(car, 1.0), 2),
                theta_per_day_usd=round((th_l - th_s) * 100.0, 2),
                delta_dollars=round((dl_l - dl_s) * 100.0 * spot, 2),
                walk_away_label=ladder.abort_label,
                caution="Friction >25% reward" if (100.0 * fric / max_prof) > 25.0 else "",
            )
        )

    # 4. Wide Credit Spread (60-120 DTE 5%-wide — directly reflects Test O1 LEAD geometry!)
    ot_cr = "put" if d_upper in ("BULLISH", "NEUTRAL") else "call"
    sl_wide = _get_slice("wide_credit", max(90, h_days), 60, max(120, h_days + 30))
    target_ksh = spot * (0.94 if ot_cr == "put" else 1.06)
    target_klg = target_ksh + (-0.05 * spot if ot_cr == "put" else 0.05 * spot)
    real_sh = sl_wide.match_contract(ot_cr, target_ksh, max_rel_dist=0.06) if sl_wide is not None else None
    real_lg = None
    if real_sh is not None and sl_wide is not None:
        t_lg_exact = real_sh.strike + (-0.05 * spot if ot_cr == "put" else 0.05 * spot)
        real_lg = sl_wide.match_contract(ot_cr, t_lg_exact, max_rel_dist=0.06)
        if real_lg is not None and (
            (ot_cr == "put" and real_lg.strike >= real_sh.strike)
            or (ot_cr == "call" and real_lg.strike <= real_sh.strike)
        ):
            real_lg = None

    if real_sh is not None and real_lg is not None and sl_wide is not None and (real_sh.mid - real_lg.mid) > 0.05:
        t_cr = max(sl_wide.dte, 30) / 365.0
        k_sh, k_lg = real_sh.strike, real_lg.strike
        w_cr = abs(k_sh - k_lg)
        cr_mid = real_sh.mid - real_lg.mid
        cr_cross = max(0.02, real_sh.bid - real_lg.ask)
        half_cr = max(0.02, cr_mid - cr_cross)
        _, d_sh_bs, th_sh_bs = _bs_px_delta(spot, k_sh, t_cr, r, real_sh.iv or eff_iv, ot_cr)
        _, d_lg_bs, th_lg_bs = _bs_px_delta(spot, k_lg, t_cr, r, real_lg.iv or eff_iv, ot_cr)
        d_sh = real_sh.delta if real_sh.delta is not None else d_sh_bs
        d_lg = real_lg.delta if real_lg.delta is not None else d_lg_bs
        th_sh = real_sh.theta if real_sh.theta is not None else th_sh_bs
        th_lg = real_lg.theta if real_lg.theta is not None else th_lg_bs
        q_cr_mkt = sl_wide.rnd_density if sl_wide.rnd_density is not None else _build_lognormal_density(spot, sl_wide.atm_iv or eff_iv, t_cr, r)
        cr_dte_lbl = f"{sl_wide.dte}d"
    else:
        t_cr = max(90.0 / 365.0, t_h)
        k_sh = _round_strike(target_ksh)
        k_lg = _round_strike(k_sh + (-0.05 * spot if ot_cr == "put" else 0.05 * spot))
        if abs(k_sh - k_lg) < 1.0:
            k_lg = k_sh + (-5.0 if ot_cr == "put" else 5.0)
        w_cr = abs(k_sh - k_lg)
        p_sh, d_sh, th_sh = _bs_px_delta(spot, k_sh, t_cr, r, eff_iv, ot_cr)
        p_lg, d_lg, th_lg = _bs_px_delta(spot, k_lg, t_cr, r, eff_iv, ot_cr)
        cr_mid = max(0.10, p_sh - p_lg)
        half_cr = max(0.03, (p_sh + p_lg) * (eff_half_pct * 0.5))
        cr_cross = max(0.05, cr_mid - half_cr)
        q_cr_mkt = _build_lognormal_density(spot, eff_iv, t_cr, r)
        cr_dte_lbl = f"{int(round(t_cr * 365))}d"

    car_cr = max(1.0, (w_cr - cr_cross) * 100.0)
    max_prof_cr = cr_cross * 100.0
    fric_cr = 2.0 * half_cr * 100.0
    # Do NOT linearly extrapolate a short-horizon view (e.g. +8%/30d -> +24%/90d):
    # apply the user's stated target move once over their view horizon (risk-free thereafter).
    q_cr_view = apply_view(q_cr_mkt, drift_pct=view.target_move_pct, vol_mult=view.vol_mult)

    def _cr_payoff(K: np.ndarray) -> np.ndarray:
        if ot_cr == "put":
            loss = np.clip(k_sh - K, 0.0, w_cr)
        else:
            loss = np.clip(K - k_sh, 0.0, w_cr)
        return 100.0 * (cr_mid - loss)

    ev_mkt_cr = q_cr_mkt.expected_payoff(_cr_payoff) - fric_cr
    ev_view_cr = q_cr_view.expected_payoff(_cr_payoff) - fric_cr
    pop_cr = q_cr_view.prob_payoff_exceeds(_cr_payoff, fric_cr)
    ladder_cr = build_walk_away_ladder(
        structure="Wide Credit Spread",
        is_credit=True,
        p_mid=cr_mid,
        p_cross=cr_cross,
        ev_view_gross_per_share=cr_mid,
    )
    results.append(
        VehicleComparison(
            vehicle="Wide Credit Spread",
            description=f"{cr_dte_lbl} {'Bull Put' if ot_cr == 'put' else 'Bear Call'} {k_sh:g}/{k_lg:g} @ ${cr_mid:.2f} cr",
            legs=2,
            tfsa_legal=True,
            capital_at_risk_usd=round(car_cr, 2),
            cash_outlay_usd=round(car_cr, 2),
            max_profit_usd=round(max_prof_cr, 2),
            friction_usd=round(fric_cr, 2),
            friction_pct_of_reward=round(100.0 * fric_cr / max(max_prof_cr, 1.0), 2),
            ev_mkt_net_usd=round(ev_mkt_cr, 2),
            ev_view_net_usd=round(ev_view_cr, 2),
            pop_view=round(pop_cr, 3),
            rr_ratio=round(max_prof_cr / car_cr, 2),
            theta_per_day_usd=round((-th_sh + th_lg) * 100.0, 2),
            delta_dollars=round((-d_sh + d_lg) * 100.0 * spot, 2),
            walk_away_label=ladder_cr.abort_label,
            caution="LIMIT ONLY (k<=0.35)" if (100.0 * fric_cr / max(max_prof_cr, 1.0)) <= 25.0 else "Friction >25% credit",
        )
    )

    # 5. Covered Call (only if owns_100_shares=True and not strongly bearish)
    if view.owns_100_shares and d_upper in ("NEUTRAL", "BULLISH"):
        sl_cc = _get_slice("covered_call", 45, 25, 65)
        real_cc = sl_cc.match_contract("call", spot * 1.05, max_rel_dist=0.06) if sl_cc is not None else None
        if real_cc is not None and sl_cc is not None:
            k_cc = real_cc.strike
            t_cc = max(sl_cc.dte, 14) / 365.0
            p_cc = real_cc.mid
            cc_cross = max(0.02, real_cc.bid)
            half_cc = max(0.01, p_cc - cc_cross)
            _, d_cc_bs, th_cc_bs = _bs_px_delta(spot, k_cc, t_cc, r, real_cc.iv or eff_iv, "call")
            d_cc = real_cc.delta if real_cc.delta is not None else d_cc_bs
            th_cc = real_cc.theta if real_cc.theta is not None else th_cc_bs
            q_cc_mkt = sl_cc.rnd_density if sl_cc.rnd_density is not None else _build_lognormal_density(spot, sl_cc.atm_iv or eff_iv, t_cc, r)
            cc_dte_lbl = f"{sl_cc.dte}d"
        else:
            k_cc = _round_strike(spot * 1.05)
            t_cc = 45.0 / 365.0
            p_cc, d_cc, th_cc = _bs_px_delta(spot, k_cc, t_cc, r, eff_iv, "call")
            half_cc = max(0.02, p_cc * eff_half_pct)
            cc_cross = max(0.02, p_cc - half_cc)
            q_cc_mkt = _build_lognormal_density(spot, eff_iv, t_cc, r)
            cc_dte_lbl = "45d"

        fric_cc = half_cc * 100.0  # 1-leg open (usually held/expired)
        q_cc_view = apply_view(q_cc_mkt, drift_pct=view.target_move_pct, vol_mult=view.vol_mult)

        def _cc_payoff(K: np.ndarray) -> np.ndarray:
            return 100.0 * (p_cc - np.maximum(K - k_cc, 0.0))

        ev_mkt_cc = q_cc_mkt.expected_payoff(_cc_payoff) - fric_cc
        ev_view_cc = q_cc_view.expected_payoff(_cc_payoff) - fric_cc
        pop_cc = q_cc_view.prob_payoff_exceeds(_cc_payoff, fric_cc)
        ladder_cc = build_walk_away_ladder(
            structure="Covered Call",
            is_credit=True,
            p_mid=p_cc,
            p_cross=cc_cross,
            ev_view_gross_per_share=p_cc,
        )
        results.append(
            VehicleComparison(
                vehicle="Covered Call",
                description=f"Sell {cc_dte_lbl} {k_cc:g}C (~{d_cc:.2f}Δ) against 100 sh @ ${p_cc:.2f} cr",
                legs=1,
                tfsa_legal=True,
                capital_at_risk_usd=0.0,
                cash_outlay_usd=0.0,
                max_profit_usd=round(cc_cross * 100.0, 2),
                friction_usd=round(fric_cc, 2),
                friction_pct_of_reward=round(100.0 * fric_cc / max(cc_cross * 100.0, 1.0), 2),
                ev_mkt_net_usd=round(ev_mkt_cc, 2),
                ev_view_net_usd=round(ev_view_cc, 2),
                pop_view=round(pop_cc, 3),
                rr_ratio=0.0,
                theta_per_day_usd=round(-th_cc * 100.0, 2),
                delta_dollars=round(-d_cc * 100.0 * spot, 2),
                walk_away_label=ladder_cc.abort_label,
            )
        )

    # 6. Single Option (~0.40Δ, 30-60 DTE)
    if d_upper in ("BULLISH", "BEARISH"):
        ot = "call" if d_upper == "BULLISH" else "put"
        target_ko = spot * (1.03 if ot == "call" else 0.97)
        real_o = sl_horizon.match_contract(ot, target_ko, max_rel_dist=0.06) if sl_horizon is not None else None
        if real_o is not None and sl_horizon is not None:
            k_opt = real_o.strike
            t_opt = max(sl_horizon.dte, 7) / 365.0
            p_o = real_o.mid
            p_cross_o = real_o.ask
            half_o = max(0.02, real_o.half_spread)
            _, d_o_bs, th_o_bs = _bs_px_delta(spot, k_opt, t_opt, r, real_o.iv or eff_iv, ot)
            d_o = real_o.delta if real_o.delta is not None else d_o_bs
            th_o = real_o.theta if real_o.theta is not None else th_o_bs
            so_dte_lbl = f"{sl_horizon.dte}d"
        else:
            k_opt = _round_strike(target_ko)
            t_opt = max(35.0 / 365.0, t_h)
            p_o, d_o, th_o = _bs_px_delta(spot, k_opt, t_opt, r, eff_iv, ot)
            half_o = max(0.03, p_o * eff_half_pct)
            p_cross_o = p_o + half_o
            so_dte_lbl = f"{int(t_opt*365)}d"

        car_o = p_cross_o * 100.0
        fric_o = 2.0 * half_o * 100.0

        def _so_payoff(K: np.ndarray) -> np.ndarray:
            intr = np.maximum(K - k_opt, 0.0) if ot == "call" else np.maximum(k_opt - K, 0.0)
            return 100.0 * (intr - p_o)

        ev_mkt_o = q_mkt.expected_payoff(_so_payoff) - fric_o
        ev_view_o = q_view.expected_payoff(_so_payoff) - fric_o
        pop_o = q_view.prob_payoff_exceeds(_so_payoff, fric_o)
        ladder_o = build_walk_away_ladder(
            structure="Single Option",
            is_credit=False,
            p_mid=p_o,
            p_cross=p_cross_o,
            ev_view_gross_per_share=p_o + (ev_view_o + fric_o) / 100.0,
            exit_half_spread_per_share=half_o,
        )
        results.append(
            VehicleComparison(
                vehicle="Single Option",
                description=f"Buy {so_dte_lbl} {k_opt:g}{ot[0].upper()} (~{abs(d_o):.2f}Δ) @ ${p_o:.2f}",
                legs=1,
                tfsa_legal=True,
                capital_at_risk_usd=round(car_o, 2),
                cash_outlay_usd=round(car_o, 2),
                max_profit_usd=round(car_o * 3.0, 2),
                friction_usd=round(fric_o, 2),
                friction_pct_of_reward=round(100.0 * fric_o / max(car_o, 1.0), 2),
                ev_mkt_net_usd=round(ev_mkt_o, 2),
                ev_view_net_usd=round(ev_view_o, 2),
                pop_view=round(pop_o, 3),
                rr_ratio=round(max(0.0, ev_view_o) / max(car_o, 1.0), 2),
                theta_per_day_usd=round(th_o * 100.0, 2),
                delta_dollars=round(d_o * 100.0 * spot, 2),
                walk_away_label=ladder_o.abort_label,
                caution="Negative EV under view" if ev_view_o <= 0 else "",
            )
        )

    # Sort by EV(view) net of friction descending, tie-breaking on lowest friction %
    results.sort(key=lambda v: (-v.ev_view_net_usd, v.friction_pct_of_reward))
    # Recommend the best positive-EV vehicle that actually fits the user's risk budget
    # (never crown a $14,000 LEAPS for a $500 budget when an affordable positive-EV vehicle exists).
    affordable_pos = [
        i for i, v in enumerate(results)
        if v.ev_view_net_usd > 0 and v.capital_at_risk_usd <= view.risk_budget_usd * 1.15
    ]
    rec_idx = affordable_pos[0] if affordable_pos else (0 if (results and results[0].ev_view_net_usd > 0) else None)
    if rec_idx is not None:
        from dataclasses import replace

        results[rec_idx] = replace(results[rec_idx], recommended=True)
    return results
