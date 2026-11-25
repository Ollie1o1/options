"""Route a view into the structures whose measured breakeven it can clear.

Five filters, each recording WHY it rejected something - the user learns the
logic instead of obeying a number.
"""
import json
from typing import Dict, List, Optional, Tuple

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
) -> List[VehicleComparison]:
    """Compare Shares, Deep-ITM LEAPS, Debit Vertical, Wide 60-120d Credit Spread,
    Covered Call (if 100 sh owned), and Single Option side-by-side under Probability Lab RND.

    All vehicles are evaluated under BOTH market RND (`q_mkt`) and tilted user view (`q_view`)
    net of realistic round-trip crossing friction.
    """
    import numpy as np
    from src.probability_lab.view import apply_view

    t_h = max(int(view.horizon_days), 5) / 365.0
    q_mkt = market_density if market_density is not None else _build_lognormal_density(spot, iv, t_h, r)
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
        k_leaps = _round_strike(spot * (0.85 if ot == "call" else 1.15))
        t_leaps = 270.0 / 365.0
        px, dl, th = _bs_px_delta(spot, k_leaps, t_leaps, r, iv, ot)
        half = max(0.05, px * option_half_spread_pct)
        p_cross = px + half
        car = p_cross * 100.0
        fric = 2.0 * half * 100.0
        t_rem = max(t_leaps - t_h, 30.0 / 365.0)

        def _leaps_payoff(K: np.ndarray) -> np.ndarray:
            # Approximate horizon mark via intrinsic + remaining time value scaled smoothly
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
                description=f"Buy 270d {k_leaps:g}{ot[0].upper()} (~{abs(dl):.2f}Δ) @ ${px:.2f} mid",
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
        k_long = _round_strike(spot)
        k_short = _round_strike(spot * (1.06 if ot == "call" else 0.94))
        if abs(k_short - k_long) < 1.0:
            k_short = k_long + (5.0 if ot == "call" else -5.0)
        width = abs(k_short - k_long)
        pl, dl_l, th_l = _bs_px_delta(spot, k_long, t_h, r, iv, ot)
        ps, dl_s, th_s = _bs_px_delta(spot, k_short, t_h, r, iv, ot)
        debit_mid = max(0.05, pl - ps)
        half = max(0.04, (pl + ps) * option_half_spread_pct)
        p_cross = min(width - 0.05, debit_mid + half)
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
                description=f"{view.horizon_days}d {k_long:g}/{k_short:g} {ot.title()} Debit @ ${debit_mid:.2f}",
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
    t_cr = max(90.0 / 365.0, t_h)
    k_sh = _round_strike(spot * (0.94 if ot_cr == "put" else 1.06))
    k_lg = _round_strike(k_sh + (-0.05 * spot if ot_cr == "put" else 0.05 * spot))
    if abs(k_sh - k_lg) < 1.0:
        k_lg = k_sh + (-5.0 if ot_cr == "put" else 5.0)
    w_cr = abs(k_sh - k_lg)
    p_sh, d_sh, th_sh = _bs_px_delta(spot, k_sh, t_cr, r, iv, ot_cr)
    p_lg, d_lg, th_lg = _bs_px_delta(spot, k_lg, t_cr, r, iv, ot_cr)
    cr_mid = max(0.10, p_sh - p_lg)
    half_cr = max(0.03, (p_sh + p_lg) * (option_half_spread_pct * 0.5))
    cr_cross = max(0.05, cr_mid - half_cr)
    car_cr = max(1.0, (w_cr - cr_cross) * 100.0)
    max_prof_cr = cr_cross * 100.0
    fric_cr = 2.0 * half_cr * 100.0
    q_cr_mkt = _build_lognormal_density(spot, iv, t_cr, r)
    q_cr_view = apply_view(q_cr_mkt, drift_pct=view.target_move_pct * (t_cr / t_h), vol_mult=view.vol_mult)

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
            description=f"90d {'Bull Put' if ot_cr == 'put' else 'Bear Call'} {k_sh:g}/{k_lg:g} @ ${cr_mid:.2f} cr",
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
        k_cc = _round_strike(spot * 1.05)
        t_cc = 45.0 / 365.0
        p_cc, d_cc, th_cc = _bs_px_delta(spot, k_cc, t_cc, r, iv, "call")
        half_cc = max(0.02, p_cc * option_half_spread_pct)
        cc_cross = max(0.02, p_cc - half_cc)
        fric_cc = half_cc * 100.0  # 1-leg open (usually held/expired)
        q_cc_mkt = _build_lognormal_density(spot, iv, t_cc, r)
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
                description=f"Sell 45d {k_cc:g}C (~{d_cc:.2f}Δ) against 100 sh @ ${p_cc:.2f} cr",
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
        k_opt = _round_strike(spot * (1.03 if ot == "call" else 0.97))
        t_opt = max(35.0 / 365.0, t_h)
        p_o, d_o, th_o = _bs_px_delta(spot, k_opt, t_opt, r, iv, ot)
        half_o = max(0.03, p_o * option_half_spread_pct)
        p_cross_o = p_o + half_o
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
                description=f"Buy {int(t_opt*365)}d {k_opt:g}{ot[0].upper()} (~{abs(d_o):.2f}Δ) @ ${p_o:.2f}",
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
    if results and results[0].ev_view_net_usd > 0:
        from dataclasses import replace

        results[0] = replace(results[0], recommended=True)
    return results
