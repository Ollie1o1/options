"""Leverage execution-vehicle selector (display-only).

For a single-leg directional pick, compare *buying the option* against expressing
the same directional view with a *leveraged delta-1 position* (a synthetic-equity
perp / tokenized-stock margin trade — e.g. gTrade on Base, or xStocks on Solana
collateralised on Kamino, funded with USDC/USDT).

Anchor = RISK-MATCHED: size leverage so the margin you put at risk equals the
option premium you'd otherwise pay. That equalises dollars-at-risk and makes the
comparison honest — you then see the implied leverage, where you'd get liquidated,
and the funding drag, instead of pretending leverage is "cheaper".

The switch (which vehicle wins) keys off IV richness: when the contract is RICH
(positive surface residual / short-vol-favourable VRP) you're overpaying for vol,
so a delta-1 leveraged position skips the vol tax — leverage favoured. When it's
CHEAP, the option's convexity and defined risk are worth the premium — option
favoured. Otherwise it's a toss-up (defined risk vs no theta).

Never raises; returns None when inputs are insufficient.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# Built-in defaults. Overridden by config["leverage_selector"] when supplied.
# funding_rate_daily is intentionally None: funding drifts with rates, so it is
# derived live (short rate + carry spread) rather than hardcoded. Set it to a
# positive number only to pin a manual override.
_DEFAULTS = {
    "enabled": True,
    "funding_rate_daily": None,     # None -> derive live from rate + carry_spread_annual
    "carry_spread_annual": 0.05,    # venue markup over the financing base (perp/borrow)
    "maintenance_margin": 0.005,    # liquidation buffer (fraction of notional)
    "max_leverage": 50.0,           # venue cap (gTrade-ish); clamps the implied figure
    "fallback_risk_free": 0.045,    # used only if the live rate lookup fails
}


def _live_risk_free(rate_fetcher, fallback: float) -> float:
    """Live annual risk-free rate, or fallback. Never raises."""
    try:
        if rate_fetcher is None:
            from src.data_fetching import get_risk_free_rate as rate_fetcher
        rf = float(rate_fetcher())
        if rf > 0:
            return rf
    except Exception:
        pass
    return fallback


def _daily_funding(s, rate_fetcher):
    """(daily_funding_fraction, annual_rate_used). Live unless explicitly pinned."""
    override = s.get("funding_rate_daily")
    if override is not None:
        try:
            ov = float(override)
            if ov > 0:
                return ov, ov * 365.0
        except (TypeError, ValueError):
            pass
    rf = _live_risk_free(rate_fetcher, float(s.get("fallback_risk_free", 0.045)))
    annual = rf + float(s.get("carry_spread_annual", 0.05))
    return annual / 365.0, annual


def _settings(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve settings: explicit config -> config.json -> built-in defaults."""
    s = dict(_DEFAULTS)
    block: Optional[Dict[str, Any]] = None
    if config is not None:
        block = config.get("leverage_selector")
    else:
        try:  # lazy, best-effort load of the project config
            import json
            import os
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(here, "config.json"), "r") as fh:
                block = json.load(fh).get("leverage_selector")
        except Exception:
            block = None
    if isinstance(block, dict):
        s.update({k: v for k, v in block.items() if k in _DEFAULTS})
    return s


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


_TRADING_DAYS = 252.0


def _bs_vega_per_contract(spot: float, strike: float, T: float, iv: float,
                          r: float) -> Optional[float]:
    """Black-Scholes vega for one contract (100 shares), in $ per 1.0 of
    absolute vol (i.e. multiply by a residual expressed as a vol fraction to get
    the dollar over/under-payment). None when inputs are degenerate."""
    if spot <= 0 or strike <= 0 or T <= 0 or iv <= 0:
        return None
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return spot * phi * math.sqrt(T) * 100.0


def leverage_vehicle(row: Dict[str, Any],
                     config: Optional[Dict[str, Any]] = None,
                     rate_fetcher=None) -> Optional[Dict[str, Any]]:
    """Risk-matched option-vs-leverage comparison for a single-leg pick.

    Returns a dict of {direction, leverage, capped, margin, notional, liq_price,
    liq_move_frac, funding_cost, vehicle, premium, spot} or None if the pick can't
    be evaluated (missing/invalid premium or delta, or feature disabled).
    """
    try:
        s = _settings(config)
        if not s.get("enabled", True):
            return None

        opt_type = str(row.get("type") or "").lower()
        if opt_type not in ("call", "put"):
            return None

        spot = _num(row.get("underlying"))
        premium = _num(row.get("premium"))
        delta = _num(row.get("delta"))
        if spot is None or spot <= 0:
            return None
        if premium is None or premium <= 0:
            return None
        if delta is None or abs(delta) <= 0:
            return None

        dte = max(0.0, (_num(row.get("T_years")) or 0.0) * 365.0)
        adelta = abs(delta)
        direction = "long" if opt_type == "call" else "short"

        notional = adelta * 100.0 * spot          # delta-matched share exposure
        risk_matched_margin = premium * 100.0      # same dollars at risk as the option
        implied_lev = notional / risk_matched_margin  # = |delta|*spot/premium

        max_lev = float(s["max_leverage"])
        capped = implied_lev > max_lev
        leverage = min(implied_lev, max_lev)
        # If clamped, you cannot match the option's small premium with that little
        # margin — you must post more. Report the margin actually required.
        margin = notional / leverage

        maint = float(s["maintenance_margin"])
        liq_move_frac = (1.0 / leverage) * (1.0 - maint)
        if direction == "long":
            liq_price = spot * (1.0 - liq_move_frac)
        else:
            liq_price = spot * (1.0 + liq_move_frac)

        daily_funding, funding_annual = _daily_funding(s, rate_fetcher)
        funding_cost = notional * daily_funding * dte

        # Dollar vol tax skipped by going leverage: |residual| priced through
        # BS vega. Only when IV is present (real scan rows carry it); else None
        # and the verdict falls back to the sign threshold.
        iv = _num(row.get("impliedVolatility"))
        resid = _num(row.get("iv_surface_residual"))
        vol_tax_usd = None
        if iv is not None and iv > 0 and resid is not None:
            rf = _live_risk_free(rate_fetcher, float(s.get("fallback_risk_free", 0.045)))
            vega = _bs_vega_per_contract(spot, _num(row.get("strike")) or spot,
                                         dte / 365.0, iv, rf)
            if vega is not None:
                vol_tax_usd = vega * resid

        # Liquidation safety vs the name's own daily vol: how many daily sigmas
        # of adverse move it takes to hit the liq wick. < ~1.5 is dangerous.
        liq_sigma_ratio = None
        liq_tight = False
        if iv is not None and iv > 0:
            daily_sigma = iv / math.sqrt(_TRADING_DAYS)
            if daily_sigma > 0:
                liq_sigma_ratio = liq_move_frac / daily_sigma
                liq_tight = liq_sigma_ratio < 1.5

        vehicle = _verdict(row, vol_tax_usd=vol_tax_usd, funding_cost=funding_cost)

        return {
            "direction": direction,
            "leverage": leverage,
            "capped": capped,
            "margin": margin,
            "notional": notional,
            "liq_price": liq_price,
            "liq_move_frac": liq_move_frac,
            "funding_cost": funding_cost,
            "funding_annual": funding_annual,
            "vehicle": vehicle,
            "vol_tax_usd": vol_tax_usd,
            "liq_sigma_ratio": liq_sigma_ratio,
            "liq_tight": liq_tight,
            "premium": risk_matched_margin,
            "spot": spot,
            "dte": dte,
        }
    except Exception:
        return None


def _verdict(row: Dict[str, Any], vol_tax_usd: Optional[float] = None,
             funding_cost: Optional[float] = None) -> str:
    """LEVERAGE / OPTION / TOSS-UP from IV richness, magnitude-aware.

    Primary signal is the SVI surface residual (negative = cheap, positive =
    rich). When the contract is RICH *and* we can price the trade in dollars,
    leverage only wins if the vol tax you skip (`vol_tax_usd`) exceeds the
    funding you'd pay (`funding_cost`) — a thin richness that funding would eat
    is a TOSS-UP, not a leverage call. Without dollar inputs (e.g. IV missing)
    it falls back to the sign threshold. VRP regime nudges only when the
    residual is flat/absent.
    """
    resid = _num(row.get("iv_surface_residual"))
    if resid is not None:
        if resid >= 0.01:
            if vol_tax_usd is not None and funding_cost is not None:
                return "LEVERAGE" if vol_tax_usd > funding_cost else "TOSS-UP"
            return "LEVERAGE"
        if resid <= -0.01:
            return "OPTION"
    # residual flat or missing: consult VRP regime keywords
    vrp = str(row.get("vrp_regime") or "").upper()
    if any(k in vrp for k in ("RICH", "SHORT", "SELL", "EXPENSIVE")):
        return "LEVERAGE"
    if any(k in vrp for k in ("CHEAP", "LONG", "BUY")):
        return "OPTION"
    return "TOSS-UP"


def leverage_vehicle_line(row: Dict[str, Any],
                          config: Optional[Dict[str, Any]] = None,
                          rate_fetcher=None) -> Optional[str]:
    """One-line, decision-time read comparing the option to a risk-matched
    leveraged position. Display-only; returns None when not applicable."""
    try:
        v = leverage_vehicle(row, config, rate_fetcher=rate_fetcher)
        if v is None:
            return None

        side = "long" if v["direction"] == "long" else "short"
        liq_pct = (v["liq_price"] / v["spot"] - 1.0) * 100.0
        head = (f"same ${v['premium']:,.0f} at risk = {v['leverage']:.1f}x {side}, "
                f"liq @ ${v['liq_price']:,.2f} ({liq_pct:+.1f}%)")
        if v["capped"]:
            head += f" [capped — needs ${v['margin']:,.0f} margin]"
        funding = (f"funding ~${v['funding_cost']:,.0f} over hold "
                   f"(live {v['funding_annual'] * 100:.1f}%/yr)")

        vol_tax = v.get("vol_tax_usd")
        if v["vehicle"] == "LEVERAGE":
            tail = "IV RICH -> leverage favored: skip the vol tax; risk is the liq wick, not premium"
            if vol_tax is not None:
                tail = (f"IV RICH -> leverage favored: vol tax skipped ~${vol_tax:,.0f} "
                        f"> funding ~${v['funding_cost']:,.0f}; risk is the liq wick")
        elif v["vehicle"] == "OPTION":
            tail = "IV CHEAP -> option favored: cheap convexity + defined risk"
        else:
            tail = "toss-up: option = defined risk, leverage = no theta"
            if vol_tax is not None and vol_tax > 0:
                tail = (f"toss-up: rich, but funding ~${v['funding_cost']:,.0f} eats the "
                        f"~${vol_tax:,.0f} vol tax — leverage edge is washed")

        # Liquidation-vs-volatility caveat: a leverage call is dangerous when the
        # liq wick sits inside ~1.5 daily sigma of this name's own vol.
        if v.get("liq_tight") and v.get("liq_sigma_ratio") is not None:
            tail += (f"  [liq tight: ~{v['liq_sigma_ratio']:.1f}x daily sigma to liq"
                     f"{' — size down' if v['vehicle'] == 'LEVERAGE' else ''}]")

        # Short-DTE picks often have no SVI fit; say so rather than implying a
        # surface-based read.
        if _num(row.get("iv_surface_residual")) is None:
            tail += "  (no surface fit — VRP-only read)"

        return f"Leverage read: {head}  |  {funding}  |  {tail}"
    except Exception:
        return None
