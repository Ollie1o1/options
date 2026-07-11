"""Honest metrics for far-OTM lottery tickets.

The live Lottery Ticket mode used to estimate upside with a hand-wave
(``em_multiple = delta * move * 2.5``). This module replaces that with real
Black-Scholes repricing over a lognormal terminal-price distribution, so every
number on the board is defensible:

  breakeven_move_pct  how far the underlying must move (favourable direction)
                      just to break even at expiry, as a fraction of spot
  expected_move_pct   the straddle-implied expected move, same units — the
                      honest "is the breakeven even reachable" comparison
  tail_multiple       BS-repriced payoff multiple if the stock travels N
                      expected-moves partway through the option's life
  hit_probability     closed-form P[ticket returns >= hit_multiple x debit] if
                      held to expiry, under a lognormal move distribution
  edge_flag           does this pass the evidence bar (cheap IV + reachable
                      strike + a real catalyst/aligned momentum + exitable)?
  crush_trap          rich IV into an event before expiry -> shown, never picked

Everything is a pure function of explicit inputs; ``contract_read`` is the
row-adapter that extracts fields defensively and returns the whole bundle, and
is safe to call on ANY option row (not just lottery candidates).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from src.utils import bs_price, norm_cdf, safe_float


# ── config defaults (overridable via config.json "lottery_sleeve") ──────────────
DEFAULT_EDGE_CFG: Dict[str, Any] = {
    "hit_multiple": 3.0,        # a "hit" = ticket returns >= 3x the debit
    "max_iv_rank_cheap": 0.40,  # IV rank at/below this reads as "cheap"
    "rich_iv_rank": 0.70,       # IV rank at/above this reads as "rich" (crush risk)
    "max_edge_iv_rank": 0.60,   # edge disqualified above this unless realized>implied
    "max_strike_sigma": 2.5,    # Boyer-Vorkink: max-skew (deep OTM) = worst returns
    "catalyst_dte": 45,         # an earnings/event within this many days counts
    "max_spread_pct": 0.70,     # exitability gate for the edge flag
}


def _cfg(cfg: Optional[dict]) -> Dict[str, Any]:
    out = dict(DEFAULT_EDGE_CFG)
    if cfg:
        out.update({k: v for k, v in cfg.items() if k in DEFAULT_EDGE_CFG})
    return out


def _is_call(opt_type: Optional[str]) -> bool:
    return str(opt_type or "call").lower().startswith("c")


def strike_sigma(spot: float, strike: float, iv: float, t_years: float) -> Optional[float]:
    """|ln(K/S)| in units of one standard deviation of the terminal move."""
    if not spot or spot <= 0 or not strike or strike <= 0:
        return None
    if not iv or iv <= 0 or not t_years or t_years <= 0:
        return None
    unit = iv * math.sqrt(t_years)
    if unit <= 0:
        return None
    return abs(math.log(strike / spot)) / unit


def breakeven_move_pct(
    spot: float, strike: float, opt_type: str, premium: float
) -> Optional[float]:
    """Favourable move (fraction of spot) needed to break even at expiry."""
    if not spot or spot <= 0 or premium is None or premium < 0 or not strike:
        return None
    if _is_call(opt_type):
        be_price = strike + premium
        return (be_price - spot) / spot
    be_price = strike - premium
    return (spot - be_price) / spot


def hit_probability(
    spot: float,
    strike: float,
    opt_type: str,
    premium: float,
    t_years: float,
    sigma: float,
    hit_multiple: float = 3.0,
    r: float = 0.0,
    q: float = 0.0,
) -> Optional[float]:
    """P[ intrinsic at expiry >= hit_multiple * debit ] under a lognormal terminal
    price. Hold-to-expiry (no take-profit cap), so it reduces to a single normal CDF.

    ``sigma`` should be the honest physical vol — pass realized vol when you have it,
    else the option's IV. Drift is risk-neutral (r - q): no assumed directional alpha.
    """
    if (not spot or spot <= 0 or not strike or strike <= 0 or premium is None
            or premium <= 0 or not sigma or sigma <= 0 or not t_years or t_years <= 0):
        return None
    call = _is_call(opt_type)
    if call:
        barrier = strike + hit_multiple * premium          # S_T must exceed this
    else:
        barrier = strike - hit_multiple * premium          # S_T must fall below this
        if barrier <= 0:
            return 0.0                                      # unreachable (put capped at K)
    mu = r - q
    unit = sigma * math.sqrt(t_years)
    d = (math.log(spot / barrier) + (mu - 0.5 * sigma * sigma) * t_years) / unit
    return float(norm_cdf(d) if call else norm_cdf(-d))


def tail_multiple(
    spot: float,
    strike: float,
    opt_type: str,
    premium: float,
    t_years: float,
    sigma: float,
    em_dollars: float,
    n_em: float,
    r: float = 0.0,
    q: float = 0.0,
    time_elapsed_frac: float = 0.5,
) -> Optional[float]:
    """BS-repriced payoff multiple if the underlying travels ``n_em`` expected-moves
    in the favourable direction, marked with part of the option's life gone.

    Not held-to-expiry: assumes the move lands with ``time_elapsed_frac`` of the DTE
    burned and IV unchanged. That's the realistic "it ran, what's my ticket worth"
    number a trader would actually see mid-life.
    """
    if (not spot or spot <= 0 or premium is None or premium <= 0 or not sigma
            or sigma <= 0 or not t_years or t_years <= 0 or em_dollars is None
            or em_dollars <= 0):
        return None
    move = n_em * em_dollars
    s_new = spot + move if _is_call(opt_type) else max(spot - move, 0.01)
    t_rem = max(t_years * (1.0 - time_elapsed_frac), 1.0 / 365.0)
    val = bs_price(opt_type, s_new, strike, t_rem, r, sigma, q)
    if val is None:
        return None
    return float(val) / premium


def iv_state(iv_rank: Optional[float], cfg: Optional[dict] = None) -> str:
    """'cheap' | 'fair' | 'rich' from IV rank (0..1)."""
    c = _cfg(cfg)
    if iv_rank is None:
        return "fair"
    if iv_rank <= c["max_iv_rank_cheap"]:
        return "cheap"
    if iv_rank >= c["rich_iv_rank"]:
        return "rich"
    return "fair"


def crush_trap(
    iv_rank: Optional[float],
    catalyst_dte: Optional[float],
    cfg: Optional[dict] = None,
) -> str:
    """A crush trap = rich IV with an event before/near expiry. Returns a reason
    string (truthy) when tripped, else "".
    """
    c = _cfg(cfg)
    if iv_rank is None or catalyst_dte is None:
        return ""
    if iv_rank >= c["rich_iv_rank"] and 0 <= catalyst_dte <= c["catalyst_dte"]:
        return f"IV rank {iv_rank:.2f} into event in {int(catalyst_dte)}d"
    return ""


def edge_flag(
    *,
    spot: float,
    strike: float,
    opt_type: str,
    iv: float,
    t_years: float,
    iv_rank: Optional[float],
    realized_vol: Optional[float] = None,
    has_catalyst: bool = False,
    momentum_aligned: bool = False,
    spread_pct: Optional[float] = None,
    cfg: Optional[dict] = None,
) -> bool:
    """Does this long-shot clear the evidence bar? Cheap IV AND reachable strike AND
    (a real catalyst OR aligned momentum) AND exitable. Mirrors the shelved
    src/lottery/selector.py guardrails, applied to a live row.
    """
    c = _cfg(cfg)
    ks = strike_sigma(spot, strike, iv, t_years)
    if ks is None or ks > c["max_strike_sigma"]:
        return False
    cheap = False
    if iv_rank is not None and iv_rank <= c["max_edge_iv_rank"]:
        cheap = True
    if realized_vol is not None and iv is not None and realized_vol > iv:
        cheap = True
    if not cheap:
        return False
    if not (has_catalyst or momentum_aligned):
        return False
    if spread_pct is not None and spread_pct > c["max_spread_pct"]:
        return False
    return True


# ── row adapter ─────────────────────────────────────────────────────────────────
def _row(row: Any, *keys, default=None):
    """Get the first present key from a dict-like or pandas row."""
    for k in keys:
        try:
            if hasattr(row, "get"):
                v = row.get(k, None)
            elif k in getattr(row, "index", []):
                v = row[k]
            else:
                v = None
        except Exception:
            v = None
        v = safe_float(v, None) if not isinstance(v, str) else v
        if v is not None:
            return v
    return default


def contract_read(row: Any, cfg: Optional[dict] = None, play_type: str = "") -> Dict[str, Any]:
    """Full honest read for one option row. Safe on any row; missing fields degrade
    to None rather than raising. Returns the reusable bundle used by the board, the
    tearsheet, and the cross-pick 'lottery lens'.
    """
    c = _cfg(cfg)
    spot = _row(row, "underlying", "spot", "stock_price")
    strike = _row(row, "strike")
    opt_type = None
    for k in ("type", "option_type", "opt_type"):
        try:
            v = row.get(k) if hasattr(row, "get") else (row[k] if k in getattr(row, "index", []) else None)
        except Exception:
            v = None
        if isinstance(v, str) and v:
            opt_type = v
            break
    opt_type = opt_type or "call"
    premium = _row(row, "premium", "entry_price", "mid", "last")
    t_years = _row(row, "T_years", "t_years")
    if t_years is None:
        dte = _row(row, "dte", "DTE")
        t_years = (dte / 365.0) if dte else None
    iv = _row(row, "iv", "impliedVolatility", "entry_iv")
    realized_vol = _row(row, "realized_vol", "hv_30d", "hv_ewma", "hv_30", "rv_30")
    iv_rank = _row(row, "iv_rank", "iv_rank_score", "iv_percentile_30")
    em_dollars = _row(row, "expected_move", "expected_move_dollars")
    catalyst_dte = _row(row, "earnings_dte", "days_to_earnings", "catalyst_dte")
    spread_pct = _row(row, "spread_pct")
    momentum = _row(row, "momentum", "momentum_raw")

    # Catalyst can come as a numeric dte OR the pipeline's "Earnings Play" flag
    # (= the option's expiry falls beyond the earnings date, i.e. event in life).
    earnings_flag = False
    try:
        ep = row.get("Earnings Play") if hasattr(row, "get") else (
            row["Earnings Play"] if "Earnings Play" in getattr(row, "index", []) else None)
        earnings_flag = isinstance(ep, str) and ep.upper() == "YES"
    except Exception:
        earnings_flag = False
    numeric_event = catalyst_dte is not None and 0 <= catalyst_dte <= c["catalyst_dte"]
    has_catalyst = bool(numeric_event or earnings_flag)

    hm = c["hit_multiple"]
    phys_vol = realized_vol if (realized_vol and realized_vol > 0) else iv
    be = breakeven_move_pct(spot, strike, opt_type, premium) if (spot and strike and premium is not None) else None
    em_pct = (em_dollars / spot) if (em_dollars and spot) else None
    momentum_aligned = bool(
        momentum is not None and (
            (momentum > 0 and _is_call(opt_type)) or (momentum < 0 and not _is_call(opt_type))
        )
    )

    crush = crush_trap(iv_rank, catalyst_dte, c)
    if not crush and earnings_flag and iv_rank is not None and iv_rank >= c["rich_iv_rank"]:
        crush = f"IV rank {iv_rank:.2f} into earnings before expiry"

    read = {
        "play_type": play_type or "",
        "iv_state": iv_state(iv_rank, c),
        "crush_trap": crush,
        "breakeven_move_pct": be,
        "expected_move_pct": em_pct,
        "breakeven_vs_em": (be / em_pct) if (be is not None and em_pct and em_pct > 0) else None,
        "hit_prob": (
            hit_probability(spot, strike, opt_type, premium, t_years, phys_vol, hm)
            if (spot and strike and premium and t_years and phys_vol) else None
        ),
        "tail_x_1em": (
            tail_multiple(spot, strike, opt_type, premium, t_years, iv, em_dollars, 1.0)
            if (spot and strike and premium and t_years and iv and em_dollars) else None
        ),
        "tail_x_2em": (
            tail_multiple(spot, strike, opt_type, premium, t_years, iv, em_dollars, 2.0)
            if (spot and strike and premium and t_years and iv and em_dollars) else None
        ),
        "edge": (
            edge_flag(
                spot=spot, strike=strike, opt_type=opt_type, iv=iv, t_years=t_years,
                iv_rank=iv_rank, realized_vol=realized_vol, has_catalyst=has_catalyst,
                momentum_aligned=momentum_aligned, spread_pct=spread_pct, cfg=c,
            )
            if (spot and strike and iv and t_years) else False
        ),
    }
    return read


__all__ = [
    "DEFAULT_EDGE_CFG", "strike_sigma", "breakeven_move_pct", "hit_probability",
    "tail_multiple", "iv_state", "crush_trap", "edge_flag", "contract_read",
]
