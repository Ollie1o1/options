"""Decision-grade Trade Ticket. One block per signal — never a data dump.

Colorized through src/formatting so it matches the equity/crypto screeners.
Color is auto-disabled when stdout is not a TTY (piped / cron / tests), so the
plain-text content is unchanged in those contexts."""
from __future__ import annotations
from .signals import Signal
from .sizing import Sizing

try:
    from src import formatting as _fmt
    _C = _fmt.Colors
except Exception:  # pragma: no cover - formatting always importable in repo
    _fmt = None
    _C = None


def _c(text: str, color_attr: str, bold: bool = False) -> str:
    if _fmt is None:
        return text
    return _fmt.colorize(text, getattr(_C, color_attr), bold=bold)


def _leverage_band(eff_lev: float) -> tuple[str, str]:
    """Return (label, color_attr) describing where eff leverage sits vs the
    preferred 3-6x band (2.5x acceptable per spec; >6x is capped upstream)."""
    if 3.0 <= eff_lev <= 6.0:
        return "in 3-6x band", "BRIGHT_GREEN"
    if 2.0 <= eff_lev < 3.0:
        return "below 3-6x band", "BRIGHT_YELLOW"
    return "outside 3-6x band", "BRIGHT_YELLOW"


def render(sig: Signal, sizing: Sizing, liq_price: float, safe: bool) -> str:
    base = sig.symbol.replace("USDT", "")
    stop_pct = (sig.stop - sig.entry) / sig.entry * 100.0
    tgt_pct = (sig.target - sig.entry) / sig.entry * 100.0
    liq_dist = abs(liq_price - sig.entry) / sig.entry * 100.0
    rr = abs(tgt_pct) / abs(stop_pct) if stop_pct else 0.0
    qty = f"{sizing.qty:.4f}" if sizing.qty is not None else "n/a"

    side_color = "BRIGHT_GREEN" if sig.side == "long" else "BRIGHT_RED"
    side = _c(sig.side.upper(), side_color, bold=True)
    flag = (_c("SAFE", "BRIGHT_GREEN", bold=True) if safe
            else _c("REJECT (stop too close to liquidation)", "BRIGHT_RED", bold=True))
    band_label, band_color = _leverage_band(sizing.eff_leverage)
    band = _c(f"({band_label})", band_color)
    liq_color = "BRIGHT_GREEN" if liq_dist >= 15.0 else "BRIGHT_YELLOW"

    return "\n".join([
        f"{_c(base, 'BRIGHT_WHITE', bold=True)}  {side}  @ {sig.entry:,.0f}   "
        f"conf {sig.confidence:.2f}   session: {sig.session}",
        f"stop   {sig.stop:,.0f}  ({_c(f'{stop_pct:+.2f}%', 'BRIGHT_RED')} / "
        f"-${sizing.risk_usd:,.0f} risk, {sizing.risk_frac*100:.1f}% equity)",
        f"target {sig.target:,.0f}  ({_c(f'{tgt_pct:+.2f}%', 'BRIGHT_GREEN')})  "
        f"R:R {_c(f'{rr:.2f} : 1', 'BRIGHT_CYAN', bold=True)}   "
        f"trail after +1.5 ATR (@ {sig.trail_trigger:,.0f})",
        f"size   {qty} {base}  notional ${sizing.notional:,.0f}   "
        f"eff leverage {sizing.eff_leverage:.1f}x  {band}",
        f"liq    ~{liq_price:,.0f}  ({_c(f'{liq_dist:.1f}% away', liq_color)})  {flag}",
    ])
