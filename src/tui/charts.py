"""High-resolution Unicode Braille & Half-Block Terminal Telemetry Charts (`src/tui/charts.py`).

Pure, deterministic, dependency-free (numpy + stdlib) visual renderers designed for both
the Textual TUI (`opt`) and Rich CLI (`opt today / check / book / discover`):
  - Unified theme palette lookup honoring `settings.json` (`amber_crt`, `quant_desk`,
    `matrix_terminal`, `cyberpunk_neon`) while keeping CVD-safe good/warn/bad polarity.
  - `braille_price_chart`  — 2x4 sub-char Braille price tape with SMA200 + S/R guides.
  - `braille_sector_rrg`   — 4-Quadrant Relative Rotation Graph (Leading/Weakening/Lagging/Improving).
  - `braille_payoff_and_rnd` — Market vs Your-View RND curve + Vehicle Expiry Payoff curve.
  - `braille_ghost_race`   — 3-line normalized equity race (VM_OUTLOOK+SGOV vs SPY vs Real Book).
  - `vix_curve_strip`      — Compact 4-point VIX term-structure indicator (VIX9D..VIX6M).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.structure.types import DetailedView, VehicleComparison

# Unified theme palettes (mirrors src/tui/themes.py + src/formatting.py)
PALETTES: Dict[str, Dict[str, str]] = {
    "quant_desk": {
        "primary": "#82aad2",
        "secondary": "#5c7fa3",
        "accent": "#82aad2",
        "good": "#5ec98d",
        "warn": "#d6a452",
        "bad": "#e06c75",
        "muted": "#8b949e",
    },
    "amber_crt": {
        "primary": "#ffb03b",
        "secondary": "#b8731a",
        "accent": "#e69628",
        "good": "#5ec98d",
        "warn": "#d6a452",
        "bad": "#e06c75",
        "muted": "#a88d65",
    },
    "matrix_terminal": {
        "primary": "#50ff78",
        "secondary": "#28a745",
        "accent": "#3cdc5a",
        "good": "#5ec98d",
        "warn": "#d6a452",
        "bad": "#e06c75",
        "muted": "#6ca378",
    },
    "cyberpunk_neon": {
        "primary": "#50e6ff",
        "secondary": "#9b5de5",
        "accent": "#ff3cdc",
        "good": "#5ec98d",
        "warn": "#d6a452",
        "bad": "#e06c75",
        "muted": "#9d8ab8",
    },
}


def theme_palette(name: Optional[str] = None) -> Dict[str, str]:
    """Return active hex color tokens matching `settings.json` (default `quant_desk`)."""
    if name and name in PALETTES:
        return dict(PALETTES[name])
    try:
        from src import settings as _settings

        saved = _settings.get_theme()
        if saved in PALETTES:
            return dict(PALETTES[saved])
    except Exception:
        pass
    return dict(PALETTES["quant_desk"])


# Unicode Braille dot bit masks: (dx in 0..1, dy in 0..3)
_DOT_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


class _BrailleGrid:
    """Lightweight 2x4 sub-character Braille canvas supporting layered series colors."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols = max(8, int(cols))
        self.rows = max(3, int(rows))
        self.pw = self.cols * 2
        self.ph = self.rows * 4
        self.bits: List[List[int]] = [[0] * self.cols for _ in range(self.rows)]
        self.colors: List[List[Optional[str]]] = [[None] * self.cols for _ in range(self.rows)]

    def set_px(self, x: int, y: int, color: Optional[str] = None) -> None:
        if 0 <= x < self.pw and 0 <= y < self.ph:
            cx, rx = divmod(x, 2)
            cy, ry = divmod(y, 4)
            self.bits[cy][cx] |= _DOT_BITS[ry][rx]
            if color:
                self.colors[cy][cx] = color

    def draw_hline(self, y: int, *, step: int = 2, color: Optional[str] = None) -> None:
        for x in range(0, self.pw, max(1, step)):
            self.set_px(x, y, color=color)

    def draw_vline(self, x: int, *, step: int = 2, color: Optional[str] = None) -> None:
        for y in range(0, self.ph, max(1, step)):
            self.set_px(x, y, color=color)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Optional[str] = None) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            self.set_px(x, y, color=color)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def plot_series(
        self,
        ys: Sequence[float],
        lo: float,
        hi: float,
        *,
        color: Optional[str] = None,
        dotted: bool = False,
    ) -> None:
        vals = [float(v) for v in ys if v is not None and math.isfinite(float(v))]
        if len(vals) < 2:
            return
        span = max(hi - lo, 1e-9)
        pts: List[Tuple[int, int]] = []
        n = len(vals)
        for i, v in enumerate(vals):
            px = int(round(i * (self.pw - 1) / max(n - 1, 1)))
            py = int(round((1.0 - (v - lo) / span) * (self.ph - 1)))
            py = max(0, min(self.ph - 1, py))
            pts.append((px, py))
        for i in range(1, len(pts)):
            if dotted and (i % 2 == 0):
                continue
            self.draw_line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1], color=color)

    def render_lines(self, *, rich: bool = True) -> List[str]:
        out: List[str] = []
        for r in range(self.rows):
            if not rich:
                out.append("".join(chr(0x2800 + b) for b in self.bits[r]))
                continue
            parts: List[str] = []
            cur_col: Optional[str] = None
            buf: List[str] = []
            for c in range(self.cols):
                ch = chr(0x2800 + self.bits[r][c])
                col = self.colors[r][c] if self.bits[r][c] else None
                if col != cur_col:
                    if buf:
                        seg = "".join(buf)
                        parts.append(f"[{cur_col}]{seg}[/]" if cur_col else seg)
                        buf = []
                    cur_col = col
                buf.append(ch)
            if buf:
                seg = "".join(buf)
                parts.append(f"[{cur_col}]{seg}[/]" if cur_col else seg)
            out.append("".join(parts))
        return out


# ── 1. VIX Term-Structure Curve Strip ────────────────────────────────────────


def vix_curve_strip(
    *,
    vix9d: Optional[float] = None,
    vix: Optional[float] = None,
    vix3m: Optional[float] = None,
    vix6m: Optional[float] = None,
    rich: bool = True,
) -> str:
    """Compact 4-point VIX curve indicator showing contango vs backwardation at a glance."""
    pts = [(lbl, val) for lbl, val in (("9D", vix9d), ("1M", vix), ("3M", vix3m), ("6M", vix6m)) if val is not None]
    if not pts:
        return "VIX curve: —"
    vals = [float(v) for _, v in pts]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 0.5)
    blocks = "▁▂▃▄▅▆▇█"
    spark = "".join(blocks[min(7, int((v - lo) / span * 7.99))] for v in vals)
    is_back = (vix is not None and vix3m is not None and float(vix) > float(vix3m))
    tag = "BACKWARDATION ↘" if is_back else ("CONTANGO ↗" if (vix is not None and vix3m is not None) else "")
    pts_s = "·".join(f"{k}:{v:.1f}" for k, v in pts)
    if not rich:
        return f"VIX Curve [{spark}] {pts_s} {tag}".strip()
    pal = theme_palette()
    col = pal["bad"] if is_back else pal["good"]
    return f"[bold]VIX Curve[/] [{col}]{spark} {tag}[/] [dim]({pts_s})[/]"


# ── 2. Braille Price + SMA + S/R Levels Chart ────────────────────────────────


def braille_price_chart(
    closes: Sequence[float],
    *,
    supports: Optional[Sequence[Dict[str, Any]]] = None,
    resistances: Optional[Sequence[Dict[str, Any]]] = None,
    sma200_level: Optional[float] = None,
    width: int = 38,
    height: int = 5,
    rich: bool = True,
) -> str:
    """High-res Braille price tape with optional support/resistance and 200d reference lines."""
    vals = [float(v) for v in (closes or []) if v is not None and math.isfinite(float(v))]
    if len(vals) < 3:
        return ""
    pal = theme_palette()
    sup_lvls = [float(s["level"]) for s in (supports or [])[:1] if s.get("level") is not None]
    res_lvls = [float(r["level"]) for r in (resistances or [])[:1] if r.get("level") is not None]

    all_v = list(vals) + sup_lvls + res_lvls + ([float(sma200_level)] if sma200_level is not None else [])
    lo, hi = min(all_v), max(all_v)
    pad = max((hi - lo) * 0.06, 1e-4)
    lo -= pad
    hi += pad
    span = max(hi - lo, 1e-6)

    grid = _BrailleGrid(width, height)

    def _y(v: float) -> int:
        return max(0, min(grid.ph - 1, int(round((1.0 - (v - lo) / span) * (grid.ph - 1)))))

    if sma200_level is not None:
        grid.draw_hline(_y(float(sma200_level)), step=3, color=pal["muted"])
    for lv in sup_lvls:
        grid.draw_hline(_y(lv), step=4, color=pal["good"])
    for lv in res_lvls:
        grid.draw_hline(_y(lv), step=4, color=pal["bad"])

    up = vals[-1] >= vals[0]
    line_col = pal["good"] if up else pal["bad"]
    grid.plot_series(vals, lo, hi, color=line_col)

    rows = grid.render_lines(rich=rich)
    out: List[str] = []
    for idx, line in enumerate(rows):
        if idx == 0:
            lbl = f"${hi - pad:7.2f}"
        elif idx == len(rows) - 1:
            lbl = f"${lo + pad:7.2f}"
        elif idx == len(rows) // 2:
            lbl = f"${vals[-1]:7.2f}"
        else:
            lbl = " " * 8
        out.append(f"[dim]{lbl}[/] │{line}" if rich else f"{lbl} │{line}")
    return "\n".join(out)


# ── 3. 4-Quadrant Sector Relative Rotation Graph (RRG) ───────────────────────


def braille_sector_rrg(
    board_rows: Sequence[Any],
    *,
    width: int = 44,
    height: int = 9,
    rich: bool = True,
) -> str:
    """Plot the 16 Outlook ETFs across a 4-Quadrant RRG (Leading / Weakening / Lagging / Improving).

    X-axis = 3-month RS vs SPY (`rs_63d_pct`); Y-axis = short-term RS acceleration (`rs_21d_pct - rs_63d_pct / 3`).
    Core sleeve ETFs (`weight > 0`) are highlighted with `★`.
    """
    pts: List[Tuple[str, float, float, bool, str]] = []
    for r in board_rows:
        rs63 = getattr(r, "rs_63d_pct", None)
        rs21 = getattr(r, "rs_21d_pct", None)
        if rs63 is None or rs21 is None:
            continue
        x = float(rs63)
        y = float(rs21) - (x / 3.0)
        is_core = float(getattr(r, "weight", 0.0) or 0.0) > 0
        posture = str(getattr(r, "posture", "") or "")
        pts.append((str(getattr(r, "etf", "")), x, y, is_core, posture))
    if not pts:
        return "[dim]Sector RRG unavailable (no RS data).[/]" if rich else "Sector RRG unavailable."

    pal = theme_palette()
    max_x = max(max(abs(p[1]) for p in pts), 4.0) * 1.15
    max_y = max(max(abs(p[2]) for p in pts), 2.0) * 1.15

    # Character grid for crisp readable ticker labels inside quadrants
    w = max(36, int(width))
    h = max(7, int(height))
    canvas = [[" " for _ in range(w)] for _ in range(h)]
    colors: List[List[Optional[str]]] = [[None for _ in range(w)] for _ in range(h)]
    mid_x = w // 2
    mid_y = h // 2

    for c in range(w):
        canvas[mid_y][c] = "─"
        colors[mid_y][c] = pal["muted"]
    for r in range(h):
        canvas[r][mid_x] = "│"
        colors[r][mid_x] = pal["muted"]
    canvas[mid_y][mid_x] = "┼"

    # Place core ETFs first so they never get overwritten by non-core labels
    ordered = sorted(pts, key=lambda p: (0 if p[3] else 1, -abs(p[1])))
    occupied: set[Tuple[int, int]] = set()
    for etf, x, y, is_core, posture in ordered:
        cx = int(round(((x + max_x) / (2.0 * max_x)) * (w - 6))) + 1
        cy = int(round(((max_y - y) / (2.0 * max_y)) * (h - 1)))
        cx = max(0, min(w - 5, cx))
        cy = max(0, min(h - 1, cy))
        # Nudge vertically if slot already occupied
        for dy in (0, -1, 1, -2, 2):
            ry = max(0, min(h - 1, cy + dy))
            if ry == mid_y and h >= 7:
                continue
            tag = (f"★{etf}" if is_core else f"·{etf}")[:5]
            slots = {(ry, cx + k) for k in range(len(tag))}
            if not (slots & occupied):
                col = pal["good"] if is_core else (pal["bad"] if posture == "AVOID" else pal["primary"])
                for k, ch in enumerate(tag):
                    if cx + k < w:
                        canvas[ry][cx + k] = ch
                        colors[ry][cx + k] = col
                occupied |= slots
                break

    hdr = (
        f"[{pal['warn']}]↖ IMPROVING[/]" + " " * max(2, w - 22) + f"[{pal['good']}]LEADING ↗[/]"
        if rich
        else "↖ IMPROVING" + " " * max(2, w - 22) + "LEADING ↗"
    )
    ftr = (
        f"[{pal['bad']}]↙ LAGGING[/]" + " " * max(2, w - 22) + f"[{pal['muted']}]WEAKENING ↘[/]"
        if rich
        else "↙ LAGGING" + " " * max(2, w - 22) + "WEAKENING ↘"
    )

    lines = [hdr]
    for r in range(h):
        if not rich:
            lines.append("".join(canvas[r]))
        else:
            segs: List[str] = []
            cur_c: Optional[str] = None
            buf: List[str] = []
            for c in range(w):
                cell_col: Optional[str] = colors[r][c]
                if cell_col != cur_c:
                    if buf:
                        s = "".join(buf)
                        segs.append(f"[{cur_c}]{s}[/]" if cur_c else s)
                        buf = []
                    cur_c = cell_col
                buf.append(canvas[r][c])
            if buf:
                s = "".join(buf)
                segs.append(f"[{cur_c}]{s}[/]" if cur_c else s)
            lines.append("".join(segs))
    lines.append(ftr)
    return "\n".join(lines)


# ── 4. Workbench Braille RND + Payoff Curve ──────────────────────────────────


def braille_payoff_and_rnd(
    spot: float,
    iv: float,
    view: DetailedView,
    vehicle: Optional[VehicleComparison] = None,
    *,
    width: int = 44,
    height: int = 5,
    rich: bool = True,
) -> str:
    """Compact Braille overlay of Market RND (grey) vs Your Tilted View (green/red) across ±20% spot."""
    if spot <= 0:
        return ""
    pal = theme_palette()
    sig = max(0.05, float(iv or 0.25))
    t = max(int(view.horizon_days), 5) / 365.0
    sd = max(sig * math.sqrt(t), 0.02)
    ks = np.linspace(spot * 0.80, spot * 1.20, width * 2)

    mu_mkt = math.log(spot) - 0.5 * sig * sig * t
    pdf_mkt = np.exp(-0.5 * ((np.log(ks) - mu_mkt) / sd) ** 2) / (ks * sd * math.sqrt(2.0 * math.pi))

    shift = float(view.target_move_pct)
    sd_v = max(sd * float(view.vol_mult or 1.0), 0.02)
    mu_view = math.log(spot * max(0.3, 1.0 + shift)) - 0.5 * (sd_v ** 2)
    pdf_view = np.exp(-0.5 * ((np.log(ks) - mu_view) / sd_v) ** 2) / (ks * sd_v * math.sqrt(2.0 * math.pi))

    hi_pdf = max(float(np.max(pdf_mkt)), float(np.max(pdf_view)), 1e-9)
    grid = _BrailleGrid(width, height)
    # Center spot vertical guide
    grid.draw_vline(grid.pw // 2, step=2, color=pal["muted"])
    grid.plot_series(pdf_mkt.tolist(), 0.0, hi_pdf, color=pal["muted"], dotted=True)
    v_col = pal["good"] if shift >= 0 else pal["bad"]
    grid.plot_series(pdf_view.tolist(), 0.0, hi_pdf, color=v_col)

    rows = grid.render_lines(rich=rich)
    title = (
        f"[bold]RND Curve ({view.horizon_days}d):[/] [{pal['muted']}]┄ Market[/] vs "
        f"[{v_col}]━ View ({shift*100:+.0f}%)[/] · Spot ${spot:.2f}"
        if rich
        else f"RND Curve ({view.horizon_days}d): Market vs View ({shift*100:+.0f}%) · Spot ${spot:.2f}"
    )
    axis = f"${spot*0.80:6.1f}" + " " * max(1, width - 19) + f"▲${spot:5.1f}" + " " * max(1, width - 25) + f"${spot*1.20:6.1f}"
    lines = [title] + rows + ([f"[dim]{axis[:width]}[/]"] if rich else [axis[:width]])
    if vehicle is not None:
        v_tag = f"Selected: {vehicle.vehicle} · Max Risk ${vehicle.capital_at_risk_usd:,.0f} · View EV ${vehicle.ev_view_net_usd:+,.0f}"
        lines.append(f"[{pal['primary']}]{v_tag}[/]" if rich else v_tag)
    return "\n".join(lines)


# ── 5. Ghost Benchmark Race Braille Chart ────────────────────────────────────


def braille_ghost_race(
    race: Any,
    *,
    width: int = 52,
    height: int = 6,
    rich: bool = True,
) -> str:
    """3-line Braille equity race: Mechanical VM_OUTLOOK+SGOV vs SPY vs Your Real Book."""
    spy = list(getattr(race, "spy", None) or [])
    vm = list(getattr(race, "vm_outlook_sgov", None) or [])
    real = list(getattr(race, "real_book", None) or [])
    if len(spy) < 3 or len(vm) < 3:
        return "[dim]Ghost Benchmark Race unavailable (insufficient local OHLCV history).[/]" if rich else "Ghost Race unavailable."

    pal = theme_palette()
    all_v = spy + vm + real
    lo, hi = min(all_v), max(all_v)
    pad = max((hi - lo) * 0.08, 0.5)
    lo -= pad
    hi += pad

    grid = _BrailleGrid(width, height)
    y100 = max(0, min(grid.ph - 1, int(round((1.0 - (100.0 - lo) / max(hi - lo, 1e-6)) * (grid.ph - 1)))))
    grid.draw_hline(y100, step=3, color=pal["muted"])
    grid.plot_series(spy, lo, hi, color=pal["muted"], dotted=True)
    grid.plot_series(vm, lo, hi, color=pal["primary"])
    if len(real) >= 2:
        r_col = pal["good"] if (real[-1] >= 100.0) else pal["bad"]
        grid.plot_series(real, lo, hi, color=r_col)

    rows = grid.render_lines(rich=rich)
    vm_ret = float(getattr(race, "vm_ret_pct", 0.0))
    spy_ret = float(getattr(race, "spy_ret_pct", 0.0))
    boost = float(getattr(race, "vm_cash_only_boost_pct", 0.0))
    real_ret = getattr(race, "real_ret_pct", None)
    ov_core = getattr(race, "override_vs_core_pct", None)
    d0 = (getattr(race, "dates", None) or [""])[0]
    d1 = (getattr(race, "dates", None) or [""])[-1]

    if real_ret is not None:
        r_str = f"Real Book {real_ret:+.1f}% (vs Core {ov_core:+.1f}%)"
    else:
        r_str = "Real Book — (log picks with [l] to race)"

    hdr = (
        f"[bold]GHOST BENCHMARK RACE[/] [dim]({d0}..{d1})[/]  "
        f"[{pal['primary']}]━ VM+SGOV {vm_ret:+.1f}% (+{boost:.1f}% T-bill)[/]  "
        f"[{pal['muted']}]┄ SPY {spy_ret:+.1f}%[/]  "
        f"[{pal['good'] if (real_ret or 0) >= 0 else pal['bad']}]━ {r_str}[/]"
        if rich
        else f"GHOST RACE ({d0}..{d1}) VM+SGOV {vm_ret:+.1f}% | SPY {spy_ret:+.1f}% | {r_str}"
    )
    out = [hdr]
    for i, ln in enumerate(rows):
        lbl = f"{hi - pad:5.1f}" if i == 0 else (f"{lo + pad:5.1f}" if i == len(rows) - 1 else "  100" if i == len(rows) // 2 else "     ")
        out.append(f"[dim]{lbl}[/] │{ln}" if rich else f"{lbl} │{ln}")
    return "\n".join(out)
