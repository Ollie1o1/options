"""Render layer for the Intel Briefing — clean boxed terminal panels.

All width math accounts for ANSI color codes (via formatting._strip_ansi) so
colored content still aligns. Pure rendering: takes already-computed data and
returns a list of printable lines; the orchestrators (briefing/market) print.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

try:
    from src import formatting as _fmt
    _C = _fmt.Colors
    _B = _fmt.BoxChars
    _HAS = True
except Exception:  # pragma: no cover
    _fmt = None
    _HAS = False


def _vlen(s: str) -> int:
    """Visible length, ignoring ANSI escapes."""
    return len(_fmt._strip_ansi(s)) if _HAS else len(s)


# Legacy ANSI constants → semantic theme tokens, so intel panels pick up the
# quant-desk palette (truecolor when available) instead of raw 16-color codes.
_SEMANTIC = {}
if _HAS:
    _SEMANTIC = {_C.GREEN: 'good', _C.RED: 'bad', _C.YELLOW: 'warn',
                 _C.CYAN: 'accent', _C.BRIGHT_CYAN: 'heading',
                 _C.DIM: 'muted', _C.BRIGHT_WHITE: 'emph'}


def color(s: str, c: str, bold: bool = False) -> str:
    if not _HAS:
        return s
    token = _SEMANTIC.get(c)
    if token:
        return _fmt.style(s, token, bold=bold)
    return _fmt.colorize(s, c, bold=bold)


def _sparkline(values: Sequence[float]) -> str:
    """Tiny unicode sparkline of a numeric series."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return ""
    bars = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(bars[min(len(bars) - 1, int((v - lo) / rng * (len(bars) - 1)))]
                   for v in vals)


def box(title_left: str, title_right: str, body: List[str], width: int = 64) -> List[str]:
    """Draw a titled box around pre-rendered body lines.

    body lines may contain ANSI codes and a sentinel "\x00" meaning "horizontal
    divider here". Lines are padded (not truncated) to the inner width.
    """
    inner = width - 2
    if _HAS:
        tl, tr, bl, br = _B.TOP_LEFT, _B.TOP_RIGHT, _B.BOTTOM_LEFT, _B.BOTTOM_RIGHT
        h, v = _B.HORIZONTAL, _B.VERTICAL
    else:
        tl = tr = bl = br = "+"
        h, v = "-", "|"

    # Title bar:  ┌─ left ──────────── right ─┐
    left = f"{h} {title_left} " if title_left else h * 2
    right = f" {title_right} {h}" if title_right else h * 2
    mid = inner - _vlen(left) - _vlen(right)
    mid = max(0, mid)
    top = color(tl + left + h * mid + right + tr, _C.CYAN, True) if _HAS else tl + left + h * mid + right + tr

    out = [top]
    for line in body:
        if line == "\x00":
            out.append(color(v + h * inner + v, _C.CYAN) if _HAS else v + h * inner + v)
            continue
        # Safety net: a plain (un-colored) line that would overflow the border
        # gets truncated with an ellipsis. Colored lines are built to fit.
        if "\x1b" not in line and _vlen(line) > inner - 1:
            line = line[: inner - 2] + "…"
        pad = inner - _vlen(line) - 1
        pad = max(0, pad)
        content = " " + line + " " * pad
        out.append((v + content + v))
    out.append(color(bl + h * inner + br, _C.CYAN, True) if _HAS else bl + h * inner + br)
    return out


def row(label: str, value: str, label_w: int = 9, color_value: str = "") -> str:
    """A 'LABEL   value' body row with a fixed-width colored label."""
    lab = color(f"{label:<{label_w}}", _C.BRIGHT_WHITE, True) if _HAS else f"{label:<{label_w}}"
    val = color(value, color_value) if (color_value and _HAS) else value
    return f"{lab} {val}"


def verdict_color(call: str) -> str:
    if not _HAS:
        return ""
    return {"BUY": _C.GREEN, "WAIT": _C.YELLOW, "NEUTRAL": _C.YELLOW,
            "AVOID": _C.RED}.get(call, _C.YELLOW)


def direction_color(value: float) -> str:
    if not _HAS:
        return ""
    return _C.GREEN if value > 0.05 else (_C.RED if value < -0.05 else _C.YELLOW)


def wrap(text: str, width: int, indent: str = "") -> List[str]:
    """Word-wrap plain text (no ANSI) to width, with optional hanging indent."""
    words = text.split()
    lines: List[str] = []
    cur = indent
    for w in words:
        if _vlen(cur) + len(w) + 1 > width and cur.strip():
            lines.append(cur)
            cur = indent + w
        else:
            cur = (cur + " " + w) if cur.strip() else (indent + w)
    if cur.strip():
        lines.append(cur)
    return lines


SPARK = _sparkline
