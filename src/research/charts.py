"""Research-desk SVG chart primitives. Pure, deterministic, dependency-free.

Every stroke/fill references a CSS custom property so a theme swap re-inks the
charts without a second render. Crosshair-capable charts carry class="xh" plus
data-* geometry attributes that the page JS reads; the JS lives in render.py.

Color discipline: polarity is always encoded by direction from a baseline AND
a signed text label; --good/--bad ink is reinforcement, never the sole channel.
"""
import html as _html
import json as _json

_INK = "var(--ink)"
_STRONG = "var(--ink-strong)"
_MUTED = "var(--muted)"
_GOOD = "var(--good)"
_BAD = "var(--bad)"
_ACCENT = "var(--accent)"
_RULE = "var(--rule)"
_GRID = "var(--grid)"
_MONO = "ui-monospace,Menlo,monospace"


def _finite(vals):
    return [float(v) for v in (vals or [])
            if isinstance(v, (int, float)) and v == v]


def _poly(pts):
    return " ".join("{:.1f},{:.1f}".format(x, y) for x, y in pts)


def _esc(s):
    return _html.escape(str(s), quote=True)


def _svg_open(w, h, extra=""):
    return ('<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;'
            'aspect-ratio:{w}/{h}"{x}>').format(w=w, h=h, x=extra)


def _grid_lines(x0, x1, y0, y1, lo, span, fmt):
    """Three horizontal gridlines with right-aligned axis value labels."""
    parts = []
    for frac in (0.0, 0.5, 1.0):
        gy = y1 - frac * (y1 - y0)
        parts.append('<line x1="{a}" y1="{y:.1f}" x2="{b}" y2="{y:.1f}" '
                     'stroke="{g}" stroke-width="1"/>'.format(
                         a=x0, b=x1, y=gy, g=_GRID))
        parts.append('<text x="{x}" y="{y:.1f}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="end">{v}</text>'.format(
                         x=x0 - 6, y=gy + 3, m=_MUTED, f=_MONO,
                         v=_esc(fmt.format(lo + frac * span))))
    return parts


def _crosshair(y0, y1):
    return ('<line class="ch-line" x1="0" x2="0" y1="{a}" y2="{b}" stroke="{m}" '
            'stroke-width="1" stroke-dasharray="3,3" visibility="hidden"/>'
            '<circle class="ch-dot" r="3.2" fill="{s}" visibility="hidden"/>'
            .format(a=y0, b=y1, m=_MUTED, s=_STRONG))


def area_chart(values, labels, uid, w=760, h=200, fmt="{:,.2f}"):
    """Gradient area chart with gridlines, end labels, and crosshair hooks."""
    vals = _finite(values)
    if len(vals) < 2:
        return ""
    labels = [str(x) for x in (labels or [])]
    if len(labels) != len(vals):
        labels = ["" for _ in vals]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad, axis_w = 14, 56
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 22
    step = (x1 - x0) / (len(vals) - 1)

    def _xy(i, v):
        return x0 + i * step, y1 - (v - lo) / span * (y1 - y0)

    pts = [_xy(i, v) for i, v in enumerate(vals)]
    hover = [fmt.format(v) + ((" · " + labels[i]) if labels[i] else "")
             for i, v in enumerate(vals)]
    ys = [round(y, 1) for _, y in pts]
    parts = [_svg_open(w, h,
             ' class="xh" data-x0="{x0}" data-step="{st:.4f}" '
             "data-ys='{ys}' data-labels='{lb}'".format(
                 x0=x0, st=step, ys=_esc(_json.dumps(ys)),
                 lb=_esc(_json.dumps(hover))))]
    parts.append('<defs><linearGradient id="g{u}" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0" stop-color="{c}" stop-opacity="0.22"/>'
                 '<stop offset="1" stop-color="{c}" stop-opacity="0"/>'
                 "</linearGradient></defs>".format(u=_esc(uid), c=_ACCENT))
    parts += _grid_lines(x0, x1, y0, y1, lo, span, fmt)
    parts.append('<polygon points="{p}" fill="url(#g{u})"/>'.format(
        p=_poly(pts + [(x1, y1), (x0, y1)]), u=_esc(uid)))
    parts.append('<polyline points="{p}" fill="none" stroke="{c}" '
                 'stroke-width="1.8"/>'.format(p=_poly(pts), c=_STRONG))
    lx, ly = pts[-1]
    parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}"/>'.format(
        x=lx, y=ly, c=_ACCENT))
    if labels[0]:
        parts.append('<text x="{x}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}">{t}</text>'.format(
                         x=x0, y=h - 6, m=_MUTED, f=_MONO, t=_esc(labels[0])))
    if labels[-1]:
        parts.append('<text x="{x}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="end">{t}</text>'.format(
                         x=x1, y=h - 6, m=_MUTED, f=_MONO, t=_esc(labels[-1])))
    parts.append(_crosshair(y0, y1))
    parts.append("</svg>")
    return "".join(parts)


def price_chart(closes, ma50, ma200, support, resist, labels, uid, w=760, h=280):
    """Close line + dashed MAs + support/resistance level bands + crosshair.

    The y-scale spans closes AND every drawn level/MA so a real level can
    never silently fall off-canvas.
    """
    vals = _finite(closes)
    if len(vals) < 2:
        return ""
    labels = [str(x) for x in (labels or [])]
    if len(labels) != len(vals):
        labels = ["" for _ in vals]
    m50 = _finite(ma50)[-len(vals):]
    m200 = _finite(ma200)[-len(vals):]
    levels = [lv for lv in (support, resist) if lv and lv.get("level") is not None]
    allv = vals + m50 + m200 + [float(lv["level"]) for lv in levels]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    pad, axis_w = 14, 56
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 22
    step = (x1 - x0) / (len(vals) - 1)

    def _y(v):
        return y1 - (float(v) - lo) / span * (y1 - y0)

    def _line(series, colour, dash, label):
        if len(series) < 2:
            return ""
        offset = len(vals) - len(series)
        pts = [(x0 + (offset + i) * step, _y(v)) for i, v in enumerate(series)]
        tail = ('<text x="{x:.1f}" y="{y:.1f}" font-size="9" fill="{c}" '
                'font-family="{f}">{t}</text>'.format(
                    x=min(x1 - 24, pts[-1][0] + 4), y=pts[-1][1] - 4,
                    c=colour, f=_MONO, t=_esc(label))) if label else ""
        return ('<polyline points="{p}" fill="none" stroke="{c}" '
                'stroke-width="1.2" stroke-dasharray="{d}"/>'.format(
                    p=_poly(pts), c=colour, d=dash) + tail)

    pts = [(x0 + i * step, _y(v)) for i, v in enumerate(vals)]
    hover = ["${:,.2f}".format(v) + ((" · " + labels[i]) if labels[i] else "")
             for i, v in enumerate(vals)]
    ys = [round(y, 1) for _, y in pts]
    parts = [_svg_open(w, h,
             ' class="xh" data-x0="{x0}" data-step="{st:.4f}" '
             "data-ys='{ys}' data-labels='{lb}'".format(
                 x0=x0, st=step, ys=_esc(_json.dumps(ys)),
                 lb=_esc(_json.dumps(hover))))]
    parts += _grid_lines(x0, x1, y0, y1, lo, span, "{:,.2f}")
    for lv, colour in ((resist, _BAD), (support, _GOOD)):
        if not lv or lv.get("level") is None:
            continue
        y = _y(lv["level"])
        parts.append('<rect x="{a}" y="{y:.1f}" width="{ww}" height="5" '
                     'fill="{c}" opacity="0.18"/>'.format(
                         a=x0, y=max(y0, y - 2.5), ww=x1 - x0, c=colour))
        parts.append('<text x="{x}" y="{y:.1f}" font-size="9" fill="{c}" '
                     'font-family="{f}" text-anchor="end">{t} {v:,.2f}</text>'.format(
                         x=x1, y=max(y0 + 8, y - 6), c=colour, f=_MONO,
                         t=_esc(lv.get("label", "")), v=float(lv["level"])))
    parts.append(_line(m200, _MUTED, "5,4", "200d"))
    parts.append(_line(m50, _ACCENT, "5,4", "50d"))
    parts.append('<polyline points="{p}" fill="none" stroke="{c}" '
                 'stroke-width="1.8"/>'.format(p=_poly(pts), c=_STRONG))
    if labels[0]:
        parts.append('<text x="{x}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}">{t}</text>'.format(
                         x=x0, y=h - 6, m=_MUTED, f=_MONO, t=_esc(labels[0])))
    if labels[-1]:
        parts.append('<text x="{x}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="end">{t}</text>'.format(
                         x=x1, y=h - 6, m=_MUTED, f=_MONO, t=_esc(labels[-1])))
    parts.append(_crosshair(y0, y1))
    parts.append("</svg>")
    return "".join(parts)


def rsi_strip(values, w=760, h=70):
    """RSI line with 30/70 guides and the neutral band shaded."""
    vals = _finite(values)
    if len(vals) < 2:
        return ""
    pad, axis_w = 6, 56
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 8

    def _y(v):
        return y1 - (min(100.0, max(0.0, v)) / 100.0) * (y1 - y0)

    step = (x1 - x0) / (len(vals) - 1)
    pts = [(x0 + i * step, _y(v)) for i, v in enumerate(vals)]
    parts = [_svg_open(w, h)]
    parts.append('<rect x="{a}" y="{t:.1f}" width="{ww}" height="{hh:.1f}" '
                 'fill="{g}" opacity="0.5"/>'.format(
                     a=x0, t=_y(70), ww=x1 - x0, hh=_y(30) - _y(70), g=_GRID))
    for lvl in (30, 70):
        parts.append('<line x1="{a}" y1="{y:.1f}" x2="{b}" y2="{y:.1f}" '
                     'stroke="{r}" stroke-width="1" stroke-dasharray="2,3"/>'
                     .format(a=x0, b=x1, y=_y(lvl), r=_RULE))
        parts.append('<text x="{x}" y="{y:.1f}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="end">{v}</text>'.format(
                         x=x0 - 6, y=_y(lvl) + 3, m=_MUTED, f=_MONO, v=lvl))
    parts.append('<polyline points="{p}" fill="none" stroke="{c}" '
                 'stroke-width="1.4"/>'.format(p=_poly(pts), c=_INK))
    parts.append('<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="{c}" '
                 'font-family="{f}">RSI {v:.0f}</text>'.format(
                     x=min(x1 - 46, pts[-1][0] + 5), y=pts[-1][1] + 3,
                     c=_STRONG, f=_MONO, v=vals[-1]))
    parts.append("</svg>")
    return "".join(parts)


def hbar_diverging(rows, w=560, fmt="{:+.1f}", unit=""):
    """Signed horizontal bars from a center baseline.

    Sign is encoded by bar direction AND the signed value label; good/bad ink
    is reinforcement only (the pair alone is not CVD-safe).
    """
    data = [(str(l), float(v)) for l, v in (rows or [])
            if isinstance(v, (int, float)) and v == v]
    if not data:
        return ""
    row_h, gap, label_w, val_w = 22, 6, 96, 70
    h = len(data) * (row_h + gap) + 6
    mid = label_w + (w - label_w - val_w) / 2.0
    half = (w - label_w - val_w) / 2.0 - 6
    max_abs = max(abs(v) for _, v in data) or 1.0
    parts = [_svg_open(w, h)]
    parts.append('<line x1="{m:.1f}" y1="0" x2="{m:.1f}" y2="{h}" '
                 'stroke="{r}" stroke-width="1"/>'.format(m=mid, h=h, r=_RULE))
    y = 3
    for label, v in data:
        bw = max(2.0, abs(v) / max_abs * half)
        colour = _GOOD if v >= 0 else _BAD
        bx = mid if v >= 0 else mid - bw
        cy = y + row_h / 2.0
        parts.append('<rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="{bh}" '
                     'rx="2" fill="{c}" opacity="0.85"/>'.format(
                         x=bx, y=y + 3, bw=bw, bh=row_h - 6, c=colour))
        parts.append('<text x="{x}" y="{cy:.1f}" font-size="11" fill="{i}" '
                     'font-family="{f}" text-anchor="end" '
                     'dominant-baseline="middle">{t}</text>'.format(
                         x=label_w - 8, cy=cy, i=_INK, f=_MONO, t=_esc(label)))
        parts.append('<text x="{x}" y="{cy:.1f}" font-size="11" fill="{c}" '
                     'font-family="{f}" text-anchor="end" '
                     'dominant-baseline="middle">{v}</text>'.format(
                         x=w - 4, cy=cy, c=colour, f=_MONO,
                         v=_esc(fmt.format(v) + unit)))
        y += row_h + gap
    parts.append("</svg>")
    return "".join(parts)


def cone_chart(rows, w=560, h=220):
    """Realized-vol cone: p25-p75 envelope, median line, current markers."""
    rows = [r for r in (rows or [])
            if all(isinstance(r.get(k), (int, float)) for k in
                   ("window", "p25", "median", "p75"))]
    if len(rows) < 2:
        return ""
    rows = sorted(rows, key=lambda r: r["window"])
    lows = [float(r["p25"]) for r in rows]
    highs = [float(r["p75"]) for r in rows]
    meds = [float(r["median"]) for r in rows]
    curs = [float(r["current"]) for r in rows
            if isinstance(r.get("current"), (int, float))]
    allv = lows + highs + meds + curs
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    pad, axis_w = 14, 56
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 22
    step = (x1 - x0) / (len(rows) - 1)

    def _pt(i, v):
        return x0 + i * step, y1 - (v - lo) / span * (y1 - y0)

    parts = [_svg_open(w, h)]
    parts += _grid_lines(x0, x1, y0, y1, lo, span, "{:.0%}")
    top = [_pt(i, v) for i, v in enumerate(highs)]
    bot = [_pt(i, v) for i, v in enumerate(lows)][::-1]
    parts.append('<polygon points="{p}" fill="{r}" opacity="0.5"/>'.format(
        p=_poly(top + bot), r=_RULE))
    parts.append('<polyline points="{p}" fill="none" stroke="{i}" '
                 'stroke-width="1.6"/>'.format(
                     p=_poly([_pt(i, v) for i, v in enumerate(meds)]), i=_STRONG))
    for i, r in enumerate(rows):
        cur = r.get("current")
        if isinstance(cur, (int, float)):
            cx, cy = _pt(i, float(cur))
            parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" '
                         'fill="{c}"/>'.format(x=cx, y=cy, c=_ACCENT))
        x, _ = _pt(i, meds[i])
        parts.append('<text x="{x:.1f}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="middle">{d}d</text>'.format(
                         x=x, y=h - 6, m=_MUTED, f=_MONO, d=int(r["window"])))
    parts.append("</svg>")
    return "".join(parts)


def term_chart(term, w=560, h=200):
    """ATM IV term structure: (dte, iv) curve with labelled expiry points."""
    pts_in = [(int(d), float(v)) for d, v in (term or [])
              if isinstance(v, (int, float)) and v == v]
    if len(pts_in) < 2:
        return ""
    pts_in.sort()
    ivs = [v for _, v in pts_in]
    lo, hi = min(ivs), max(ivs)
    span = (hi - lo) or 1.0
    pad, axis_w = 14, 56
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 30
    step = (x1 - x0) / (len(pts_in) - 1)
    xy = [(x0 + i * step, y1 - (v - lo) / span * (y1 - y0))
          for i, (_, v) in enumerate(pts_in)]
    parts = [_svg_open(w, h)]
    parts += _grid_lines(x0, x1, y0, y1, lo, span, "{:.0%}")
    parts.append('<polyline points="{p}" fill="none" stroke="{i}" '
                 'stroke-width="1.6"/>'.format(p=_poly(xy), i=_STRONG))
    for (x, y), (d, v) in zip(xy, pts_in):
        parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" '
                     'fill="{i}"/>'.format(x=x, y=y, i=_STRONG))
        parts.append('<text x="{x:.1f}" y="{ty}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="middle">{d}d {v:.0f}%</text>'
                     .format(x=x, ty=h - 8, m=_MUTED, f=_MONO, d=d, v=v * 100))
    parts.append("</svg>")
    return "".join(parts)
