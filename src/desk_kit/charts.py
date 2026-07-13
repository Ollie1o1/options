"""Desk-kit SVG chart primitives. Pure, deterministic, dependency-free.

Every stroke/fill references a CSS custom property so a theme swap re-inks the
charts without a second render. Crosshair-capable charts carry class="xh" plus
data-* geometry attributes that the shell JS reads.

Color discipline: polarity is always encoded by direction from a baseline AND
a signed text label; --good/--bad ink is reinforcement, never the sole channel
(the dark-theme pair alone is not CVD-safe). Gridlines are solid hairlines;
dashes are reserved for *semantic* lines — thresholds and model projections.
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
_PAPER = "var(--paper)"
_PANEL = "var(--panel)"
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
            '<circle class="ch-dot" r="3.6" fill="{s}" stroke="{p}" '
            'stroke-width="2" visibility="hidden"/>'
            .format(a=y0, b=y1, m=_MUTED, s=_STRONG, p=_PANEL))


def _xh_attrs(x0, step, ys, hover):
    return (' class="xh" data-x0="{x0}" data-step="{st:.4f}" '
            "data-ys='{ys}' data-labels='{lb}'".format(
                x0=x0, st=step, ys=_esc(_json.dumps(ys)),
                lb=_esc(_json.dumps(hover))))


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
    parts = [_svg_open(w, h, _xh_attrs(x0, step, ys, hover))]
    parts.append('<defs><linearGradient id="g{u}" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0" stop-color="{c}" stop-opacity="0.18"/>'
                 '<stop offset="1" stop-color="{c}" stop-opacity="0"/>'
                 "</linearGradient></defs>".format(u=_esc(uid), c=_ACCENT))
    parts += _grid_lines(x0, x1, y0, y1, lo, span, fmt)
    parts.append('<polygon points="{p}" fill="url(#g{u})"/>'.format(
        p=_poly(pts + [(x1, y1), (x0, y1)]), u=_esc(uid)))
    parts.append('<polyline points="{p}" fill="none" stroke="{c}" '
                 'stroke-width="2" stroke-linejoin="round"/>'.format(
                     p=_poly(pts), c=_STRONG))
    lx, ly = pts[-1]
    parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" '
                 'stroke="{p}" stroke-width="2"/>'.format(
                     x=lx, y=ly, c=_ACCENT, p=_PANEL))
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
    never silently fall off-canvas. MA dashes are semantic (derived series).
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
    parts = [_svg_open(w, h, _xh_attrs(x0, step, ys, hover))]
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
                 'stroke-width="2" stroke-linejoin="round"/>'.format(
                     p=_poly(pts), c=_STRONG))
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
    """RSI line with 30/70 guides and the neutral band shaded.

    The 30/70 dashes are thresholds — semantic, not grid decoration."""
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
    is reinforcement only."""
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
                 'stroke-width="2" stroke-linejoin="round"/>'.format(
                     p=_poly([_pt(i, v) for i, v in enumerate(meds)]), i=_STRONG))
    for i, r in enumerate(rows):
        cur = r.get("current")
        if isinstance(cur, (int, float)):
            cx, cy = _pt(i, float(cur))
            parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" '
                         'stroke="{p}" stroke-width="2"/>'.format(
                             x=cx, y=cy, c=_ACCENT, p=_PANEL))
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
    # wider right margin: the last point's centered "NNd NN%" label would
    # otherwise clip at the canvas edge
    x0, x1, y0, y1 = axis_w, w - 30, pad, h - 30
    step = (x1 - x0) / (len(pts_in) - 1)
    xy = [(x0 + i * step, y1 - (v - lo) / span * (y1 - y0))
          for i, (_, v) in enumerate(pts_in)]
    parts = [_svg_open(w, h)]
    parts += _grid_lines(x0, x1, y0, y1, lo, span, "{:.0%}")
    parts.append('<polyline points="{p}" fill="none" stroke="{i}" '
                 'stroke-width="2" stroke-linejoin="round"/>'.format(
                     p=_poly(xy), i=_STRONG))
    for (x, y), (d, v) in zip(xy, pts_in):
        parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{i}" '
                     'stroke="{p}" stroke-width="2"/>'.format(
                         x=x, y=y, i=_STRONG, p=_PANEL))
        parts.append('<text x="{x:.1f}" y="{ty}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="middle">{d}d {v:.0f}%</text>'
                     .format(x=x, ty=h - 8, m=_MUTED, f=_MONO, d=d, v=v * 100))
    parts.append("</svg>")
    return "".join(parts)


def sparkline(values, w=150, h=30):
    """Tiny inline trend line for table cells. No axes, no labels."""
    vals = _finite(values)
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = 3
    step = (w - 2 * pad) / (len(vals) - 1)
    pts = [(pad + i * step, h - pad - (v - lo) / span * (h - 2 * pad))
           for i, v in enumerate(vals)]
    lx, ly = pts[-1]
    return (_svg_open(w, h)
            + '<polyline points="{p}" fill="none" stroke="{i}" '
              'stroke-width="1.5"/>'.format(p=_poly(pts), i=_INK)
            + '<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{c}"/>'.format(
                x=lx, y=ly, c=_ACCENT)
            + "</svg>")


def meter(value, target, w=None):
    """Progress meter (HTML): filled share of a lighter same-ramp track.

    Fill is accent while short of target, good at/after it — a status change,
    not decoration. Returns '' when target is unusable."""
    try:
        v, t = float(value), float(target)
    except (TypeError, ValueError):
        return ""
    if t <= 0:
        return ""
    pct = max(0.0, min(1.0, v / t))
    fill = _GOOD if v >= t else _ACCENT
    return ('<div class="meter"><div class="meter-fill" '
            'style="width:{p:.0f}%;background:{c}"></div></div>').format(
                p=pct * 100, c=fill)


def _split_runs(pts_pnl):
    """Split a sampled (x, pnl) series into sign-runs with interpolated
    zero-crossings inserted, for the payoff region fills."""
    runs, cur, sign = [], [], None
    for i, (x, v) in enumerate(pts_pnl):
        s = (v > 0) - (v < 0)
        if sign is None or s == sign or s == 0:
            cur.append((x, v))
            sign = s if s != 0 else sign
            continue
        # sign flip: interpolate the crossing
        px, pv = pts_pnl[i - 1]
        t = pv / (pv - v) if (pv - v) else 0.0
        cx = px + t * (x - px)
        cur.append((cx, 0.0))
        runs.append((sign, cur))
        cur, sign = [(cx, 0.0), (x, v)], s
    if cur:
        runs.append((sign, cur))
    return runs


def payoff_chart(spot, strike, opt_type, premium, breakeven=None,
                 today_prices=None, today_pnl=None, w=760, h=250):
    """Payoff at expiry for a single-leg option, per CONTRACT (×100).

    Solid line: P&L at expiry (pure arithmetic over the inputs). Dashed accent
    line: model P&L today, when the sidecar provides it — dashed because it is
    a projection. Profit/loss regions washed good/bad at 10%; sign is also
    carried by position vs the zero axis and the signed axis labels.
    """
    try:
        sp, k, prem = float(spot), float(strike), float(premium)
    except (TypeError, ValueError):
        return ""
    if sp <= 0 or k <= 0 or prem < 0:
        return ""
    is_call = str(opt_type or "call").lower().startswith("c")
    mult = 100.0

    ladder = _finite(today_prices)
    if len(ladder) >= 2:
        ladder = sorted(ladder)
    else:
        lo_p = min(sp, k) * 0.80
        hi_p = max(sp, k) * 1.20
        n = 81
        ladder = [lo_p + i * (hi_p - lo_p) / (n - 1) for i in range(n)]

    def _expiry(p):
        intrinsic = max(0.0, p - k) if is_call else max(0.0, k - p)
        return (intrinsic - prem) * mult

    exp_pnl = [_expiry(p) for p in ladder]
    tod = _finite(today_pnl)
    if len(tod) != len(ladder):
        tod = []

    allv = exp_pnl + tod
    lo_v, hi_v = min(allv), max(allv)
    lo_v, hi_v = min(lo_v, 0.0), max(hi_v, 0.0)
    span = (hi_v - lo_v) or 1.0
    pad, axis_w = 14, 62
    x0, x1, y0, y1 = axis_w, w - pad, pad, h - 22
    p_lo, p_hi = ladder[0], ladder[-1]
    p_span = (p_hi - p_lo) or 1.0

    def _x(p):
        return x0 + (p - p_lo) / p_span * (x1 - x0)

    def _y(v):
        return y1 - (v - lo_v) / span * (y1 - y0)

    zero_y = _y(0.0)
    pts = [(_x(p), _y(v)) for p, v in zip(ladder, exp_pnl)]
    hover = ["@{:,.2f} → {:+,.0f}".format(p, v) for p, v in zip(ladder, exp_pnl)]
    ys = [round(y, 1) for _, y in pts]
    step = (x1 - x0) / (len(ladder) - 1)

    parts = [_svg_open(w, h, _xh_attrs(x0, step, ys, hover))]
    parts += _grid_lines(x0, x1, y0, y1, lo_v, span, "{:+,.0f}")

    # profit/loss washes between the expiry line and the zero axis
    for sign, run in _split_runs(list(zip([_x(p) for p in ladder], exp_pnl))):
        if sign == 0 or sign is None or len(run) < 2:
            continue
        poly = [(x, _y(v)) for x, v in run]
        poly += [(run[-1][0], zero_y), (run[0][0], zero_y)]
        parts.append('<polygon points="{p}" fill="{c}" opacity="0.10"/>'.format(
            p=_poly(poly), c=_GOOD if sign > 0 else _BAD))

    # zero axis: the one line that means break-even at expiry
    parts.append('<line x1="{a}" y1="{y:.1f}" x2="{b}" y2="{y:.1f}" '
                 'stroke="{r}" stroke-width="1"/>'.format(
                     a=x0, b=x1, y=zero_y, r="var(--rule-hard)"))

    # spot marker (threshold semantics → dashed)
    sx = _x(sp)
    if x0 <= sx <= x1:
        parts.append('<line x1="{x:.1f}" y1="{a}" x2="{x:.1f}" y2="{b}" '
                     'stroke="{m}" stroke-width="1" stroke-dasharray="3,3"/>'
                     .format(x=sx, a=y0, b=y1, m=_MUTED))
        parts.append('<text x="{x:.1f}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="middle">spot {v:,.2f}</text>'
                     .format(x=sx, y=y0 + 8, m=_MUTED, f=_MONO, v=sp))

    # model P&L today (projection → dashed accent)
    if tod:
        tpts = [(_x(p), _y(v)) for p, v in zip(ladder, tod)]
        parts.append('<polyline points="{p}" fill="none" stroke="{c}" '
                     'stroke-width="2" stroke-dasharray="5,4"/>'.format(
                         p=_poly(tpts), c=_ACCENT))
        parts.append('<text x="{x:.1f}" y="{y:.1f}" font-size="9" fill="{c}" '
                     'font-family="{f}" text-anchor="end">today</text>'.format(
                         x=x1 - 2, y=tpts[-1][1] - 5, c=_ACCENT, f=_MONO))

    # expiry payoff line
    parts.append('<polyline points="{p}" fill="none" stroke="{i}" '
                 'stroke-width="2" stroke-linejoin="round"/>'.format(
                     p=_poly(pts), i=_STRONG))

    # breakeven marker on the zero axis
    be = breakeven
    try:
        be = float(be) if be is not None else None
    except (TypeError, ValueError):
        be = None
    if be is not None and p_lo <= be <= p_hi:
        bx = _x(be)
        parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{c}" '
                     'stroke="{p}" stroke-width="2"/>'.format(
                         x=bx, y=zero_y, c=_ACCENT, p=_PANEL))
        parts.append('<text x="{x:.1f}" y="{y:.1f}" font-size="9" fill="{c}" '
                     'font-family="{f}" text-anchor="middle">BE {v:,.2f}</text>'
                     .format(x=bx, y=zero_y - 8, c=_ACCENT, f=_MONO, v=be))

    # x-axis price labels: ends + strike
    for p, anchor in ((p_lo, "start"), (k, "middle"), (p_hi, "end")):
        parts.append('<text x="{x:.1f}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="{a}">{v:,.0f}</text>'.format(
                         x=_x(p), y=h - 6, m=_MUTED, f=_MONO, a=anchor, v=p))
    parts.append(_crosshair(y0, y1))
    parts.append("</svg>")
    return "".join(parts)


def waterfall(items, w=560, h=190, fmt="{:+,.0f}"):
    """Connected column waterfall: running total, signed columns, net column.

    Sign is carried by direction from the running baseline AND the signed cap
    label; good/bad ink is reinforcement. Returns '' without items.
    """
    vals = [(str(l), float(v)) for l, v in (items or [])
            if isinstance(v, (int, float)) and v == v]
    if not vals:
        return ""
    total = sum(v for _, v in vals)
    cols = vals + [("Net", total)]
    run, levels = 0.0, [0.0]
    for _, v in vals:
        run += v
        levels.append(run)
    lo_v = min(levels + [0.0, total])
    hi_v = max(levels + [0.0, total])
    span = (hi_v - lo_v) or 1.0
    pad, axis_w, label_band = 12, 56, 30
    x0, x1, y0, y1 = axis_w, w - pad, pad + 10, h - label_band

    def _y(v):
        return y1 - (v - lo_v) / span * (y1 - y0)

    n = len(cols)
    slot = (x1 - x0) / n
    bar_w = min(24.0, slot * 0.55)
    parts = [_svg_open(w, h)]
    parts += _grid_lines(x0, x1, y0, y1, lo_v, span, fmt)
    parts.append('<line x1="{a}" y1="{y:.1f}" x2="{b}" y2="{y:.1f}" '
                 'stroke="{r}" stroke-width="1"/>'.format(
                     a=x0, b=x1, y=_y(0.0), r="var(--rule-hard)"))
    run = 0.0
    prev_edge_x = None
    for i, (label, v) in enumerate(cols):
        is_net = i == n - 1
        lo_bar, hi_bar = (0.0, v) if is_net else (run, run + v)
        if not is_net:
            run += v
        top, bot = _y(max(lo_bar, hi_bar)), _y(min(lo_bar, hi_bar))
        cx = x0 + i * slot + slot / 2.0
        bx = cx - bar_w / 2.0
        colour = _GOOD if v >= 0 else _BAD
        parts.append('<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     'height="{bh:.1f}" rx="2" fill="{c}" opacity="{o}"/>'.format(
                         x=bx, y=top, bw=bar_w, bh=max(2.0, bot - top),
                         c=colour, o="1" if is_net else "0.85"))
        if prev_edge_x is not None:
            ly = _y(lo_bar if not is_net else hi_bar)
            # connector from the previous column's finishing level
            parts.append('<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" '
                         'y2="{y:.1f}" stroke="{r}" stroke-width="1"/>'.format(
                             a=prev_edge_x, b=bx, y=_y(lo_bar) if not is_net
                             else ly, r=_RULE))
        prev_edge_x = bx + bar_w
        # signed value on the cap
        vy = top - 4 if v >= 0 else bot + 11
        parts.append('<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="{c}" '
                     'font-family="{f}" text-anchor="middle">{v}</text>'.format(
                         x=cx, y=vy, c=colour, f=_MONO, v=_esc(fmt.format(v))))
        # column label, wrapped to the slot
        words = label.split()
        line1 = words[0] if words else ""
        line2 = " ".join(words[1:])
        parts.append('<text x="{x:.1f}" y="{y}" font-size="9" fill="{m}" '
                     'font-family="{f}" text-anchor="middle">{t}</text>'.format(
                         x=cx, y=h - label_band + 13, m=_MUTED, f=_MONO,
                         t=_esc(line1)))
        if line2:
            parts.append('<text x="{x:.1f}" y="{y}" font-size="9" fill="{m}" '
                         'font-family="{f}" text-anchor="middle">{t}</text>'.format(
                             x=cx, y=h - label_band + 24, m=_MUTED, f=_MONO,
                             t=_esc(line2)))
    parts.append("</svg>")
    return "".join(parts)


def heat_grid(stress, heat_inks_fn):
    """Spot × IV P&L grid (HTML). Each cell carries BOTH theme inks; CSS picks.

    `heat_inks_fn` is theme.heat_inks — injected so this module stays free of
    a theme import cycle. Signed values ride every cell."""
    moves = (stress or {}).get("moves") or []
    rows = (stress or {}).get("rows") or []
    if not moves or not rows:
        return ""
    span = max((abs(p) for r in rows for p in r["pnls"]), default=1.0) or 1.0
    cols = "64px " + " ".join(["1fr"] * len(moves))
    out = ['<div class="heat" style="grid-template-columns:{}">'.format(cols)]
    out.append('<div class="rh"></div>')
    for m in moves:
        out.append('<div class="rh eye">{:+.0%}</div>'.format(float(m)))
    for r in rows:
        iv = float(r["iv"])
        label = "IV flat" if iv == 0 else "IV {:+.0f}pp".format(iv * 100)
        out.append('<div class="rh">{}</div>'.format(_esc(label)))
        for pnl in r["pnls"]:
            hl, hd = heat_inks_fn(pnl, span)
            v = float(pnl)
            # sub-dollar cells print unsigned "0": "-0" reads as a bug
            txt = "0" if abs(v) < 0.5 else "{:+,.0f}".format(v)
            out.append('<div class="hc" style="--hl:{hl};--hd:{hd}">'
                       "{v}</div>".format(hl=hl, hd=hd, v=txt))
    out.append("</div>")
    return "".join(out)
