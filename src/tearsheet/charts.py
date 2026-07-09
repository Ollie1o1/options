"""Inline-SVG chart primitives. Pure, deterministic, dependency-free.

Every stroke/fill references a CSS custom property, so a theme swap re-inks the
charts with no second render and no JS redraw.
"""
import html as _html

_INK = "var(--ink)"
_MUTED = "var(--muted)"
_GOOD = "var(--good)"
_BAD = "var(--bad)"
_RULE = "var(--rule)"


def _finite(vals):
    return [float(v) for v in vals
            if v is not None and isinstance(v, (int, float)) and v == v]


def _scale(vals, w, h, pad=10):
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    step = (w - 2 * pad) / max(1, n - 1)
    return [(pad + i * step, h - pad - (v - lo) / span * (h - 2 * pad))
            for i, v in enumerate(vals)]


def _poly(pts):
    return " ".join("{:.1f},{:.1f}".format(x, y) for x, y in pts)


def line_chart(series, w=300, h=90):
    """Single polyline over a numeric series. '' if fewer than 2 finite points."""
    vals = _finite(series)
    if len(vals) < 2:
        return ""
    pts = _scale(vals, w, h)
    return (
        '<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px">'
        '<polyline points="{p}" fill="none" stroke="{ink}" stroke-width="1.5"/>'
        "</svg>"
    ).format(w=w, h=h, p=_poly(pts), ink=_INK)


def price_with_bands(closes, supports, resistances, w=300, h=104):
    """Price polyline with support (good) and resistance (bad) bands.

    The y-scale spans the closes AND every drawn level. Scaling to the closes
    alone pushes a level outside the visible range off-canvas, which would make
    a real support line silently disappear.
    """
    vals = _finite(closes)
    if len(vals) < 2:
        return ""
    pad = 10
    bands = [(resistances or [])[:2], (supports or [])[:2]]
    levels = [float(lv["level"]) for band in bands for lv in band]
    lo, hi = min(vals + levels), max(vals + levels)
    span = (hi - lo) or 1.0

    def _y(v):
        return h - pad - (float(v) - lo) / span * (h - 2 * pad)

    n = len(vals)
    step = (w - 2 * pad) / max(1, n - 1)
    pts = [(pad + i * step, _y(v)) for i, v in enumerate(vals)]

    parts = ['<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px">'.format(w=w, h=h)]
    for band, colour in zip(bands, (_BAD, _GOOD)):
        for lv in band:
            y = _y(lv["level"])
            parts.append(
                '<rect x="{x}" y="{y:.1f}" width="{ww}" height="6" fill="{c}" opacity="0.16"/>'
                '<text x="{tx}" y="{ty:.1f}" font-size="7" fill="{c}" '
                'font-family="monospace" text-anchor="end">{lbl} {lev:.2f}</text>'.format(
                    x=pad, y=max(0, y - 3), ww=w - 2 * pad, c=colour, tx=w - pad,
                    ty=max(7, y - 5), lbl=_html.escape(str(lv["label"])),
                    lev=float(lv["level"])))
    parts.append('<polyline points="{p}" fill="none" stroke="{ink}" stroke-width="1.5"/>'
                 .format(p=_poly(pts), ink=_INK))
    parts.append("</svg>")
    return "".join(parts)


def vol_cone(cone, current_iv, w=300, h=96):
    """p25-p75 envelope + median line + a marker for current IV. '' if no cone."""
    if not cone:
        return ""
    rows = sorted(cone, key=lambda r: r["window"])
    lows = [float(r["p25"]) for r in rows]
    highs = [float(r["p75"]) for r in rows]
    meds = [float(r["median"]) for r in rows]
    allv = lows + highs + meds + ([float(current_iv)] if current_iv else [])
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or 1.0
    pad = 10
    n = len(rows)
    step = (w - 2 * pad) / max(1, n - 1)

    def _pt(i, v):
        return pad + i * step, h - pad - (v - lo) / span * (h - 2 * pad)

    top = [_pt(i, v) for i, v in enumerate(highs)]
    bot = [_pt(i, v) for i, v in enumerate(lows)][::-1]
    parts = ['<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px">'.format(w=w, h=h)]
    parts.append('<polygon points="{p}" fill="{r}" opacity="0.55"/>'.format(
        p=_poly(top + bot), r=_RULE))
    parts.append('<polyline points="{p}" fill="none" stroke="{ink}" stroke-width="1.4"/>'
                 .format(p=_poly([_pt(i, v) for i, v in enumerate(meds)]), ink=_INK))
    if current_iv:
        cx, cy = _pt(n // 2, float(current_iv))
        parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{c}"/>'.format(
            x=cx, y=cy, c=_BAD))
    for i, r in enumerate(rows):
        x, _ = _pt(i, meds[i])
        parts.append('<text x="{x:.1f}" y="{y}" font-size="7" fill="{m}" '
                     'font-family="monospace" text-anchor="middle">{d}d</text>'.format(
                         x=x, y=h - 1, m=_MUTED, d=int(r["window"])))
    parts.append("</svg>")
    return "".join(parts)


def term_curve(term, w=300, h=96):
    """(dte, iv) curve with a labelled point per expiry. '' if fewer than 2."""
    pts = [(int(d), float(v)) for d, v in (term or []) if v is not None]
    if len(pts) < 2:
        return ""
    pts.sort()
    ivs = [v for _, v in pts]
    lo, hi = min(ivs), max(ivs)
    span = (hi - lo) or 1.0
    pad, n = 14, len(pts)
    step = (w - 2 * pad) / max(1, n - 1)
    xy = [(pad + i * step, h - 26 - (v - lo) / span * (h - 44))
          for i, (_, v) in enumerate(pts)]
    parts = ['<svg viewBox="0 0 {w} {h}" style="width:100%;height:{h}px">'.format(w=w, h=h)]
    parts.append('<polyline points="{p}" fill="none" stroke="{ink}" stroke-width="1.4"/>'
                 .format(p=_poly(xy), ink=_INK))
    for (x, y), (d, v) in zip(xy, pts):
        parts.append('<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{ink}"/>'.format(
            x=x, y=y, ink=_INK))
        parts.append('<text x="{x:.1f}" y="{ty}" font-size="7" fill="{m}" '
                     'font-family="monospace" text-anchor="middle">{d}d {v:.0%}</text>'.format(
                         x=x, ty=h - 6, m=_MUTED, d=d, v=v))
    parts.append("</svg>")
    return "".join(parts)


def waterfall_bars(items, w=260):
    """HTML (not SVG) signed bars for the cost waterfall, plus a total row."""
    vals = [(str(l), float(v)) for l, v in (items or [])]
    if not vals:
        return ""
    max_abs = max(abs(v) for _, v in vals) or 1.0
    out = ['<div class="wf">']
    for label, v in vals:
        width = max(2, int(round(abs(v) / max_abs * w)))
        colour = _GOOD if v >= 0 else _BAD
        out.append(
            '<div><span class="wfl">{l}</span>'
            '<span class="wfbar" style="width:{px}px;background:{c}"></span>'
            '<span class="wfv" style="color:{c}">{v:+,.0f}</span></div>'.format(
                l=_html.escape(label), px=width, c=colour, v=v))
    total = sum(v for _, v in vals)
    tc = _GOOD if total >= 0 else _BAD
    out.append('<div class="wftot"><span class="wfl"><strong>Net EV</strong></span>'
               '<span class="wfv" style="color:{c}"><strong>{v:+,.0f}</strong></span></div>'
               .format(c=tc, v=total))
    out.append("</div>")
    return "".join(out)
