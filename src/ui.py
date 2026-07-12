"""Quant-desk UI component kit.

Primitives shared by every display surface: rules, fixed-gutter kv rows,
banners, cards, tables, and meters. All width math is ANSI-aware — pad on
stripped width, then emit — so colored cells never break alignment.
"""
import math
import re
import sys
import threading

from . import formatting as fmt

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

LABEL_W = 11  # fixed label gutter for kv_line

SPINNER_FRAMES = '◐◓◑◒'  # spinning circle


def visible_len(text) -> int:
    """Printable width of text, ignoring ANSI escapes."""
    return len(_ANSI_RE.sub('', str(text)))


def _title_style(title: str) -> str:
    """Style a panel title as a heading, unless the caller already styled it.

    Callers that need a semantic color the kit doesn't own (e.g. a severity
    banner) pass pre-styled text; re-wrapping it would emit a dead escape.
    """
    title = str(title)
    return title if _ANSI_RE.search(title) else fmt.style(title, 'heading')


def clip(text, max_len: int, ellipsis: str = '…') -> str:
    """Truncate to max_len visible chars, preferring a word boundary.

    Avoids hard mid-word cuts like 'Strong R/R (2…'. Falls back to a hard
    cut only when a single word already overflows the budget.
    """
    text = str(text)
    if visible_len(text) <= max_len:
        return text
    budget = max(1, max_len - len(ellipsis))
    cut = text[:budget]
    sp = cut.rfind(' ')
    if sp >= budget * 0.6:  # only back up if it doesn't shave too much
        cut = cut[:sp]
    return cut.rstrip(' ,•|-') + ellipsis


def pad(text, width: int, align: str = 'left') -> str:
    """Pad text to a visible width. Never truncates."""
    text = str(text)
    gap = max(0, width - visible_len(text))
    if align == 'right':
        return ' ' * gap + text
    if align == 'center':
        left = gap // 2
        return ' ' * left + text + ' ' * (gap - left)
    return text + ' ' * gap


def rule(width: int, title: str = None) -> str:
    """Light horizontal rule, optionally with a left-anchored heading."""
    if not title:
        return fmt.style('─' * width, 'muted')
    prefix = '─ '
    suffix_len = max(0, width - len(prefix) - visible_len(title) - 1)
    return (fmt.style(prefix, 'muted') + _title_style(title)
            + ' ' + fmt.style('─' * suffix_len, 'muted'))


def heavy_rule(width: int, title: str = None) -> str:
    """Heavy horizontal rule (━) for pick boundaries; light rule is `rule`."""
    if not title:
        return fmt.style('━' * width, 'heading')
    prefix = '━ '
    suffix_len = max(0, width - len(prefix) - visible_len(title) - 1)
    return (fmt.style(prefix, 'heading') + fmt.style(title, 'heading')
            + ' ' + fmt.style('━' * suffix_len, 'heading'))


def tier(text) -> str:
    """Dim a depth-tier line (still visible, de-emphasized)."""
    return fmt.style(str(text), 'muted')


def kv_line(label: str, segments, indent: int = 2, sep: str = '  ') -> str:
    """Fixed-gutter labeled row: dim label, joined value segments."""
    if isinstance(segments, str):
        segments = [segments]
    body = sep.join(str(s) for s in segments if s)
    return ' ' * indent + fmt.style(pad(label, LABEL_W), 'label') + ' ' + body


def banner(title: str, context_lines=(), width: int = 100) -> str:
    """Top-level report header — the only double-line element per run."""
    t = f' {title} '
    left = max(0, (width - visible_len(t)) // 2)
    right = max(0, width - visible_len(t) - left)
    out = [fmt.style('═' * left, 'muted') + fmt.style(t, 'heading')
           + fmt.style('═' * right, 'muted')]
    for ln in context_lines:
        out.append('  ' + str(ln))
    out.append(rule(width))
    return '\n'.join(out)


def card(title: str, body_lines, width: int, boxed: bool = False,
         accent: bool = False) -> str:
    """Card: titled rule + body (+ closing rule), or a full box when boxed.

    accent colors the border only — body text keeps its own styling.
    """
    border = 'accent' if accent else 'muted'
    if not boxed:
        out = [rule(width, title)]
        out.extend(str(ln) for ln in body_lines)
        out.append(rule(width))
        return '\n'.join(out)
    inner = width - 4  # "│ " + body + " │"
    top_fill = max(0, width - visible_len(title) - 5)
    top = (fmt.style('┌─ ', border) + _title_style(title)
           + ' ' + fmt.style('─' * top_fill + '┐', border))
    v = fmt.style('│', border)
    out = [top]
    for ln in body_lines:
        out.append(f"{v} {pad(ln, inner)} {v}")
    out.append(fmt.style('└' + '─' * (width - 2) + '┘', border))
    return '\n'.join(out)


def table(cols, rows, indent: int = 2) -> str:
    """Aligned table. cols: [{'h': str, 'w': int, 'align': 'left'|'right'}].

    Cells may already contain ANSI styling.
    """
    lead = ' ' * indent
    header = lead + ' '.join(
        fmt.style(pad(c['h'], c['w'], c.get('align', 'left')), 'label', bold=True)
        for c in cols)
    total = sum(c['w'] for c in cols) + len(cols) - 1
    sep = lead + fmt.style('─' * total, 'muted')
    out = [header, sep]
    for r in rows:
        out.append(lead + ' '.join(
            pad(cell, c['w'], c.get('align', 'left'))
            for cell, c in zip(r, cols)))
    out.append(sep)
    # rstrip drops trailing padding on a left-aligned final column (no visible
    # change on screen; keeps piped output / copy-paste clean).
    return '\n'.join(line.rstrip() for line in out)


def meter(pct, width: int = 16, style_name: str = None) -> str:
    """Percentile bar: ████░░░░."""
    try:
        pct = max(0.0, min(1.0, float(pct)))
    except (TypeError, ValueError):
        pct = 0.0
    filled = int(round(pct * width))
    bar = '█' * filled + '░' * (width - filled)
    return fmt.style(bar, style_name) if style_name else bar


def _lerp(a, b, t: float):
    """Linear-interpolate two RGB triples; t in [0,1]."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _heat_prefix(value, span) -> str:
    """ANSI fg prefix on a diverging loss→neutral→gain scale, symmetric about 0.

    `span` is the |value| that saturates the color. '' when color is
    unavailable or span <= 0.
    """
    if not fmt.supports_color():
        return ''
    try:
        v = float(value)
        s = float(span)
    except (TypeError, ValueError):
        return ''
    if s <= 0:
        return ''
    t = max(-1.0, min(1.0, v / s))
    if fmt.supports_truecolor():
        neutral = fmt._THEME_RGB['muted']
        if t >= 0:
            r, g, b = _lerp(neutral, fmt._THEME_RGB['good'], t)
        else:
            r, g, b = _lerp(neutral, fmt._THEME_RGB['bad'], -t)
        return fmt.rgb_fg(r, g, b)
    if v > 0:
        return fmt._THEME_ANSI['good']
    if v < 0:
        return fmt._THEME_ANSI['bad']
    return fmt._THEME_ANSI['muted']


_HEAT_SHADES = ' ░▒▓█'


def heat_cell(text, value, span, glyph: bool = True) -> str:
    """`text` colored on the diverging heat scale.

    When glyph=True, prefix a shade char whose density tracks |value|/span.
    Plain text when color is off.
    """
    prefix = _heat_prefix(value, span)
    text = str(text)
    if glyph:
        try:
            mag = 0.0 if float(span) <= 0 else min(1.0, abs(float(value)) / float(span))
        except (TypeError, ValueError):
            mag = 0.0
        text = _HEAT_SHADES[min(4, int(round(mag * 4)))] + text
    if not prefix:
        return text
    return f"{prefix}{text}{fmt.Colors.RESET}"


_SPARK_BARS = '▁▂▃▄▅▆▇█'


def _is_finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def sparkline(series, style_name: str = None) -> str:
    """Single-line block sparkline over a numeric series.

    None/NaN render as a space. A flat series renders at mid-level. Empty → ''.
    """
    vals = list(series)
    if not vals:
        return ''
    finite = [float(v) for v in vals if _is_finite(v)]
    if not finite:
        return ' ' * len(vals)
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    out = []
    for v in vals:
        if not _is_finite(v):
            out.append(' ')
            continue
        idx = int(round((float(v) - lo) / span * (len(_SPARK_BARS) - 1)))
        out.append(_SPARK_BARS[max(0, min(len(_SPARK_BARS) - 1, idx))])
    bar = ''.join(out)
    return fmt.style(bar, style_name) if style_name else bar


def waterfall(items, bar_w: int = 28, total_label: str = 'Total'):
    """Signed contribution waterfall: each bar starts where the last one ended.

    `items` is [(label, value)]. Returns one line per item plus a total line.
    Bars share one scale spanning the running total's full excursion, so segment
    lengths are directly comparable. A non-zero item always draws >=1 block, so
    small contributors stay visible next to large ones. [] if everything is 0.
    """
    items = [(str(l), float(v)) for l, v in items]
    if not items or all(v == 0 for _, v in items):
        return []

    # Running-total excursion fixes the axis; always include the 0 baseline.
    cum, bounds = 0.0, [0.0]
    for _, v in items:
        cum += v
        bounds.append(cum)
    lo, hi = min(bounds), max(bounds)
    span = (hi - lo) or 1.0
    max_abs = max(abs(v) for _, v in items) or 1.0
    label_w = max(len(l) for l, _ in items + [(total_label, 0.0)])

    def _col(x):
        return int(round((x - lo) / span * bar_w))

    def _row(label, value, start, end, style_name=None):
        c0, c1 = _col(start), _col(end)
        left = min(c0, c1)
        fill = max(1, abs(c1 - c0)) if value else 1
        left = min(left, bar_w - fill)          # never overflow the track
        bar = ' ' * left + '█' * fill + ' ' * max(0, bar_w - left - fill)
        if style_name:
            bar = fmt.style(bar, style_name)
        else:
            bar = heat_cell(bar, value, max_abs, glyph=False)
        return f"  {pad(label, label_w)}  {bar}  {value:>+10,.0f}"

    out, run = [], 0.0
    for label, value in items:
        out.append(_row(label, value, run, run + value))
        run += value
    out.append(_row(total_label, run, 0.0, run,
                    style_name='good' if run >= 0 else 'bad'))
    return out


def _downsample(vals, n):
    """Bucket-average `vals` down to at most n points (mean per bucket)."""
    if len(vals) <= n:
        return vals
    out = []
    step = len(vals) / n
    for i in range(n):
        a = int(i * step)
        b = int((i + 1) * step) or a + 1
        chunk = vals[a:b] or vals[a:a + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def braille_chart(series, width: int = 72, height: int = 5, style_name: str = 'good'):
    """Multi-row braille line chart of a 1-D numeric series.

    Downsamples to fit width*2 pixels. Returns styled lines; [] if <2 finite
    points.
    """
    from .visual_surface import BrailleCanvas
    finite = [float(v) for v in series if _is_finite(v)]
    if len(finite) < 2:
        return []
    pts = _downsample(finite, width * 2)
    canvas = BrailleCanvas(width, height)
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    n = len(pts)
    color = ''
    if fmt.supports_color():
        rgb = fmt._THEME_RGB.get(style_name)
        if rgb and fmt.supports_truecolor():
            color = fmt.rgb_fg(*rgb)
        else:
            color = fmt._THEME_ANSI.get(style_name, '')

    def _px(i, v):
        x = int(round(i / (n - 1) * (canvas.pw - 1)))
        y = int(round((1 - (v - lo) / span) * (canvas.ph - 1)))
        return x, y

    def _seg(x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            canvas.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    prev = _px(0, pts[0])
    for i in range(1, n):
        cur = _px(i, pts[i])
        _seg(prev[0], prev[1], cur[0], cur[1])
        prev = cur
    return canvas.render_lines()


class Spinner:
    """Animated single-line progress indicator for blocking work.

    Renders a rotating circle + label on its own line, refreshed in a daemon
    thread, so the user sees motion instead of a frozen screen while data
    loads. No-op when stdout is not a TTY (keeps piped/CI/log output clean).

    Usage:
        with Spinner("Fetching VIX…"):
            vix = get_vix_level()
    """

    def __init__(self, label: str = '', frames: str = SPINNER_FRAMES,
                 interval: float = 0.1, stream=None):
        self.label = label
        self.frames = frames
        self.interval = interval
        self.stream = stream if stream is not None else sys.stdout
        self._stop = threading.Event()
        self._thread = None
        self._enabled = bool(getattr(self.stream, 'isatty', lambda: False)())

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self.frames[i % len(self.frames)]
            self.stream.write('\r  ' + fmt.style(frame, 'accent')
                              + ' ' + fmt.style(self.label, 'muted'))
            self.stream.flush()
            i += 1
            self._stop.wait(self.interval)

    def start(self) -> 'Spinner':
        if self._enabled and self._thread is None:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
        if self._enabled:
            self.stream.write('\r' + ' ' * (visible_len(self.label) + 4) + '\r')
            self.stream.flush()

    def __enter__(self) -> 'Spinner':
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False


def spinner(label: str = '', **kw) -> Spinner:
    """Convenience constructor for `Spinner` used as a context manager."""
    return Spinner(label, **kw)


def error_line(msg: str) -> str:
    """One consistent error treatment for interactive flows."""
    return fmt.style(f'  ✖ {msg}', 'bad')
