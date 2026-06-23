"""Quant-desk UI component kit.

Primitives shared by every display surface: rules, fixed-gutter kv rows,
banners, cards, tables, and meters. All width math is ANSI-aware — pad on
stripped width, then emit — so colored cells never break alignment.
"""
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
    return (fmt.style(prefix, 'muted') + fmt.style(title, 'heading')
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
    top = (fmt.style('┌─ ', border) + fmt.style(title, 'heading')
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
