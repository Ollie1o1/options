"""Quant-desk UI component kit.

Primitives shared by every display surface: rules, fixed-gutter kv rows,
banners, cards, tables, and meters. All width math is ANSI-aware — pad on
stripped width, then emit — so colored cells never break alignment.
"""
import re

from . import formatting as fmt

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')

LABEL_W = 11  # fixed label gutter for kv_line


def visible_len(text) -> int:
    """Printable width of text, ignoring ANSI escapes."""
    return len(_ANSI_RE.sub('', str(text)))


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
    return '\n'.join(out)


def meter(pct, width: int = 16, style_name: str = None) -> str:
    """Percentile bar: ████░░░░."""
    try:
        pct = max(0.0, min(1.0, float(pct)))
    except (TypeError, ValueError):
        pct = 0.0
    filled = int(round(pct * width))
    bar = '█' * filled + '░' * (width - filled)
    return fmt.style(bar, style_name) if style_name else bar
