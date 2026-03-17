#!/usr/bin/env python3
"""
Formatting and display utilities for enhanced CLI output.
Provides color coding, box drawing, and visual hierarchy.
"""

import os
import sys
from typing import Optional, Tuple


# ANSI Color Codes
class Colors:
    """ANSI color codes for terminal output"""
    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'

    # Reset
    RESET = '\033[0m'


class BoxChars:
    """Unicode box drawing characters"""
    # Single line
    HORIZONTAL = '─'
    VERTICAL = '│'
    TOP_LEFT = '┌'
    TOP_RIGHT = '┐'
    BOTTOM_LEFT = '└'
    BOTTOM_RIGHT = '┘'
    T_DOWN = '┬'
    T_UP = '┴'
    T_RIGHT = '├'
    T_LEFT = '┤'
    CROSS = '┼'

    # Double line
    D_HORIZONTAL = '═'
    D_VERTICAL = '║'
    D_TOP_LEFT = '╔'
    D_TOP_RIGHT = '╗'
    D_BOTTOM_LEFT = '╚'
    D_BOTTOM_RIGHT = '╝'


# Global setting for color support
_COLOR_ENABLED = None


def supports_color() -> bool:
    """
    Detect if terminal supports ANSI colors.

    Returns:
        bool: True if colors are supported
    """
    global _COLOR_ENABLED

    if _COLOR_ENABLED is not None:
        return _COLOR_ENABLED

    # Check environment
    if os.getenv('NO_COLOR'):
        _COLOR_ENABLED = False
        return False

    if os.getenv('FORCE_COLOR'):
        _COLOR_ENABLED = True
        return True

    # Check if output is a terminal
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        _COLOR_ENABLED = False
        return False

    # Check TERM environment variable
    term = os.getenv('TERM', '')
    if 'color' in term or term in ('xterm', 'xterm-256color', 'screen', 'screen-256color', 'tmux', 'tmux-256color'):
        _COLOR_ENABLED = True
        return True

    # Windows terminal check
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ANSI escape sequences on Windows 10+
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            _COLOR_ENABLED = True
            return True
        except:
            _COLOR_ENABLED = False
            return False

    _COLOR_ENABLED = False
    return False


def set_color_enabled(enabled: bool):
    """Manually enable/disable color output"""
    global _COLOR_ENABLED
    _COLOR_ENABLED = enabled


def colorize(text: str, color: str, bold: bool = False) -> str:
    """
    Add color to text if terminal supports it.

    Args:
        text: Text to colorize
        color: Color code from Colors class
        bold: Whether to make text bold

    Returns:
        Colorized text string
    """
    if not supports_color():
        return text

    prefix = color
    if bold:
        prefix = Colors.BOLD + prefix

    return f"{prefix}{text}{Colors.RESET}"


def format_metric(value: float, good_threshold: float, bad_threshold: float,
                 higher_is_better: bool = True, format_spec: str = ".2f",
                 suffix: str = "") -> str:
    """
    Format a metric with color coding based on thresholds.

    Args:
        value: Metric value
        good_threshold: Threshold for green color
        bad_threshold: Threshold for red color
        higher_is_better: True if higher values are better
        format_spec: Python format specification
        suffix: Suffix to add (e.g., '%', 'x')

    Returns:
        Color-coded formatted string
    """
    formatted = f"{value:{format_spec}}{suffix}"

    if not supports_color():
        return formatted

    if higher_is_better:
        if value >= good_threshold:
            return colorize(formatted, Colors.GREEN, bold=True)
        elif value <= bad_threshold:
            return colorize(formatted, Colors.RED)
    else:
        if value <= good_threshold:
            return colorize(formatted, Colors.GREEN, bold=True)
        elif value >= bad_threshold:
            return colorize(formatted, Colors.RED)

    return colorize(formatted, Colors.YELLOW)


def format_percentage(value: float, good: float = 60, bad: float = 45,
                     higher_is_better: bool = True) -> str:
    """
    Format a percentage value with color coding.

    Args:
        value: Value as decimal (0.5 = 50%)
        good: Good threshold (as percentage)
        bad: Bad threshold (as percentage)
        higher_is_better: True if higher is better

    Returns:
        Formatted percentage string with color
    """
    pct = value * 100
    return format_metric(pct, good, bad, higher_is_better, ".1f", "%")


def format_money(value: float, threshold_positive: float = 0) -> str:
    """
    Format a money value with color coding.

    Args:
        value: Dollar amount
        threshold_positive: Threshold above which to use green

    Returns:
        Formatted money string with color
    """
    formatted = f"${abs(value):,.2f}"

    if not supports_color():
        return formatted if value >= 0 else f"-{formatted}"

    if value > threshold_positive:
        return colorize(formatted, Colors.GREEN, bold=True)
    elif value < -threshold_positive:
        return colorize(f"-{formatted}", Colors.RED)
    else:
        return colorize(formatted if value >= 0 else f"-{formatted}", Colors.YELLOW)


def format_quality_score(score: float) -> Tuple[str, str]:
    """
    Convert quality score to stars and color.

    Args:
        score: Quality score (0-1)

    Returns:
        Tuple of (stars_string, color_code)
    """
    if score >= 0.90:
        stars = "★★★★★"
        color = Colors.BRIGHT_GREEN
    elif score >= 0.75:
        stars = "★★★★☆"
        color = Colors.GREEN
    elif score >= 0.60:
        stars = "★★★☆☆"
        color = Colors.YELLOW
    elif score >= 0.45:
        stars = "★★☆☆☆"
        color = Colors.BRIGHT_RED
    else:
        stars = "★☆☆☆☆"
        color = Colors.RED

    if not supports_color():
        return stars, ""

    return colorize(stars, color), color


def draw_box(title: str, width: int = 80, double: bool = False) -> str:
    """
    Draw a box header with title.

    Args:
        title: Box title text
        width: Total width of box
        double: Use double-line characters

    Returns:
        Box header string
    """
    if double:
        tl, tr, h, v = BoxChars.D_TOP_LEFT, BoxChars.D_TOP_RIGHT, BoxChars.D_HORIZONTAL, BoxChars.D_VERTICAL
    else:
        tl, tr, h, v = BoxChars.TOP_LEFT, BoxChars.TOP_RIGHT, BoxChars.HORIZONTAL, BoxChars.VERTICAL

    # Title with padding
    title_with_space = f" {title} "
    title_len = len(title_with_space)
    padding_left = (width - title_len - 2) // 2
    padding_right = width - title_len - 2 - padding_left

    top_line = tl + h * padding_left + title_with_space + h * padding_right + tr

    if supports_color():
        return colorize(top_line, Colors.BRIGHT_BLUE, bold=True)

    return top_line


def draw_separator(width: int = 80, char: str = None) -> str:
    """
    Draw a horizontal separator line.

    Args:
        width: Line width
        char: Character to use (default: ─)

    Returns:
        Separator line string
    """
    char = char or BoxChars.HORIZONTAL
    line = char * width

    if supports_color():
        return colorize(line, Colors.BRIGHT_BLACK)

    return line


def format_header(text: str, emoji: str = "") -> str:
    """
    Format a section header with optional emoji.

    Args:
        text: Header text
        emoji: Optional emoji prefix

    Returns:
        Formatted header string
    """
    header_text = f"{emoji} {text}" if emoji else text

    if supports_color():
        return colorize(header_text, Colors.BRIGHT_CYAN, bold=True)

    return header_text


def format_warning(text: str) -> str:
    """Format a warning message."""
    if supports_color():
        return colorize(f"⚠️  {text}", Colors.YELLOW, bold=True)
    return f"⚠️  {text}"


def format_error(text: str) -> str:
    """Format an error message."""
    if supports_color():
        return colorize(f"❌ {text}", Colors.RED, bold=True)
    return f"❌ {text}"


def format_success(text: str) -> str:
    """Format a success message."""
    if supports_color():
        return colorize(f"✓ {text}", Colors.GREEN, bold=True)
    return f"✓ {text}"


def format_info(text: str) -> str:
    """Format an info message."""
    if supports_color():
        return colorize(f"ℹ️  {text}", Colors.BRIGHT_BLUE)
    return f"ℹ️  {text}"


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_len: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def align_columns(data: list, widths: list, separators: str = " ") -> str:
    """
    Align data into columns with specified widths.

    Args:
        data: List of column values
        widths: List of column widths
        separators: Separator between columns

    Returns:
        Aligned row string
    """
    aligned = []
    for value, width in zip(data, widths):
        str_val = str(value)
        aligned.append(str_val.ljust(width))

    return separators.join(aligned)


# Presets for common scenarios
def format_pop(pop: float) -> str:
    """Format probability of profit"""
    return format_percentage(pop, good=60, bad=45, higher_is_better=True)


def format_spread(spread: float) -> str:
    """Format bid-ask spread"""
    return format_percentage(spread, good=0.10, bad=0.20, higher_is_better=False)


def format_rr(rr: float) -> str:
    """Format risk/reward ratio"""
    return format_metric(rr, 1.5, 0.7, True, ".1f", "x")


def format_ev(ev: float) -> str:
    """Format expected value"""
    return format_money(ev, threshold_positive=10)


def format_iv_rank_bar(iv_pct: float, hv: float, iv: float, width: int = 20, iv_confidence: str = "") -> str:
    """
    Format IV percentile as a visual bar with HV/IV ratio and regime label.

    Args:
        iv_pct: IV percentile (0.0-1.0)
        hv: Historical volatility (0.0-1.0, e.g. 0.28 = 28%)
        iv: Implied volatility (0.0-1.0, e.g. 0.31 = 31%)
        width: Bar character width

    Returns:
        Formatted IV rank bar string
    """
    try:
        iv_pct = max(0.0, min(1.0, float(iv_pct) if iv_pct is not None else 0.5))
    except (TypeError, ValueError):
        iv_pct = 0.5

    filled = int(round(iv_pct * width))
    empty = width - filled
    bar = '\u2588' * filled + '\u2591' * empty
    pct_label = f"{iv_pct * 100:.0f}%ile"

    # Regime
    if iv_pct >= 0.90:
        regime = "EXTREME"
        regime_color = Colors.RED
        regime_bold = True
    elif iv_pct >= 0.70:
        regime = "HIGH"
        regime_color = Colors.BRIGHT_RED
        regime_bold = False
    elif iv_pct >= 0.20:
        regime = "NORMAL"
        regime_color = Colors.YELLOW
        regime_bold = False
    else:
        regime = "LOW"
        regime_color = Colors.BRIGHT_GREEN
        regime_bold = False

    # HV/IV ratio and cheap/rich label
    ratio_str = ""
    cheap_rich = ""
    try:
        hv_f = float(hv) if hv else 0.0
        iv_f = float(iv) if iv else 0.0
        if hv_f > 0 and iv_f > 0:
            ratio = iv_f / hv_f
            ratio_str = f"  HV:{hv_f*100:.0f}% IV:{iv_f*100:.0f}% ratio:{ratio:.2f}"
            if ratio < 0.90:
                cheap_rich = "  " + (colorize("CHEAP", Colors.GREEN) if supports_color() else "CHEAP")
            elif ratio > 1.20:
                cheap_rich = "  " + (colorize("RICH", Colors.RED) if supports_color() else "RICH")
    except (TypeError, ValueError):
        pass

    if supports_color():
        bar_str = colorize(bar, regime_color)
        regime_str = colorize(f"[{regime}]", regime_color, bold=regime_bold)
    else:
        bar_str = bar
        regime_str = f"[{regime}]"

    conf_suffix = ""
    if iv_confidence == "Low":
        conf_suffix = "  ⚠ Low Conf" if not supports_color() else "  " + colorize("⚠ Low Conf", Colors.DIM)

    return f"IV: {bar_str} {pct_label}{ratio_str}  {regime_str}{cheap_rich}{conf_suffix}"


def format_delta(delta: float, is_call: bool = True) -> str:
    """Format delta value"""
    # Show alignment-aware sign: + when directionally expected, - otherwise
    aligned = (is_call and delta > 0) or (not is_call and delta < 0)
    formatted = f"+{abs(delta):.2f}" if aligned else f"{delta:.2f}"

    if not supports_color():
        return formatted

    # Color based on absolute magnitude
    abs_delta = abs(delta)
    if abs_delta > 0.70:
        return colorize(formatted, Colors.BRIGHT_GREEN)
    elif abs_delta < 0.20:
        return colorize(formatted, Colors.RED)
    else:
        return colorize(formatted, Colors.YELLOW)


__all__ = [
    'Colors',
    'BoxChars',
    'supports_color',
    'set_color_enabled',
    'colorize',
    'format_metric',
    'format_percentage',
    'format_money',
    'format_quality_score',
    'draw_box',
    'draw_separator',
    'format_header',
    'format_warning',
    'format_error',
    'format_success',
    'format_info',
    'truncate',
    'align_columns',
    'format_pop',
    'format_spread',
    'format_rr',
    'format_ev',
    'format_delta',
    'format_iv_rank_bar',
]
