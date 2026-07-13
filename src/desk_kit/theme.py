"""Semantic ink tokens and the diverging heat ramp — the desk's single source.

Dark is the terminal's own palette and the default applied theme; light is the
paper research note. Both tables MUST define the same keys: a token missing
from one theme renders invisible text.

Polarity discipline (validated 2026-07-13 with the dataviz six-checks script):
the light good/bad pair passes outright; the dark pair sits in the 8-12 CVD
floor band, which is legal ONLY with secondary encoding. Every chart in
desk_kit.charts therefore encodes sign by direction from a baseline AND a
signed text label; good/bad ink is reinforcement, never the sole channel.
"""

# Role names mirror formatting._THEME_RGB.
LIGHT = {
    "paper": "#faf9f6", "ink": "#23282e", "ink_strong": "#14181c",
    "rule": "#ddd9d0", "rule_hard": "#23282e", "panel": "#f3f1ec",
    "muted": "#8a8378", "good": "#1f6f43", "bad": "#9d2c33", "warn": "#8a6516",
    # accent nudged from #8a6a2b: that value sat below the 0.1 chroma floor
    # (reads gray) and it is the primary chart-line hue in light mode.
    "accent": "#9a6a15", "grid": "#e6e2d9",
    "chip_bad_bg": "#f7eeee", "chip_bad_bd": "#c9a2a5",
    "chip_ok_bg": "#eef5f1", "chip_ok_bd": "#9dc0ac",
    "chip_wn_bg": "#f8f3e8", "chip_wn_bd": "#d4bd8c",
}

DARK = {
    "paper": "#0e1116", "ink": "#c3cad3", "ink_strong": "#f0f0f0",
    "rule": "#232830", "rule_hard": "#3a424d", "panel": "#141821",
    "muted": "#626870", "good": "#5ec98d", "bad": "#e06c75", "warn": "#d6a452",
    "accent": "#e0a54a", "grid": "#1c2129",
    "chip_bad_bg": "#1d1416", "chip_bad_bd": "#4a2327",
    "chip_ok_bg": "#101d17", "chip_ok_bd": "#1f4a35",
    "chip_wn_bg": "#1e1a11", "chip_wn_bd": "#4d3f1e",
}

# Diverging ramp anchors: (loss, neutral, gain) per theme, as RGB triples.
_RAMP = {
    "light": ((201, 139, 146), (246, 236, 236), (150, 200, 170)),
    "dark": ((143, 60, 69), (38, 42, 49), (31, 87, 56)),
}


def _lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _ink(value, span, theme_name):
    loss, neutral, gain = _RAMP[theme_name]
    try:
        v, s = float(value), float(span)
    except (TypeError, ValueError):
        return _hex(neutral)
    if s <= 0 or v != v:  # non-positive span, or NaN
        return _hex(neutral)
    t = max(-1.0, min(1.0, v / s))
    return _hex(_lerp(neutral, gain, t) if t >= 0 else _lerp(neutral, loss, -t))


def heat_inks(value, span):
    """(light_hex, dark_hex) for one heat cell on a symmetric diverging scale.

    Both inks are computed once at build time; CSS chooses. Nothing is filtered
    or inverted at view time, so red/green semantics can never flip.
    """
    return _ink(value, span, "light"), _ink(value, span, "dark")


def css_tokens() -> str:
    """The `:root` light block plus the `[data-theme="dark"]` override block."""
    def _block(table):
        return "\n".join("  --{}: {};".format(k.replace("_", "-"), v)
                         for k, v in sorted(table.items()))
    return (":root {\n" + _block(LIGHT) + "\n}\n"
            '[data-theme="dark"] {\n' + _block(DARK) + "\n}\n")
