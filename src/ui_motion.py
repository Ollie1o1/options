"""Ambient ASCII-art masthead for the mode menu — and only the mode menu.

The wordmark is printed as part of the menu; while the menu waits for input a
painter thread re-styles those same rows in place with a moving shimmer band.
No information lives in the animation (no ticker bar): it is pure art, so
killing it never costs data.

Safety model: the painter only rewrites the wordmark rows, which sit `offset`
rows above the prompt (the mode list is printed between them), so the prompt
line and its echo are untouched; it breaks its loop the moment stdin has a
completed line; and it paints inside try/except so a closed stream can never
crash the app. Callers must gate on motion_allowed() — non-TTY, plain mode,
and dumb terminals get the static wordmark.
"""
import os
import select
import sys
import threading
import time

# ANSI-shadow block font, per-letter so the wordmark is assembled
# programmatically (hand-aligned multiline art rots the first time it's edited).
_FONT = {
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "P": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔═══╝ ", "██║     ", "╚═╝     "],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "D": ["██████╗ ", "██╔══██╗", "██║  ██║", "██║  ██║", "██████╔╝", "╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "K": ["██╗  ██╗", "██║ ██╔╝", "█████╔╝ ", "██╔═██╗ ", "██║  ██╗", "╚═╝  ╚═╝"],
    " ": ["   ", "   ", "   ", "   ", "   ", "   "],
}

WORDMARK = "OPTIONS DESK"
FALLBACK = "◤ OPTIONS DESK ◢"
_SHIMMER_W = 10


def _assemble(text: str) -> list:
    rows = ["".join(_FONT[ch][i] for ch in text if ch in _FONT)
            for i in range(6)]
    return rows


_ART = _assemble(WORDMARK)
_ART_W = max(len(r) for r in _ART)


def art_lines(width: int) -> list:
    """Static wordmark fitted to `width`: full block art, or a one-liner."""
    if width >= _ART_W:
        pad = " " * ((width - _ART_W) // 2)
        return [pad + r for r in _ART]
    return [FALLBACK.center(max(len(FALLBACK), width))[:width]]


def art_frame(width: int, tick: int = None) -> list:
    """One shimmer frame: the wordmark in muted ink with a bright band that
    sweeps left→right. Plain text comes back unstyled when color is off."""
    lines = art_lines(width)
    from src import formatting as fmt
    if not fmt.supports_color():
        return lines
    if tick is None:
        tick = int(time.monotonic() * 14)
    span = max(len(l) for l in lines) + _SHIMMER_W * 2
    a = tick % span - _SHIMMER_W
    b = a + _SHIMMER_W
    out = []
    for l in lines:
        lo, hi = max(0, a), max(0, min(len(l), b))
        out.append(fmt.style(l[:lo], 'muted')
                   + fmt.style(l[lo:hi], 'accent', bold=True)
                   + fmt.style(l[hi:], 'muted'))
    return out


def motion_allowed(interactive: bool) -> bool:
    if not interactive:
        return False
    try:
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
    except Exception:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    from src import formatting as fmt
    return bool(fmt.supports_color())


class HeaderMotion:
    """Repaints `n_lines` rows with frames from `frame_fn(width)`. The band's
    bottom row sits `offset` rows above the cursor row (offset=0 means the
    band is directly above the prompt)."""

    def __init__(self, n_lines: int, frame_fn, fps: int = 12, offset: int = 0):
        self.n_lines = n_lines
        self.frame_fn = frame_fn
        self.offset = offset
        self.tick = 1.0 / max(1, fps)
        self._stop = threading.Event()
        self._thread = None

    def _width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def _paint(self, lines) -> None:
        out = ["\0337"]  # save cursor
        for i, line in enumerate(lines[: self.n_lines]):
            up = self.offset + self.n_lines - i
            out.append(f"\033[{up}A\r\033[2K{line}\0338\0337")
        out.append("\0338")  # restore cursor
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            # A completed input line means the user acted: stop repainting
            # BEFORE the app prints anything below the prompt.
            try:
                ready, _, _ = select.select([sys.stdin], [], [], self.tick)
                if ready:
                    return
            except Exception:
                return
            try:
                self._paint(self.frame_fn(self._width()))
            except Exception:
                return

    def start(self) -> "HeaderMotion":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
