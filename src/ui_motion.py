"""Ambient menu-header motion: a scrolling ticker tape repainted in place
while the mode menu waits for input.

Safety model: the painter thread only ever rewrites the N lines directly
ABOVE the cursor (ANSI save/restore), so the prompt line and its echo are
untouched; it breaks its loop the moment stdin has a completed line; and it
paints inside try/except so a closed stream can never crash the app. Callers
must gate on motion_allowed() — non-TTY, plain mode, and dumb terminals get
the static header.
"""
import os
import select
import sys
import threading

_TAGLINE = ("options desk — quotes 15+ min delayed · display-only until "
            "the gate fires")
_SEP = " · "

_tape_segments: list = []
_tape_lock = threading.Lock()


def set_tape(segments) -> None:
    with _tape_lock:
        _tape_segments[:] = [s for s in (segments or []) if s]


def tape_text() -> str:
    with _tape_lock:
        segs = list(_tape_segments)
    return _SEP.join(segs) if segs else _TAGLINE


def tape_frame(offset: int, width: int) -> str:
    """A `width`-char window into the looped tape, shifted by `offset`."""
    text = tape_text() + _SEP
    if not text.strip():
        text = _TAGLINE + _SEP
    doubled = text * (2 + width // max(1, len(text)))
    start = offset % len(text)
    return doubled[start:start + width]


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
    """Repaints `n_lines` above the cursor with frames from `frame_fn(width)`.

    frame_fn returns a list of exactly n_lines strings (pre-styled OK); each
    is clipped to the terminal width by the painter.
    """

    def __init__(self, n_lines: int, frame_fn, fps: int = 8):
        self.n_lines = n_lines
        self.frame_fn = frame_fn
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
            up = self.n_lines - i
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
                width = self._width()
                self._paint([l[:width] for l in self.frame_fn(width)])
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
