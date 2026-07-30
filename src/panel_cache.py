"""Cache-first rendering for startup panels.

Time-to-menu was ~11s, nearly all of it live market fetches repeated on every
launch: the world pulse (3.54s) and the VIX/regime fetch (1.09s). The
sector-outlook box already served itself from cache; this generalises that so
any print-style panel can do the same.

The contract:

- A panel is cached as its *rendered text*, keyed by (panel, width). Caching the
  text rather than the underlying data means a panel can be cached without
  understanding its internals, and a width change can never replay a box drawn
  to the wrong size.
- Every entry stores when it was produced, so the caller can stamp
  "as of HH:MM". Stale market data must never be presented as live.
- A stale entry is still served instantly; the refresh happens behind the user.

**Refresh runs in a detached subprocess, never a thread.** ``redirect_stdout``
replaces the *global* ``sys.stdout``, so a thread capturing a panel while the
main thread prints the menu swallows the menu — the blank-UI race this repo hit
once already (see the comment above the regime dashboard call site). A
subprocess has its own stdout and cannot interfere.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Callable, Optional, Tuple

DEFAULT_TTL = 300  # 5 minutes
DEFAULT_CACHE_DIR = os.path.join("logs", "panel_cache")

def _render_regime_dashboard(width: int) -> None:
    from src.regime_dashboard import print_regime_dashboard
    print_regime_dashboard(width)


# Panels the refresh subprocess knows how to re-render, by cache key.
#
# These MUST be defined here, not registered by importers. The refresh runs as
# `python -m src.panel_cache`, which imports only this module — a registry
# populated from the screener would be empty in that process, so every
# background refresh would silently do nothing and the cache would freeze at
# its first render. The imports inside each renderer stay lazy so importing
# this module costs nothing at startup.
_RENDERERS: dict = {
    "regime_dashboard": _render_regime_dashboard,
}


def panel_renderer(key: str):
    """The registered renderer for a cache key, or None."""
    return _RENDERERS.get(key)


def _path(key: str, width: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"{key}_{int(width)}.json")


def load_panel(key: str, width: int, cache_dir: str = DEFAULT_CACHE_DIR):
    """Return ``(text, asof)`` for a cached panel, or None if unusable.

    Any problem — missing, unparseable, missing or invalid timestamp — is a miss
    rather than an error: a bad cache file must never break startup.
    """
    try:
        with open(_path(key, width, cache_dir)) as f:
            blob = json.load(f)
        return blob["text"], datetime.fromisoformat(blob["asof"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def store_panel(key: str, width: int, text: str, asof: datetime,
                cache_dir: str = DEFAULT_CACHE_DIR) -> None:
    """Write a rendered panel atomically.

    A refresh subprocess can be killed mid-write; writing to a temp file in the
    same directory and renaming means a half-written file never becomes the
    cache, because rename is atomic within a filesystem.
    """
    try:
        os.makedirs(cache_dir, exist_ok=True)
        blob = {"text": text, "asof": asof.isoformat(), "width": int(width)}
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(blob, f)
            os.replace(tmp, _path(key, width, cache_dir))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except (OSError, ValueError):
        pass  # An uncacheable panel is a slow panel, not a broken one.


def _spawn_refresh(key: str, width: int) -> None:
    """Re-render a panel in a detached subprocess (see module docstring)."""
    with contextlib.suppress(Exception):
        subprocess.Popen(
            [sys.executable, "-m", "src.panel_cache", "--refresh", key,
             "--width", str(int(width))],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )


def render_cached(
    key: str,
    width: int,
    renderer: Callable[[], None],
    ttl: int = DEFAULT_TTL,
    cache_dir: str = DEFAULT_CACHE_DIR,
    now: Optional[datetime] = None,
    spawn_refresh: Optional[Callable[[str, int], None]] = None,
) -> Tuple[str, Optional[datetime], bool]:
    """Render a panel cache-first.

    Returns ``(text, asof, from_cache)``:

    - fresh cache  -> stored text, no render, no refresh
    - stale cache  -> stored text *immediately*, refresh spawned behind the user
    - no cache     -> render inline (the user has to wait once), then store

    ``asof`` is None only when there was no cache and the render failed.
    """
    now = now or datetime.now()
    spawn_refresh = spawn_refresh or _spawn_refresh

    cached = load_panel(key, width, cache_dir)
    if cached is not None:
        text, asof = cached
        if (now - asof).total_seconds() > ttl:
            spawn_refresh(key, width)
        return text, asof, True

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            renderer()
    except Exception:
        return "", None, False

    text = buf.getvalue()
    store_panel(key, width, text, now, cache_dir)
    return text, now, False


def asof_note(asof: Optional[datetime], now: Optional[datetime] = None,
              min_age: int = 30) -> str:
    """A short "as of HH:MM" note, or "" when the data is effectively live.

    Suppressed under ``min_age`` so a just-rendered panel is not labelled with
    the current time, which would be noise.
    """
    if asof is None:
        return ""
    now = now or datetime.now()
    if (now - asof).total_seconds() < min_age:
        return ""
    return f"as of {asof:%H:%M}"


def main(argv=None) -> None:
    """`python -m src.panel_cache --refresh <key> [--width N]`."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--refresh" not in argv:
        return
    key = argv[argv.index("--refresh") + 1]
    width = 100
    if "--width" in argv:
        with contextlib.suppress(ValueError, IndexError):
            width = int(argv[argv.index("--width") + 1])
    renderer = panel_renderer(key)
    if renderer is None:
        return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            renderer(width)
    except Exception:
        return
    store_panel(key, width, buf.getvalue(), datetime.now())


if __name__ == "__main__":
    main()
