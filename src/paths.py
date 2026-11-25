"""Repo-root path resolution — one answer to "which config.json?".

A bare filename like ``"config.json"`` is not a location. It resolves against
whatever directory the process happened to start in, so the same code reads a
different file depending on how it was launched. The launcher starts from the
repo root, which is why this held together; anything else — a cron entry, a
LaunchAgent, a shell in a subdirectory, a test that chdir'd — silently got
something else.

The failure is quiet by construction. Every config reader in this codebase
wraps its ``open`` in ``except``, falls back to ``{}`` or a hardcoded default,
and carries on. Measured 2026-08-07 from ``/tmp``: ``load_config`` returned a
9-weight fallback with ``pop=0.18`` and no ``vrp``/``iv_velocity``/
``term_structure`` at all, against the live config's 27 weights with
``pop=0.0354`` and ``vrp=0.1755``. A scan would have ranked on a different
scorer and said nothing.

The rule, applied at the point of use rather than at ~55 declaration sites:

* **relative in → repo root.** Includes the bare ``"config.json"`` default.
* **absolute in → unchanged.** Every caller that injects its own path — the
  doctor's temp fixtures, ``--config``, the calibration harnesses, every test
  that builds a config in a ``TemporaryDirectory`` — is untouched.

Deliberately dependency-free (stdlib only), for the same reason
``ledger_filters`` is: it gets imported by modules that treat pandas as
optional, and a path helper must never be the thing that drags a heavy import
into a light module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# src/paths.py -> src/ -> repo root. Same idiom as api.py, backtester.py,
# ai_cache.py, which anchored themselves individually before this existed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Strings SQLite understands that are NOT filenames. Anchoring `:memory:`
# created a real 36KB database at the repo root literally named `:memory:`,
# committed in 54ec402 and rewritten by every test run — and the seven test
# modules passing the literal for a private in-memory database were all
# sharing that one file instead. An empty string is SQLite's anonymous
# on-disk temporary database, and is left alone for the same reason.
#
# Handled here rather than at each call site because this function is the
# thing that decides what counts as a path, and these never did. `file:` is
# matched by prefix since a URI carries a query string; a plain name that
# merely begins with the letters, like `filed_trades.db`, still resolves.
_SQLITE_KEYWORDS = frozenset({":memory:", ""})


def repo_path(path: Union[str, "os.PathLike[str]"]) -> str:
    """Resolve ``path`` against the repo root unless it is already absolute.

    Accepts ``Path`` as well as ``str`` — ``backtest_optimizer.save_to_config``
    takes a ``Path``. The type parameter on ``PathLike`` is not decoration:
    bare ``os.PathLike`` makes ``os.fspath`` return ``str | bytes``, and the
    bytes branch would not survive the ``PROJECT_ROOT / text`` below.

    Returns a ``str`` because nearly every caller hands the result straight to
    ``open()`` or stores it where other code compares it as text.
    """
    text = os.fspath(path)
    if text in _SQLITE_KEYWORDS or text.startswith("file:"):
        return text
    return text if os.path.isabs(text) else str(PROJECT_ROOT / text)


def load_project_env(
    env_path: Optional[Union[str, "os.PathLike[str]"]] = None,
    *,
    override: bool = False,
) -> int:
    """Load key=value pairs from the project's `.env` into `os.environ` (stdlib-only).

    Called at application entrypoints (`opt`, `run.py`, `dashboard.py`,
    `trade_alerts.py`, `notifications.py`) so keys like `POLYGON_API_KEY`,
    `ALPHAVANTAGE_API_KEY`, `SEC_EDGAR_CONTACT`, and `DISCORD_WEBHOOK_URL` are
    available even when AI scoring is disabled. Never overwrites already-set
    environment variables unless ``override=True``. Returns count of vars set.
    """
    target = Path(repo_path(env_path)) if env_path is not None else (PROJECT_ROOT / ".env")
    if not target.is_file():
        return 0
    loaded = 0
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if override or key not in os.environ or not os.environ.get(key):
            os.environ[key] = val
            loaded += 1
    return loaded
