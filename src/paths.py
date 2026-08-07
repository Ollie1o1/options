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
from typing import Union

# src/paths.py -> src/ -> repo root. Same idiom as api.py, backtester.py,
# ai_cache.py, which anchored themselves individually before this existed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    return text if os.path.isabs(text) else str(PROJECT_ROOT / text)
