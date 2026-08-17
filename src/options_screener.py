#!/usr/bin/env python3
"""
Options Screener (Top 5 low / 5 medium / 5 high by premium)

Features:
- Fetches options chains via yfinance (Yahoo Finance data; check terms).
- Scores contracts by liquidity (volume/OI), spread tightness, delta quality, and IV balance.
- Categorizes by premium into low/medium/high and picks top 5 in each.
- User-friendly prompts, input validation, and formatted console output.

Note:
- Not financial advice. For personal/informational use only.
- Data availability and timeliness depend on the data provider.
"""

import sys
import os
import csv
import json
import logging
import hashlib as _hashlib
import uuid
import time
import threading as _threading
from datetime import datetime, timezone, timedelta
from pathlib import Path as _Path
from typing import Optional, Tuple, List, Dict, Union, Any
from .schemas import ScanResult
from . import absolute_scores
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import warnings
import contextlib


# Dependency checks
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import numpy as np
except Exception:
    missing.append("numpy")
if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)

from .data_fetching import (
    get_risk_free_rate,
    get_vix_level,
    determine_vix_regime,
    get_market_context,
    fetch_options_yfinance,
    get_dynamic_tickers,
)
from .utils import (
    safe_float,
    norm_cdf,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    bs_rho,
    bs_charm,
    bs_vanna,
    early_exercise_premium,
    _d1d2,
)
from .filters import (
    filter_iv_smile_outliers,
    categorize_by_premium,
    pick_top_per_bucket
)
from .paper_manager import PaperManager
from .capital_risk import pick_within_budget
from . import budget_view
from src.execution_costs import FALLBACK_COMMISSION_PER_CONTRACT

# Enhanced CLI modules
try:
    from . import formatting as fmt
    from . import ui
    from . import settings
    from .trade_analysis import (
        generate_trade_thesis,
        calculate_entry_exit_levels,
        calculate_confidence_score,
        categorize_by_strategy,
        assess_risk_factors,
        format_trade_plan,
        explain_quality_score,
        format_risk_alerts,
        build_scenario_table,
    )
    from tqdm import tqdm
    # tqdm's default write lock allocates a multiprocessing RLock — a kernel
    # semaphore — the first time a bar is built, and never releases it, so every
    # exit printed "leaked semaphore objects to clean up at shutdown". That lock
    # only matters for bars driven from separate processes; every bar here is
    # driven from one process (the fetch pool is a ThreadPoolExecutor), so a
    # thread lock is the correct primitive and leaves nothing to leak.
    tqdm.set_lock(_threading.RLock())
    HAS_ENHANCED_CLI = True
except ImportError as e:
    HAS_ENHANCED_CLI = False
    print(f"Enhanced CLI features unavailable: {e}")
    print("Install with: pip install colorama tqdm")


def _spinner(label: str):
    """Animated loading spinner if the enhanced UI is available, else a no-op."""
    if HAS_ENHANCED_CLI:
        return ui.spinner(label)
    import contextlib
    return contextlib.nullcontext()


def _print_via_spinner(label: str, fn, *args, **kwargs):
    """Run a printing callable with a spinner over its slow gather phase.

    The callable's stdout is captured into a buffer while the spinner animates
    on the real terminal (the spinner binds sys.stdout at construction, before
    the redirect), then emitted once the spinner clears — same pattern as the
    startup regime render, so spinner frames and box output never interleave.
    Partial output is still printed if the callable raises.
    """
    import io as _io
    buf = _io.StringIO()
    try:
        with _spinner(label), contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
    finally:
        out = buf.getvalue()
        if out:
            print(out, end="")


# Hard wall-clock cap on the intel HTML report builders ([d] morning, [e]
# research). Their network phase bounds itself (collect budget ~25s), but that
# promise lives in Python-level thread joins — a wedged layer underneath (DNS,
# a stuck disk, an import lock) has none, and on 2026-07-15 one froze the
# intel menu indefinitely with no diagnostics. On overrun the builder thread
# is abandoned (daemon), every thread's stack is dumped to _INTEL_HANG_LOG so
# the next hang identifies itself, and the menu stays usable.
_INTEL_BUILD_TIMEOUT_S = 90.0
_INTEL_HANG_LOG = os.path.join("logs", "intel_hangs.log")


def _build_report_bounded(label, fn, timeout_s=None, **kwargs):
    """Run a report builder on a worker thread with a hard time cap.

    Returns the builder's result, re-raises its exception, or returns None on
    overrun (after logging all thread stacks to _INTEL_HANG_LOG)."""
    import faulthandler
    import queue
    import threading as _t
    if timeout_s is None:
        timeout_s = _INTEL_BUILD_TIMEOUT_S
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put(("ok", fn(**kwargs)))
        except BaseException as exc:
            q.put(("err", exc))

    _t.Thread(target=_run, daemon=True).start()
    try:
        kind, val = q.get(timeout=timeout_s)
    except queue.Empty:
        try:
            os.makedirs(os.path.dirname(_INTEL_HANG_LOG), exist_ok=True)
            with open(_INTEL_HANG_LOG, "a") as f:
                f.write("\n[{:%Y-%m-%d %H:%M:%S}] {} exceeded {:.0f}s; "
                        "thread stacks:\n".format(datetime.now(), label,
                                                  timeout_s))
                faulthandler.dump_traceback(file=f)
        except Exception:
            pass
        return None
    if kind == "err":
        raise val
    return val


# How long startup will WAIT for background exit-enforcement before showing the
# menu. Exit enforcement (auto-closing positions past their stops) runs in a
# daemon thread and is idempotent, so overrunning it is safe to leave running in
# the background — but blocking the interactive menu on it is not. update_positions
# already parallelizes its fetches; a healthy run finishes in a few seconds. The
# old 60s bound (matched to update_positions' worst-case internal yfinance
# timeouts) turned a rate-limited data feed into a minute-long startup hang.
#
# Now 0.0 — startup does not wait on exit enforcement at all.
#
# update_positions() measures ~8.9s against the real book, so it was never going
# to finish inside any sane bound; the join was simply a fixed toll on every
# launch. Once the dashboard became cache-first this was the ENTIRE remaining
# startup cost (a 2.0 bound measured as exactly 2.00s to menu).
#
# Safe because the work is unchanged, only unwaited: the thread still runs to
# completion behind the menu during the session, update_positions is idempotent,
# and the cron/automation path enforces exits via `--enforce-exits` rather than
# through here. The one trade-off: quitting within a few seconds of launch may
# end the process before enforcement finishes, which the next launch redoes.
_EXIT_ENFORCE_JOIN_TIMEOUT = 0.0


def _render_regime_with_exit_enforcement(pm, width, spinner_factory=None,
                                         cache_dir=None, ttl=None):
    """Render the market-regime dashboard to a string while paper-trade exits
    are enforced in the background. Returns the captured dashboard text
    ('' on failure).

    Race-free by construction: the dashboard renders SYNCHRONOUSLY in the
    calling thread inside ``redirect_stdout`` (a guaranteed-restore context
    manager), so the global ``sys.stdout`` is always handed back before the
    caller prints anything else. Only the non-printing ``update_positions()``
    runs in the daemon thread.

    The previous design inverted this: it rendered the dashboard in a daemon
    thread that reassigned the global ``sys.stdout`` and restored it only in its
    own ``finally``. When ``fetch_market_regime`` overran the 6s join, the thread
    was abandoned mid-render with stdout still pointing at a dead buffer — so the
    mode menu printed afterwards vanished into that buffer and the UI looked
    blank/frozen. ``fetch_market_regime`` already caps itself at ~6s internally,
    so the external thread+timeout bought nothing and only created the race."""
    import threading as _t
    spinner_factory = spinner_factory or _spinner

    def _enforce_exits():
        try:
            pm.update_positions()
            # Record that exits were enforced inline so the automation-health
            # check (which keys off this log's mtime) reflects reality.
            try:
                import os as _os
                from datetime import datetime as _now_dt
                _os.makedirs("logs", exist_ok=True)
                with open("logs/enforce_exits.log", "a") as _elog:
                    _elog.write(f"[{_now_dt.now():%Y-%m-%d %H:%M:%S}] "
                                "exits enforced inline at screener startup\n")
            except Exception:
                pass
        except Exception:
            pass

    text = ""
    try:
        from . import regime_dashboard as _rd
        from .panel_cache import DEFAULT_CACHE_DIR, DEFAULT_TTL, asof_note, render_cached
        exit_thread = _t.Thread(target=_enforce_exits, daemon=True)
        with spinner_factory("Loading market data…"):
            exit_thread.start()
            # Cache-first: the dashboard is 5-10s of live fetches (world pulse
            # 3.5s, VIX/regime 1.1s) that were repeated on every launch. A stale
            # entry is still served instantly and refreshed behind the user, so
            # the wait happens at most once per TTL rather than every time.
            #
            # Resolved through the module, not a from-import, so the existing
            # monkeypatch-the-renderer tests still intercept it.
            text, _asof, _from_cache = render_cached(
                "regime_dashboard", width,
                lambda: _rd.print_regime_dashboard(width),
                ttl=DEFAULT_TTL if ttl is None else ttl,
                cache_dir=DEFAULT_CACHE_DIR if cache_dir is None else cache_dir,
            )
            # Never present aged market data as live.
            _note = asof_note(_asof)
            if _note and text:
                _line = f"  market data {_note}"
                text = text.rstrip("\n") + "\n" + (
                    fmt.colorize(_line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else _line
                ) + "\n"
            # Exits are NOT waited on (see _EXIT_ENFORCE_JOIN_TIMEOUT): the
            # thread is a daemon and idempotent, so it settles behind the menu.
            exit_thread.join(timeout=_EXIT_ENFORCE_JOIN_TIMEOUT)
    except Exception:
        pass
    return text

# Scan-level warning counter (incremented in except blocks, reported at end of scan)
_SCAN_WARNINGS = [0]

# Squeeze Hunt fetches a longer window than Discovery so the calls board has
# something to floor on. squeeze.board.SQUEEZE_MIN_DTE is 60 calendar days
# (the 42-trading-day measured window); fetching to 45 would leave the floor
# with nothing to select, and the nearest 4 expirations on these weekly-heavy
# names are all inside a month.
#
# Wide, because these ladders are sparse past the weeklies and the gap is not
# uniform: on 2026-08-07 QUBT and SOUN both listed 2026-10-16 (70d), while ONDS
# jumped from 2026-09-18 (42d) straight to 2026-12-18 (133d). A 75-day window
# lands in that hole and reports "no calls past the floor" on a name that has
# four of them.
#
# The bound is the January LEAPS. Being listed is not enough — the intermediate
# monthlies are frequently too thin near the money to trade. RH that day had
# 105d and 133d expiries fetched and no in-band call under a 15% spread in
# either, because its tight quotes sat at $120-$160 against a $195.68 spot,
# outside the 15% moneyness band; 2027-01-15 (161d) had four calls clearing
# every filter on up to 1,288 open interest. That expiry lands at index 8-10
# across these names, so the count has to clear it too. Overpaying for time is
# penalised by the ranking itself — more premium, lower multiple.
SQUEEZE_MAX_DTE = 180
SQUEEZE_MAX_EXPIRIES = 12

# Optional imports (relative to this package)
try:
    from .simulation import monte_carlo_pop, batch_monte_carlo_pop
    HAS_SIMULATION = True
except ImportError:
    HAS_SIMULATION = False

try:
    from .visualize_results import create_visualizations
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

try:
    from .vol_analytics import print_vol_cone, print_iv_surface, classify_vol_regime, print_regime_summary
    from .backtester import print_paper_trade_ic
    HAS_VOL_ANALYTICS = True
except ImportError:
    HAS_VOL_ANALYTICS = False





from .cli_display import (
    get_display_width, print_executive_summary,
    print_report, print_news_panel,
    print_credit_spreads_report, print_iron_condor_report,
    print_lottery_ticket_report, print_per_risk_table,
)
from .watchlist import (
    load_watchlist, add_to_watchlist, remove_from_watchlist,
)
from .oi_snapshot import load_oi_snapshot
from .utils import safe_float
from .spread_scoring import enrich_credit_spreads, enrich_iron_condors


def _progress_bar(total: int, desc: str, enabled: bool = True, stream=None):
    """Scan-phase progress bar. Canonical implementation lives in ``ui`` so the
    portfolio viewer and reports share one answer to "is this wait indicated?"."""
    return ui.progress_bar(total, desc, enabled=enabled and HAS_ENHANCED_CLI,
                           stream=stream)


@contextlib.contextmanager
def _suppress_scan_noise():
    """Suppress noisy third-party logging/warnings during parallel scan."""
    _noisy = ['yfinance', 'urllib3', 'peewee', 'charset_normalizer',
              'requests', 'asyncio', 'httpx', 'httpcore']
    _saved = {}
    for name in _noisy:
        lg = logging.getLogger(name)
        _saved[name] = lg.level
        lg.setLevel(logging.CRITICAL)
    # Also silence the root logger's stderr handler temporarily
    _root = logging.getLogger()
    _saved_root = _root.level
    # Capture and discard warnings from third-party libs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            yield
        finally:
            for name, level in _saved.items():
                logging.getLogger(name).setLevel(level)
            _root.setLevel(_saved_root)


def _strategy_label_for_mode(mode: str, opt_type) -> str:
    """Map screener mode + option type to a paper-trade strategy_name.

    Premium-Selling logs are short-premium (Short Put / Short Call).
    All other single-leg modes (Discovery, Budget, Single-stock) generate buy
    signals and are logged as Long Put / Long Call so the P&L sign and
    Greeks signing in check_pnl read correctly.
    """
    from .trade_analysis import strategy_label_for_mode
    return strategy_label_for_mode(mode, opt_type)


def _cohort_min_dte(cfg: dict) -> int:
    """Minimum DTE-at-entry for a Long Call to be eligible for the validation
    cohort. Horizon-aware: an explicit auto_log.cohort_min_dte wins; otherwise
    it derives from the time-exit so the floor can never drift below it —
    time_exit_dte + cohort_min_runway_days (default 9) so an entry has real
    swing runway before the time-exit can force-close it."""
    al = (cfg.get("auto_log") or {})
    explicit = al.get("cohort_min_dte")
    if explicit is not None:
        try:
            return int(explicit)
        except (TypeError, ValueError):
            pass
    time_exit = (cfg.get("exit_rules") or {}).get("time_exit_dte", 21)
    runway = al.get("cohort_min_runway_days", 9)
    try:
        return int(time_exit) + int(runway)
    except (TypeError, ValueError):
        return 30


def _trade_dte(trade: dict):
    """DTE-at-entry for a trade dict. Prefers an explicit 'dte'; else derives
    from 'expiration' (YYYY-MM-DD) relative to today. Returns None if unknown."""
    dte = trade.get("dte")
    if dte is not None:
        try:
            return int(dte)
        except (TypeError, ValueError):
            pass
    exp = trade.get("expiration")
    if exp:
        from datetime import datetime as _dt, date as _date
        try:
            d = _dt.strptime(str(exp)[:10], "%Y-%m-%d").date()
            return (d - _date.today()).days
        except (ValueError, TypeError):
            pass
    return None


from .paths import PROJECT_ROOT as _PROJECT_ROOT
from .paths import repo_path as _repo_path


def auto_log_budget_cap(cfg_path: str = "config.json"):
    """The per-position budget the auto-log feeder must respect, or None.

    Reads the same ``auto_log.max_capital_at_risk`` key ``PaperManager`` enforces
    at insert time, with the same coercion, so the scan-side pre-filter and the
    ledger-side gate can never disagree about what is affordable.

    Any config problem yields None (no constraint): a cap that fails to load
    must not silently filter every candidate out of the scan.
    """
    import json
    try:
        with open(_repo_path(cfg_path)) as f:
            cfg = json.load(f)
        cap = (cfg.get("auto_log") or {}).get("max_capital_at_risk")
        return float(cap) if cap is not None and cap not in ("", 0) else None
    except Exception:
        return None


def apply_auto_log_allowlist(trade: dict, cfg_path: str = "config.json") -> tuple:
    """
    Phase 1 cohort quarantine. Returns one of:
      ("insert", 0)  → log normally, eligible for the Long-Call validation cohort
      ("insert", 1)  → log with paper_only=1, excluded from the cohort
      ("drop", None) → skip auto-log entirely

    Precedence: if a strategy appears in BOTH allowed_strategies and
    paper_only_strategies (config typo), allowed wins and a warning is logged.

    Cohort horizon guard: an allowed Long Call entered below the cohort DTE
    floor (see _cohort_min_dte) is logged as paper_only=1 — kept for data but
    excluded from the gate, because the time-exit would force-close it before
    the swing thesis can play out (contaminating the IC with 3-day returns).
    A trade with unknown DTE is left eligible (no false quarantine).
    """
    import json, logging
    try:
        with open(_repo_path(cfg_path)) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    al = (cfg.get("auto_log") or {})
    allowed = set(al.get("allowed_strategies") or [])
    paper_only = set(al.get("paper_only_strategies") or [])
    strat = str(trade.get("strategy_name") or "")
    overlap = allowed & paper_only
    if strat in overlap:
        logging.warning(
            "auto_log config: '%s' appears in both allowed_strategies and "
            "paper_only_strategies — treating as allowed.", strat
        )
    if strat in allowed:
        floor = _cohort_min_dte(cfg)
        dte = _trade_dte(trade)
        if dte is not None and dte < floor:
            logging.info(
                "cohort horizon: %s at %dDTE < floor %dDTE — logging paper_only=1 "
                "(data only, excluded from gate).", strat, dte, floor
            )
            return ("insert", 1)
        return ("insert", 0)
    if strat in paper_only:
        return ("insert", 1)
    return ("drop", None)


def rank_by_verdict(df, win_rates: Optional[Dict[str, float]] = None):
    """Order candidates by what survives their own trading costs.

    Replaces `sort_values("quality_score")` on the display paths. The composite
    cannot rank: Spearman against return on capital is -0.132 on the
    long-premium book and +0.047 on short premium, and its top bucket is the
    worst cell in the ledger — 31.6% win rate, -19.9% return on capital, versus
    +5.2% for the [0.55, 0.65) bucket. Sorting by it selects the worst
    candidates. It survives here only as a tie-breaker.

    What replaces it is the round-trip crossing cost as a share of the reward,
    measured per candidate from its own quotes: 0.7-1.7% for a single leg
    against 33% for a two-leg credit spread.

    Failure-safe by design, matching the rest of the scan path: if quotes are
    missing or anything raises, the old `quality_score` ordering is returned so
    a scan still produces a report. See src/candidate_verdict.py and
    docs/EXECUTION_TRUTH.md.
    """
    if df is None or len(df) == 0:
        return df
    try:
        from . import candidate_verdict as _cv
        ranked = _cv.rank(df.to_dict("records"), win_rates=win_rates)
        out = pd.DataFrame(ranked)
        verdicts = out.pop("verdict")
        out["verdict_passed"] = [v.passed for v in verdicts]
        out["verdict_reason"] = [v.reason for v in verdicts]
        out["friction_pct"] = [v.round_trip_pct for v in verdicts]
        out["breakeven_win_rate"] = [v.breakeven for v in verdicts]
        return out.reset_index(drop=True)
    except Exception:
        if "quality_score" in getattr(df, "columns", []):
            return df.sort_values("quality_score", ascending=False).reset_index(drop=True)
        return df


def rank_single_legs_by_verdict(df, mode: str):
    """Order single-leg candidates for the auto-logger by what survives costs.

    `rank_by_verdict` replaced `sort_values("quality_score")` on the *display*
    paths, but the auto-log path kept sorting by the composite — so the score
    decided both which leg per symbol survived the per-symbol dedup and which
    symbols reached the top-N. Every row in the ledger was selected by it.

    That score is measured at rank IC **-0.10** against friction-adjusted
    return on the 335-row long-premium cohort, negative in 5 of 5 walk-forward
    windows, while the 27-component composite it is built from is flat
    (+0.004) — the post-composite adjustment stack carries the whole negative.
    See `docs/ADJUSTMENT_STACK_20260807.md`.

    Labelling comes first and is not incidental. `candidate_verdict._legs_of`
    reads the buy/sell side off `strategy_name`, and these rows carry only
    `type` at this point, so a Premium Selling short put would be priced as a
    debit *buy*. That flips `is_credit`, which in turn skips both the
    "credit disappears once the spread is crossed" check and the breakeven
    check — the two gates that matter most for short premium.

    Ordering only: every input row is returned. The allowlist and budget
    filters downstream do the dropping, and removing candidates here would
    starve the forward cohort.
    """
    if df is None or len(df) == 0 or "type" not in getattr(df, "columns", []):
        return df
    out = df.copy()
    try:
        out["strategy_name"] = [
            _strategy_label_for_mode(mode, t) for t in out["type"]
        ]
    except Exception:
        logging.getLogger(__name__).debug("strategy labelling failed", exc_info=True)
    return rank_by_verdict(out)


def structure_strategy_name(row) -> str:
    """Structure label for a spread-scan row.

    Condors carry a `total_credit`; verticals are named by their short leg's
    type. `candidate_verdict._legs_of` reads the leg layout off this name, so
    it has to be set before a structure can be priced.
    """
    _keys = row.index if hasattr(row, "index") else row.keys()
    if ("total_credit" in _keys) and not pd.isna(row.get("total_credit")):
        return "Iron Condor"
    return "Bear Call" if str(row.get("type", "")).strip().lower() == "call" else "Bull Put"


def rank_structures_by_verdict(df):
    """Order spreads and condors for the auto-logger by what survives costs.

    The single-leg path moved off `quality_score` first; this path stayed on it
    because a condor could not be priced at all — `find_iron_condors` emitted
    no per-leg quotes, so `_legs_of` refused every one of them and routing them
    through the gate would have starved the cohort rather than ranked it. The
    quotes are now carried, so the ordering can apply here too.

    Friction is the point: four crossings against one credit runs roughly twice
    the two-leg burden, which already measured ~33% of the credit on the logged
    Bull Puts against 1-4% for a single leg.

    Ordering only — every input row is returned. See
    `rank_single_legs_by_verdict` for why nothing is dropped here.
    """
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    try:
        out["strategy_name"] = [structure_strategy_name(r) for _, r in out.iterrows()]
    except Exception:
        logging.getLogger(__name__).debug("structure labelling failed", exc_info=True)
    return rank_by_verdict(out)


def gate_and_report(df, board: str, *, label_structures: bool = False,
                    verbose: bool = True):
    """Remove what the ledger measures as a loser, and say what was removed.

    The counterpart to `rank_by_verdict`, which ordered boards. Ordering was
    tested and failed — no key beat `quality_score` at the #1 slot out of
    sample (23 of 48 paired cells, Wilcoxon p=0.89) — while removal held in
    five folds out of five. So this refuses rather than ranks, and an empty
    board is a real answer rather than a failure to find one.

    Returns the surviving frame. Callers keep their existing "nothing found"
    branch for a genuinely empty scan; this handles the case where candidates
    were found and every one of them was refused.
    """
    from . import pick_ranking as _pr

    result = _pr.gate_board(df, label_structures=label_structures)
    if verbose and result.refused is not None and len(result.refused):
        _print_refusals(result, board)
    return result.kept


def _print_refusals(result, board: str) -> None:
    """What the board declined to show, and on what evidence."""
    width = get_display_width()
    kept, scanned = len(result.kept), result.scanned
    if kept == 0:
        print("\n" + ui.rule(width, title=f"{board} — NO QUALIFYING CONTRACT"))
        print(f"  {scanned} scanned · {scanned} refused\n")
    else:
        print("\n" + ui.rule(width, title=f"{board} — {kept} of {scanned} shown"))
        print(f"  {scanned - kept} refused\n")

    for line in result.summary_lines():
        print("  " + (fmt.style(line, 'muted') if HAS_ENHANCED_CLI else line))

    if kept == 0:
        msg = "Nothing here cleared the gates. That is the answer, not an error."
        print("\n  " + (fmt.style(msg, 'emph') if HAS_ENHANCED_CLI else msg))
    print()


def _budget_board(df, label_fn, budget: Optional[float], *, verbose: bool = True):
    """Annotate a board per dollar of capital at risk, then keep what fits.

    Call this AFTER `gate_and_report`. The refusal block is the
    best-evidenced output this scan produces and a budget must never be able
    to suppress it, so the gate speaks first and the budget narrows what is
    left. Call it BEFORE the top-N bucket cut: filtering after the cut lets
    unaffordable rows consume every slot and log nothing, which is exactly how
    the short-put window starved on 2026-07-30.

    `label_fn(row) -> str` names the strategy per ROW, not per frame.
    `budget_view` takes one strategy name for a whole frame, but a Premium
    Selling board mixes Short Put (collateral-backed, sizable) with Short Call
    (unbounded, never sizable), and a spread board mixes Bull Put with Bear
    Call. One label for the frame would cost the other kind its capital at
    risk, and an unsizable row fails a set budget by design.

    DISPLAY ONLY. Row order is preserved exactly — ranking was disproven out
    of sample (Wilcoxon p=0.89), so nothing here may become a sort key.
    """
    if df is None or len(df) == 0:
        return df
    # `within_budget` reads a non-positive cap as no cap; say the same thing
    # here so a 0 can never print "0 of 14 fit" beside an unfiltered board.
    if budget is not None and budget <= 0:
        budget = None
    try:
        labels = [label_fn(df.iloc[i]) for i in range(len(df))]
    except Exception:
        logging.getLogger(__name__).debug("budget labelling failed", exc_info=True)
        return df

    # Positional throughout: a scan frame concatenated across tickers can carry
    # duplicate index labels, and grouping on those would scramble the board.
    work = df.reset_index(drop=True)
    kept: List[int] = []
    cells: Dict[int, tuple] = {}
    for label in dict.fromkeys(labels):
        positions = [i for i, lab in enumerate(labels) if lab == label]
        sub = budget_view.annotate(work.iloc[positions], label)
        if budget is not None:
            sub = budget_view.affordable(sub, budget, label)
        for pos in sub.index:
            kept.append(int(pos))
            cells[int(pos)] = (sub.at[pos, "capital_at_risk"],
                               sub.at[pos, "reward_per_risk"],
                               sub.at[pos, "net_ev_per_risk"])
    kept.sort()  # back into the order the board arrived in

    out = df.iloc[kept].copy()
    for slot, col in enumerate(("capital_at_risk", "reward_per_risk",
                                "net_ev_per_risk")):
        # An object-dtype ndarray, assigned positionally: a plain list of mixed
        # None and float upcasts to float64 and turns None into NaN, which is a
        # third state on top of budget_view's None-vs-0 contract.
        out[col] = np.array([cells[p][slot] for p in kept], dtype=object)
    if verbose and budget is not None and len(out) < len(df):
        _line = (f"Budget ${budget:,.0f} per position: "
                 f"{len(out)} of {len(df)} surviving candidates fit.")
        print("  " + (fmt.style(_line, 'muted') if HAS_ENHANCED_CLI else _line))
    return out


def _print_per_risk_table(df, label_fn, budget: Optional[float]) -> None:
    """Spec s4's common-axis table, printed after the board it describes.

    `label_fn` is the same one `_budget_board` sized with, so the Structure
    column cannot name a different strategy than the risk figure beside it.
    Display only: `print_per_risk_table` reprints the board's order and never
    re-sorts it.
    """
    if budget is not None and budget <= 0:
        budget = None
    print_per_risk_table(df, label_fn, budget)


def _print_budget_use(df, budget: Optional[float]) -> None:
    """The "you could hold N of these" line, printed after the board."""
    if budget is not None and budget <= 0:
        return
    line = budget_view.budget_use_line(df, budget)
    if line:
        print("    " + (fmt.style(line, 'muted') if HAS_ENHANCED_CLI else line))


def _with_session_budget(trade_dict: dict, budget_was_chosen: bool,
                         session_budget: Optional[float]) -> dict:
    """Attach the session budget to a trade dict ONLY if a prompt happened.

    `log_trade` reads KEY PRESENCE, not value: present-and-None means the
    operator explicitly chose no limit, present-and-float means that ceiling,
    and ABSENT means fall back to `auto_log.max_capital_at_risk`.

    A mode that never reaches the prompt — ALL, LOTTERY, SQUEEZE, or a bare
    ticker typed at the menu — is the same case as a cron run that never
    reaches it. Neither chose anything, so neither may be handed an explicit
    "no limit": doing so silently uncapped modes that config had been holding
    at $4,000.

    Takes a separate flag rather than sniffing `session_budget`, because
    `None` is itself a legitimate answer to the prompt and must stay
    distinguishable from "never asked". A sentinel in `session_budget` would
    destroy that distinction.
    """
    if budget_was_chosen:
        trade_dict["budget_at_entry"] = session_budget
    return trade_dict


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file with fallback defaults.

    Relative paths resolve against the repo root — see `_repo_path`.
    """
    default_config = {
        # Composite quality score weights (can be overridden in config.json)
        "composite_weights": {
            "pop": 0.18,
            "em_realism": 0.12,
            "rr": 0.15,
            "momentum": 0.10,
            "iv_rank": 0.10,
            "liquidity": 0.15,
            "catalyst": 0.05,
            "theta": 0.10,
            "ev": 0.05,
            "trader_pref": 0.10
        },
        "moneyness_band": 0.15,
        "target_delta": 0.40,
        "earnings_buffer_days": 5,
        "monte_carlo_simulations": 10000,
        "exit_rules": {
            "take_profit": 0.50,
            "stop_loss": -0.25
        }
    }
    
    try:
        with open(_repo_path(config_path), 'r') as f:
            config = json.load(f)
            # Merge with defaults
            for key in default_config:
                if key not in config:
                    config[key] = default_config[key]
            return config
    except FileNotFoundError:
        # First-run case — silently use defaults.
        return default_config
    except json.JSONDecodeError as e:
        # Malformed JSON: surface loudly so a typo in config.json doesn't silently
        # erase the user's tuned weights/filters/exit rules during a cron run.
        sys.stderr.write(
            f"\n[load_config] ERROR: {config_path} failed to parse — using DEFAULTS. "
            f"Fix and re-run.\n  {type(e).__name__}: {e}\n"
        )
        return default_config
    except Exception as e:
        sys.stderr.write(
            f"\n[load_config] WARNING: unexpected error reading {config_path} — using defaults.\n"
            f"  {type(e).__name__}: {e}\n"
        )
        return default_config


def ai_scoring_disabled(config: Optional[Dict]) -> bool:
    """True when AI scoring is turned off in config (``ai_scoring.enabled: false``).

    Lets a user disable the AI ranking persistently — equivalent to passing
    --no-ai on every run, which interactive sessions can't do. Defaults to
    enabled (False) so absence preserves existing behavior. Never raises.
    """
    try:
        return not bool((config or {}).get("ai_scoring", {}).get("enabled", True))
    except Exception:
        return False


_IC_WEIGHTS_CACHE: dict | None = None
_IC_RECALIB_RUNNING: bool = False
_IC_RECALIB_LOCK = _threading.Lock()
_CACHE_MAX_AGE_DAYS = 7


def _maybe_trigger_recalib(cache_path: str) -> None:
    """Fire-and-forget: recalibrate IC weights in background if cache is stale and ≥30 closed trades exist."""
    global _IC_RECALIB_RUNNING
    if _IC_RECALIB_RUNNING:
        return
    cache_stale = True
    try:
        mtime = os.path.getmtime(cache_path)
        cache_stale = (time.time() - mtime) > (_CACHE_MAX_AGE_DAYS * 86400)
    except OSError:
        cache_stale = True
    if not cache_stale:
        return

    def _run():
        global _IC_RECALIB_RUNNING, _IC_WEIGHTS_CACHE
        with _IC_RECALIB_LOCK:
            _IC_RECALIB_RUNNING = True
            try:
                import sqlite3 as _sqlite3
                from contextlib import closing as _closing
                with _closing(_sqlite3.connect(_repo_path("paper_trades.db"))) as _conn:
                    n = _conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL"
                    ).fetchone()[0]
                if n < 30:
                    return
                from .backtester import run_paper_trade_ic
                ic_data = run_paper_trade_ic()
                if ic_data.get("n_trades", 0) >= 30:
                    with open(cache_path, "w") as _f:
                        json.dump(ic_data, _f, indent=2)
                    _IC_WEIGHTS_CACHE = None  # invalidate in-memory cache so next call re-reads
                    logging.getLogger(__name__).info(
                        "IC weights auto-recalibrated from %d trades", ic_data["n_trades"]
                    )
            except Exception as _e:
                logging.getLogger(__name__).debug("IC auto-recalib failed: %s", _e)
            finally:
                _IC_RECALIB_RUNNING = False

    t = _threading.Thread(target=_run, daemon=True)
    t.start()


def load_ic_adjusted_weights(config: Dict, cache_path: str = "ic_weights_cache.json") -> Dict:
    """Blend config composite weights with IC-derived weights from paper trade analysis.

    Blending formula: final_weight = 0.7 * config_weight + 0.3 * ic_weight,
    where ic_weight is the component's IC (floored at 0) as a share of the IC
    measured across **every candidate component**, not just the ones that clear
    the p-gate.

    The denominator is the whole fix. It used to be the sum over survivors,
    which made the reallocation a function of how many components happened to
    be significant rather than of the evidence:

    * With one survivor the ratio is 1.0 by construction, so that component took
      the entire 0.30 budget however weak its IC was. Measured live 2026-08-07,
      `theta` held **24.3%** of the composite off IC=+0.082 against a base
      weight of 0.0197 — a 16x lift the calibration never intended.
    * Doubling that component's IC moved its weight by exactly 0.0000, while an
      unrelated component crossing p=0.10 stripped 26% of it. A weight that does
      not respond to its own evidence and does respond to someone else's is not
      carrying evidence at all.

    Normalising over all candidates keeps the p-gate deciding *eligibility* and
    lets the IC magnitudes decide *the split*, so both properties hold: more
    measured IC means more weight, and a neighbour's significance changes
    nothing. Returns plain config weights on any failure.
    """
    global _IC_WEIGHTS_CACHE
    if _IC_WEIGHTS_CACHE is not None:
        return _IC_WEIGHTS_CACHE
    # Relative in -> repo root, absolute in -> unchanged. 54ec402 anchored the
    # ledger and config readers and missed this pair: the bare cache filename
    # and the `paper_trades.db` the recalib thread counts trades in. A scan
    # launched from anywhere but the repo root blended its weights off a
    # different (or absent) ledger and cache, and said nothing — every read
    # here is wrapped in `except` and falls back to the plain config weights.
    cache_path = _repo_path(cache_path)
    _maybe_trigger_recalib(cache_path)
    base_weights = config.get("composite_weights", {}) or {}
    try:
        with open(cache_path, "r") as f:
            cache = json.load(f)
        component_ic = cache.get("component_ic", {})
        if not component_ic:
            _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        # Map component_ic keys (e.g. "pop_score") to weight keys (e.g. "pop")
        key_map = {
            "pop_score": "pop", "ev_score": "ev", "rr_score": "rr",
            "liquidity_score": "liquidity", "momentum_score": "momentum",
            "iv_rank_score": "iv_rank", "theta_score": "theta",
        }
        component_pvalues = cache.get("component_pvalues", {})
        ic_vals = {}
        # Denominator spans every candidate, eligible or not — see docstring.
        # Components with a negative IC contribute 0 here exactly as they do to
        # the numerator, so a factor that points the wrong way cannot inflate
        # the share of one that points the right way.
        ic_scale = 0.0
        for ic_key, w_key in key_map.items():
            ic_raw = component_ic.get(ic_key)
            if ic_raw is None or not isinstance(ic_raw, (int, float)):
                continue
            ic_pos = max(0.0, float(ic_raw))
            ic_scale += ic_pos
            if component_pvalues.get(ic_key, 1.0) < 0.10:
                ic_vals[w_key] = ic_pos
        if not ic_vals or ic_scale <= 0.0:
            _IC_WEIGHTS_CACHE = base_weights
            return _IC_WEIGHTS_CACHE
        ic_total = ic_scale
        blended = dict(base_weights)
        for w_key, ic_raw in ic_vals.items():
            if w_key in blended:
                ic_norm = ic_raw / ic_total
                blended[w_key] = 0.7 * float(blended[w_key]) + 0.3 * ic_norm
        _IC_WEIGHTS_CACHE = blended
        return _IC_WEIGHTS_CACHE
    except Exception:
        _IC_WEIGHTS_CACHE = base_weights
        return _IC_WEIGHTS_CACHE


def _invalidate_ic_weights_cache() -> None:
    global _IC_WEIGHTS_CACHE
    _IC_WEIGHTS_CACHE = None




def calculate_probability_of_profit(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray], premium: Union[float, np.ndarray], r: Union[float, np.ndarray] = 0.0, q: Union[float, np.ndarray] = 0.0) -> Union[float, np.ndarray, None]:
    """Calculate probability of profit at expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        premium = np.asanyarray(premium)
        r = np.asanyarray(r, dtype=float)
        q = np.asanyarray(q, dtype=float)
        # Clip T to 1 hour minimum to prevent division-by-zero on expiration day
        T = np.maximum(T, 1.0 / (365.0 * 24.0))

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        # Break-even point
        breakeven = np.where(is_call, K + premium, K - premium)

        # Forward price: F = S * exp((r - q) * T)
        F = S * np.exp((r - q) * T)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = (np.log(F / breakeven) - (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

        pop = np.where(is_call, norm_cdf(d), 1.0 - norm_cdf(d))

        if np.isscalar(option_type) and np.isscalar(S):
            return float(pop)
        return pop
    except Exception:
        return None


def calculate_expected_move(S: Union[float, np.ndarray], sigma: Union[float, np.ndarray], T: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate expected move (1 standard deviation) until expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        sigma = np.asanyarray(sigma)
        T = np.asanyarray(T)
        move = S * sigma * np.sqrt(T)
        if move.ndim == 0:
            return float(move)
        return move
    except Exception:
        return None


def calculate_probability_of_touch(option_type: Union[str, np.ndarray], S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray, None]:
    """Calculate probability that option will touch the strike price before expiration (Vectorized)."""
    try:
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        T = np.asanyarray(T)
        sigma = np.asanyarray(sigma)
        # Clip T to 1 hour minimum to prevent division-by-zero on expiration day
        T = np.maximum(T, 1.0 / (365.0 * 24.0))

        scalar_input = isinstance(option_type, str) and S.ndim == 0  # type: ignore[union-attr]

        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        # Probability of touching is approximately 2 * delta for ATM options
        # More precise: P(touch) ≈ 2 * N(d2)
        with np.errstate(divide='ignore', invalid='ignore'):
            d2 = (np.log(S / K) - (0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

        # Scalar fast-path: avoid boolean indexing on 0-d arrays
        if scalar_input:
            is_otm = (K > S) if is_call else (K < S)
            if is_otm:
                pot_val = 2 * norm_cdf(float(d2)) if is_call else 2 * (1.0 - norm_cdf(float(d2)))
                return float(np.clip(pot_val, 0.0, 1.0))
            return 1.0

        pot = np.ones_like(S, dtype=float)
        call_otm = is_call & (K > S)
        put_otm = (~is_call) & (K < S)
        pot[call_otm] = 2 * norm_cdf(d2[call_otm])
        pot[put_otm] = 2 * (1.0 - norm_cdf(d2[put_otm]))
        return np.clip(pot, 0.0, 1.0)
    except Exception:
        return None


def calculate_risk_reward(
    option_type: Union[str, np.ndarray],
    premium: Union[float, np.ndarray],
    S: Union[float, np.ndarray],
    K: Union[float, np.ndarray],
    expected_move: Optional[Union[float, np.ndarray]] = None,
    em_factor: float = 0.68,
) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray, None], Union[float, np.ndarray, None]]:
    """Calculate max loss, break-even, and risk/reward ratio (Vectorized).

    Uses:
      - target_price = stock_price ± em_factor * EM  (default 0.68 = 1σ)
      - RR = max_gain_if_target_hit / premium
    where gains and premium are measured per share.
    """
    try:
        premium = np.asanyarray(premium)
        S = np.asanyarray(S)
        K = np.asanyarray(K)
        
        if isinstance(option_type, str):
            is_call = option_type.lower() == "call"
        else:
            is_call = np.char.lower(np.asanyarray(option_type).astype(str)) == "call"

        max_loss: Any = premium * 100  # Per contract

        # Break-even price
        breakeven = np.where(is_call, K + premium, K - premium)

        # Compute max gain at target using expected move when available
        if expected_move is not None:
            expected_move = np.asanyarray(expected_move)
            target_price = np.where(is_call, S + em_factor * expected_move, S - em_factor * expected_move)
            payoff_per_share = np.where(is_call, np.maximum(0.0, target_price - K), np.maximum(0.0, K - target_price))
        else:
            # Fallback: simple heuristic target if EM is unavailable
            target_price = np.where(is_call, S * 1.5, S * 0.5)
            payoff_per_share = np.where(is_call, np.maximum(0.0, target_price - K), np.maximum(0.0, K - target_price))

        max_gain_per_share = np.maximum(0.0, payoff_per_share - premium)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            risk_reward_ratio = np.where(premium > 0, max_gain_per_share / premium, 0.0)

        if premium.ndim == 0:  # type: ignore[union-attr]
            return float(max_loss), float(breakeven), float(risk_reward_ratio)
            
        return max_loss, breakeven, risk_reward_ratio
    except Exception:
        return None, None, None


def calculate_metrics(
    df: pd.DataFrame,
    risk_free_rate: float,
    earnings_date: Optional[datetime],
    config: Dict,
    iv_rank: Optional[float],
    iv_percentile: Optional[float],
    sentiment_score: Optional[float],
    macro_risk_active: bool,
    sector_perf: Dict,
    tnx_change_pct: float,
    short_interest: Optional[float] = None,
    next_ex_div: Optional[object] = None,
    earnings_move_data: Optional[dict] = None,
    mode: str = "Single-stock",
    dividend_yield: float = 0.0,
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """Calculates all objective mathematical metrics and merges external data.

    ``as_of`` is the instant the chain is priced at; None means wall-clock now.
    It reaches here only to date the Monte Carlo seed. Without it, pinning the
    instant makes a scan reproducible today and NOT tomorrow, because the seed
    rolls at midnight — a guarantee that silently expires is worse than none.
    """
    
    # --- Institutional Flow & Sentiment ---
    df["Vol_OI_Ratio"] = (df["volume"] / df["openInterest"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["Unusual_Whale"] = (df["Vol_OI_Ratio"] > 1.5) & (df["volume"] > 500)
    df["high_premium_turnover"] = (df["premium"] * df["volume"] * 100) > 25000

    def _sentiment_tag(score):
        if score is None or pd.isna(score):
            return "Neutral"
        if score > 0.05:
            return "Bullish"
        elif score < -0.05:
            return "Bearish"
        else:
            return "Neutral"

    df["sentiment_tag"] = df["sentiment_score"].apply(_sentiment_tag)

    # --- Earnings Volatility Logic ---
    df["Earnings Play"] = "NO"
    _now_utc = datetime.now(timezone.utc)
    if earnings_date and earnings_date > _now_utc:   # only flag future earnings
        df.loc[(df["exp_dt"] > earnings_date), "Earnings Play"] = "YES"

    df["is_underpriced"] = False
    earnings_mask = df["Earnings Play"] == "YES"
    if earnings_mask.any():
        df.loc[earnings_mask, "is_underpriced"] = df.loc[earnings_mask, "impliedVolatility"] < df.loc[earnings_mask, "hv_30d"]

    # --- Trend Alignment Filter ---
    # Require price above BOTH SMA-20 and SMA-50 for calls (and below both for puts)
    # to confirm a genuine medium-term trend rather than just short-term noise.
    df["Trend_Aligned"] = False
    has_sma50 = "sma_50" in df.columns and df["sma_50"].notna().any()
    if has_sma50:
        df.loc[
            (df["type"] == "call") & (df["underlying"] > df["sma_20"]) & (df["underlying"] > df["sma_50"]),
            "Trend_Aligned"
        ] = True
        df.loc[
            (df["type"] == "put") & (df["underlying"] < df["sma_20"]) & (df["underlying"] < df["sma_50"]),
            "Trend_Aligned"
        ] = True
    else:
        df.loc[(df["type"] == "call") & (df["underlying"] > df["sma_20"]), "Trend_Aligned"] = True
        df.loc[(df["type"] == "put") & (df["underlying"] < df["sma_20"]), "Trend_Aligned"] = True

    # --- VECTORIZED GREEKS ---
    S_vals = df["underlying"].values
    K_vals = df["strike"].values
    T_vals = df["T_years"].values
    IV_vals = np.maximum(1e-9, df["impliedVolatility"].values)
    types_vals = df["type"].values

    _q = float(dividend_yield) if dividend_yield else 0.0
    df["delta"] = bs_delta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["abs_delta"] = np.abs(df["delta"].values)
    df["gamma"] = bs_gamma(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["vega"] = bs_vega(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["theta"] = bs_theta(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["rho"] = bs_rho(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["charm"] = bs_charm(types_vals, S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)
    df["vanna"] = bs_vanna(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, _q)

    # --- Early Exercise Premium (American vs European put) ---
    # For puts where early exercise is materially valuable, flag the contract.
    # Threshold: early exercise premium > 3% of market premium → flag "EARLY_EX".
    # This warns that BS Greeks understate true option value for these rows.
    try:
        _put_mask = (df["type"].str.lower() == "put") & df["underlying"].notna() & df["impliedVolatility"].notna() & df["premium"].notna() & (df["T_years"] > 0) & (df["impliedVolatility"] > 0)
        df["early_exercise_flag"] = ""
        if _put_mask.any():
            _ee_func = np.vectorize(
                lambda S, K, T, IV: early_exercise_premium("put", float(S), float(K), float(T), risk_free_rate, float(IV), _q),
                otypes=[float],
            )
            _sub = df.loc[_put_mask]
            _ee_vals = _ee_func(_sub["underlying"].values, _sub["strike"].values, _sub["T_years"].values, _sub["impliedVolatility"].values)
            df.loc[_put_mask, "early_exercise_flag"] = np.where(
                _ee_vals > _sub["premium"].values * 0.03, "EARLY_EX", ""
            )
    except Exception:
        df["early_exercise_flag"] = ""

    # --- PCR per Expiration ---
    is_call = np.char.lower(types_vals.astype(str)) == "call"
    try:
        pcr_map = {}
        for exp, grp in df.groupby("expiration"):
            call_vol = grp.loc[grp["type"] == "call", "volume"].sum()
            put_vol = grp.loc[grp["type"] == "put", "volume"].sum()
            pcr_val = float(put_vol) / float(call_vol) if pd.notna(call_vol) and call_vol > 0 else np.nan
            pcr_map[exp] = pcr_val
        df["pcr"] = df["expiration"].map(pcr_map)
        def _pcr_signal(v):
            if pd.isna(v):
                return ""
            if v > 1.5:
                return "HEAVY HEDGING"
            if v < 0.5:
                return "BULLISH FLOW"
            return ""
        df["pcr_signal"] = df["pcr"].apply(_pcr_signal)
    except Exception:
        df["pcr"] = np.nan
        df["pcr_signal"] = ""

    # --- GEX (Gamma Exposure) by Strike ---
    try:
        gex_per_contract = df["gamma"].values * df["openInterest"].values * 100.0 * S_vals ** 2
        df["gex"] = np.where(is_call, gex_per_contract, -gex_per_contract)
        gex_by_strike = df.groupby("strike")["gex"].sum().sort_index()
        cumulative_gex = gex_by_strike.cumsum()
        negative_strikes = cumulative_gex[cumulative_gex < 0]
        gex_flip = float(negative_strikes.index[0]) if not negative_strikes.empty else None
        df["gex_flip_price"] = gex_flip
        # Max gamma strike pinning
        try:
            gex_by_strike_abs = df.groupby("strike")["gex"].apply(lambda x: x.abs().sum())
            max_gamma_strike = float(gex_by_strike_abs.idxmax()) if not gex_by_strike_abs.empty else None
            df["max_gamma_strike"] = max_gamma_strike
            _price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
            if max_gamma_strike and _price_scalar > 0:
                df["gamma_pin_dist_pct"] = abs(max_gamma_strike - _price_scalar) / _price_scalar * 100
            else:
                df["gamma_pin_dist_pct"] = pd.NA
        except Exception:
            df["max_gamma_strike"] = pd.NA
            df["gamma_pin_dist_pct"] = pd.NA
    except Exception:
        df["gex"] = 0.0
        df["gex_flip_price"] = None
        df["max_gamma_strike"] = pd.NA
        df["gamma_pin_dist_pct"] = pd.NA

    # Max pain distance from current price. The data-fetch path used by
    # `enrich_chain_for_scan` writes the strike to `max_pain_strike`, but the
    # older path in `fetch_options_yfinance` writes it to `max_pain` (no
    # suffix). Honor either so the dist_pct — and therefore the max_pain_score
    # — isn't pinned to its NaN-fallback constant.
    if "max_pain_strike" not in df.columns:
        df["max_pain_strike"] = pd.NA
    _mp_price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
    _mp_src = pd.to_numeric(df["max_pain_strike"], errors="coerce")
    if not _mp_src.notna().any() and "max_pain" in df.columns:
        _mp_src = pd.to_numeric(df["max_pain"], errors="coerce")
    if _mp_src.notna().any() and _mp_price_scalar > 0:
        df["max_pain_dist_pct"] = ((_mp_src - _mp_price_scalar) / _mp_price_scalar * 100).abs()
    else:
        df["max_pain_dist_pct"] = pd.NA

    # --- Option RVOL unusual activity flag ---
    if "option_rvol" in df.columns:
        df["unusual_options_activity"] = df["option_rvol"] > 5.0
    else:
        df["unusual_options_activity"] = False

    # --- OI Change (Day-over-Day) ---
    _oi_prev = load_oi_snapshot()
    if _oi_prev:
        def _oi_delta(row):
            key = f"{row.get('symbol','')}_{row.get('strike','')}_{row.get('expiration','')}_{row.get('type','')}"
            prev = _oi_prev.get(key)
            if prev is not None:
                return int(row.get("openInterest", 0)) - prev
            return 0
        df["oi_change"] = df.apply(_oi_delta, axis=1)
    else:
        df["oi_change"] = 0

    # --- Short Interest ---
    df["short_interest"] = short_interest if short_interest is not None else pd.NA

    # --- Dividend Warning ---
    df["div_warning"] = ""
    if next_ex_div is not None:
        try:
            _exp_dates = df["exp_dt"].dt.tz_localize(None).dt.date
            _div_mask = (
                (df["type"] == "call")
                & (df["abs_delta"] > 0.70)
                & (_exp_dates >= next_ex_div)
            )
            df.loc[_div_mask, "div_warning"] = f"EX-DIV {next_ex_div}"
        except Exception:
            pass

    # --- Earnings Implied Move vs Historical ---
    df["implied_earnings_move"] = pd.NA
    df["hist_earnings_move"] = pd.NA
    df["earnings_beat_rate"] = pd.NA
    df["earnings_iv_cheap"] = pd.NA
    if earnings_move_data:
        emd = earnings_move_data
        df["implied_earnings_move"] = emd.get("implied_move_pct")
        df["hist_earnings_move"] = emd.get("hist_avg_move")
        df["earnings_beat_rate"] = emd.get("hist_beat_rate")
        df["earnings_iv_cheap"] = emd.get("is_cheap")

    # --- Earnings IV Crush Prediction ---
    df["predicted_iv_crush"] = pd.NA
    df["crush_confidence"] = ""
    if earnings_move_data:
        df["predicted_iv_crush"] = earnings_move_data.get("predicted_iv_crush")
        df["crush_confidence"] = earnings_move_data.get("crush_confidence", "")

    # --- ADVANCED METRICS ---
    _em_result = calculate_expected_move(S_vals, IV_vals, T_vals)
    if _em_result is None:
        _em_result = S_vals * IV_vals * np.sqrt(T_vals)
    df["expected_move"] = _em_result
    is_call = np.char.lower(types_vals.astype(str)) == "call"

    # Probability of Profit: breakeven-based formula P(S_T > K+prem) for calls,
    # P(S_T < K-prem) for puts — correctly accounts for premium cost unlike 1-delta.
    prem_vals = df["premium"].values
    pop_arr = calculate_probability_of_profit(types_vals, S_vals, K_vals, T_vals, IV_vals, prem_vals, r=risk_free_rate, q=_q)
    if pop_arr is None:
        pop_arr = np.where(is_call, 1.0 - df["delta"].values, 1.0 + df["delta"].values)
    df["prob_profit"] = np.clip(pop_arr, 0.0, 1.0)

    _pot_result = calculate_probability_of_touch(types_vals, S_vals, K_vals, T_vals, IV_vals)
    df["prob_touch"] = _pot_result if _pot_result is not None else np.nan
    _em_factor = config.get("rr_expected_move_factor", 0.68)
    max_loss, breakeven, rr_ratio = calculate_risk_reward(types_vals, prem_vals, S_vals, K_vals, df["expected_move"].values, em_factor=_em_factor)
    df["max_loss"] = max_loss
    df["breakeven"] = breakeven
    df["rr_ratio"] = rr_ratio

    # Break-even realism
    be_vals = np.where(is_call, K_vals + prem_vals, K_vals - prem_vals)
    req_move = np.where(is_call, np.maximum(0.0, be_vals - S_vals), np.maximum(0.0, S_vals - be_vals))
    em = df["expected_move"].values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(em > 0, req_move / em, np.nan)
    em_realism = np.full_like(ratio, 0.5)
    em_realism[ratio <= 0.5] = 1.0
    em_realism[(ratio > 0.5) & (ratio <= 1.0)] = 0.7
    em_realism[ratio > 1.0] = np.maximum(0.1, em[ratio > 1.0] / (req_move[ratio > 1.0] + 1e-9))
    df["required_move"] = req_move
    df["em_realism_score"] = em_realism

    # Theta Decay Pressure: use actual Greek theta as fraction of premium
    # (replaces old premium/DTE/delta formula which inflated OTM options)
    _theta_abs = np.abs(df["theta"].values)
    _prem_safe = np.nan_to_num(df["premium"].values, nan=0.0)
    df["theta_decay_pressure"] = _theta_abs / np.maximum(_prem_safe, 0.01)

    # IV vs HV comparison
    if "hv_30d" in df.columns and df["hv_30d"].notna().any():
        df["iv_vs_hv"] = df["impliedVolatility"] - df["hv_30d"]
        df["iv_hv_ratio"] = df["impliedVolatility"] / df["hv_30d"].replace(0, float('nan'))
    else:
        df["iv_vs_hv"] = 0.0
        df["iv_hv_ratio"] = 1.0
    
    # --- Risk Checks (OI wall + gamma ramp, delegated to risk_engine) ---
    from .risk_engine import run_risk_checks
    current_price_scalar = float(S_vals[0]) if len(S_vals) > 0 else 0.0
    df = run_risk_checks(df, current_price=current_price_scalar, config=config)

    # IV Skew — 25-delta risk reversal (standard vol surface signal)
    # Finds the call and put closest to 0.25 abs_delta per expiration,
    # then computes skew = Put_25Δ_IV − Call_25Δ_IV.
    # Positive = market paying more for downside protection (fear/hedging demand).
    # Negative = call skew (momentum/squeeze regime).
    df["iv_skew"] = 0.0
    try:
        for exp, exp_grp in df.groupby("expiration"):
            calls_exp = exp_grp[exp_grp["type"] == "call"]
            puts_exp  = exp_grp[exp_grp["type"] == "put"]
            if calls_exp.empty or puts_exp.empty:
                continue
            call_25d = calls_exp.iloc[(calls_exp["abs_delta"] - 0.25).abs().argsort()[:1]]
            put_25d  = puts_exp.iloc[(puts_exp["abs_delta"] - 0.25).abs().argsort()[:1]]
            if call_25d.empty or put_25d.empty:
                continue
            put_iv = put_25d["impliedVolatility"].iloc[0]
            call_iv = call_25d["impliedVolatility"].iloc[0]
            if pd.isna(put_iv) or pd.isna(call_iv):
                continue
            skew_val = float(put_iv) - float(call_iv)
            df.loc[exp_grp.index, "iv_skew"] = skew_val
    except Exception:
        pass
    df["iv_skew"] = df["iv_skew"].fillna(0.0)

    # IV Skew Directional Alignment
    # Positive skew (put IV > call IV) = market hedging downside → favour puts
    # Negative/flat skew = normal regime → favours calls
    skew_vals = df["iv_skew"].values
    df["skew_alignment_score"] = np.where(
        df["type"] == "call",
        np.clip(0.5 - skew_vals * 4.0, 0.0, 1.0),   # calls: better when skew is low/negative
        np.clip(0.5 + skew_vals * 4.0, 0.0, 1.0),   # puts:  better when skew is positive
    )

    # Gamma/Theta Efficiency: explosive payoff potential per unit of daily time decay
    # Higher ratio = more leverage per dollar of daily premium bleed
    df["gamma_theta_ratio"] = np.abs(df["gamma"].values) / np.maximum(np.abs(df["theta"].values), 1e-9)

    # Flags
    df["liquidity_flag"] = "GOOD"
    df.loc[(df["volume"] < 10) & (df["openInterest"] < 100), "liquidity_flag"] = "POOR"
    df.loc[(df["volume"] >= 10) & (df["volume"] < 50) & (df["openInterest"] >= 100) & (df["openInterest"] < 500), "liquidity_flag"] = "FAIR"
    df["spread_flag"] = "OK"
    df.loc[df["spread_pct"] > 0.10, "spread_flag"] = "WIDE"
    df.loc[df["spread_pct"] > 0.20, "spread_flag"] = "VERY_WIDE"

    # --- Vega Dollar Exposure ---
    # Dollar P&L change per 1 volatility point (1%) move in IV, per contract (100 shares).
    # A vega_dollar of $50 means IV moving +1% adds $50 to the position value.
    df["vega_dollar"] = np.abs(df["vega"].values) * 100.0

    # --- Breakeven Distance % ---
    # What % move in the underlying is required to reach breakeven at expiration.
    # Low = more achievable; >1x expected move = structurally difficult.
    with np.errstate(divide='ignore', invalid='ignore'):
        be_dist = np.where(
            df["underlying"].values > 0,
            df["required_move"].values / df["underlying"].values * 100.0,
            np.nan
        )
    df["be_dist_pct"] = np.where(np.isfinite(be_dist), be_dist, np.nan)

    # --- Annualized Return (premium selling context) ---
    # Annualizes the premium collected relative to the strike price.
    # Standard metric for cash-secured puts / covered calls: (premium/strike) * (365/DTE).
    with np.errstate(divide='ignore', invalid='ignore'):
        ann_ret = np.where(
            (df["strike"].values > 0) & (df["T_years"].values > 0),
            (df["premium"].values / df["strike"].values) * (1.0 / df["T_years"].values),
            np.nan
        )
    df["annualized_return"] = np.where(np.isfinite(ann_ret), ann_ret, np.nan)
    
    # External data
    df["iv_rank"] = iv_rank if iv_rank is not None else pd.NA
    df["iv_percentile"] = iv_percentile if iv_percentile is not None else pd.NA
    df["event_flag"] = "OK"
    if earnings_date is not None:
        eb_days = config.get("earnings_buffer_days", 5)
        for idx, row in df.iterrows():
            if pd.notna(row["exp_dt"]):
                days_to_e = abs((row["exp_dt"].replace(tzinfo=None) - earnings_date.replace(tzinfo=None)).days)
                if days_to_e <= eb_days:
                    df.at[idx, "event_flag"] = "EARNINGS_NEARBY"
    
    # Monte Carlo
    if HAS_SIMULATION:
        n_sims = config.get("monte_carlo_simulations", 10000)
        _short_modes = {"Premium Selling", "Credit Spreads", "Iron Condor"}
        _is_short_mode = mode in _short_modes
        # Deterministic seed for reproducible PoP across runs on the same date.
        #
        # This used `hash()` on a tuple containing a string, and Python
        # randomises string hashing per process unless PYTHONHASHSEED is set —
        # so the seed, and therefore every PoP and every `quality_score` built
        # on one, differed between processes on identical input. Measured
        # 2026-08-10: up to 2.0e-02 of score movement on the same chain, which
        # is 2% of the score's range, and enough to make CI disagree with
        # itself across interpreter versions (run 31413572815, py3.11 red and
        # py3.12 green on the same commit).
        #
        # blake2b over a canonical string is stable across processes,
        # interpreters and platforms, which is what the line always claimed.
        _mc_seed_input = "|".join(str(x) for x in (
            tuple(df["underlying"].to_numpy(dtype=float, na_value=0.0)[:5]),
            tuple(df["strike"].to_numpy(dtype=float, na_value=0.0)[:5]),
            len(df),
            (as_of or datetime.now()).strftime("%Y-%m-%d"),
        ))
        _mc_seed = int.from_bytes(
            _hashlib.blake2b(_mc_seed_input.encode(), digest_size=4).digest(),
            "big") % (2**31)
        pop_arr, pot_arr = batch_monte_carlo_pop(
            S_arr=df["underlying"].to_numpy(dtype=float, na_value=0.0),
            K_arr=df["strike"].to_numpy(dtype=float, na_value=0.0),
            T_arr=df["T_years"].to_numpy(dtype=float, na_value=0.0),
            sigma_arr=df["impliedVolatility"].to_numpy(dtype=float, na_value=0.0),
            r=risk_free_rate,
            premium_arr=df["premium"].to_numpy(dtype=float, na_value=0.0),
            option_types=df["type"].to_numpy(),
            n_simulations=n_sims,
            is_short=_is_short_mode,
            random_seed=_mc_seed,
            q_arr=df["dividend_yield"].to_numpy(dtype=float, na_value=0.0) if "dividend_yield" in df.columns else None,
        )
        df["pop_sim"] = pop_arr
        df["pot_sim"] = pot_arr
    else:
        df["pop_sim"], df["pot_sim"] = pd.NA, pd.NA

    # Blend MC PoP (60%) with analytical PoP (40%) when simulation data is available.
    # MC captures path-dependency and jump risk; analytical gives a stable floor.
    if HAS_SIMULATION:
        mc_valid = df["pop_sim"].notna()
        if mc_valid.any():
            df.loc[mc_valid, "prob_profit"] = (
                0.6 * df.loc[mc_valid, "pop_sim"].astype(float)
                + 0.4 * df.loc[mc_valid, "prob_profit"]
            ).clip(0.0, 1.0)

    # Earnings IV crush: recompute PoP with crush-adjusted IV for earnings plays
    # (bakes crush into probability instead of relying solely on post-hoc score penalty)
    _earn_pop_mask = df.get("Earnings Play", pd.Series("NO", index=df.index)) == "YES"
    if _earn_pop_mask.any() and "predicted_iv_crush" in df.columns:
        _crush_raw = pd.to_numeric(df["predicted_iv_crush"], errors="coerce").fillna(0.0)
        _crush_valid = _earn_pop_mask & (_crush_raw > 0.01)
        if _crush_valid.any():
            _crush_adj_iv = IV_vals[_crush_valid] * (1.0 - _crush_raw.values[_crush_valid])
            _crush_adj_iv = np.maximum(_crush_adj_iv, 1e-9)
            _pop_crush = calculate_probability_of_profit(
                types_vals[_crush_valid], S_vals[_crush_valid], K_vals[_crush_valid],
                T_vals[_crush_valid], _crush_adj_iv, prem_vals[_crush_valid], r=risk_free_rate, q=_q,
            )
            if _pop_crush is not None:
                # Blend: 70% crush-adjusted, 30% raw (uncertainty in crush magnitude)
                df.loc[_crush_valid, "prob_profit"] = np.clip(
                    0.7 * _pop_crush + 0.3 * df.loc[_crush_valid, "prob_profit"].values, 0.0, 1.0
                )

    # For Premium Selling, flip PoP to reflect the SELLER's perspective.
    # calculate_probability_of_profit() returns the BUYER's PoP (P option expires ITM).
    # Seller profits when that same option expires worthless, so seller's PoP = 1 − buyer's PoP.
    # e.g. OTM put buyer: 30% PoP → seller: 70% PoP (which is what we want to score highly).
    if mode == "Premium Selling":
        df["prob_profit"] = (1.0 - df["prob_profit"]).clip(0.0, 1.0)

    # Theoretical value and P(ITM) using market IV (for display/reference)
    d1, d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, IV_vals, q=_q)
    p_itm = np.where(is_call, norm_cdf(d2), norm_cdf(-d2))
    disc = np.exp(-risk_free_rate * T_vals)
    _q_disc_theo = np.exp(-_q * T_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        theo_payoff = np.where(is_call,
            S_vals * _q_disc_theo * norm_cdf(d1) - K_vals * disc * norm_cdf(d2),
            K_vals * disc * norm_cdf(-d2) - S_vals * _q_disc_theo * norm_cdf(-d1))
    df["p_itm"], df["theo_value"] = p_itm, theo_payoff

    # HV-adjusted EV: BS(realized_vol) - market_price
    # Positive = options cheap vs realized vol (edge for buyers)
    # Negative = options expensive vs realized vol (edge for sellers)
    #
    # Prefer the LONG realized-vol window. EWMA (span 20) was chosen for being
    # "more responsive to recent moves", but responsiveness is the defect for a
    # multi-month option: checked against live quotes 2026-08-04, a 163-DTE
    # MSFT 535 call quoted $28.80 mid and Black-Scholes at 252-day realized
    # (31.6%) prices it at $28.85 — a $5/contract edge. The short window read
    # 51.8% off a single earnings gap and the model reported +$4,664, roughly
    # 900x overstated, on calls and puts simultaneously.
    #
    # If HV is unavailable, flag as unreliable instead of silently using IV
    # (which defeats the purpose of the EV).
    _hv_cols = [c for c in ("hv_252d", "hv_ewma", "hv_30d") if c in df.columns]
    hv_for_ev = df[_hv_cols[0]] if _hv_cols else df["hv_30d"]
    for _c in _hv_cols[1:]:
        hv_for_ev = hv_for_ev.fillna(df[_c])
    _hv_raw = hv_for_ev.fillna(df["hv_30d"])
    _hv_fallback_mask = _hv_raw.isna()
    hv_arr = np.maximum(_hv_raw.fillna(df["impliedVolatility"]).values, 1e-9)

    # Forward-looking adjustment: IV term structure slope signals whether market
    # expects vol to rise (contango) or fall (backwardation). Adjust HV used for
    # EV by ±5% based on term structure direction.
    try:
        if "expiration" in df.columns and "impliedVolatility" in df.columns:
            exp_iv_mean = df.groupby("expiration")["impliedVolatility"].transform("mean")
            exp_dte = (df["T_years"] * 365).fillna(30)
            # Per-row slope: compare this expiration's avg IV to the chain mean
            chain_iv_mean = df["impliedVolatility"].mean()
            ts_signal = np.where(
                (exp_iv_mean > chain_iv_mean * 1.02) & (exp_dte > 20), 1.05,
                np.where(
                    (exp_iv_mean < chain_iv_mean * 0.98) & (exp_dte > 20), 0.95,
                    1.0
                )
            )
            hv_arr = hv_arr * ts_signal
    except Exception as _ts_exc:
        _SCAN_WARNINGS[0] += 1
        logging.getLogger(__name__).debug("IV term structure adjustment failed: %s", _ts_exc)
    hv_d1, hv_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, hv_arr, q=_q)
    _q_disc = np.exp(-_q * T_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        hv_payoff = np.where(is_call,
            S_vals * _q_disc * norm_cdf(hv_d1) - K_vals * disc * norm_cdf(hv_d2),
            K_vals * disc * norm_cdf(-hv_d2) - S_vals * _q_disc * norm_cdf(-hv_d1))
    # Net-of-cost EV: rank by what survives REAL round-trip costs (cost is the wall).
    # Was: deducted only one half-spread (entry). Now: full round-trip spread crossing
    # (open + close) plus commission both sides. gross/cost broken out for transparency.
    from .trade_analysis import net_ev_per_contract as _net_ev
    _commission = float(config.get("paper_trading", {}).get("commission_per_contract", FALLBACK_COMMISSION_PER_CONTRACT))
    _spread_arr = df["spread_pct"].fillna(0.0).values
    _gross_edge = hv_payoff - prem_vals
    df["ev_gross_per_contract"] = 100.0 * _gross_edge
    df["ev_cost_per_contract"] = (100.0 * prem_vals * _spread_arr) + (2.0 * _commission)
    df["ev_per_contract"] = _net_ev(_gross_edge, prem_vals, _spread_arr,
                                    commission_per_contract=_commission)
    # Null out EV where HV was missing (IV fallback produces meaningless ~0 values)
    df["ev_hv_fallback"] = _hv_fallback_mask
    df.loc[_hv_fallback_mask, "ev_per_contract"] = np.nan

    # ...and where realized and implied vol are too far apart to both describe
    # one market. A stale earnings gap leaves realized far above implied and
    # the EV reads the difference as free money on every strike of the name,
    # calls and puts alike. Same treatment as a missing HV: an EV built on a
    # basis this suspect is absent, not zero.
    from .trade_analysis import implausible_vol_gap as _bad_gap
    _gap_mask = pd.Series(
        _bad_gap(_hv_raw.values, df["impliedVolatility"].values), index=df.index)
    df["ev_vol_gap_refused"] = _gap_mask
    df.loc[_gap_mask, "ev_per_contract"] = np.nan

    # The error bar the EV is judged against, carried on the row rather than
    # recomputed by each consumer. `decide_verdict`, the pick_ranking EV gate,
    # the WORTH grade and the persisted `entry_ev_noise` must all be the same
    # number; deriving it in four places is how they stop being.
    try:
        from .tearsheet.collect import ev_noise as _ev_noise
        df["ev_noise"] = [_ev_noise(r) for _, r in df.iterrows()]
    except Exception:
        logging.getLogger(__name__).debug("ev_noise unavailable", exc_info=True)
        df["ev_noise"] = np.nan

    # Earnings-adjusted EV for earnings plays
    df["ev_earnings"] = pd.NA
    _earn_mask_ev = df.get("Earnings Play", pd.Series("NO", index=df.index)) == "YES"
    if _earn_mask_ev.any():
        try:
            emd_hist = pd.to_numeric(
                df.get("hist_earnings_move", pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            emd_impl = pd.to_numeric(
                df.get("implied_earnings_move", pd.Series(np.nan, index=df.index)), errors="coerce"
            )
            eff_sigma = emd_hist.where(emd_hist.notna(), emd_impl * 1.2)
            valid_sigma = _earn_mask_ev & eff_sigma.notna() & (eff_sigma > 0)
            if valid_sigma.any():
                ev_sig_full = np.where(
                    valid_sigma.values,
                    np.maximum(eff_sigma.fillna(0).values, 1e-9),
                    hv_arr,
                )
                ev_d1, ev_d2 = _d1d2(S_vals, K_vals, T_vals, risk_free_rate, ev_sig_full, q=_q)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ev_earn_payoff = np.where(
                        is_call,
                        S_vals * _q_disc * norm_cdf(ev_d1) - K_vals * disc * norm_cdf(ev_d2),
                        K_vals * disc * norm_cdf(-ev_d2) - S_vals * _q_disc * norm_cdf(-ev_d1),
                    )
                ev_earn_raw = 100.0 * (ev_earn_payoff - prem_vals)
                df.loc[valid_sigma, "ev_earnings"] = ev_earn_raw[valid_sigma.values]
        except Exception as _ev_exc:
            logging.getLogger(__name__).warning("ev_earnings computation failed: %s", _ev_exc)

    # Warnings
    df["Theta_Burn_Rate"] = np.where(df["premium"] > 0, np.abs(df["theta"].values) / df["premium"].values, 0.0)
    df["decay_warning"] = df["Theta_Burn_Rate"] > 0.06
    df["sr_warning"] = ""
    df.loc[(df["type"] == "call") & (df["underlying"] > df["high_20"] * 0.98), "sr_warning"] = "NEAR RESISTANCE"
    df.loc[(df["type"] == "put") & (df["underlying"] < df["low_20"] * 1.02), "sr_warning"] = "NEAR SUPPORT"

    # Professional Filters
    df["macro_warning"] = "⛔ MACRO RISK" if macro_risk_active else ""
    df["max_pain_warning"] = ""
    if sector_perf:
        stock_ret, sector_ret = sector_perf.get("ticker_return", 0.0), sector_perf.get("sector_return", 0.0)
        if "max_pain" in df.columns:
            mp, und, dte = pd.to_numeric(df["max_pain"], errors='coerce'), pd.to_numeric(df["underlying"], errors='coerce'), pd.to_numeric(df["T_years"], errors='coerce') * 365.0
            mask_mp = mp.notna() & und.notna() & (dte < 3)
            df.loc[mask_mp & ((und - mp).abs() / mp > 0.05), "max_pain_warning"] = "⚠️ FIGHTING MAX PAIN"
        if stock_ret > 0 and sector_ret < -0.015:
            df["macro_warning"] = np.where(df["macro_warning"] != "", df["macro_warning"] + " | FAKE-OUT DIVERGENCE", "FAKE-OUT DIVERGENCE")
    RATE_SENSITIVE = set(config.get("rate_sensitive_tickers", ["QQQ", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX"]))
    if tnx_change_pct > 0.025:
        df["yield_warning"] = np.where(df["symbol"].isin(RATE_SENSITIVE), "📉 RATES UP", "")
    else:
        df["yield_warning"] = ""

    # SVI IV surface fitting — detect mispriced contracts vs fitted vol smile
    try:
        from .iv_surface import fit_svi_surface
        df = fit_svi_surface(df)
    except Exception as _svi_exc:
        _SCAN_WARNINGS[0] += 1
        logging.getLogger(__name__).debug("SVI surface fit failed: %s", _svi_exc)
        df["iv_surface_residual"] = np.nan
        df["iv_surface_confidence"] = 0.0
        df["iv_surface_fitted"] = False

    # NaN gate: drop contracts where ALL key columns are NaN (no usable data)
    _nan_key_cols = ["impliedVolatility", "volume", "openInterest"]
    _nan_present = [c for c in _nan_key_cols if c in df.columns]
    if _nan_present:
        _nan_count = df[_nan_present].isna().sum(axis=1)
        _nan_drop = _nan_count >= len(_nan_present)
        if _nan_drop.any():
            logging.getLogger(__name__).info(
                "NaN gate: dropped %d/%d contracts (all key columns NaN)",
                _nan_drop.sum(), len(df),
            )
            df = df[~_nan_drop].copy()

    return df


def _cross_section_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw quality_score to the display [0, 1] scale using a fixed absolute reference.

    The raw quality_score is a weighted average of 23+ components compressed into
    roughly the 0.28–0.82 range. This function stretches that to [0, 1] using a
    fixed linear mapping — no batch effects, no relative ranking, no caps.

    A contract with raw 0.55 always maps to the same display score regardless of
    what else is in the scan. Weak market days honestly show 2-star setups;
    strong market days show 3–5 star setups.

    Reference:
        raw 0.28 → 0.00  (near-minimum: all signals weak + penalties)
        raw 0.40 → 0.30  (2-star threshold)
        raw 0.50 → 0.52  (3-star threshold)
        raw 0.60 → 0.70  (4-star threshold)
        raw 0.70 → 0.85  (5-star threshold)
        raw 0.82 → 1.00  (near-maximum: all signals strong + bonuses)

    Now that the rank-based EV tiebreaker is gone (see below), this is a PURE
    per-row function: one raw score always maps to one display score. The
    ``n <= 1`` early return that used to sit here was a leftover from when the
    mapping was batch-relative, and it made a single-contract scan skip
    normalisation entirely — the same contract displaying on two scales
    depending on how many others were fetched alongside it.
    """
    if "quality_score" not in getattr(df, "columns", []) or len(df) == 0:
        return df

    raw = df["quality_score"].copy()

    # Wider range [0.28, 0.82] (span=0.54) accommodates penalty accumulation.
    # Power curve (x^0.65) expands the middle range where scores cluster.
    base = ((raw - 0.28) / 0.54).clip(0, 1)
    normalized = base ** 0.65

    # There was an "EV tiebreaker" here, reading `df["ev"]`. No path in the repo
    # ever creates a bare `ev` column — the scan carries `ev_per_contract`,
    # `ev_score` and `ev_gross_per_contract` — so the branch never once ran and
    # the docstring above described a ±0.015 tilt that was not being applied.
    # Removed rather than repointed: `ev_per_contract` already enters the raw
    # score through `ev_score`, and wiring a second, unmeasured EV tilt into the
    # display scale would be a new signal, not a bug fix. The mapping is now
    # exactly what the docstring says — a pure function of the raw score.

    df["quality_score"] = normalized.round(4)
    return df


SI_HEAVY_FRACTION = 0.20      # short interest as a share of float
SI_HEAVY_BONUS = 0.05


def _score_adjustment_flags(df: pd.DataFrame) -> pd.Series:
    """Which of the post-composite adjustments fired, per row, as a flat string.

    `quality_score` is a 27-component weighted average and then roughly twenty
    hand-set additions and multipliers. Measured 2026-08-07: those adjustments
    can subtract 1.28 and add 0.47, against a composite whose entire documented
    range spans 0.54 and whose observed spread on a clean chain was 0.29. A
    single `decay_warning` at -0.20 outweighs any component; two penalties
    outweigh all 27 together.

    Not one of those constants has ever been measured, and the ledger could not
    measure them: it stores the component scores but no record of which flags
    fired, so `flag -> outcome` was unanswerable. This records them at entry so
    it becomes answerable — the honest alternative to re-tuning twenty numbers
    by taste. It changes no score; it only writes down what happened.

    Deliberately the CONDITIONS rather than the deltas. The conditions are what
    a calibration needs to regress against, they are stable if a constant is
    later retuned, and they can be reconstructed here from columns that already
    exist without touching the 24 mutation sites.

    Format is a comma-separated sorted list of the flags that fired, empty
    string when none did — compact enough for a TEXT column, and greppable.
    """
    idx = df.index
    n = len(df)

    def _bool(col) -> pd.Series:
        s = df.get(col)
        if s is None:
            return pd.Series(False, index=idx)
        return s.fillna(False).astype(bool)

    def _nonempty(col) -> pd.Series:
        s = df.get(col)
        if s is None:
            return pd.Series(False, index=idx)
        return s.fillna("").astype(str).str.strip().ne("")

    def _num(col, default=float("nan")) -> pd.Series:
        return pd.to_numeric(df.get(col, pd.Series(default, index=idx)),
                             errors="coerce")

    earn_play = df.get("Earnings Play", pd.Series("", index=idx)).astype(str).eq("YES")
    seasonal = _num("seasonal_win_rate")
    spread_pct = _num("spread_pct").fillna(0.0)
    squeeze = _bool("squeeze_play")

    flags = {
        # Penalties
        "decay_warning": _bool("decay_warning"),
        "gamma_ramp": _bool("gamma_ramp"),
        "sr_warning": _nonempty("sr_warning"),
        "oi_wall_warning": _nonempty("oi_wall_warning"),
        "div_warning": _nonempty("div_warning"),
        "macro_risk": df.get("macro_warning", pd.Series("", index=idx))
                        .astype(str).str.contains("MACRO RISK", na=False),
        "stale_quote": df.get("quote_freshness", pd.Series("", index=idx))
                         .astype(str).eq("stale"),
        "low_pop": _num("prob_profit").lt(0.25).fillna(False),
        "spread_gt_10pct": spread_pct.gt(0.10) & spread_pct.le(0.15),
        "spread_gt_15pct": spread_pct.gt(0.15),
        "seasonal_weak": seasonal.le(0.2).fillna(False),
        # Bonuses
        "trend_aligned": _bool("Trend_Aligned"),
        "seasonal_strong": seasonal.ge(0.8).fillna(False),
        "squeeze_play": squeeze,
        "squeeze_confirmed": squeeze & _bool("Trend_Aligned"),
        "si_heavy": _num("short_interest").fillna(0.0).gt(SI_HEAVY_FRACTION),
        # Earnings — reached the score from up to five places at once, which is
        # the single biggest reason this record exists.
        "earnings_nearby": df.get("event_flag", pd.Series("", index=idx))
                             .astype(str).eq("EARNINGS_NEARBY"),
        "earnings_play": earn_play,
        "earnings_underpriced": earn_play & _bool("is_underpriced"),
        "earnings_iv_cheap": earn_play & _bool("earnings_iv_cheap"),
    }

    out = pd.Series([""] * n, index=idx, dtype=object)
    parts: list = [[] for _ in range(n)]
    for name in sorted(flags):
        mask = flags[name].reindex(idx).fillna(False).astype(bool).to_numpy()
        for pos in np.nonzero(mask)[0]:
            parts[pos].append(name)

    # risk_flag_count is the multiplier stage, and every one of the five flags
    # it counts ALSO fired as an additive penalty above — the double-count.
    # Recorded as a level so the two stages can be separated in analysis.
    #
    # `row` rather than reusing `pos`: that name is bound above from
    # np.nonzero, which yields a numpy signedinteger, and rebinding it to a
    # plain int here is the assignment mypy rejects.
    rfc = _num("risk_flag_count").fillna(0).astype(int).to_numpy()
    for row in range(n):
        if rfc[row] >= 3:
            parts[row].append(f"risk_mult_{int(rfc[row])}")
        out.iloc[row] = ",".join(parts[row])
    return out


# Scales applied to the post-composite adjustment stack, split by sign.
#
# Measured 2026-08-08 (scripts/measure_adjustment_stack.py) on closed ledger
# rows, rank IC vs friction-adjusted return:
#
#                         Long Call/Put   Short Put   negative windows
#   as shipped                -0.0995      -0.0970      5/5 , 4/5
#   stack OFF                 +0.0038      +0.0330      4/5 , 2/5
#   penalties only            -0.0291      +0.0429      5/5 , 2/5
#   BONUSES only              -0.1029      -0.1546      5/5 , 5/5
#
# The bonuses rank backwards in five windows of five in BOTH families, and rows
# the stack net-bonuses underperform the ones it net-penalises. The penalties
# are mixed and are the best single variant for Short Put, so they stay.
#
# These constants were hand-set and never fitted to this ledger, so measuring
# them negative on it is already out-of-sample evidence — unlike the IV-rank
# result that failed its holdout the same day.
#
# Set {"bonus": 1.0, "penalty": 1.0} in config.scoring.adjustment_scales to
# restore the old behaviour.
DEFAULT_ADJUSTMENT_SCALES = {"bonus": 0.0, "penalty": 1.0}


def _short_interest_bonus(df: pd.DataFrame, mode: str) -> pd.Series:
    """The heavy-short-interest bonus, pointed the way it was measured.

    This used to add ``+0.05`` to every contract on a name with SI > 20% —
    calls, puts and short premium alike. The squeeze study measures the effect
    of heavy short interest on the *shape* of the forward distribution, and the
    shape is one-sided. Over the 810,266-row panel at 42 trading days, float
    assumed at 80% of shares outstanding:

    ================================  ==========  ============
    cohort                            P(+2 sigma)  P(-2 sigma)
    ================================  ==========  ============
    SI > 20% (the bonus fired)          10.26%        3.77%
    SI <= 20%                            6.87%        5.05%
    ================================  ==========  ============

    The up-tail lifts **+3.39pp** and the down-tail *falls* **1.28pp**, 95% CI
    [-2.07, -0.51] on a moving-block bootstrap over 200 settlement dates. So a
    long call is paid for the effect and a long put is charged for it: the put's
    own tail is measurably thinner than the base rate on exactly the names that
    were collecting the bonus. Short premium is on the other side of the fatter
    tail again, and gets nothing.

    (The raw P(-20%) rate *is* higher on these names, +18.67pp — they are simply
    more volatile. That is already in the premium via IV, which is why the
    sigma-normalised tail is the one that decides the sign here.)
    """
    zero = pd.Series(0.0, index=df.index)
    if "short_interest" not in df.columns or "type" not in df.columns:
        return zero
    if mode in ("Premium Selling", "Credit Spreads", "Iron Condor"):
        return zero
    si = pd.to_numeric(df["short_interest"], errors="coerce").fillna(0.0)
    is_call = df["type"].astype(str).str.lower() == "call"
    return zero.mask((si > SI_HEAVY_FRACTION) & is_call, SI_HEAVY_BONUS)


def calculate_scores(
    df: pd.DataFrame,
    config: Dict,
    vix_regime_weights: Dict,
    trader_profile: str,
    mode: str,
    min_dte: int,
    max_dte: int,
    sector_etf: Optional[str] = None,
) -> pd.DataFrame:
    """Calculates subjective quality scores using normalization and weights."""
    
    def rank_norm(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1:
            return pd.Series([0.5] * n, index=s.index)
        r = s.rank(method="average", na_option="keep")
        return (r - 1.0) / (n - 1.0)

    def _sigmoid_scale(x, center: float = 0.0, scale: float = 12.0):
        """Smooth [0,1] mapping that preserves information at extremes."""
        return 1.0 / (1.0 + np.exp(-scale * (x - center)))

    # Base features — absolute liquidity scoring via sigmoid (cross-ticker comparable)
    _liq_cfg = config.get("liquidity_thresholds", {})
    _vol_center = _liq_cfg.get("volume_center", 500)
    _vol_scale = _liq_cfg.get("volume_scale", 0.005)
    _oi_center = _liq_cfg.get("oi_center", 2000)
    _oi_scale = _liq_cfg.get("oi_scale", 0.001)
    vol_raw = df["volume"].fillna(0).astype(float)
    oi_raw = df["openInterest"].fillna(0).astype(float)
    vol_n = pd.Series(1.0 / (1.0 + np.exp(-_vol_scale * (vol_raw.values - _vol_center))), index=df.index)
    oi_n = pd.Series(1.0 / (1.0 + np.exp(-_oi_scale * (oi_raw.values - _oi_center))), index=df.index)
    sp_cap = max(config.get("spread_score_cap", 0.25), 0.01)
    sp = pd.to_numeric(df["spread_pct"], errors="coerce").fillna(1.0).clip(lower=0)
    spread_score = pd.Series(
        1.0 / (1.0 + np.exp(20.0 * (sp.values / sp_cap - 0.7))),
        index=df.index,
    ).clip(0, 1)
    d_target = config.get("target_delta", 0.40)
    delta_quality = (1.0 - (df["abs_delta"] - d_target).abs() / max(d_target, 1e-6)).clip(0, 1)
    iv_n = rank_norm(df["impliedVolatility"].fillna(df["impliedVolatility"].median()))
    iv_quality = 1.0 - (2.0 * (iv_n - 0.5).abs())
    liquidity = 0.5 * (vol_n + oi_n)
    pop_score = df["prob_profit"].fillna(0.5).clip(0, 1)
    rr_raw = pd.to_numeric(df["rr_ratio"], errors='coerce').fillna(0.0)
    # Smooth linear mapping [0.5 → 0, 4.0 → 1] instead of hard step thresholds
    rr_score = np.clip((rr_raw - 0.5) / 3.5, 0.0, 1.0)
    _ev_for_rank = df["ev_per_contract"].copy()
    if mode == "Premium Selling":
        _ev_for_rank = -_ev_for_rank  # seller's edge = prem_vals - hv_payoff
    ev_score = rank_norm(_ev_for_rank.fillna(_ev_for_rank.median()))
    # Blend ev_score with ev_earnings_score for earnings plays (Improvement 6)
    if "ev_earnings" in df.columns and "Earnings Play" in df.columns:
        try:
            _earn_play_mask = df["Earnings Play"] == "YES"
            _ev_earn_num = pd.to_numeric(df["ev_earnings"], errors="coerce")
            _ev_earn_valid = _earn_play_mask & _ev_earn_num.notna()
            if _ev_earn_valid.any():
                _ev_earn_for_rank = _ev_earn_num.copy()
                if mode == "Premium Selling":
                    _ev_earn_for_rank = -_ev_earn_for_rank
                ev_earnings_score = rank_norm(_ev_earn_for_rank.fillna(_ev_for_rank.median()))
                ev_score = ev_score.copy()
                ev_score.loc[_ev_earn_valid] = (
                    0.5 * ev_score.loc[_ev_earn_valid]
                    + 0.5 * ev_earnings_score.loc[_ev_earn_valid]
                )
        except Exception as _ev_blend_exc:
            _SCAN_WARNINGS[0] += 1
            logging.getLogger(__name__).debug("Earnings EV blend failed: %s", _ev_blend_exc)
    em_realism_score = pd.to_numeric(df["em_realism_score"], errors='coerce').fillna(0.5).clip(0, 1)
    # Absolute rather than rank_norm over the chain. calculate_scores runs once
    # per symbol, so the rank ranked each contract only against its own chain,
    # while the composite it feeds is compared ACROSS tickers — a contract's
    # score moved with whatever else happened to be fetched. Constants are
    # frozen from the chain archive; see src/absolute_scores and
    # docs/ABSOLUTE_SCORES_20260807.md.
    theta_score = absolute_scores.theta_pressure_score(
        df["theta"], df["premium"], mode == "Premium Selling")
    theta_score = theta_score.where((df["T_years"] * 365.0) > 7, theta_score * 0.7)
    
    # Multi-timeframe momentum confluence (replaces simple momentum_score)
    ret5 = pd.to_numeric(df.get("ret_5d", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    rsi_vals = pd.to_numeric(df.get("rsi_14", pd.Series(50.0, index=df.index)), errors="coerce").fillna(50.0)
    price_vs_sma20 = (df.get("underlying", pd.Series(0.0, index=df.index)).values /
                      df.get("sma_20", pd.Series(1.0, index=df.index)).replace(0, np.nan).values) - 1.0
    price_vs_vwap = (df.get("underlying", pd.Series(0.0, index=df.index)).values /
                     df.get("vwap", pd.Series(1.0, index=df.index)).replace(0, np.nan).values) - 1.0

    is_call_mom = df["type"].str.lower() == "call"

    # Each signal returns 1 (aligned) or 0 (not aligned) for the option's direction
    # For calls: want momentum UP, RSI not overbought, price above SMA/VWAP
    # For puts: want momentum DOWN, RSI not oversold, price below SMA/VWAP
    sig_ret5 = np.where(is_call_mom, (ret5 > 0).astype(float), (ret5 < 0).astype(float))
    # ADX trend-strength gate: reduce RSI penalty by 70% when ADX > 25 (strong trend)
    # prevents penalizing overbought momentum in trending markets
    adx_vals = pd.to_numeric(df.get("adx_14", pd.Series(20.0, index=df.index)), errors="coerce").fillna(20.0)
    _rsi_penalty_mult = np.where(adx_vals > 25, 0.3, 1.0)
    sig_rsi  = np.where(is_call_mom,
                        np.clip(1.0 - _rsi_penalty_mult * (rsi_vals - 50.0).clip(0, 30) / 30.0, 0.0, 1.0),
                        np.clip(1.0 - _rsi_penalty_mult * (50.0 - rsi_vals).clip(0, 30) / 30.0, 0.0, 1.0))
    sig_sma  = np.where(is_call_mom, (price_vs_sma20 > 0).astype(float), (price_vs_sma20 < 0).astype(float))
    sig_vwap = np.where(is_call_mom, (price_vs_vwap > 0).astype(float), (price_vs_vwap < 0).astype(float))

    # Weighted confluence: configurable via momentum_weights
    mw = config.get("momentum_weights", {})
    w_rsi  = mw.get("rsi", 0.35)
    w_ret5 = mw.get("ret5d", 0.30)
    w_sma  = mw.get("sma", 0.20)
    w_vwap = mw.get("vwap", 0.15)
    momentum_score = pd.Series(
        w_ret5 * sig_ret5 + w_rsi * sig_rsi + w_sma * sig_sma + w_vwap * sig_vwap,
        index=df.index
    ).clip(0, 1)
    df["momentum_score"] = momentum_score
    df["momentum_confluence"] = momentum_score
    
    # Blend 30-day and 90-day IV percentile for a more stable IV rank signal.
    # 90-day context prevents over-reacting to short-term vol spikes.
    iv_pct_30 = pd.to_numeric(df.get("iv_percentile_30", df.get("iv_percentile", pd.Series(np.nan, index=df.index))), errors="coerce")
    iv_pct_90 = pd.to_numeric(df.get("iv_percentile_90", pd.Series(np.nan, index=df.index)), errors="coerce")
    # Where 90-day is available, blend 60/40 (30d/90d); otherwise fall back to 30d alone
    iv_pct_series = iv_pct_30.where(iv_pct_90.isna(), 0.6 * iv_pct_30 + 0.4 * iv_pct_90)
    iv_rank_score = iv_pct_series.clip(0, 1).fillna(0.5) if mode == "Premium Selling" else (1.0 - iv_pct_series.clip(0, 1)).fillna(0.5)
    # Dampen IV rank score by history confidence (Low = 0.5, Medium = 0.8, High = 1.0)
    iv_conf = df.get("iv_confidence", pd.Series("High", index=df.index))
    conf_mult = iv_conf.map({"High": 1.0, "Medium": 0.8, "Low": 0.5}).fillna(0.7)
    iv_rank_score = iv_rank_score * conf_mult
    catalyst_score = pd.Series(0.3, index=df.index).mask(df["event_flag"] == "EARNINGS_NEARBY", 0.8)
    dte_norm = ((df["T_years"] * 365.0 - min_dte) / max(1, (max_dte - min_dte))).clip(0, 1)
    trader_pref_score = (0.6 * liquidity + 0.4 * spread_score) if trader_profile.lower().startswith("day") else (0.5 * delta_quality + 0.5 * dte_norm)

    # IV Edge Score: sigmoid mapping preserves tail information (replaces ±20% linear clip)
    iv_vs_hv = df.get("iv_vs_hv", pd.Series(0.0, index=df.index)).fillna(0.0)
    if mode == "Premium Selling":
        iv_edge_score = pd.Series(_sigmoid_scale(iv_vs_hv.values, center=0.0, scale=8.0), index=df.index)
    else:
        iv_edge_score = pd.Series(_sigmoid_scale(-iv_vs_hv.values, center=0.0, scale=8.0), index=df.index)

    # IV Skew Directional Alignment (computed in calculate_metrics)
    skew_align_score = pd.to_numeric(
        df.get("skew_alignment_score", pd.Series(0.5, index=df.index)), errors='coerce'
    ).fillna(0.5).clip(0, 1)

    # Gamma/Theta Efficiency (rank-normalised, capped at 95th pct to handle outliers)
    gt_raw = pd.to_numeric(
        df.get("gamma_theta_ratio", pd.Series(0.0, index=df.index)), errors='coerce'
    ).fillna(0.0)
    gt_cap = gt_raw.quantile(0.95) if len(gt_raw) > 10 else gt_raw.max()
    gamma_theta_score = rank_norm(gt_raw.clip(upper=max(gt_cap, 1e-9))).fillna(0.5)

    # Initialise components only computed in the non-Premium-Selling branch.
    # Stays as NaN in Premium Selling mode → calibrator dropna() handles it cleanly.
    pcr_score = pd.Series(float("nan"), index=df.index)
    gex_score = pd.Series(float("nan"), index=df.index)
    oi_change_score = pd.Series(float("nan"), index=df.index)
    sentiment_score_component = pd.Series(float("nan"), index=df.index)

    # Weight Application
    if mode == "Premium Selling":
        weights = config.get("premium_selling_weights", {})
        ror_score = rank_norm(df["return_on_risk"].fillna(df["return_on_risk"].median()))
        w = {k: weights.get(k, 0.0) for k in ["pop", "return_on_risk", "iv_rank", "liquidity", "theta", "ev", "trader_pref"]}
        w["em_realism"] = weights.get("em_realism", 0.05)
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (w["pop"]*pop_score + w["return_on_risk"]*ror_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity + w["theta"]*theta_score + w["ev"]*ev_score + w["trader_pref"]*trader_pref_score + w["em_realism"]*em_realism_score) / w_sum
        try:
            _cdf = pd.DataFrame({"PoP": w["pop"]*pop_score, "RoR": w["return_on_risk"]*ror_score,
                                  "IV rank": w["iv_rank"]*iv_rank_score, "Liq": w["liquidity"]*liquidity,
                                  "Theta": w["theta"]*theta_score, "EV": w["ev"]*ev_score,
                                  "EM real": w["em_realism"]*em_realism_score}, index=df.index)
            df["score_drivers"] = _cdf.apply(lambda r: " · ".join(r.nlargest(3).index.tolist()), axis=1)
        except Exception:
            df["score_drivers"] = ""
    else:
        is_call_series = df["type"].str.lower() == "call"

        # Build weight dict early so zero-weight guards can reference it
        dw = {
            # Rebalanced: profitability (pop+rr+ev=30%), direction (momentum=10%),
            # vol signals (iv_rank+iv_edge+vrp=21%), execution (liq+spread+theta=15%)
            "pop": 0.13, "iv_mispricing": 0.05, "rr": 0.10, "momentum": 0.10,
            "iv_rank": 0.08, "liquidity": 0.08, "catalyst": 0.00, "theta": 0.06,
            "ev": 0.07, "trader_pref": 0.00, "iv_edge": 0.08, "skew_align": 0.02,
            "gamma_theta": 0.00, "pcr": 0.00, "gex": 0.01, "oi_change": 0.00,
            "sentiment": 0.00, "option_rvol": 0.00, "vrp": 0.05, "gamma_pin": 0.00,
            "max_pain": 0.01, "iv_velocity": 0.05, "em_realism": 0.00,
            "gamma_magnitude": 0.03, "vega_risk": 0.03, "term_structure": 0.04,
            "spread": 0.01,
        }
        cw = load_ic_adjusted_weights(config)
        w = {k: cw.get(k, dw[k]) for k in dw}

        # PCR score — always compute. The weight (`w["pcr"]`) only governs how
        # much this contributes to the composite. Skipping computation when
        # weight=0 pins the stored column to a constant, which prevents the
        # calibrator from ever rediscovering signal (self-reinforcing).
        pcr_vals = pd.to_numeric(df.get("pcr", pd.Series(np.nan, index=df.index)), errors="coerce")
        pcr_score_call = (1 - np.clip(pcr_vals / 2.0, 0, 1)).fillna(0.5)
        pcr_score_put = np.clip(pcr_vals / 2.0, 0, 1).fillna(0.5)
        pcr_score = pd.Series(np.where(is_call_series, pcr_score_call, pcr_score_put), index=df.index)

        # GEX score: distance-based sigmoid (replaces binary 0.3/0.7)
        gex_flip = pd.to_numeric(df.get("gex_flip_price", pd.Series(np.nan, index=df.index)), errors="coerce")
        underlying_s = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce")
        gex_flip_valid = gex_flip.notna() & (underlying_s > 0)
        gex_dist_pct = ((underlying_s - gex_flip) / underlying_s).fillna(0.0)
        # Calls: above GEX flip (positive gamma) is favorable; puts: below is favorable
        gex_score_call = pd.Series(_sigmoid_scale(gex_dist_pct.values, center=0.0, scale=30.0), index=df.index)
        gex_score_put = pd.Series(_sigmoid_scale(-gex_dist_pct.values, center=0.0, scale=30.0), index=df.index)
        gex_score = pd.Series(np.where(is_call_series, gex_score_call, gex_score_put), index=df.index)
        gex_score = gex_score.where(gex_flip_valid, 0.5)  # neutral when no flip data

        # OI change score — always compute (see PCR comment re: weight gate).
        oi_chg = pd.to_numeric(df.get("oi_change", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        oi_change_score = rank_norm(oi_chg)

        # Sentiment score — always compute.
        raw_sent = pd.to_numeric(
            df.get("sentiment_score", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0).clip(-1.0, 1.0)
        sent_call = ((raw_sent + 0.5) / 1.0).clip(0, 1)
        sent_put = ((0.5 - raw_sent) / 1.0).clip(0, 1)
        sentiment_score_component = pd.Series(
            np.where(is_call_series, sent_call, sent_put), index=df.index
        )

        # Option RVOL score — always compute.
        option_rvol_score = rank_norm(
            pd.to_numeric(df.get("option_rvol", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        ).clip(0, 1)
        df["option_rvol_score"] = option_rvol_score

        # Skew combined score: blend directional alignment with percentile rank
        iv_skew_rank_vals = pd.to_numeric(
            df.get("iv_skew_rank", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        skew_rank_score = pd.Series(
            np.where(is_call_series, 1.0 - iv_skew_rank_vals, iv_skew_rank_vals),
            index=df.index,
        ).clip(0, 1)
        skew_combined_score = (0.5 * skew_align_score + 0.5 * skew_rank_score).clip(0, 1)
        df["skew_align_score"] = skew_combined_score

        # VRP score: sigmoid scaling preserves information at extremes (vs old linear clip)
        vrp_vals = pd.to_numeric(
            df.get("vrp_mean", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        if mode == "Premium Selling":
            vrp_score = _sigmoid_scale(vrp_vals, center=0.025, scale=12.0)
        else:
            vrp_score = _sigmoid_scale(-vrp_vals, center=0.025, scale=12.0)
        df["vrp_score"] = vrp_score

        # Vega risk score: penalizes high vega exposure when IV rank is already elevated
        # High vega + high IV rank = mean-reversion risk (IV likely to compress)
        # Absolute rather than .rank(pct=True) over the chain — same reason as
        # theta above. See docs/ABSOLUTE_SCORES_20260807.md.
        vega_risk_score = absolute_scores.vega_risk_score_absolute(
            df.get("vega", pd.Series(0.0, index=df.index)),
            df.get("iv_percentile_30", pd.Series(0.5, index=df.index)))
        df["vega_risk_score"] = vega_risk_score

        # Gamma pin score — always compute. Active only for near-expiry trades
        # (dte ≤ 14) where pin risk is real; longer-dated trades get the neutral
        # 0.5 directly from the np.where, which is fine — the variance
        # available to the calibrator comes from the short-dated rows.
        gamma_pin_dist = pd.to_numeric(
            df.get("gamma_pin_dist_pct", pd.Series(100.0, index=df.index)), errors="coerce"
        ).fillna(100.0)
        dte_arr = df["T_years"].values * 365.0
        gamma_pin_score = pd.Series(
            np.where(
                dte_arr <= 14,
                np.clip(1.0 - (gamma_pin_dist.values / 10.0), 0.0, 1.0),
                0.5,
            ),
            index=df.index,
        )
        df["gamma_pin_score"] = gamma_pin_score

        # Gamma magnitude score: high gamma near expiry = outsized convexity opportunity
        gamma_vals_mag = pd.to_numeric(df.get("gamma", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        underlying_vals = pd.to_numeric(df.get("underlying", pd.Series(100.0, index=df.index)), errors="coerce").fillna(100.0)
        gamma_dollar = (gamma_vals_mag * underlying_vals).abs()
        dte_vals = df["T_years"].values * 365.0
        expiry_boost = np.where(dte_vals < 14, 1.5, np.where(dte_vals < 21, 1.2, 1.0))
        gamma_mag_raw = gamma_dollar * expiry_boost
        gamma_magnitude_score = gamma_mag_raw.rank(pct=True).fillna(0.5)
        df["gamma_magnitude_score"] = gamma_magnitude_score

        # Max pain score: near max pain is favorable for the market structure (sellers get paid)
        # Score high if within 3%, drop off linearly to 0.3 at 10%+
        max_pain_dist = pd.to_numeric(df.get("max_pain_dist_pct", pd.Series(pd.NA, index=df.index)), errors="coerce").fillna(10.0)
        max_pain_score = pd.Series(np.clip(1.0 - (max_pain_dist.values / 8.0), 0.3, 1.0), index=df.index)
        df["max_pain_score"] = max_pain_score

        # IV Velocity score: rewards sellers when IV is expanding, buyers when contracting
        iv_trend_vals = df.get("iv_trend", pd.Series("stable", index=df.index))
        if not isinstance(iv_trend_vals, pd.Series):
            iv_trend_vals = pd.Series("stable", index=df.index)
        # Sellers want IV to contract (option cheapens); buyers want IV to expand (vega profit)
        _is_seller_mode = (mode == "Premium Selling")
        iv_velocity_raw = np.where(
            iv_trend_vals == "expanding",
            0.0 if _is_seller_mode else 1.0,
            np.where(
                iv_trend_vals == "contracting",
                1.0 if _is_seller_mode else 0.0,
                0.5,
            ),
        )
        iv_velocity_score = pd.Series(iv_velocity_raw, index=df.index)
        df["iv_velocity_score"] = iv_velocity_score

        # Term structure score: contango/backwardation impacts trade viability
        ts_spread = pd.to_numeric(
            df.get("term_structure_spread", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        if mode == "Premium Selling":
            # Sellers benefit from contango (positive spread = front IV < back IV)
            term_structure_score = _sigmoid_scale(ts_spread, center=0.0, scale=8.0)
        else:
            # Buyers benefit from backwardation (negative spread = front IV > back IV)
            term_structure_score = _sigmoid_scale(-ts_spread, center=0.0, scale=8.0)
        df["term_structure_score"] = term_structure_score

        # IV surface mispricing score: rewards buying cheap vs surface, selling rich vs surface
        resid = df.get("iv_surface_residual", pd.Series(0.0, index=df.index))
        resid = pd.to_numeric(resid, errors="coerce").fillna(0.0)
        # For buyers (calls): negative residual = cheap = good → score = clip(-resid * 5, 0, 1)
        # For sellers (puts in premium selling): positive residual = rich = good
        is_sell = (mode == "Premium Selling")
        if is_sell:
            iv_mispricing_score = pd.Series(np.clip(resid * 5, 0, 1), index=df.index)
        else:
            iv_mispricing_score = pd.Series(np.clip(-resid * 5, 0, 1), index=df.index)
        # Dampen mispricing score by surface fit confidence
        # Series default, not a bare float: pd.to_numeric(1.0) returns a float,
        # which has no .fillna, so the scalar fallback raised AttributeError
        # rather than degrading. Unreachable in production — enrich_and_score
        # always sets the column, via fit_svi_surface or its except branch —
        # but it made calculate_scores uncallable on a hand-built frame.
        surf_conf = pd.to_numeric(
            df.get("iv_surface_confidence", pd.Series(1.0, index=df.index)),
            errors="coerce").fillna(0.0)
        iv_mispricing_score = iv_mispricing_score * surf_conf.clip(0, 1)
        # Bug fix: when no SVI surface fit exists (surf_conf ≈ 0), the multiplication
        # above yields 0.0 — not the neutral 0.5.  Restore neutrality for unfit contracts.
        iv_mispricing_score = iv_mispricing_score.where(surf_conf > 0.05, 0.5)
        df["iv_mispricing_score"] = iv_mispricing_score

        # Apply VIX regime multipliers — adjust weights based on current vol environment
        regime_mults = config.get("vix_regime_multipliers", {}).get(
            vix_regime_weights.get("regime", "normal") if isinstance(vix_regime_weights, dict) else "normal",
            {}
        )
        for k, mult in regime_mults.items():
            if k in w:
                w[k] *= mult
        # Re-normalize so weights sum to 1.0
        w_sum = sum(w.values()) or 1.0
        df["quality_score"] = (
            w["pop"]*pop_score + w["em_realism"]*em_realism_score + w["rr"]*rr_score
            + w["momentum"]*momentum_score + w["iv_rank"]*iv_rank_score + w["liquidity"]*liquidity
            + w["catalyst"]*catalyst_score + w["theta"]*theta_score + w["ev"]*ev_score
            + w["trader_pref"]*trader_pref_score + w["iv_edge"]*iv_edge_score
            + w["skew_align"]*skew_combined_score + w["gamma_theta"]*gamma_theta_score
            + w["pcr"]*pcr_score + w["gex"]*gex_score + w["oi_change"]*oi_change_score
            + w["sentiment"]*sentiment_score_component + w["option_rvol"]*option_rvol_score
            + w["vrp"]*vrp_score + w["gamma_pin"]*gamma_pin_score
            + w["max_pain"]*max_pain_score + w["iv_velocity"]*iv_velocity_score
            + w["iv_mispricing"]*iv_mispricing_score
            + w["gamma_magnitude"]*gamma_magnitude_score
            + w["vega_risk"]*vega_risk_score
            + w["term_structure"]*term_structure_score
            + w.get("spread", 0)*spread_score
        ) / w_sum
        try:
            _cdf = pd.DataFrame({
                "PoP": w["pop"]*pop_score, "EV": w["ev"]*ev_score,
                "RR": w["rr"]*rr_score, "IV edge": w["iv_edge"]*iv_edge_score,
                "Liq": w["liquidity"]*liquidity, "Theta": w["theta"]*theta_score,
                "Mom": w["momentum"]*momentum_score, "Skew": w["skew_align"]*skew_combined_score,
                "Sent": w["sentiment"]*sentiment_score_component,
                "VRP": w["vrp"]*vrp_score, "IV vel": w["iv_velocity"]*iv_velocity_score,
                "SVI": w["iv_mispricing"]*iv_mispricing_score,
                "Gam": w["gamma_magnitude"]*gamma_magnitude_score,
                "Vega": w["vega_risk"]*vega_risk_score,
                "TSpr": w["term_structure"]*term_structure_score,
            }, index=df.index)
            # There was a `_neg_cdf` frame here whose two columns were both
            # hardcoded to 0.0, filtered on `v < 0` — so the negative half of
            # score_drivers could never render, and never once did. Removed
            # rather than wired up, on the same reasoning as the EV tiebreaker
            # in _cross_section_normalize: the real negatives are appended
            # further down the stack as they fire (" -spread(...)",
            # " -stale_quote(-0.05)", " earnings(...)"), so nothing is lost,
            # and inventing a second negative-driver path under cover of a
            # dead-code fix would be adding display behaviour, not removing a
            # defect. See docs/SCORE_AUDIT_20260807.md item 7.
            def _fmt_drivers(row):
                top3 = row.nlargest(3)
                return " ".join(f"+{k}({v:.2f})" for k, v in top3.items() if v > 0)
            df["score_drivers"] = [
                _fmt_drivers(_cdf.iloc[i]) for i in range(len(_cdf))
            ]
        except Exception:
            df["score_drivers"] = ""

    # Snapshot the composite before the hand-set adjustments, so their net
    # effect can be rescaled by sign below — and so it stays measurable.
    df["quality_score_pre_adjust"] = df["quality_score"].astype(float)

    # Adjustments — earnings penalty scaled by historical IV crush magnitude
    _earn_nearby_mask = df["event_flag"] == "EARNINGS_NEARBY"
    if _earn_nearby_mask.any():
        is_seller = (mode == "Premium Selling")
        try:
            from .data_fetching import get_historical_iv_crush
            _ticker = df["symbol"].iloc[0] if "symbol" in df.columns else ""
            crush_data = get_historical_iv_crush(_ticker) if _ticker else {}
        except Exception:
            crush_data = {}
        avg_crush = crush_data.get("avg_crush", 0.25)
        # Continuous scaling: penalty proportional to crush magnitude
        # Buyers: high crush = bigger penalty. Sellers: high crush = opportunity
        if is_seller:
            _earn_penalty = max(0.01, min(0.08, avg_crush * 0.15))
        else:
            _earn_penalty = max(-0.15, min(-0.02, -0.05 * (avg_crush / 0.20)))
        df.loc[_earn_nearby_mask, "quality_score"] += _earn_penalty
        # Store crush info for thesis display
        df.loc[_earn_nearby_mask, "avg_iv_crush"] = avg_crush
        if "score_drivers" in df.columns:
            crush_str = f" earnings({_earn_penalty:+.2f},crush={avg_crush:.0%})"
            for _idx in df.index[_earn_nearby_mask]:
                df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + crush_str
    elif "score_drivers" in df.columns:
        pass  # no earnings nearby — no adjustment needed
    # Reward earnings plays where IV is actually underpriced vs realized vol
    if "Earnings Play" in df.columns and "is_underpriced" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["is_underpriced"]), "quality_score"] += 0.08
    df.loc[df["Trend_Aligned"], "quality_score"] += 0.05
    df.loc[df["decay_warning"], "quality_score"] -= 0.20
    # Gamma ramp: near-expiry gamma explosion is a structural risk — penalise hard
    if "gamma_ramp" in df.columns:
        df.loc[df["gamma_ramp"], "quality_score"] -= 0.15
    df.loc[df["sr_warning"] != "", "quality_score"] -= 0.10
    if "seasonal_win_rate" in df.columns:
        df.loc[df["seasonal_win_rate"] >= 0.8, "quality_score"] += 0.10
        df.loc[df["seasonal_win_rate"] <= 0.2, "quality_score"] -= 0.10
    df.loc[df["oi_wall_warning"] != "", "quality_score"] -= 0.10
    df["squeeze_play"] = (df.get("is_squeezing", pd.Series(False, index=df.index))) & (df.get("Unusual_Whale", pd.Series(False, index=df.index)))
    # Confirmed squeeze (trend-aligned) gets larger bonus; unconfirmed gets small nudge
    _squeeze_confirmed = df["squeeze_play"] & (df.get("Trend_Aligned", pd.Series(False, index=df.index)))
    df.loc[_squeeze_confirmed, "quality_score"] += 0.10
    df.loc[df["squeeze_play"] & ~_squeeze_confirmed, "quality_score"] += 0.04
    df.loc[df["macro_warning"].str.contains("MACRO RISK", na=False), "quality_score"] -= 0.10

    # Short interest squeeze potential — long calls only. See _short_interest_bonus.
    df["quality_score"] += _short_interest_bonus(df, mode)

    # Dividend early exercise warning
    if "div_warning" in df.columns:
        df.loc[df["div_warning"] != "", "quality_score"] -= 0.08

    # Earnings implied move: if IV is cheap vs historical, boost earnings plays
    if "earnings_iv_cheap" in df.columns:
        df.loc[(df["Earnings Play"] == "YES") & (df["earnings_iv_cheap"] == True), "quality_score"] += 0.06
        df.loc[(df["Earnings Play"] == "YES") & (df["earnings_iv_cheap"] == False), "quality_score"] -= 0.06

    # --- Charm / Vanna Greek Adjustments ---
    greek_adj = config.get("greek_adjustments", {})

    # Charm penalty: near-expiry OTM options with rapid delta decay
    charm_thresh  = greek_adj.get("charm_penalty_threshold", -0.05)
    charm_penalty = greek_adj.get("charm_penalty_value", -0.05)
    if "charm" in df.columns and "dte" not in df.columns:
        df["dte"] = df["T_years"] * 365.0
    if "charm" in df.columns:
        charm_mask = (df["dte"] < 7) & (pd.to_numeric(df["charm"], errors="coerce").fillna(0) < charm_thresh)
        df.loc[charm_mask, "quality_score"] += charm_penalty

    # Vanna reward: positive vanna in rising IV environment
    vanna_iv_min = greek_adj.get("vanna_reward_iv_rank_min", 0.50)
    vanna_reward  = greek_adj.get("vanna_reward_value", 0.03)
    if "vanna" in df.columns:
        iv_rank_col = df.get("iv_rank_30", pd.Series(np.nan, index=df.index))
        vanna_mask = (
            (pd.to_numeric(df["vanna"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(iv_rank_col, errors="coerce").fillna(0) > vanna_iv_min)
        )
        df.loc[vanna_mask, "quality_score"] += vanna_reward

    # Macro event penalty (sector-aware)
    try:
        from .macro_analyzer import get_macro_penalty
        _macro_pen, _macro_active, _macro_desc = get_macro_penalty(config, sector_etf=sector_etf)
        if _macro_active and _macro_pen != 0.0:
            df["quality_score"] += _macro_pen
            df["macro_event_flag"] = _macro_desc
    except Exception:
        pass

    # Tiered bid-ask spread penalty (reduced — spread_score now weighted in composite)
    _spread_pct = pd.to_numeric(df.get("spread_pct", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    _spread_penalty = pd.Series(0.0, index=df.index)
    _spread_penalty = _spread_penalty.where(_spread_pct <= 0.10, -0.04)
    _spread_penalty = _spread_penalty.where(_spread_pct <= 0.15, -0.08)
    df["quality_score"] += _spread_penalty
    # Append spread penalty to score_drivers
    if "score_drivers" in df.columns:
        _has_spread_penalty = _spread_penalty < 0
        for _idx in df.index[_has_spread_penalty]:
            _pen_val = float(_spread_penalty.loc[_idx])
            df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + f" -spread({_pen_val:.2f})"

    # Stale-quote penalty: honestly downgrade (do NOT drop) contracts whose
    # quote is stale per src.data_quality.classify_quote_freshness.
    if "quote_freshness" in df.columns:
        _stale_mask = df["quote_freshness"] == "stale"
        if _stale_mask.any():
            df.loc[_stale_mask, "quality_score"] += -0.05
            if "score_drivers" in df.columns:
                for _idx in df.index[_stale_mask]:
                    df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + " -stale_quote(-0.05)"

    # Rescale the stack's NET per-row effect by sign. bonus=0.0 suppresses the
    # upward adjustments, which measured anti-predictive in every window of
    # both strategy families; penalty=1.0 keeps the downward ones, which did
    # not. The default is EXACTLY the measured "composite + penalties only"
    # variant — that variant is `composite + stack.clip(upper=0)`, the same
    # net-per-row gate computed here, so the shipped numbers are the measured
    # ones and not an approximation of them.
    _adj_scales = dict(DEFAULT_ADJUSTMENT_SCALES)
    _adj_scales.update((config.get("scoring") or {}).get("adjustment_scales") or {})
    _pre = pd.to_numeric(df["quality_score_pre_adjust"], errors="coerce").fillna(0.0)
    _delta = pd.to_numeric(df["quality_score"], errors="coerce").fillna(0.0) - _pre
    df["quality_score"] = (
        _pre
        + float(_adj_scales.get("bonus", 0.0)) * _delta.clip(lower=0.0)
        + float(_adj_scales.get("penalty", 1.0)) * _delta.clip(upper=0.0)
    )
    df["quality_score"] = df["quality_score"].fillna(0.0).clip(0, 1)

    # PoP soft floor for buyer modes: dampen very low probability trades.
    # Lottery Ticket is exempt — low PoP is by design for far-OTM options.
    _buyer_modes = {"Single-stock", "Scan", "Watchlist"}
    _pop_floor_exempt = {"Premium Selling", "Credit Spreads", "Iron Condor", "Lottery Ticket"}
    if mode in _buyer_modes or mode not in _pop_floor_exempt:
        _low_pop = df["prob_profit"] < 0.25
        if _low_pop.any():
            df.loc[_low_pop, "quality_score"] *= 0.6
            if "score_drivers" in df.columns:
                for _idx in df.index[_low_pop]:
                    df.at[_idx, "score_drivers"] = str(df.at[_idx, "score_drivers"]) + " -low_pop(-40%)"

    # Residual earnings crush penalty (reduced — primary crush adjustment now in PoP/EV)
    if "predicted_iv_crush" in df.columns and "Earnings Play" in df.columns:
        _crush_vals = pd.to_numeric(df["predicted_iv_crush"], errors="coerce").fillna(0.0)
        _earn_mask = df["Earnings Play"] == "YES"
        crush_penalty = (_crush_vals * 0.3).clip(0, 0.06)
        df.loc[_earn_mask, "quality_score"] -= crush_penalty[_earn_mask]

    # Catastrophic risk gate: if 3+ structural risks overlap, hard-cap quality_score
    _risk_flags = pd.DataFrame({
        "gamma_ramp":       df.get("gamma_ramp", pd.Series(False, index=df.index)).astype(bool),
        "decay_warning":    df.get("decay_warning", pd.Series(False, index=df.index)).astype(bool),
        "earnings_nearby":  (df.get("event_flag", pd.Series("", index=df.index)) == "EARNINGS_NEARBY"),
        "macro_risk":       df.get("macro_warning", pd.Series("", index=df.index)).str.contains("MACRO RISK", na=False),
        "sr_warning":       df.get("sr_warning", pd.Series("", index=df.index)) != "",
    }).astype(int)
    _risk_count = _risk_flags.sum(axis=1)
    # Graduated scaling instead of hard cap: 3→×0.85, 4→×0.70, 5+→cap 0.40
    _risk_mult = pd.Series(1.0, index=df.index)
    _risk_mult[_risk_count == 3] = 0.85
    _risk_mult[_risk_count == 4] = 0.70
    _risk_mult[_risk_count >= 5] = 0.50
    df["quality_score"] = df["quality_score"] * _risk_mult
    df.loc[_risk_count >= 5, "quality_score"] = df.loc[_risk_count >= 5, "quality_score"].clip(upper=0.40)
    df["risk_flag_count"] = _risk_count

    # Save components
    df["ev_score"] = ev_score
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"], df["delta_quality"], df["iv_quality"] = liquidity, delta_quality, iv_quality
    df["spread_score"], df["theta_score"], df["momentum_score"] = spread_score, theta_score, momentum_score
    df["iv_rank_score"], df["catalyst_score"] = iv_rank_score, catalyst_score
    df["iv_advantage_score"] = iv_edge_score  # mode-aware: buyers rewarded for IV < HV
    df["pop_score"] = pop_score   # stored for backtester IC analysis
    df["rr_score"]  = rr_score    # stored for backtester IC analysis
    df["trader_pref_score"] = trader_pref_score
    df["gamma_theta_score"] = gamma_theta_score
    df["pcr_score"]         = pcr_score
    df["gex_score"]         = gex_score
    df["oi_change_score"]   = oi_change_score
    # Save normalised sentiment under a distinct name so the raw `sentiment_score`
    # input column (used by display) is preserved.
    df["sentiment_score_norm"] = sentiment_score_component

    # Long Gamma mode: compute dedicated score with inverted IV rank
    # Done after all component scores are saved so we can reference them by column name
    if mode == "Long Gamma":
        lg_w = config.get("long_gamma_weights", {
            "iv_cheap": 0.30, "squeeze": 0.25, "rvol": 0.20, "momentum": 0.15, "liquidity": 0.10
        })
        # Invert IV rank: low IV = good for buying options
        _iv_cheap = (1.0 - pd.to_numeric(
            df.get("iv_rank_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)).clip(0, 1)
        # Invert BB width: tight bands = compressed vol = primed to expand
        _bb_raw = pd.to_numeric(
            df.get("bb_width_pct", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _squeeze = (1.0 - _bb_raw).clip(0, 1)
        # Relative volume: rank-normalised (high rvol = catalyst brewing)
        _rvol_raw = pd.to_numeric(
            df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce"
        ).fillna(1.0)
        _n = len(_rvol_raw)
        _rvol_n = ((_rvol_raw.rank(method="average", na_option="keep") - 1.0) / max(_n - 1, 1)).clip(0, 1) if _n > 1 else pd.Series(0.5, index=df.index)
        _mom = pd.to_numeric(
            df.get("momentum_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _liq = pd.to_numeric(
            df.get("liquidity_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _lg_sum = sum(lg_w.values()) or 1.0
        df["long_gamma_score"] = (
            lg_w.get("iv_cheap", 0.30) * _iv_cheap
            + lg_w.get("squeeze", 0.25) * _squeeze
            + lg_w.get("rvol", 0.20) * _rvol_n
            + lg_w.get("momentum", 0.15) * _mom
            + lg_w.get("liquidity", 0.10) * _liq
        ) / _lg_sum
        # Override quality_score so existing sort/display logic works unchanged
        df["quality_score"] = df["long_gamma_score"]
    else:
        df["long_gamma_score"] = pd.NA

    # Lottery Ticket mode: score far-OTM cheap options for extreme-move potential.
    # Looks for the kind of options that explode on surprise earnings, short squeezes,
    # FDA rulings, or sudden macro events — small premium, massive upside if it moves.
    if mode == "Lottery Ticket":
        lt_w = config.get("lottery_ticket_weights", {
            "payoff": 0.25, "catalyst": 0.25, "unusual": 0.20,
            "iv_cheap": 0.15, "squeeze": 0.10, "otm_sweet": 0.05,
        })

        # Payoff asymmetry: how many multiples can this option return on a big move?
        # rr_ratio maps [2, 25] → [0, 1]; RR above 25 is capped at 1.0
        _rr = pd.to_numeric(df.get("rr_ratio", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _payoff = np.clip((_rr - 2.0) / 23.0, 0.0, 1.0)

        # Catalyst proximity: earnings nearby = 0.8, macro flag adds bonus, base 0.3
        _catalyst = pd.to_numeric(
            df.get("catalyst_score", pd.Series(0.3, index=df.index)), errors="coerce"
        ).fillna(0.3)
        # Earnings play (exp beyond earnings date) captures pre-event positioning
        if "Earnings Play" in df.columns:
            _catalyst = _catalyst.where(df["Earnings Play"] != "YES", _catalyst.clip(lower=0.6))

        # Unusual activity: rank-normalised relative volume (smart money accumulation)
        _rvol_lt = pd.to_numeric(
            df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce"
        ).fillna(1.0)
        _n_lt = len(_rvol_lt)
        _unusual = (
            (_rvol_lt.rank(method="average", na_option="keep") - 1.0) / max(_n_lt - 1, 1)
        ).clip(0, 1) if _n_lt > 1 else pd.Series(0.5, index=df.index)
        # Whale flag confirms unusual (discrete boost)
        if "Unusual_Whale" in df.columns:
            _unusual = _unusual.where(~df["Unusual_Whale"].astype(bool), (_unusual + 0.2).clip(0, 1))

        # IV cheapness: low IV percentile means the option is historically cheap → more upside
        _iv_pct_lt = pd.to_numeric(
            df.get("iv_rank_score", pd.Series(0.5, index=df.index)), errors="coerce"
        ).fillna(0.5)
        _iv_cheap_lt = (1.0 - _iv_pct_lt).clip(0, 1)

        # Squeeze potential: high short interest + unusual call buying = squeeze setup
        _si = pd.to_numeric(
            df.get("short_interest", pd.Series(0.0, index=df.index)), errors="coerce"
        ).fillna(0.0)
        # Normalise short interest: 0% → 0, 30%+ → 1.0
        _si_norm = np.clip(_si / 0.30, 0.0, 1.0)
        _whale_flag = df.get("Unusual_Whale", pd.Series(False, index=df.index)).astype(float).fillna(0.0)
        _squeeze_lt = (0.6 * _si_norm + 0.4 * _whale_flag).clip(0, 1)

        # OTM sweet spot: Gaussian centred on abs_delta=0.08 with σ=0.04.
        # Rewards the delta range where leverage and non-zero probability balance best.
        _ad_lt = pd.to_numeric(df.get("abs_delta", pd.Series(0.08, index=df.index)), errors="coerce").fillna(0.08)
        _otm_sweet = np.exp(-0.5 * ((_ad_lt - 0.08) / 0.04) ** 2)

        _lt_sum = sum(lt_w.values()) or 1.0
        df["lottery_ticket_score"] = (
            lt_w.get("payoff",    0.25) * _payoff
            + lt_w.get("catalyst",  0.25) * _catalyst
            + lt_w.get("unusual",   0.20) * _unusual
            + lt_w.get("iv_cheap",  0.15) * _iv_cheap_lt
            + lt_w.get("squeeze",   0.10) * _squeeze_lt
            + lt_w.get("otm_sweet", 0.05) * _otm_sweet
        ) / _lt_sum

        # Annotate: expected multiple if stock moves one full expected move
        if "expected_move" in df.columns and "premium" in df.columns:
            _em_lt = pd.to_numeric(df["expected_move"], errors="coerce").fillna(0.0)
            _prem_lt = pd.to_numeric(df["premium"], errors="coerce").fillna(1.0).replace(0, 1.0)
            _new_delta = (_ad_lt + 0.35).clip(0, 0.70)  # rough post-move delta approximation
            _intrinsic_at_em = (
                df.get("strike", pd.Series(0.0, index=df.index)).astype(float).sub(
                    df.get("underlying", pd.Series(0.0, index=df.index)).astype(float)
                ).abs()
            )
            # Rough option value after 1 EM move (delta * move approximation)
            _move_value = (_ad_lt * _em_lt * 2.5).clip(lower=_prem_lt * 0.1)
            df["em_multiple"] = (_move_value / _prem_lt).clip(0, 50).round(1)
        else:
            df["em_multiple"] = pd.NA

        df["quality_score"] = df["lottery_ticket_score"]

        # ── Honest lottery read: play archetype + BS-repriced metrics + edge/trap ──
        # Additive, defensive: any failure leaves the plain score untouched. Crush
        # traps (rich IV into an event) are demoted so they surface but never top
        # the board or get auto-logged. See src/lottery/{plays,metrics}.py.
        try:
            from src.lottery.plays import classify_play
            from src.lottery.metrics import contract_read, DEFAULT_EDGE_CFG
            _edge_cfg = dict(DEFAULT_EDGE_CFG)
            _edge_cfg.update(config.get("lottery_sleeve", {}) if isinstance(config, dict) else {})
            _plays, _edges, _crush, _hp = [], [], [], []
            _t1, _t2, _bem, _emm, _bve = [], [], [], [], []
            for _i in range(len(df)):
                _r = df.iloc[_i]
                _pt = classify_play(_r, _edge_cfg.get("catalyst_dte", 45),
                                    _edge_cfg.get("max_iv_rank_cheap", 0.40))
                _rd = contract_read(_r, _edge_cfg, play_type=_pt)
                _plays.append(_pt)
                _edges.append(bool(_rd["edge"]))
                _crush.append(_rd["crush_trap"] or "")
                _hp.append(_rd["hit_prob"])
                _t1.append(_rd["tail_x_1em"])
                _t2.append(_rd["tail_x_2em"])
                _bem.append(_rd["breakeven_move_pct"])
                _emm.append(_rd["expected_move_pct"])
                _bve.append(_rd["breakeven_vs_em"])
            df["lottery_play"] = _plays
            df["lottery_edge"] = _edges
            df["lottery_crush"] = _crush
            df["lottery_hit_prob"] = _hp
            df["lottery_tail1"] = _t1
            df["lottery_tail2"] = _t2
            df["lottery_be_move"] = _bem
            df["lottery_em_move"] = _emm
            df["lottery_be_vs_em"] = _bve
            # Demote crush traps: shown, never picked (0.55x keeps them below clean picks).
            _crush_mask = df["lottery_crush"].astype(bool)
            df.loc[_crush_mask, "quality_score"] = df.loc[_crush_mask, "quality_score"] * 0.55
        except Exception as _lt_exc:  # never let the read break the scan
            logging.getLogger(__name__).debug("lottery read skipped: %s", _lt_exc)
    else:
        df["lottery_ticket_score"] = pd.NA

    df["score_adjustments"] = _score_adjustment_flags(df)

    # Final clamp — the LAST thing that touches the score.
    #
    # There is a clip(0, 1) partway through the adjustment stack, but three
    # mutations follow it and none restores a floor: the residual crush penalty
    # SUBTRACTS, and the risk-flag clip is `upper=` only. So the score could go
    # negative, and then the risk multiplier inverted — at -0.030, three flags
    # gave -0.026 while five gave -0.015, i.e. MORE structural risk scoring
    # HIGHER. Masked on the logged path (the display normaliser clips at zero,
    # which is why no negative appears in 947 ledger rows) but not on surfaces
    # that read the raw score, such as the squeeze board and the filter sorts.
    df["quality_score"] = pd.to_numeric(
        df["quality_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    return df


def _compute_quote_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive mid/premium from bid/ask, trusting only usable two-sided quotes.

    A quote is usable when bid > 0, ask > 0, AND ask >= bid. Inverted (crossed)
    quotes are broken prints: their mid is meaningless and their negative
    spread would rank as the tightest in the chain, so they get mid = NaN and
    fall back to lastPrice (then die at the max-spread filter, spread = inf).
    Individually invalid bid/ask values are set to NaN so the spread math
    can't use them.
    """
    valid_bid = (df["bid"].notna()) & (df["bid"] > 0)
    valid_ask = (df["ask"].notna()) & (df["ask"] > 0)
    valid_quotes = valid_bid & valid_ask & (df["ask"] >= df["bid"])

    df["mid"] = np.where(valid_quotes, (df["bid"] + df["ask"]) / 2.0, np.nan)
    df["premium"] = df["mid"].where(df["mid"].notna() & (df["mid"] > 0.0), df["lastPrice"])

    # For spread calculation, set bid/ask to NaN if invalid (filled later)
    df.loc[~valid_bid, "bid"] = np.nan
    df.loc[~valid_ask, "ask"] = np.nan
    return df


def enrich_and_score(
    df: pd.DataFrame,
    min_dte: int,
    max_dte: int,
    risk_free_rate: float,
    config: Dict,
    vix_regime_weights: Dict,
    trader_profile: str = "swing",
    mode: str = "Single-stock",
    iv_rank: Optional[float] = None,
    iv_percentile: Optional[float] = None,
    earnings_date: Optional[datetime] = None,
    sentiment_score: Optional[float] = None,
    seasonal_win_rate: Optional[float] = None,
    term_structure_spread: Optional[float] = None,
    macro_risk_active: bool = False,
    sector_perf: Dict = {},
    tnx_change_pct: float = 0.0,
    short_interest: Optional[float] = None,
    next_ex_div: Optional[object] = None,
    earnings_move_data: Optional[dict] = None,
    hv_ewma: Optional[float] = None,
    hv_252d: Optional[float] = None,
    vrp_data: Optional[dict] = None,
    news_data=None,
    dividend_yield: float = 0.0,
    squeeze_out: Optional[Dict[str, pd.DataFrame]] = None,
    as_of: Optional[datetime] = None,
) -> pd.DataFrame:
    """Score a chain and apply the mode's filters.

    ``squeeze_out``: opt-in side channel. When Squeeze Hunt passes a dict, the
    scored calls are written to ``squeeze_out["calls"]`` before the delta band
    drops them. It is a parameter rather than a ``df.attrs`` entry because
    pandas' ``__finalize__`` compares ``attrs`` dicts across the frames being
    concatenated, and a DataFrame there raises on ``==``.
    """
    # Use richer multi-source news sentiment when available
    if news_data is not None and hasattr(news_data, "aggregate_sentiment"):
        sentiment_score = news_data.aggregate_sentiment

    # Prepare.
    #
    # `as_of` is the instant this chain is priced at. Defaulting to None keeps
    # a live scan exactly as it was — it prices at wall-clock now, which is
    # correct: a contract really does have less time left a second later.
    #
    # It is injectable because that correctness makes the scorer irreproducible.
    # `T_years` carries sub-second resolution, so two processes 1.371 seconds
    # apart produced `T_years` 1.371 seconds apart, and that propagates through
    # Black-Scholes into every Greek, `prob_profit`, `pop_score` and finally
    # `quality_score` at ~7e-8 (measured 2026-08-10). Small, but it means you
    # cannot re-run a scan and get the scan back, and it made two tests compare
    # floats that were never going to be equal.
    #
    # Not to be confused with the seed bug fixed in 4bceef5, which was worth
    # 2.0e-02 — 2% of the score's range — and was a genuine defect. This one is
    # correct behaviour with an escape hatch for reproducing a result.
    now = as_of or datetime.now(timezone.utc)
    df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
    df = df[df["exp_dt"].notna()].copy()
    df["T_years"] = (df["exp_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    df = df[(df["T_years"] >= min_dte / 365.0) & (df["T_years"] <= max_dte / 365.0)].copy()

    for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "underlying"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Data-quality provenance: classify each contract's quote freshness so it
    # survives all downstream filtering and lands in ScanResult.picks. Pure;
    # market state and classifier both come from src.data_quality.
    from src.data_quality import check_market_hours, classify_quote_freshness
    _market_open = check_market_hours()[0]
    if "quote_age_min" not in df.columns:
        df["quote_age_min"] = np.nan
    if "quote_source" not in df.columns:
        df["quote_source"] = "yfinance"
    if "quote_as_of" not in df.columns:
        df["quote_as_of"] = pd.NA
    if "data_fetched_at" not in df.columns:
        df["data_fetched_at"] = pd.NA
    df["quote_freshness"] = [
        classify_quote_freshness(a, _market_open) for a in df["quote_age_min"]
    ]

    mb = config.get("moneyness_band", 0.15)
    if "underlying" in df.columns and "strike" in df.columns:
        df = df[(df["strike"] >= df["underlying"] * (1 - mb)) & (df["strike"] <= df["underlying"] * (1 + mb))].copy()

    # Only use valid bid/ask prices (> 0), otherwise fall back to lastPrice
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["lastPrice"] = pd.to_numeric(df["lastPrice"], errors="coerce")

    # Detect systemic stale-quote condition: yfinance sometimes serves
    # bid=0/ask=0 for the whole chain even when the market is open.
    # If >=80% of rows are zero-quoted but lastPrice exists, synthesize
    # a conservative bid/ask around lastPrice so downstream filters don't
    # drop every contract. Flag on df so the caller can surface it.
    zero_quote_mask = (
        ((df["bid"].fillna(0) <= 0) & (df["ask"].fillna(0) <= 0))
        & df["lastPrice"].notna() & (df["lastPrice"] > 0)
    )
    stale_ratio = float(zero_quote_mask.mean()) if len(df) else 0.0
    stale_quotes_active = stale_ratio >= 0.80 and len(df) >= 5
    if stale_quotes_active:
        # Assume a 10% round-trip spread around lastPrice for stale rows.
        lp = df.loc[zero_quote_mask, "lastPrice"].astype(float)
        df.loc[zero_quote_mask, "bid"] = (lp * 0.95).clip(lower=0.01)
        df.loc[zero_quote_mask, "ask"] = (lp * 1.05).clip(lower=0.02)
        df.attrs["stale_quotes_active"] = True
        df.attrs["stale_quote_ratio"] = stale_ratio
        # Provenance: these bid/ask values are reconstructed, not real quotes.
        if "quote_source" in df.columns:
            df.loc[zero_quote_mask, "quote_source"] = "yfinance+synthetic_spread"

    df = _compute_quote_columns(df)

    if mode == "Premium Selling":
        df = df[df['type'] == 'put'].copy()
        if df.empty:
            return df
        df['return_on_risk'] = df['premium'] / df['strike']

    df = df[(df["premium"].notna()) & (df["premium"] > 0)].copy()
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    valid_spread = pd.to_numeric(df["spread_pct"], errors='coerce').notna() & np.isfinite(df["spread_pct"].astype(float))
    df.loc[~valid_spread, "spread_pct"] = float("inf")

    fc = config.get("filters", {})
    df = df[df["spread_pct"] <= fc.get("max_bid_ask_spread_pct", 0.40)].copy()
    df["volume"] = pd.to_numeric(df["volume"], errors='coerce').fillna(0)
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors='coerce').fillna(0)
    df = df[(df["volume"] >= fc.get("min_volume", 50)) | (df["openInterest"] >= fc.get("min_open_interest", 10))].copy()

    if df.empty:
        return df

    # IV Smile Outlier Filter: remove bad-print IV rows before enrichment
    df = filter_iv_smile_outliers(
        df,
        iv_threshold=config.get("iv_outlier_threshold", 0.30),
        min_volume=config.get("iv_outlier_min_volume", 10),
    )
    if df.empty:
        return df

    # IV cross-validation: verify Yahoo's reported IV against the IV implied by
    # each contract's own mid price (Black-Scholes inversion). The mid price is
    # more trustworthy than Yahoo's IV field on illiquid strikes, so where IV is
    # unverified (or missing) and an IV can be solved, adopt the solved IV for
    # all downstream Greeks/PoP/EV math. Contracts are flagged, never dropped.
    from src.data_quality import cross_validate_iv
    if "dividend_yield" not in df.columns:
        df["dividend_yield"] = dividend_yield
    df = cross_validate_iv(df, risk_free_rate)
    df["iv_yahoo"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    _solved_ok = df["iv_solved"].notna() & (df["iv_solved"] > 0)
    _corrected = _solved_ok & (df["iv_verified"] == False)  # noqa: E712
    df["iv_corrected"] = _corrected
    if _corrected.any():
        df.loc[_corrected, "impliedVolatility"] = df.loc[_corrected, "iv_solved"]
        _ivlog = logging.getLogger(__name__)
        # One summary line at INFO; the per-contract detail is DEBUG. The root
        # logger prints bare messages at INFO, so a per-contract loop here spews
        # dozens of lines through the middle of the report. Each corrected pick
        # already carries an "IV corrected (yahoo X% → solved Y%)" provenance tag.
        _ivlog.info("IV corrected on %d/%d contracts (Yahoo IV failed cross-check)",
                    int(_corrected.sum()), len(df))
        for _i in df.index[_corrected]:
            try:
                _ivlog.debug(
                    "IV corrected %s %s %s: yahoo %.1f%% -> solved %.1f%%",
                    df.at[_i, "symbol"], df.at[_i, "strike"], df.at[_i, "expiration"],
                    float(df.at[_i, "iv_yahoo"]) * 100.0, float(df.at[_i, "iv_solved"]) * 100.0,
                )
            except Exception:
                pass
    # Adopt solved IV where Yahoo had no usable value but a solve succeeded.
    _yahoo_missing = ~(df["iv_yahoo"] > 0) & _solved_ok
    if _yahoo_missing.any():
        df.loc[_yahoo_missing, "impliedVolatility"] = df.loc[_yahoo_missing, "iv_solved"]

    df["impliedVolatility"] = pd.to_numeric(df["impliedVolatility"], errors='coerce')
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    ov_iv_m = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(ov_iv_m if pd.notna(ov_iv_m) else 0.25)

    # Attach EWMA vol column if provided
    if hv_ewma is not None and "hv_ewma" not in df.columns:
        df["hv_ewma"] = hv_ewma
    if hv_252d is not None and "hv_252d" not in df.columns:
        df["hv_252d"] = hv_252d

    # Attach VRP data
    if vrp_data:
        df["vrp_mean"] = vrp_data.get("vrp_mean", 0.0)
        df["vrp_regime"] = vrp_data.get("vrp_regime", "UNKNOWN")
    else:
        df["vrp_mean"] = 0.0
        df["vrp_regime"] = "UNKNOWN"

    # Attach term structure spread
    if term_structure_spread is not None:
        df["term_structure_spread"] = term_structure_spread

    # 1. Call Helper: Metrics
    _div_yield = float(df["dividend_yield"].iloc[0]) if "dividend_yield" in df.columns and not df.empty else dividend_yield
    df = calculate_metrics(
        df, risk_free_rate, earnings_date, config, iv_rank, iv_percentile,
        sentiment_score, macro_risk_active, sector_perf, tnx_change_pct,
        short_interest=short_interest, next_ex_div=next_ex_div,
        earnings_move_data=earnings_move_data, mode=mode,
        dividend_yield=_div_yield, as_of=now,
    )

    # 2. Call Helper: Scores
    _sector_etf = sector_perf.get("sector_etf") if sector_perf else None
    df = calculate_scores(df, config, vix_regime_weights, trader_profile, mode, min_dte, max_dte, sector_etf=_sector_etf)

    # Strategy recommendation for Long Gamma mode
    if mode == "Long Gamma":
        _is_sq = df.get("is_squeezing", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        _adx = pd.to_numeric(df.get("adx_14", pd.Series(20.0, index=df.index)), errors="coerce").fillna(20.0)
        _rsi = pd.to_numeric(df.get("rsi_14", pd.Series(50.0, index=df.index)), errors="coerce").fillna(50.0)
        _iv_pct = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.3, index=df.index)), errors="coerce").fillna(0.3)
        _rvol = pd.to_numeric(df.get("rvol", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0)
        _underlying = pd.to_numeric(df.get("underlying", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _sma50 = pd.to_numeric(df.get("sma_50", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        _neutral_rsi = (_rsi >= 45) & (_rsi <= 55)
        _conditions = [
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct < 0.15),
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct >= 0.15),
            _is_sq & (_adx >= 20) & (_rsi > 55) & (_underlying > _sma50),
            _is_sq & (_adx >= 20) & (_rsi < 45) & (_underlying < _sma50),
            (~_is_sq) & (_rvol > 1.5) & (_adx > 25),
        ]
        _choices = ["Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Directional Debit Spread"]
        df["recommended_strategy"] = np.select(_conditions, _choices, default="Monitor")
    else:
        df["recommended_strategy"] = ""

    # Squeeze Hunt: stash the long side before the delta band removes it.
    #
    # A squeeze is expressed near the money, at |delta| ~0.5, and the generic
    # band below is 0.15-0.35 — so the calls this mode exists to find are
    # filtered out by construction, while puts on a name that just ran +20%
    # sit comfortably inside it. The 2026-08-07 run graded four SETUP names
    # and offered nine puts and no calls for exactly this reason.
    #
    # Stashed AFTER the liquidity and IV-outlier filters, so this relaxes
    # delta only: a 200%-spread contract is not a tradeable long side. Display
    # consumes this via board.call_board; it never re-enters `picks`, so
    # scoring, ranking and auto-log are untouched.
    if mode == "Squeeze Hunt" and squeeze_out is not None:
        _sq_calls = df[df["type"] == "call"].copy()
        if not _sq_calls.empty:
            # Normalised here, to the SAME display scale the main table uses.
            # The stash is taken inside enrich_and_score; _cross_section_normalize
            # runs later, on `picks` only — so the squeeze board's "Score" column
            # was showing the RAW composite while the top-N table showed the
            # normalised one, under the same header. Raw 0.55 and display 0.64
            # are the same contract; nothing on either surface said so.
            # The mapping is a pure per-row function, so applying it to this
            # frame gives exactly what `picks` would have given.
            _sq_calls = _cross_section_normalize(_sq_calls)
            # Was sorted by `quality_score`. The squeeze long side is a
            # convexity play and the composite has no standing to order it —
            # -0.131 against return on capital on long calls, and its top
            # quintile is the losing bucket. `board.call_board` ranks these on
            # convexity multiple, which is the thesis; leave the frame in the
            # order the board will re-rank rather than stamping a discredited
            # one on it first.
            squeeze_out["calls"] = _sq_calls

    # Final Filters
    if mode == "Long Gamma":
        # No delta target for straddles/strangles — they span the full delta range by design
        pass
    elif mode == "Lottery Ticket":
        # filter_lottery_ticket applies its own delta/premium/DTE gates; skip generic filter
        pass
    elif mode in ("Iron Condor", "Credit Spreads"):
        # Multi-leg modes need the deep-OTM long wings (|delta| < 0.15);
        # find_iron_condors / find_credit_spreads apply their own per-leg delta filters.
        # Cap upper |delta| so we don't carry deep-ITM contracts into the leg matcher.
        df = df[df["abs_delta"] <= fc.get("delta_max", 0.35) + 0.10].copy()
    elif mode == "Premium Selling":
        d_min = fc.get("premium_selling_delta_min", 0.15)
        d_max = fc.get("premium_selling_delta_max", 0.40)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    else:
        d_min = fc.get("delta_min", 0.15)
        d_max = fc.get("delta_max", 0.35)
        df = df[(df["abs_delta"] >= d_min) & (df["abs_delta"] <= d_max)].copy()
    if mode not in ("Premium Selling", "Long Gamma", "Iron Condor", "Credit Spreads", "Lottery Ticket"):
        df = df[df["rr_ratio"] >= 0.25].copy()

    if df.empty:
        return df

    # Sorting.
    #
    # KNOWN REMAINING USE of `quality_score` as an ordering, and the only one
    # left in the repo. It is deliberate, not an oversight.
    #
    # Everything downstream is re-ordered — `gate_and_report` refuses and then
    # sorts by carry — so this does not decide what a board shows. It does
    # decide which leg per symbol survives the per-symbol dedup and which
    # symbols reach the top-N, i.e. it is SELECTION, not display, and no
    # candidate the composite drops here is ever seen again.
    #
    # Replacing it is a behaviour change to what enters the funnel, and there
    # is no measured alternative: the composite is -0.131 against outcome on
    # long calls, but the replacement key tested on 2026-08-09 was a coin flip
    # (23 of 48 paired cells, p=0.89). Swapping a bad key for an unmeasured one
    # is not an improvement, it is a different unvalidated choice.
    #
    # To settle it: log both orderings' selections for a period and compare
    # outcomes, the way scripts/validate_gates.py settled the removal question.
    df = df.sort_values(["Unusual_Whale", "quality_score", "volume", "openInterest"], ascending=[False, False, False, False]).reset_index(drop=True)
    return df



def find_vertical_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies vertical spreads from a DataFrame of single options.
    """
    spreads = []

    # Identify "Buy" candidates
    buy_candidates = df[df["quality_score"] > 0.7].copy()

    for _, buy_leg in buy_candidates.iterrows():
        # Find potential "Sell" candidates in the same expiry
        if buy_leg["type"] == "call":
            sell_candidates = df[
                (df["expiration"] == buy_leg["expiration"]) &
                (df["type"] == buy_leg["type"]) &
                (df["symbol"] == buy_leg["symbol"]) &
                (df["strike"] > buy_leg["strike"]) & # OTM
                (df["strike"] <= buy_leg["strike"] + 2) # 1 or 2 strikes away
            ]
        else: # Put
            sell_candidates = df[
                (df["expiration"] == buy_leg["expiration"]) &
                (df["type"] == buy_leg["type"]) &
                (df["symbol"] == buy_leg["symbol"]) &
                (df["strike"] < buy_leg["strike"]) & # OTM
                (df["strike"] >= buy_leg["strike"] - 2) # 1 or 2 strikes away
            ]

        for _, sell_leg in sell_candidates.iterrows():
            if sell_leg["openInterest"] > 0 and sell_leg["volume"] > 0:
                spread_cost = buy_leg["premium"] - sell_leg["premium"]
                strike_width = abs(sell_leg["strike"] - buy_leg["strike"])
                max_profit = strike_width - spread_cost
                risk = spread_cost

                if risk > 0 and max_profit > 1.5 * risk:
                    spreads.append({
                    "symbol": buy_leg["symbol"],
                    "type": f"{buy_leg['type'].upper()} Spread",
                    "long_strike": buy_leg["strike"],
                    "short_strike": sell_leg["strike"],
                    "expiration": buy_leg["expiration"],
                    "spread_cost": spread_cost,
                    "max_profit": max_profit,
                    "risk": risk,
                    "underlying": buy_leg["underlying"]
                })

    return pd.DataFrame(spreads) if spreads else pd.DataFrame()


def find_credit_spreads(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Identifies high-probability Bull Put and Bear Call credit spreads.

    Minimum credit-to-width is read from
    ``config['filters']['credit_spreads']['min_credit_to_width']``, shipped at
    **0.20** (it was carried as a literal at both branches until 2026-08-13).

    ``1 - r`` is the HOLD-TO-EXPIRY breakeven win rate, and it is **not** the
    number to judge these trades by. Under the exits actually used
    (``config.exit_rules.spread``: TP 0.5, SL -1.0, both fractions of CREDIT)
    the required rate is set by the TP/SL ratio rather than by width. Measured
    2026-08-13 over 408 closed credit trades: Bull Put requires **50.9%** and
    delivers 66.4%, clearing its bar by 15.6pp. The families that fail their
    own breakeven are Bear Call (needs 66.7%, delivers 59.3%) and Iron Condor
    (needs 60.1%, delivers 50.0%).

    This value was tightened to 0.30 and reverted the same day: the case for
    tightening compared scan candidates at credit/width ~0.21-0.34 against
    logged trades at median credit/width 0.475 — different populations — and
    did not survive the managed-exit measurement.

    The hardcoded fallback tracks the shipped value. It is the branch that
    runs when config is missing or unreadable, and a fallback out of step with
    `config.json` silently changes which structures qualify; a test pins them
    together.
    """
    _cs_cfg = ((config or {}).get("filters", {}) or {}).get("credit_spreads") or {}
    min_c2w = float(_cs_cfg.get("min_credit_to_width", 0.20))
    spreads = []

    # --- Bull Put Spreads (Sell a Put, Buy a lower Put) ---
    # Short leg candidates: Delta between -0.15 and -0.40 (Relaxed)
    put_df = df[df['type'] == 'put'].copy()
    short_put_candidates = put_df[
        (put_df['delta'] >= -0.40) & (put_df['delta'] <= -0.15)
    ].copy()

    for _, short_leg in short_put_candidates.iterrows():
        # Find potential long legs (protection) 1 or 2 strikes lower
        # Find potential long legs (protection) 1 or 2 strikes lower
        strikes = sorted(put_df[
            (put_df['expiration'] == short_leg['expiration']) &
            (put_df['symbol'] == short_leg['symbol'])
        ]['strike'].unique(), reverse=True)

        try:
            current_strike_index = strikes.index(short_leg['strike'])
        except ValueError:
            continue

        potential_long_strikes = []
        if current_strike_index + 1 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 1])
        if current_strike_index + 2 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 2])

        long_leg_candidates = put_df[
            (put_df['expiration'] == short_leg['expiration']) &
            (put_df['symbol'] == short_leg['symbol']) &
            (put_df['strike'].isin(potential_long_strikes))
        ]

        for _, long_leg in long_leg_candidates.iterrows():
            strike_width = short_leg['strike'] - long_leg['strike']
            net_credit = short_leg['premium'] - long_leg['premium']

            # Profitability Filter: net credit must clear `min_c2w` of the
            # width. See the docstring for why this number decides the board.
            if net_credit > (min_c2w * strike_width):
                spreads.append({
                    "symbol": short_leg['symbol'],
                    "type": "Bull Put",
                    "short_strike": short_leg['strike'],
                    "long_strike": long_leg['strike'],
                    "expiration": short_leg['expiration'],
                    "net_credit": net_credit,
                    "max_profit": net_credit * 100,
                    "max_loss": (strike_width - net_credit) * 100,
                    "quality_score": (short_leg['quality_score'] + long_leg['quality_score']) / 2,
                    # Per-leg quotes so the candidate can be priced at what it
                    # would actually FILL for. `net_credit` above is built from
                    # mids (`premium` is the mid), and on the logged Bull Puts
                    # crossing cost 27% of that credit — the difference between
                    # a 58% and a 73% breakeven win rate. See candidate_verdict.
                    "short_bid": short_leg.get('bid'), "short_ask": short_leg.get('ask'),
                    "long_bid": long_leg.get('bid'), "long_ask": long_leg.get('ask'),
                    "spread_width": strike_width,
                })

    # --- Bear Call Spreads (Sell a Call, Buy a higher Call) ---
    call_df = df[df['type'] == 'call'].copy()
    # Short leg candidates: Delta between 0.15 and 0.40 (Relaxed)
    short_call_candidates = call_df[
        (call_df['delta'] >= 0.15) & (call_df['delta'] <= 0.40)
    ].copy()

    for _, short_leg in short_call_candidates.iterrows():
        # Find potential long legs (protection) 1 or 2 strikes higher
        strikes = sorted(call_df[
            (call_df['expiration'] == short_leg['expiration']) &
            (call_df['symbol'] == short_leg['symbol'])
        ]['strike'].unique())

        try:
            current_strike_index = strikes.index(short_leg['strike'])
        except ValueError:
            continue

        potential_long_strikes = []
        if current_strike_index + 1 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 1])
        if current_strike_index + 2 < len(strikes):
            potential_long_strikes.append(strikes[current_strike_index + 2])

        long_leg_candidates = call_df[
            (call_df['expiration'] == short_leg['expiration']) &
            (call_df['symbol'] == short_leg['symbol']) &
            (call_df['strike'].isin(potential_long_strikes))
        ]

        for _, long_leg in long_leg_candidates.iterrows():
            strike_width = long_leg['strike'] - short_leg['strike']
            net_credit = short_leg['premium'] - long_leg['premium']

            # Profitability Filter: net credit must clear `min_c2w` of the
            # width. See the docstring for why this number decides the board.
            if net_credit > (min_c2w * strike_width):
                spreads.append({
                    "symbol": short_leg['symbol'],
                    "type": "Bear Call",
                    "short_strike": short_leg['strike'],
                    "long_strike": long_leg['strike'],
                    "expiration": short_leg['expiration'],
                    "net_credit": net_credit,
                    "max_profit": net_credit * 100,
                    "max_loss": (strike_width - net_credit) * 100,
                    "quality_score": (short_leg['quality_score'] + long_leg['quality_score']) / 2,
                    # Per-leg quotes so the candidate can be priced at what it
                    # would actually FILL for. `net_credit` above is built from
                    # mids (`premium` is the mid), and on the logged Bull Puts
                    # crossing cost 27% of that credit — the difference between
                    # a 58% and a 73% breakeven win rate. See candidate_verdict.
                    "short_bid": short_leg.get('bid'), "short_ask": short_leg.get('ask'),
                    "long_bid": long_leg.get('bid'), "long_ask": long_leg.get('ask'),
                    "spread_width": strike_width,
                })

    # Unsorted by design. A producer that stamps an order is claiming to rank,
    # and `quality_score` cannot: -0.131 against return on capital on long
    # calls, +0.163 on Bull Puts but at p=0.06. `gate_and_report` refuses and
    # then orders by carry, so anything imposed here is either overwritten or,
    # worse, survives into a consumer that does not re-order.
    return pd.DataFrame(spreads) if spreads else pd.DataFrame()


def normalize_spreads_for_ranking(spreads_df: pd.DataFrame, mode: str = "Credit Spreads") -> pd.DataFrame:
    """
    Convert credit spread candidates into a picks-compatible DataFrame row format
    so they can be scored alongside single-leg options.

    Maps spread fields to the closest equivalent single-leg fields:
    - premium -> net_credit (the credit received is the "premium" analog)
    - prob_profit -> estimated from net_credit / spread_width (P(expire worthless))
    - delta -> short_strike delta (already computed in the source data)
    - rr_ratio -> max_profit / max_loss
    - quality_score -> existing quality_score from find_credit_spreads
    """
    if spreads_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in spreads_df.head(5).iterrows():
        net_credit = float(row.get("net_credit", 0) or 0)
        max_profit = float(row.get("max_profit", 0) or 0)
        max_loss = float(row.get("max_loss", 1) or 1)
        spread_width = abs(float(row.get("short_strike", 0)) - float(row.get("long_strike", 0)))

        # Probability of max profit: net_credit / spread_width (breakeven-based PoP proxy)
        pop_proxy = (net_credit / spread_width) if spread_width > 0 else 0.5
        pop_proxy = min(max(pop_proxy, 0.3), 0.9)

        rr = (max_profit / 100) / net_credit if net_credit > 0 else 0.0

        normalized = {
            "symbol": row.get("symbol", ""),
            "type": f"{row.get('type', 'spread').upper()} SPREAD",
            "strike": row.get("short_strike", 0),
            "expiration": row.get("expiration", ""),
            "premium": net_credit,
            "underlying": row.get("underlying", 0) if "underlying" in row else 0,
            "prob_profit": pop_proxy,
            "rr_ratio": rr,
            "quality_score": float(row.get("quality_score", 0.5)),
            "impliedVolatility": 0.0,  # not meaningful for spreads
            "delta": 0.0,
            "volume": 0,
            "openInterest": 0,
            "ev_per_contract": max_profit / 100 * pop_proxy - net_credit * (1 - pop_proxy),
            "spread_pct": 0.05,  # spreads have defined risk, treat as tight
            "_is_spread": True,
            "_spread_type": row.get("type", ""),
            "_max_profit": max_profit,
            "_max_loss": max_loss,
        }
        rows.append(normalized)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Fill required columns with safe defaults so enrich_and_score doesn't error
    for col in ["T_years", "abs_delta", "iv_rank", "iv_percentile_30", "hv_30d",
                "iv_skew_rank", "option_rvol", "gamma_pin_dist_pct", "vrp_mean"]:
        if col not in result.columns:
            result[col] = 0.0
    return result


def find_iron_condors(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Identifies Iron Condor opportunities (Bull Put Spread + Bear Call Spread).

    An Iron Condor sells premium on both sides expecting the stock to stay range-bound.
    Per-leg liquidity, delta bands, and minimum credit-to-width are tunable via
    ``config['filters']['iron_condor']``.
    """
    ic_cfg = ((config or {}).get("filters", {}) or {}).get("iron_condor", {}) or {}
    leg_min_vol = float(ic_cfg.get("leg_min_volume", 50))
    leg_min_oi = float(ic_cfg.get("leg_min_open_interest", 200))
    sd_min = float(ic_cfg.get("short_delta_min", 0.18))
    sd_max = float(ic_cfg.get("short_delta_max", 0.35))
    long_d_max = float(ic_cfg.get("long_delta_max", 0.15))
    min_c2w = float(ic_cfg.get("min_credit_to_width", 0.20))
    max_net_d = float(ic_cfg.get("max_net_delta", 0.10))

    condors = []

    # Separate puts and calls
    put_df = df[df['type'] == 'put'].copy()
    call_df = df[df['type'] == 'call'].copy()

    # Per-leg liquidity gate (config-driven)
    put_df = put_df[(put_df['volume'] > leg_min_vol) & (put_df['openInterest'] > leg_min_oi)].copy()
    call_df = call_df[(call_df['volume'] > leg_min_vol) & (call_df['openInterest'] > leg_min_oi)].copy()
    
    if put_df.empty or call_df.empty:
        return pd.DataFrame()
    
    # Group by symbol and expiration
    for (symbol, exp), group_data in df.groupby(['symbol', 'expiration']):
        put_group = put_df[(put_df['symbol'] == symbol) & (put_df['expiration'] == exp)]
        call_group = call_df[(call_df['symbol'] == symbol) & (call_df['expiration'] == exp)]
        
        if put_group.empty or call_group.empty:
            continue
        
        # --- PUT WING (Bull Put Spread) ---
        # Short Put: |delta| in [sd_min, sd_max] (signed: between -sd_max and -sd_min)
        short_put_candidates = put_group[
            (put_group['delta'] >= -sd_max) & (put_group['delta'] <= -sd_min)
        ].copy()
        
        best_put_spread = None
        best_put_ratio = 0.0

        for _, short_put in short_put_candidates.iterrows():
            # Long Put: |delta| < long_d_max (further OTM) AND lower strike
            long_put_candidates = put_group[
                (put_group['delta'].abs() < long_d_max) &
                (put_group['strike'] < short_put['strike'])
            ]

            for _, long_put in long_put_candidates.iterrows():
                put_width = short_put['strike'] - long_put['strike']
                put_credit = short_put['premium'] - long_put['premium']
                if put_width <= 0 or put_credit <= 0:
                    continue
                ratio = put_credit / put_width
                if ratio > best_put_ratio:
                    best_put_ratio = ratio
                    best_put_spread = {
                        'short_put': short_put,
                        'long_put': long_put,
                        'put_width': put_width,
                        'put_credit': put_credit
                    }
        
        if not best_put_spread:
            continue
        
        # --- CALL WING (Bear Call Spread) ---
        # Short Call: delta in [sd_min, sd_max]
        short_call_candidates = call_group[
            (call_group['delta'] >= sd_min) & (call_group['delta'] <= sd_max)
        ].copy()
        
        best_call_spread = None
        best_call_ratio = 0.0

        for _, short_call in short_call_candidates.iterrows():
            # Long Call: delta < long_d_max (further OTM) AND higher strike
            long_call_candidates = call_group[
                (call_group['delta'] < long_d_max) &
                (call_group['strike'] > short_call['strike'])
            ]

            for _, long_call in long_call_candidates.iterrows():
                call_width = long_call['strike'] - short_call['strike']
                call_credit = short_call['premium'] - long_call['premium']
                if call_width <= 0 or call_credit <= 0:
                    continue
                ratio = call_credit / call_width
                if ratio > best_call_ratio:
                    best_call_ratio = ratio
                    best_call_spread = {
                        'short_call': short_call,
                        'long_call': long_call,
                        'call_width': call_width,
                        'call_credit': call_credit
                    }
        
        if not best_call_spread:
            continue
        
        # --- COMBINE AND FILTER ---
        total_credit = best_put_spread['put_credit'] + best_call_spread['call_credit']
        max_width = max(best_put_spread['put_width'], best_call_spread['call_width'])
        max_risk = max_width - total_credit
        
        # Filter: Must collect at least min_c2w of the width as credit
        min_credit = min_c2w * max_width
        if total_credit <= min_credit or max_risk <= 0:
            continue

        # Delta Neutrality Check: |short_put_delta + short_call_delta| < max_net_d
        short_put_delta = best_put_spread['short_put']['delta']
        short_call_delta = best_call_spread['short_call']['delta']
        net_delta = short_put_delta + short_call_delta

        if abs(net_delta) >= max_net_d:
            continue  # Too directional
        
        # Calculate metrics
        return_on_risk = total_credit / max_risk if max_risk > 0 else 0
        avg_quality = (
            best_put_spread['short_put']['quality_score'] +
            best_put_spread['long_put']['quality_score'] +
            best_call_spread['short_call']['quality_score'] +
            best_call_spread['long_call']['quality_score']
        ) / 4
        
        # Per-leg quotes, so the structure can be priced at what it would
        # actually fill for. Without these `candidate_verdict._legs_of` refuses
        # every condor — four crossings against one credit is the whole reason
        # the structure is marginal, and it was invisible.
        _legs = {
            'short_put': best_put_spread['short_put'],
            'long_put': best_put_spread['long_put'],
            'short_call': best_call_spread['short_call'],
            'long_call': best_call_spread['long_call'],
        }
        _quotes = {}
        for _name, _leg in _legs.items():
            _quotes[f'{_name}_bid'] = _leg.get('bid')
            _quotes[f'{_name}_ask'] = _leg.get('ask')

        condors.append({
            'symbol': symbol,
            'expiration': exp,
            'short_put_strike': best_put_spread['short_put']['strike'],
            'long_put_strike': best_put_spread['long_put']['strike'],
            'short_call_strike': best_call_spread['short_call']['strike'],
            'long_call_strike': best_call_spread['long_call']['strike'],
            **_quotes,
            # The wider wing is what is actually at risk (max_risk = width -
            # credit), so it is the width the breakeven win rate is computed
            # against. Named spread_width to match the vertical rows and the
            # ledger column.
            'spread_width': max_width,
            'put_credit': best_put_spread['put_credit'],
            'call_credit': best_call_spread['call_credit'],
            'total_credit': total_credit,
            'max_width': max_width,
            'max_profit': total_credit * 100,  # Per contract
            'max_risk': max_risk * 100,  # Per contract
            'return_on_risk': return_on_risk,
            'net_delta': net_delta,
            'quality_score': avg_quality
        })
    
    # Unsorted by design; see find_credit_spreads. `return_on_risk` in
    # particular measures -0.216 against return on capital over 139 closed
    # condors — sorting by it put the worst candidate first.
    return pd.DataFrame(condors) if condors else pd.DataFrame()



def export_to_csv(df_picks: pd.DataFrame, mode: str,
                  budget: Optional[float] = None) -> Optional[str]:
    """Export picks to CSV with timestamp."""
    try:
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/options_picks_{mode.replace(' ', '_')}_{timestamp}.csv"
        
        # Select relevant columns for export
        export_cols = [
            "symbol", "type", "strike", "expiration", "premium", "underlying",
            "delta", "gamma", "vega", "theta", "rho", "impliedVolatility", "hv_30d", "iv_vs_hv", "iv_rank", "iv_percentile",
            "iv_rank_30", "iv_percentile_30", "iv_rank_90", "iv_percentile_90",
            "volume", "openInterest", "spread_pct", "Vol_OI_Ratio", "Unusual_Whale",
            "sentiment_score", "sentiment_tag",
            "Earnings Play", "is_underpriced",
            "prob_profit", "pop_sim", "expected_move", "required_move", "em_realism_score",
            "theta_decay_pressure", "theta_score",
            "prob_touch", "pot_sim", "p_itm",
            "max_loss", "breakeven", "rr_ratio", "return_on_risk",
            "theo_value", "ev_per_contract", "ev_earnings", "ev_score",
            "liquidity_score", "momentum_score", "iv_rank_score", "catalyst_score",
            "ret_5d", "rsi_14", "atr_trend",
            "quality_score", "liquidity_flag", "spread_flag", "event_flag", "price_bucket",
            "short_interest", "rvol", "gex_flip_price", "vwap", "high_premium_turnover",
            "quote_source", "quote_as_of", "quote_age_min", "quote_freshness",
            "iv_solved", "iv_residual_pct", "iv_verified",
        ]
        
        # Filter to existing columns
        export_cols = [c for c in export_cols if c in df_picks.columns]
        
        df_picks[export_cols].to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")
        return None


def log_trade_entry(df_picks: pd.DataFrame, mode: str) -> None:
    """Log trade entries for future P/L tracking.

    Adds a unique entry_id so rows can be reliably joined/updated later.
    """
    try:
        # Create trades_log directory if it doesn't exist
        os.makedirs("trades_log", exist_ok=True)
        
        log_file = "trades_log/entries.csv"
        file_exists = os.path.exists(log_file)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', newline='') as f:
            fieldnames = [
                'entry_id', 'timestamp', 'mode', 'symbol', 'type', 'strike', 'expiration',
                'entry_price', 'entry_underlying', 'delta', 'iv', 'hv', 'iv_rank',
                'prob_profit', 'p_itm', 'rr_ratio', 'theo_value', 'ev_per_contract',
                'quality_score', 'event_flag', 'status',
                'exit_premium', 'exit_underlying', 'exit_date', 'realized_pnl'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for _, row in df_picks.iterrows():
                entry_id = f"{datetime.now(timezone.utc).isoformat()}_{str(uuid.uuid4())[:8]}"
                writer.writerow({
                    'entry_id': entry_id,
                    'timestamp': timestamp,
                    'mode': mode,
                    'symbol': row.get('symbol', ''),
                    'type': row.get('type', ''),
                    'strike': row.get('strike', ''),
                    'expiration': row.get('expiration', ''),
                    'entry_price': row.get('premium', ''),
                    'entry_underlying': row.get('underlying', ''),
                    'delta': row.get('delta', ''),
                    'iv': row.get('impliedVolatility', ''),
                    'hv': row.get('hv_30d', ''),
                    'iv_rank': row.get('iv_rank', ''),
                    'prob_profit': row.get('prob_profit', ''),
                    'p_itm': row.get('p_itm', ''),
                    'rr_ratio': row.get('rr_ratio', ''),
                    'theo_value': row.get('theo_value', ''),
                    'ev_per_contract': row.get('ev_per_contract', ''),
                    'quality_score': row.get('quality_score', ''),
                    'event_flag': row.get('event_flag', ''),
                    'status': 'OPEN',
                    'exit_premium': '',
                    'exit_underlying': '',
                    'exit_date': '',
                    'realized_pnl': ''
                })
        
        print(f"\n  💾 Trade entries logged to {log_file}")
    except Exception as e:
        print(f"Warning: Could not log trades: {e}")


def setup_logging() -> logging.Logger:
    """Configure a simple console logger and JSONL file logger.
    LOG_LEVEL env var controls verbosity (default INFO).
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(message)s")
    logger = logging.getLogger("options_screener")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure logs are always created in the project's root 'logs' directory
    # Get the absolute path of the current script.
    script_path = os.path.abspath(__file__)
    # Navigate up two levels to get to the project root (src -> root).
    project_root = os.path.dirname(os.path.dirname(script_path))

    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.json_path = os.path.join(logs_dir, f"run_{ts}.jsonl")  # type: ignore[attr-defined]
    return logger


def log_picks_json(logger: logging.Logger, picks_df: pd.DataFrame, context: Dict):
    """Append picks to a JSONL log for later evaluation/backtesting."""
    try:
        # Create a copy to avoid modifying the original DataFrame
        log_df = picks_df.copy()

        # Convert any datetime-like columns to ISO 8601 strings
        for col in log_df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            log_df[col] = log_df[col].dt.strftime('%Y-%m-%dT%H:%M:%S%z')

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "picks": log_df.to_dict(orient="records"),
        }
        with open(logger.json_path, "a") as f:  # type: ignore[attr-defined]
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        print(f"Failed to write to log file: {e}")


_AUTO_MODE = False


def set_auto_mode(enabled: bool) -> None:
    """Record that --auto was passed, so prompts resolve to their defaults.

    Separate from the non-tty check: an operator running `run.py -ds` by hand
    has a perfectly good tty and still asked not to be prompted. Without this,
    --auto only worked when stdin happened to be closed as well.
    """
    global _AUTO_MODE
    _AUTO_MODE = bool(enabled)


def is_auto_mode() -> bool:
    return _AUTO_MODE


def suppress_prompts_for(args) -> bool:
    """Whether this invocation asked not to be prompted.

    Only --auto does. --mode/--ticker/--auto-log choose *what* gets scanned; the
    operator still has a tty and still wants the save menu when it finishes.
    Keying off the session-loop's `_interactive` flag instead swept those in and
    silently ate the Save/Export prompt on a hand-run `--ticker AAPL`. Every
    run.py shortcut that is genuinely unattended already carries --auto in its
    expansion (see run.py's _SCORING_BASE), so --auto alone is sufficient.
    """
    return bool(getattr(args, "auto", False))


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    # --auto, or a non-TTY (pipes/CI/cron): take the default rather than block.
    if _AUTO_MODE:
        return default if default is not None else ""
    if not sys.stdin.isatty() and default is not None:
        return default
    if HAS_ENHANCED_CLI:
        colored_prompt = fmt.colorize(prompt, fmt.Colors.BRIGHT_CYAN)
        if default is not None:
            colored_default = fmt.colorize(f"[{default}]", fmt.Colors.DIM)
            sys.stdout.write(f"{colored_prompt} {colored_default}: {fmt.Colors.RESET}")
        else:
            sys.stdout.write(f"{colored_prompt}: {fmt.Colors.RESET}")
        sys.stdout.flush()
        val = sys.stdin.readline().strip()
    else:
        sfx = f" [{default}]" if default is not None else ""
        val = input(f"{prompt}{sfx}: ").strip()
    return default if (not val and default is not None) else val


def _dead_scheduler_ack(dead_days: Optional[int], interactive: bool, width: int,
                        input_fn=input, print_fn=print) -> bool:
    """Escalated hard-confirm when the scheduler has been dead beyond
    `maintenance_health.DEAD_SCHEDULER_ACK_DAYS`.

    Interactive path only (`interactive` is the caller's `_interactive`, the
    same flag that guards the mode-menu loop) — automation, `--auto`,
    `--mode`/`--ticker`, and piped stdin must never block here. No bypass
    flag: exactly one Enter keypress is required, by design. Below the
    threshold — or when we can't tell how long it's been dead — behaviour is
    unchanged: no ack, the existing `health_banner` is the only output.

    Returns True iff the ack fired (for tests).
    """
    from .maintenance_health import DEAD_SCHEDULER_ACK_DAYS, dead_scheduler_ack_banner
    if not interactive or dead_days is None or dead_days <= DEAD_SCHEDULER_ACK_DAYS:
        return False
    print_fn(dead_scheduler_ack_banner(dead_days, width))
    try:
        input_fn("Press Enter to acknowledge and continue... ")
    except (EOFError, KeyboardInterrupt):
        pass
    return True


def _open_briefing_file(path: str) -> None:
    """Open a written briefing in the default browser. Never raises or hangs.

    macOS `open` blocks on LaunchServices; when that (or the browser) is
    wedged it can sit forever, and check=False used to swallow refusals — the
    report existed but nothing appeared on screen. Bound it and always leave
    the user a path they can open by hand."""
    try:
        if sys.platform == "darwin":
            import subprocess
            res = subprocess.run(["open", path], check=False, timeout=10)
            if res.returncode != 0:
                print(f"  Could not auto-open — view it at: {os.path.abspath(path)}")
        else:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(path))
    except Exception:
        try:
            print(f"  Could not auto-open — view it at: {os.path.abspath(path)}")
        except Exception:
            pass


def _run_structure_menu() -> None:
    """[13] STRUCTURE — route a directional view into the structure whose
    measured breakeven that view can clear, filtered to what the account can
    actually afford. Display-only; never logs or trades."""
    from datetime import datetime as _dt

    from src.structure.chain import fetch_candidates
    from src.structure.express import express, load_costs
    from src.structure.margins import (DEFAULT_HISTORY, apply_states,
                                       compute_league_table, load_history)
    from src.structure.report import render
    from src.structure.view import build_view

    symbol = prompt_input("Ticker", "SPY").upper().strip()
    raw_view = prompt_input(
        "Your view — [b]ullish / bea[r]ish / [n]eutral", "n").lower().strip()
    composite = {"b": 0.8, "bullish": 0.8,
                 "r": -0.8, "bearish": -0.8}.get(raw_view, 0.0)
    try:
        capital = float(prompt_input("Capital in USD", "511"))
    except (TypeError, ValueError):
        capital = 511.0

    today = _dt.now().strftime("%Y-%m-%d")
    table = apply_states(compute_league_table(),
                         load_history(DEFAULT_HISTORY), today)

    print(f"\n  Fetching {symbol} chain (30-60 DTE)...")
    cands, err = fetch_candidates(symbol, capital_usd=capital)
    if err:
        print(f"  note: {err}")

    view = build_view(symbol, composite=composite)
    commission, slippage = load_costs()
    exprs, rej = express(view, table, capital, cands,
                         commission=commission, slippage=slippage)
    print(render(view, exprs, rej, table, capital))


def _run_probability_lab_menu() -> None:
    """Probability Lab: extract the market's risk-neutral density, tilt it into
    the user's view (drift + vol multiplier), and rank listed structures by EV.

    Robust to bad input: re-prompts, validates, and never raises to the caller.
    Non-interactive callers run one pass and return.
    """
    from src import ui as _uikit
    try:
        from .probability_lab.cli import (build_context, parse_drift,
                                          render_report)
    except Exception as exc:  # pragma: no cover
        print(_uikit.error_line(f"Probability Lab unavailable: {exc}"))
        return

    _interactive = sys.stdin.isatty()
    while True:
        try:
            ticker = prompt_input("Probability Lab — ticker (or [x] back)",
                                  "").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if ticker in ("X", "BACK", "Q", "QUIT", ""):
            return
        if not ticker.isalnum():
            print("  Enter a single ticker symbol, e.g. AAPL.")
            if _interactive:
                continue
            return
        try:
            drift = parse_drift(prompt_input(
                "  Your directional view over the horizon (e.g. +3%, -2%)", "0"))
        except ValueError:
            drift = 0.0
        try:
            vol_mult = float(prompt_input(
                "  Your vol vs the market's (1.0 = same, 0.9 = calmer)", "1.0"))
        except ValueError:
            vol_mult = 1.0

        print("  Building density…")
        try:
            ctx = build_context(ticker, None, drift, vol_mult)
        except Exception as exc:
            print(_uikit.error_line(f"{ticker}: {exc}"))
            if _interactive:
                continue
            return
        print("\n".join(render_report(ctx)))
        if not _interactive:
            return


def _run_intel_menu() -> None:
    """Intel Briefing sub-menu: (a) market overview, (b) single-ticker briefing.

    Robust to bad input: re-prompts on an unrecognized choice, validates the
    ticker is non-empty alphanumeric, and never raises out to the caller.
    """
    from src import ui as _uikit
    try:
        from .intel import briefing as _brief, market as _market
    except Exception as exc:  # pragma: no cover
        print(_uikit.error_line(f"Intel module unavailable: {exc}"))
        return

    # Loop so several briefings can be run in one sitting; [x] returns to the
    # mode menu. Non-interactive callers (--mode intel, pipes) run one briefing
    # and return so they can never spin on the default choice.
    _interactive = sys.stdin.isatty()
    while True:
        try:
            choice = prompt_input(
                "Intel: [a] market overview  [b] ticker briefing  [c] macro pulse  "
                "[d] morning briefing  [e] research desk  [x] back",
                "a").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ("x", "back", "q", "quit"):
            return
        elif choice in ("c", "macro", "pulse", "3", "p"):
            try:
                from src import macro_pulse as _macro
                with _spinner("Building macro pulse (news + calendar fetch)…"):
                    pulse = _macro.run()
                print(pulse)
            except Exception as exc:
                print(_uikit.error_line(f"Could not build macro pulse: {exc}"))
        elif choice in ("a", "market", "1", "m"):
            try:
                _print_via_spinner("Gathering market overview (regime + movers)…",
                                   _market.print_market_overview,
                                   get_display_width())
            except Exception as exc:
                print(_uikit.error_line(f"Could not render market overview: {exc}"))
        elif choice in ("b", "ticker", "2", "t"):
            sym = prompt_input("Enter ticker symbol (e.g. NVDA)", "").strip().upper()
            if not sym or not sym.isalnum() or len(sym) > 6:
                print(_uikit.error_line("No valid ticker entered."))
            else:
                try:
                    _print_via_spinner(f"Building {sym} briefing (chain + news fetch)…",
                                       _brief.print_briefing, sym)
                except Exception as exc:
                    print(_uikit.error_line(f"Could not build briefing for {sym}: {exc}"))
        elif choice in ("d", "morning", "briefing", "4"):
            try:
                import src.morning as _morning
                with _uikit.spinner("Building morning briefing (up to ~30s)"):
                    result = _build_report_bounded("morning briefing",
                                                   _morning.build_and_write)
                if result is None:
                    print(_uikit.error_line(
                        f"Morning briefing stalled past {_INTEL_BUILD_TIMEOUT_S:.0f}s "
                        f"— abandoned; thread stacks in {_INTEL_HANG_LOG}"))
                else:
                    html_path, _ = result
                    print(f"  Briefing written: {html_path}")
                    _open_briefing_file(html_path)
            except Exception as exc:
                print(_uikit.error_line(f"Could not build morning briefing: {exc}"))
        elif choice in ("e", "desk", "research", "5"):
            sym = prompt_input(
                "Ticker for the deep-dive tab (blank = market only)",
                "").strip().upper()
            if sym and (not sym.isalnum() or len(sym) > 6):
                print(_uikit.error_line("Invalid ticker — building market-only desk."))
                sym = ""
            try:
                from src import research as _research
                with _uikit.spinner("Building research desk (up to ~30s)"):
                    result = _build_report_bounded("research desk",
                                                   _research.build_and_write,
                                                   symbol=sym or None)
                if result is None:
                    print(_uikit.error_line(
                        f"Research desk stalled past {_INTEL_BUILD_TIMEOUT_S:.0f}s "
                        f"— abandoned; thread stacks in {_INTEL_HANG_LOG}"))
                else:
                    html_path, _ = result
                    print(f"  Research desk written: {html_path}")
                    _open_briefing_file(html_path)
            except Exception as exc:
                print(_uikit.error_line(f"Could not build research desk: {exc}"))
        else:
            print("  Unrecognized choice. Type a / b / c / d / e, or x to go back.")

        if not _interactive:
            return


def close_trades():
    """Update trade log with closing prices and realized P/L.

    Legacy CSV log only (trades_log/entries.csv). The live book lives in
    paper_trades.db and is closed by src/maintenance.py -> update_positions;
    this path exists to settle the historical single-leg CSV entries.
    """
    import yfinance as yf

    log_file = "trades_log/entries.csv"

    if not os.path.exists(log_file):
        print("No trade log found. Run the screener first and log some trades.")
        sys.exit(1)
    
    print("=" * 80)
    print("  CLOSE TRADES - Update Trade Log with Realized P/L")
    print("=" * 80)
    
    # Read existing log
    df_trades = pd.read_csv(log_file)
    
    # Filter for OPEN trades
    open_trades = df_trades[df_trades['status'] == 'OPEN'].copy()
    
    if open_trades.empty:
        print("\nNo open trades found in log.")
        sys.exit(0)
    
    print(f"\nFound {len(open_trades)} open trades.")
    print("\nFetching current prices and calculating P/L...\n")
    
    # Header drift: rows written before the current schema carry 'entry_premium',
    # newer appends land in the same column but DictWriter never rewrites the
    # header, so accept either name rather than KeyError on the whole run.
    premium_col = next((c for c in ('entry_premium', 'entry_price')
                        if c in df_trades.columns), None)
    if premium_col is None:
        print("\n⚠️  No entry premium column found in log — nothing to settle.")
        sys.exit(1)

    updated_count = 0
    skipped_multileg = 0
    for idx, trade in open_trades.iterrows():
        symbol = trade['symbol']
        exp_date = pd.to_datetime(trade['expiration']).date()

        # Check if expired
        if exp_date > datetime.now().date():
            continue  # Skip unexpired trades

        # Single-leg intrinsic math below cannot price a spread/condor. Those
        # rows carry a strategy name in 'type' and a blank strike; settling them
        # here would write a NaN P/L and mark them CLOSED. Leave them OPEN.
        raw_type = trade['type']
        option_type = str(raw_type).strip().lower() if pd.notna(raw_type) else ''
        strike_val = safe_float(trade['strike'])  # None for blank/NaN strikes
        if option_type not in ('call', 'put') or strike_val is None:
            skipped_multileg += 1
            continue

        print(f"Processing {symbol} {option_type} ${strike_val} exp {exp_date}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get price at or near expiration
            start_date = exp_date - timedelta(days=3)
            end_date = exp_date + timedelta(days=3)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                print("  ⚠️  No price data available")
                continue
            
            # Find closest date to expiration
            hist_dates = hist.index.date
            closest_date = min(hist_dates, key=lambda d: abs((d - exp_date).days))
            filtered = hist[hist.index.date == closest_date]
            if filtered.empty:
                print("  ⚠️  No matching price for expiry date")
                continue
            exit_price = float(filtered['Close'].iloc[0])
            
            # Calculate intrinsic value at expiration (strike/type validated above)
            if option_type == 'call':
                intrinsic_value = max(0.0, exit_price - strike_val)
            else:  # put
                intrinsic_value = max(0.0, strike_val - exit_price)

            entry_price = safe_float(trade[premium_col])
            if entry_price is None:
                print("  ⚠️  No entry premium recorded — skipped")
                continue
            exit_premium = intrinsic_value
            
            # P/L per share
            pnl_per_share = exit_premium - entry_price
            realized_pnl = pnl_per_share * 100  # Per contract
            
            # Update the dataframe
            df_trades.at[idx, 'exit_premium'] = exit_premium
            df_trades.at[idx, 'exit_underlying'] = exit_price
            df_trades.at[idx, 'exit_date'] = closest_date.strftime('%Y-%m-%d')
            df_trades.at[idx, 'realized_pnl'] = realized_pnl
            df_trades.at[idx, 'status'] = 'CLOSED'
            
            updated_count += 1
            print(f"  ✓ Closed at ${exit_price:.2f} | P/L: ${realized_pnl:.2f}")
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
    
    # Save updated log
    if updated_count > 0:
        df_trades.to_csv(log_file, index=False)
        print(f"\n✓ Updated {updated_count} trades in {log_file}")
    else:
        print("\nNo trades were updated.")

    if skipped_multileg:
        print(f"  ({skipped_multileg} multi-leg/spread rows left OPEN — "
              f"single-leg intrinsic settlement does not apply to them)")
    
    print("\n" + "=" * 80)
    print("  Done!")
    print("=" * 80 + "\n")


def prompt_for_budget() -> Optional[float]:
    """Capital at risk a single position may tie up on this scan, or None.

    None means NO LIMIT and is the default: pressing ENTER must not impose a
    constraint the operator did not ask for.

    The quantity is CAPITAL AT RISK, not premium paid. For a cash-secured put
    the two differ by ~170x — AVGO pays ~$200 of credit and ties up $34,680 —
    and using the same quantity the ledger gates on means the board can never
    show a candidate that would then be refused at log time.

    Never raises and never exits. Bad input costs one re-prompt and then
    falls back to no limit: a scan dying because someone typed "5oo" is worse
    than a missing constraint.
    """
    for _ in range(2):
        raw = prompt_input(
            "Budget per position (capital at risk) in USD, ENTER for none", "")
        text = (raw or "").strip().lower().replace("$", "").replace(",", "")
        if text in ("", "none", "no", "unlimited"):
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            print("  Not a number. Enter an amount, or press ENTER for no limit.")
            continue
        if value <= 0:
            return None
        return value
    return None


def prompt_for_tickers() -> List[str]:
    """
    Prompts the user to select a ticker source and returns a list of tickers.
    """
    print("\nSelect Ticker Source:")
    print("  1. Curated Liquid (default)")
    print("  2. Top Gainers (Finviz)")
    print("  3. High IV Stocks (Finviz)")
    source_choice = prompt_input("Enter 1, 2, or 3", "1")

    if source_choice == "1":
        # Top 100 most liquid options tickers (ordered by typical volume)
        tickers = [
            # Major Indices & ETFs
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "GLD", "SLV", "TLT",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
            # Mega Cap Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "CSCO", "AVGO", "QCOM", "TXN", "AMAT", "MU", "LRCX",
            # Financial
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
            "MA", "PYPL", "XYZ", "COIN",
            # Healthcare & Pharma
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "LLY", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "CVS", "MRNA", "BNTX",
            # Consumer & Retail
            "WMT", "HD", "DIS", "NKE", "MCD", "SBUX", "TGT", "COST", "LOW", "TJX",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            # Industrial & Manufacturing
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "LMT", "RTX", "DE",
            # Communication & Media
            "T", "VZ", "CMCSA", "TMUS", "CHTR",
            # Automotive & Transportation
            "F", "GM", "RIVN", "LCID", "NIO", "UBER", "LYFT", "DAL", "UAL", "AAL"
        ]
        return tickers
    else:
        scan_type = "gainers" if source_choice == "2" else "high_iv"
        try:
            max_tickers = int(prompt_input("How many tickers to scan (1-100)", "50"))
            max_tickers = max(1, min(100, max_tickers))
            return get_dynamic_tickers(scan_type, max_tickers=max_tickers)
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)



def _score_fetched_data(
    symbol: str, data_result: dict, mode: str, min_dte: int, max_dte: int,
    rfr: float, config: dict, vix_weights: dict, trader_profile: str,
    budget=None, macro_risk_active: bool = False, tnx_change_pct: float = 0.0,
) -> dict:
    """Score and filter already-fetched options data for one symbol."""
    result: Dict[str, Any] = {
        "symbol": symbol,
        "picks": [],
        "credit_spreads": [],
        "iron_condors": [],
        "history": None,
        "success": False,
        "error": None,
    }
    try:
        df_chain = data_result["df"]
        history_df = data_result["history_df"]
        context = data_result["context"]
        result["context"] = context

        if history_df is not None and not history_df.empty:
            result["history"] = history_df

        hv = context.get("hv")
        hv_ewma = context.get("hv_ewma")
        hv_252d = context.get("hv_252d")
        iv_rank = context.get("iv_rank")
        iv_percentile = context.get("iv_percentile")
        earnings_date = context.get("earnings_date")
        earnings_move_data = context.get("earnings_move_data")
        sentiment_score = context.get("sentiment_score")
        seasonal_win_rate = context.get("seasonal_win_rate")
        term_structure_spread = context.get("term_structure_spread")
        sector_perf = context.get("sector_perf", {})
        short_interest = context.get("short_interest")
        next_ex_div = context.get("next_ex_div")
        news_data = context.get("news_data")
        vrp_data = context.get("vrp_data", {})

        context_log = []
        if hv:
            context_log.append(f"HV (30d): {hv:.2%}")
        if iv_rank:
            context_log.append(f"IV Rank: {iv_rank:.2f}")
        if earnings_date:
            context_log.append(f"Earnings: {earnings_date.strftime('%Y-%m-%d')}")
        if context.get("rvol"):
            context_log.append(f"RVOL: {context['rvol']:.2f}x")
        result["context_log"] = context_log
        result["news_data"] = news_data

        _sq_out: Dict[str, pd.DataFrame] = {}
        df_scored = enrich_and_score(
            df_chain,
            squeeze_out=_sq_out,
            min_dte=min_dte,
            max_dte=max_dte,
            risk_free_rate=rfr,
            config=config,
            vix_regime_weights=vix_weights,
            trader_profile=trader_profile,
            mode=mode,
            iv_rank=iv_rank,
            iv_percentile=iv_percentile,
            earnings_date=earnings_date,
            sentiment_score=sentiment_score,
            seasonal_win_rate=seasonal_win_rate,
            term_structure_spread=term_structure_spread,
            macro_risk_active=macro_risk_active,
            sector_perf=sector_perf,
            tnx_change_pct=tnx_change_pct,
            short_interest=short_interest,
            next_ex_div=next_ex_div,
            earnings_move_data=earnings_move_data,
            hv_ewma=hv_ewma,
            hv_252d=hv_252d,
            vrp_data=vrp_data,
            news_data=news_data,
        )

        if bool(df_scored.attrs.get("stale_quotes_active")):
            result["stale_quote_ratio"] = float(df_scored.attrs.get("stale_quote_ratio", 0.0))

        # Squeeze long side, captured before the delta band (see enrich_and_score).
        # Set before the `df_scored.empty` return below on purpose: a ticker
        # whose picks the band emptied is exactly one whose calls we still want.
        _sq_stash = _sq_out.get("calls")
        if isinstance(_sq_stash, pd.DataFrame) and not _sq_stash.empty:
            result["squeeze_calls"] = _sq_stash

        if df_scored.empty:
            result["error"] = "No contracts passed filters"
            return result

        if "symbol" not in df_scored.columns:
            result["error"] = f"'symbol' column missing from {symbol} data"
            return result

        is_budget_mode = (mode == "Budget scan")
        if is_budget_mode and budget:
            df_scored["contract_cost"] = df_scored["premium"] * 100
            df_scored = df_scored[df_scored["contract_cost"] <= budget].copy()
            if df_scored.empty:
                result["error"] = "No contracts within budget"
                return result

        if mode == "Credit Spreads":
            spreads = find_credit_spreads(df_scored, config)
            if not spreads.empty:
                spreads = enrich_credit_spreads(spreads, df_scored, config)
                result["credit_spreads"].append(spreads)
                result["success"] = True
        elif mode == "Iron Condor":
            condors = find_iron_condors(df_scored, config)
            if not condors.empty:
                condors = enrich_iron_condors(condors, df_scored, config)
                result["iron_condors"] = condors
                result["success"] = True
        elif mode == "Premium Selling":
            puts = df_scored[df_scored["type"] == "put"].copy()
            if not puts.empty:
                result["picks"].append(puts)
                result["success"] = True
        elif mode == "Long Gamma":
            from .filters import filter_long_gamma
            lg_filtered = filter_long_gamma(df_scored)
            if not lg_filtered.empty:
                result["picks"].append(lg_filtered)
                result["success"] = True
        elif mode == "Lottery Ticket":
            from .filters import filter_lottery_ticket
            lt_filtered = filter_lottery_ticket(df_scored, config)
            if not lt_filtered.empty:
                result["picks"].append(lt_filtered)
                result["success"] = True
        else:
            result["picks"].append(df_scored)
            result["success"] = True

    except Exception as e:
        import traceback, os
        tb = traceback.format_exc()
        result["error"] = str(e)
        try:
            _logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(_logdir, exist_ok=True)
            debug_path = os.path.join(_logdir, "scan_errors.log")
            # Timestamped: without it, dating a recurring failure means diffing
            # the traceback's line numbers against every commit that touched
            # this file. Same header format as data_fetching's writer.
            _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(debug_path, "a") as _f:
                _f.write(f"\n=== {_ts} | {symbol} ({mode}) ===\n{tb}\n")
        except Exception:
            pass

    return result


def process_ticker(symbol: str, mode: str, max_expiries: int, min_dte: int, max_dte: int,
                   rfr: float, config: dict, vix_weights: dict, trader_profile: str,
                   budget=None, macro_risk_active: bool = False, tnx_change_pct: float = 0.0) -> dict:
    """Thin wrapper: fetch data then score it."""
    try:
        data_result = fetch_options_yfinance(symbol, max_expiries, min_dte=min_dte, max_dte=max_dte)
        return _score_fetched_data(symbol, data_result, mode, min_dte, max_dte, rfr,
                                   config, vix_weights, trader_profile, budget,
                                   macro_risk_active, tnx_change_pct)
    except Exception as e:
        return {"symbol": symbol, "picks": [], "credit_spreads": [], "iron_condors": [],
                "history": None, "success": False, "error": str(e)}


_MULTILEG_MODES = ("Credit Spreads", "Iron Condor", "Vertical Spreads")


def _export_path(filename: str) -> str:
    """Put generated CSVs under exports/ rather than the repo root.

    Both directories already exist and are gitignored; the root is what a
    cloner sees first, and scan exports accumulating there read as though they
    were part of the project.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "exports")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError:
        return filename  # last resort: cwd, rather than failing the export
    return os.path.join(out, filename)

# Shown above the mode menu. The menu is already tall and several modes are
# tagged "no edge" or "read-only", so a newcomer cannot tell which of thirteen
# is the front of the funnel. This is the flow that works in practice:
# context first, then candidates, then drill into one, then track it.
_START_HINT = ("  New here? [10] INTEL for context, then [3] DISCOVER and "
               "drill into any pick.")


def _retry_waves_for(n_tickers: int) -> list:
    """Cool-down seconds per serial retry wave, scaled to batch size.

    The waves exist to let a big PARALLEL fetch storm subside before retrying —
    with dozens of tickers, waiting 20s then 45s meaningfully improves the hit
    rate. A single interactive ticker is not a storm: the inner fetch already
    made four attempts, so one short wave is a reasonable last try, and a 65s
    cool-down just to fail is far worse than failing fast and letting the user
    re-run.
    """
    if n_tickers <= 2:
        return [8]
    if n_tickers < 30:
        return [15, 30]
    return [20, 45]


def _config_sha(config) -> str:
    """Short content hash of the active config, for the tearsheet footer."""
    import hashlib
    import json as _json
    try:
        blob = _json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:6]
    except Exception:
        return "unknown"


def offer_tearsheet(picks_df, ctx, interactive: bool, preselect=None):
    """Offer to open an HTML tearsheet for one pick. Returns the path or None.

    Automation-safe by omission: with interactive=False and no preselect this
    returns immediately, prints nothing, and never touches input(). Cron and
    --auto runs must never be able to block here.
    """
    if picks_df is None or len(picks_df) == 0:
        return None
    if preselect is None and not interactive:
        return None
    if ctx.get("mode") in _MULTILEG_MODES:
        print("  tearsheet not supported for multi-leg structures")
        return None

    n = len(picks_df)
    choice = preselect
    if choice is None:
        try:
            raw = input("\n  Open a tearsheet?  [pick number 1-{}, Enter/n = no] > ".format(n))
        except (EOFError, KeyboardInterrupt):
            return None
        raw = (raw or "").strip()
        if not raw.isdigit():
            return None          # any non-number is "no"; never re-prompt
        choice = int(raw)
    if not (1 <= int(choice) <= n):
        return None              # out-of-range is "no", not an error

    from src.tearsheet import build, write_tearsheet
    # Board order, NOT a re-sort. This re-ranked by `quality_score`, so the
    # number the user typed indexed a different list from the one they had just
    # read — and that score is -0.131 against outcome on long calls, with its
    # top quintile the worst cell in the book. The caller passes the frame the
    # cards were numbered from; indexing it directly is what makes "pick 1"
    # mean pick 1.
    ranked = picks_df
    row = ranked.iloc[int(choice) - 1].to_dict()
    # Sibling picks give the tearsheet a real IV term structure (>=2 expiries on
    # this name). A single row can only ever produce one point, which is not a curve.
    siblings = ranked.to_dict("records")
    data = build(row, dict(ctx, rank=int(choice), n_picks=n, sibling_rows=siblings))
    html_path, _ = write_tearsheet(data)
    print("  tearsheet: {}".format(html_path))
    import webbrowser
    webbrowser.open("file://" + os.path.abspath(html_path))
    return html_path


def run_scan(mode: str, tickers: List[str], budget: Optional[float], max_expiries: int, min_dte: int, max_dte: int, trader_profile: str, logger: logging.Logger, market_trend: str, volatility_regime: str, macro_risk_active: bool = False, tnx_change_pct: float = 0.0, verbose: bool = True, custom_weights: Optional[Dict] = None, show_surface: bool = False, surface_mode: str = "braille", surface_type: str = "pnl", show_contours: bool = True, compact: bool = False, interactive: bool = False, tearsheet_pick: Optional[int] = None, session_budget: Optional[float] = None):
    """`session_budget` is the capital at risk one position may tie up, or None
    for no limit. Distinct from `budget`, which is the Budget-scan mode's cost
    of a single CONTRACT — the two are different quantities and a cash-secured
    put separates them by ~170x."""
    # Determine mode booleans for internal logic

    # === LOAD CONFIGURATION ===
    if verbose:
        print("\nLoading configuration...")
    config = load_config("config.json")

    # Merge custom weights if provided (from UI)
    if custom_weights:
        config['composite_weights'].update(custom_weights)

    if verbose:
        print("✓ Configuration loaded")

    # === SECTOR RELATIVE STRENGTH ===
    sector_ctx = None
    if config.get("sector_analysis", {}).get("enabled", True):
        try:
            from .sector_analyzer import SectorAnalyzer
            sector_ctx = SectorAnalyzer().get_sector_context()
            if verbose and sector_ctx.top_sectors:
                print(f"  Sector leaders: {', '.join(sector_ctx.top_sectors)}")
                if sector_ctx.mean_reversion_setups:
                    print(f"  Mean-reversion setups: {', '.join(sector_ctx.mean_reversion_setups)}")
        except Exception as _sa_exc:
            logger.warning("SectorAnalyzer failed: %s", _sa_exc)

    # === FETCH VIX FOR ADAPTIVE WEIGHTING ===
    with _spinner("Fetching VIX level for adaptive scoring…"):
        vix_level = get_vix_level()
    if verbose:
        if vix_level:
            print(f"✓ VIX Level: {vix_level:.2f}")
        else:
            print("⚠️  Could not fetch VIX, using default weights")

    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    if verbose:
        print(f"✓ Market Regime: {vix_regime.upper()}")

    # Fetch risk-free rate automatically
    with _spinner("Fetching current risk-free rate…"):
        rfr = get_risk_free_rate()
    if verbose:
        print(f"Using risk-free rate: {rfr*100:.2f}% (13-week Treasury)")

    # Collect data from all tickers (PARALLEL PROCESSING)
    tickers = list(set(tickers))  # Deduplicate tickers

    # Discovery scan: sort tickers in top-3 RS sectors to the front of the queue
    if mode == "Discovery scan" and sector_ctx and sector_ctx.top_sectors:
        from .data_fetching import SECTOR_MAP as _SM
        _top_set = set(sector_ctx.top_sectors)
        tickers = sorted(tickers, key=lambda s: 0 if _SM.get(s.upper()) in _top_set else 1)

    all_picks = []
    all_credit_spreads = []
    all_iron_condors = []
    ticker_histories = {} # For Portfolio Protection

    WIDTH = get_display_width()
    if verbose:
        if HAS_ENHANCED_CLI:
            print("\n" + fmt.draw_box(f"Scanning {len(tickers)} ticker(s)", WIDTH))
        else:
            print(f"\n{'='*WIDTH}")
            print(f"  Fetching data for {len(tickers)} ticker(s)...")
            print(f"{'='*WIDTH}\n")

    # ── Pre-scan active filter summary ───────────────────────────────────────
    if verbose:
        _fc = config.get("filters", {})
        _d_min = _fc.get("delta_min", 0.15)
        _d_max = _fc.get("delta_max", 0.35)
        _spread_cap = _fc.get("max_bid_ask_spread_pct", 0.40)
        _min_vol = _fc.get("min_volume", 50)
        _iv_pct_min = _fc.get("min_iv_percentile", 20)
        _filter_line = (
            f"  Filters  DTE: {min_dte}\u2013{max_dte}d"
            f"  |  \u0394: {_d_min:.2f}\u2013{_d_max:.2f}"
            f"  |  Spread \u2264{_spread_cap*100:.0f}%"
            f"  |  Vol \u2265{_min_vol}"
            f"  |  IV%ile \u2265{_iv_pct_min}"
        )
        print(fmt.colorize(_filter_line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else _filter_line)

    results_buffer: Dict[str, Any] = {}

    # Phase 1 — Pre-fetch all chains in parallel with a progress bar
    def _fetch_one(sym: str):
        try:
            return sym, fetch_options_yfinance(sym, max_expiries, min_dte=min_dte, max_dte=max_dte)
        except Exception as exc:
            return sym, {"error": str(exc)}

    def _is_transient_err(msg: str) -> bool:
        """Errors worth retrying serially: Yahoo rate-limits, transient network failures, empty-expiration races."""
        if not msg:
            return False
        m = msg.lower()
        return (
            "too many requests" in m
            or "rate limited" in m
            or "no options expirations available" in m
            or "no options data" in m
            or "could not resolve host" in m
            or "could not fetch price history" in m
            or "failed to perform" in m
        )

    raw_results: Dict[str, Any] = {}
    # Scale outer fetch concurrency with the scan size. For big scans (>=80 tickers),
    # drop parallelism to 3 so we don't exceed Yahoo's ~60-100 req/min ceiling in
    # the first 30s. The adaptive throttle in data_fetching widens intervals and
    # quarantines further once 429s start cascading.
    if len(tickers) >= 80:
        _fetch_workers = 2
    elif len(tickers) >= 30:
        _fetch_workers = 3
    else:
        _fetch_workers = min(len(tickers), 5)
    with _suppress_scan_noise():
        with ThreadPoolExecutor(max_workers=_fetch_workers) as executor:
            _future_map = {executor.submit(_fetch_one, sym): sym for sym in tickers}
            if HAS_ENHANCED_CLI and verbose:
                bar_fmt = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                _pbar = tqdm(
                    total=len(tickers), desc="  Fetching", unit="",
                    leave=False, dynamic_ncols=False, bar_format=bar_fmt, file=sys.stdout,
                )
            else:
                _pbar = None
            for _fut in as_completed(_future_map):
                _sym, _res = _fut.result()
                raw_results[_sym] = _res
                if _pbar is not None:
                    _pbar.update(1)
            if _pbar is not None:
                _pbar.close()
                print(flush=True)  # clean line after bar clears

    # Phase 1b — Serial retry pass for transient failures (rate limits, empty-expirations, DNS blips).
    # After the parallel storm subsides, Yahoo tends to serve the same tickers cleanly.
    # Up to two retry waves: first after 20s cooldown (spacing 1.5s), second after another 45s if any remain.
    import time as _time
    for _wave_idx, _wave_cooldown in enumerate(_retry_waves_for(len(tickers))):
        _retry_syms = [
            s for s in tickers
            if isinstance(raw_results.get(s), dict) and _is_transient_err(raw_results[s].get("error", ""))
        ]
        if not _retry_syms:
            break
        if verbose:
            _retry_msg = (
                f"  Retry wave {_wave_idx + 1}: {len(_retry_syms)} ticker(s) "
                f"(cooling down {_wave_cooldown}s, then serial with 1.5s spacing)..."
            )
            print(fmt.colorize(_retry_msg, fmt.Colors.DIM) if HAS_ENHANCED_CLI else _retry_msg)
        _time.sleep(_wave_cooldown)
        with _suppress_scan_noise():
            for _rs in _retry_syms:
                _sym2, _res2 = _fetch_one(_rs)
                if isinstance(_res2, dict) and "error" not in _res2:
                    raw_results[_sym2] = _res2
                _time.sleep(1.5)

    # Phase 2 — Score each fetched result.
    # Measured at 1.90s for a single ticker, so a full scan spends minutes here.
    # Without a bar the user watched the fetch bar hit 100% and then sat in front
    # of a still screen with no sign anything was running.
    _score_bar = _progress_bar(len(tickers), "Scoring", enabled=bool(verbose))
    try:
        for symbol in tickers:
            data_result = raw_results.get(symbol)
            if data_result is None or "error" in data_result:
                err_msg = (data_result or {}).get("error", "fetch returned no data")
                results_buffer[symbol] = {
                    'success': False, 'error': err_msg,
                    'context_log': [], 'picks': [],
                    'credit_spreads': [], 'iron_condors': pd.DataFrame(),
                    'history': None
                }
                _score_bar.update(1)
                continue
            results_buffer[symbol] = _score_fetched_data(
                symbol, data_result, mode, min_dte, max_dte,
                rfr, config, vix_weights, trader_profile,
                budget, macro_risk_active, tnx_change_pct
            )
            _score_bar.update(1)
    finally:
        _score_bar.close()


    # Stale-quote advisory: yfinance periodically serves bid=0/ask=0 chains
    # even when the market is open. If many tickers saw this, warn loudly —
    # scoring used a lastPrice-based synthetic mid, which is a workaround
    # and not a reliable quote.
    _stale_tickers = [
        s for s in tickers
        if float((results_buffer.get(s) or {}).get("stale_quote_ratio", 0.0)) >= 0.80
    ]
    if verbose and _stale_tickers:
        _n = len(_stale_tickers)
        _total = len(tickers)
        _msg = (
            f"  ! Stale-quote fallback active on {_n}/{_total} tickers "
            f"(yfinance bid=0/ask=0). Using lastPrice +/-5% as synthetic "
            f"bid/ask. Treat premiums as approximate; do NOT trade off these "
            f"quotes without verifying on a live feed."
        )
        print(fmt.colorize(_msg, fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else _msg)

    # Print per-ticker summary after all futures complete
    if verbose:
        ok, fail = [], []
        for symbol in tickers:
            result = results_buffer.get(symbol, {})
            if result.get('success'):
                n = (sum(len(p) for p in result.get('picks', []))
                     + sum(len(s) for s in result.get('credit_spreads', []))
                     + (len(result['iron_condors']) if isinstance(result.get('iron_condors'), pd.DataFrame) and not result['iron_condors'].empty else 0))
                ok.append((symbol, n))
            else:
                fail.append((symbol, result.get('error', 'no contracts passed filters')))

        if ok or fail:
            sep = fmt.draw_separator(WIDTH) if HAS_ENHANCED_CLI else "-" * WIDTH
            print(sep)
            for sym, n in ok:
                cov_str = ""
                try:
                    from .data_fetching import iv_history_coverage as _iv_cov
                    _cov = _iv_cov(sym)
                    cov_str = f"  IV: {_cov['days']}d ({_cov['confidence']})"
                except Exception:
                    pass
                line = f"  \u2713 {sym:<6}  {n} contract(s){cov_str}"
                print(fmt.colorize(line, fmt.Colors.GREEN) if HAS_ENHANCED_CLI else line)
            for sym, err in fail:
                # Only show brief reason, not the full stack trace
                short_err = str(err).split('\n')[0][:60]
                line = f"  \u2013 {sym:<6}  {short_err}"
                print(fmt.colorize(line, fmt.Colors.DIM) if HAS_ENHANCED_CLI else line)
            print(sep)
            print()

    # Aggregate buffered results — also collect news_data per ticker
    news_map: Dict[str, Any] = {}
    squeeze_calls_map: Dict[str, pd.DataFrame] = {}
    for symbol, result in results_buffer.items():
        # Collected outside the success branch on purpose: a ticker whose picks
        # were emptied by the delta band reports success=False, and those are
        # precisely the squeeze names whose long side we still want to show.
        _sq_stash = result.get('squeeze_calls')
        if isinstance(_sq_stash, pd.DataFrame) and not _sq_stash.empty:
            squeeze_calls_map[str(symbol)] = _sq_stash
        if result.get('success'):
            if result.get('history') is not None:
                ticker_histories[symbol] = result['history']
            for picks_df in result.get('picks', []):
                all_picks.append(picks_df)
            for spreads_df in result.get('credit_spreads', []):
                all_credit_spreads.append(spreads_df)
            condors = result.get('iron_condors')
            if isinstance(condors, pd.DataFrame) and not condors.empty:
                all_iron_condors.append(condors)
        if result.get('news_data') is not None:
            news_map[symbol] = result['news_data']

    ticker_contexts: dict = {}
    for symbol, result in results_buffer.items():
        if result.get("success") and result.get("context"):
            ticker_contexts[symbol] = result["context"]

    # --- Portfolio Protection: Correlation Warning ---
    # Only the tickers that actually produced picks matter here — correlating
    # the whole 100+ ticker scan universe just proves SPY≈QQQ and buries the
    # signal in 60 lines of noise.
    # Pick-overlap correlation: computed here, rendered once inside the
    # portfolio-guard panel (portfolio_guard.format_guard_lines) instead of its
    # own noisy section. Failure-safe — an empty list just hides correlation.
    corr_pairs: list = []
    if len(ticker_histories) > 1:
        try:
            pick_symbols = set()
            for _df in all_picks:
                if not _df.empty and "symbol" in _df.columns:
                    pick_symbols.update(_df["symbol"].unique())

            price_data = {}
            for t, h in ticker_histories.items():
                if pick_symbols and t not in pick_symbols:
                    continue
                if not h.empty and "Close" in h.columns:
                    price_data[t] = h["Close"].tail(30)  # last 30 days

            if len(price_data) > 1:
                prices_df = pd.DataFrame(price_data).ffill().dropna()
                if not prices_df.empty and len(prices_df.columns) > 1:
                    corr_matrix = prices_df.corr()
                    cols = corr_matrix.columns
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            c = corr_matrix.iloc[i, j]
                            if c > 0.80:
                                corr_pairs.append((cols[i], cols[j], float(c)))
                    corr_pairs.sort(key=lambda p: -p[2])
        except Exception:
            corr_pairs = []

    # Consolidate picks and determine underlying price
    picks = pd.DataFrame()
    credit_spreads_df = pd.DataFrame()
    iron_condors_df = pd.DataFrame()
    
    if all_picks:
        non_empty_picks = [df for df in all_picks if not df.empty]
        if non_empty_picks:
            picks = pd.concat(non_empty_picks, ignore_index=True)
            # Cross-sectional normalization across the FULL combined batch.
            # Must run here (not per-ticker) so scores reflect quality relative
            # to every contract scanned, not just the ~5-15 from one ticker's chain.
            # Risk-flag caps applied per-ticker earlier are preserved as the raw input.
            picks = _cross_section_normalize(picks)
    
    if all_credit_spreads:
        credit_spreads_df = pd.concat(all_credit_spreads, ignore_index=True)
    
    if all_iron_condors:
        iron_condors_df = pd.concat(all_iron_condors, ignore_index=True)

    # Inject credit spreads into picks pool for unified AI ranking
    if not credit_spreads_df.empty and mode not in ("Credit Spreads", "Iron Condor"):
        try:
            spread_picks = normalize_spreads_for_ranking(credit_spreads_df, mode)
            if not spread_picks.empty:
                picks = pd.concat([picks, spread_picks], ignore_index=True)
        except Exception:
            pass

    underlying_price = 0.0
    if not picks.empty and "underlying" in picks.columns:
        underlying_price = picks.iloc[0]["underlying"]

    # --- Portfolio GEX Gate ---
    # First scan of a session cold-prices the whole open book here (an IV
    # chain fetch per leg, ~20s on a 50-position book) before anything else
    # prints — without a spinner it reads as a hang.
    try:
        from .portfolio_risk import RiskAggregator, risk_off_filters_picks
        _risk = RiskAggregator(config=config)
        with _spinner("Pricing open book for the portfolio risk gate…"):
            _risk_off, _risk_reason = _risk.is_risk_off_required(config)
        _filters_picks = risk_off_filters_picks(config)
        if _risk_off and verbose:
            _warn_msg = f"RISK-OFF MODE: {_risk_reason}"
            if not _filters_picks:
                _warn_msg += " — advisory only (picks kept for research)"
            if HAS_ENHANCED_CLI:
                print(fmt.format_warning(_warn_msg))
            else:
                print(f"  ⚠️  {_warn_msg}")
        # Only trim picks when enforcement is enabled; in paper/research mode the
        # picks are validation data, so RISK-OFF stays advisory.
        if _risk_off and _filters_picks and not picks.empty and "abs_delta" in picks.columns:
            picks = picks[picks["abs_delta"] < 0.30].copy()
    except Exception:
        pass

    # Concentration warning across scan results \u2014 measured on the contracts a
    # user would actually consider (top 15 by quality), not the raw candidate
    # pool, where one liquid chain can contribute 40+ rows and trip a
    # meaningless "41 picks from MU" warning.
    if verbose and not picks.empty and len(picks) >= 5:
        # Ranked the way the report is ranked. This used to take the top 15 by
        # quality_score, which stopped being the displayed order when the scan
        # moved to rank_by_verdict — so the warning was measuring a list nobody
        # sees, and a scan whose whole visible top-5 came from one ticker could
        # pass it silently. Concentration is a risk control; it has to watch the
        # contracts actually being offered.
        _top = rank_single_legs_by_verdict(picks, mode).head(15)
        call_count = (_top["type"].str.lower() == "call").sum()
        put_count = (_top["type"].str.lower() == "put").sum()
        total = len(_top)
        if call_count / total > 0.80:
            msg = f"Concentration warning: {call_count}/{total} top picks are CALLS \u2014 selection skews heavily bullish"
            print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")
        elif put_count / total > 0.80:
            msg = f"Concentration warning: {put_count}/{total} top picks are PUTS \u2014 selection skews heavily bearish"
            print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")
        if "symbol" in _top.columns:
            symbol_counts = _top["symbol"].value_counts()
            if not symbol_counts.empty:
                dominant = symbol_counts.index[0]
                dominant_count = symbol_counts.iloc[0]
                if dominant_count >= 5:
                    msg = f"Concentration warning: {dominant_count}/{total} top picks from {dominant} \u2014 consider diversifying"
                    print(fmt.colorize(f"  \u26a0\ufe0f  {msg}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0\ufe0f  {msg}")

    # Generate Final Reports.
    # `_display_df` is the exact frame print_report numbered on screen, so
    # "pick N" means the same contract in the terminal and on a tearsheet.
    _display_df = None

    def _leg_label(row) -> str:
        """Strategy name of a single-leg scan row, for capital-at-risk sizing."""
        return _strategy_label_for_mode(mode, row.get("type"))

    if mode == "Budget scan":
        if not picks.empty:
            final_df = gate_and_report(picks, "BUDGET", verbose=verbose)
            final_df = _budget_board(final_df, _leg_label, session_budget,
                                     verbose=verbose)
        if not picks.empty and not final_df.empty:
            final_df = categorize_by_premium(final_df, budget=budget)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            _display_df = top_picks
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, budget=budget, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact, corr_pairs=corr_pairs)
                _print_per_risk_table(top_picks, _leg_label, session_budget)
                _print_budget_use(top_picks, session_budget)
        elif verbose and picks.empty:
            # Only when the scan genuinely found nothing. A board that was
            # found and then refused has already printed why, and telling the
            # reader "none found" on top of that would misreport the reason.
            print("\nNo options found within budget.")

    elif mode in ("Discovery scan", "Squeeze Hunt"):
        if not picks.empty:
            final_df = gate_and_report(picks, mode.upper(), verbose=verbose)
            final_df = _budget_board(final_df, _leg_label, session_budget,
                                     verbose=verbose)
        if not picks.empty and not final_df.empty:
            final_df = categorize_by_premium(final_df, budget=None)
            top_picks = pick_top_per_bucket(final_df, per_bucket=3, diversify_tickers=True)
            _display_df = top_picks
            if verbose:
                print_report(top_picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact, corr_pairs=corr_pairs)
                _print_per_risk_table(top_picks, _leg_label, session_budget)
                _print_budget_use(top_picks, session_budget)
        elif verbose and picks.empty:
            print("\nNo discovery picks found.")
            
    elif mode == "Credit Spreads":
        if not credit_spreads_df.empty:
            final_spreads = gate_and_report(credit_spreads_df, "CREDIT SPREADS",
                                            label_structures=True, verbose=verbose)
            final_spreads = _budget_board(final_spreads, structure_strategy_name,
                                          session_budget, verbose=verbose)
            if verbose and not final_spreads.empty:
                print_credit_spreads_report(final_spreads)
                _print_per_risk_table(final_spreads, structure_strategy_name,
                                      session_budget)
                _print_budget_use(final_spreads, session_budget)
        elif verbose:
            print("\nNo credit spreads found.")

    elif mode == "Iron Condor":
        if not iron_condors_df.empty:
            # `return_on_risk` ordered this board and measures -0.216 against
            # return on capital (n=139) — its top pick was systematically its
            # worst. Condors off the broad index are now refused outright:
            # +9.5% here against -11.8% elsewhere, p < 1e-5.
            final_condors = gate_and_report(iron_condors_df, "IRON CONDOR",
                                            label_structures=True, verbose=verbose)
            final_condors = _budget_board(final_condors, structure_strategy_name,
                                          session_budget, verbose=verbose)
            if verbose and not final_condors.empty:
                print_iron_condor_report(final_condors)
                _print_per_risk_table(final_condors, structure_strategy_name,
                                      session_budget)
                _print_budget_use(final_condors, session_budget)
        elif verbose:
            print("\nNo iron condors found.")

    elif mode == "Premium Selling":
        if not picks.empty:
            # Labelled first: _legs_of reads the side off strategy_name, and
            # these rows carry only `type`, so a short put was priced as a
            # debit BUY here — flipping is_credit and skipping both the
            # "credit disappears" and breakeven gates, the two that matter
            # most for short premium.
            final_df = rank_single_legs_by_verdict(picks, mode)
            final_df = gate_and_report(final_df, "PREMIUM SELLING", verbose=verbose)
            # Labelled per row: this board mixes Short Put, whose risk is the
            # collateral, with Short Call, whose risk cannot be bounded at all.
            final_df = _budget_board(final_df, _leg_label, session_budget,
                                     verbose=verbose)
        if not picks.empty and not final_df.empty:
            final_df = categorize_by_premium(final_df, budget=None)
            _display_df = final_df.head(10)
            if verbose:
                print_report(final_df.head(10), underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact, corr_pairs=corr_pairs)
                _print_per_risk_table(final_df.head(10), _leg_label,
                                      session_budget)
                _print_budget_use(final_df.head(10), session_budget)
        elif verbose and picks.empty:
            print("\nNo premium selling candidates found.")

    elif mode == "Lottery Ticket":
        if not picks.empty:
            final_df = picks.sort_values("lottery_ticket_score", ascending=False)
            if verbose:
                print_lottery_ticket_report(final_df, underlying_price, market_trend, volatility_regime)
        elif verbose:
            print("\nNo lottery ticket candidates found — try adding more tickers or relaxing filters.")

    else:
        # Single stock mode
        if not picks.empty:
            final_df = gate_and_report(picks, "TICKER", verbose=verbose)
            final_df = _budget_board(final_df, _leg_label, session_budget,
                                     verbose=verbose)
        if not picks.empty and not final_df.empty:
            final_df = categorize_by_premium(final_df, budget=None)
            _display_df = final_df
            if verbose:
                print_report(final_df, underlying_price, rfr, max_expiries, min_dte, max_dte, mode=mode, market_trend=market_trend, volatility_regime=volatility_regime, config=config, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=compact, corr_pairs=corr_pairs)
                _print_per_risk_table(final_df, _leg_label, session_budget)
                _print_budget_use(final_df, session_budget)
        elif verbose and picks.empty:
            print("\nNo suitable options found.")

        # Vol analytics for single-ticker scans
        if verbose and HAS_VOL_ANALYTICS and len(tickers) == 1:
            try:
                _ticker_sym = tickers[0]
                _current_iv = float(picks["impliedVolatility"].median()) if not picks.empty and "impliedVolatility" in picks.columns else None
                _current_price = underlying_price if underlying_price and underlying_price > 0 else None
                print_vol_cone(_ticker_sym, current_iv=_current_iv, width=WIDTH)
                print_iv_surface(_ticker_sym, spot=_current_price, width=WIDTH)
                print_regime_summary(_ticker_sym, current_iv=_current_iv, width=WIDTH)
            except Exception:
                pass

    # Squeeze read — display-only (src/squeeze): loud banner when a scanned
    # name grades as a short-squeeze setup, plus a calls-only mini-board so the
    # long side is visible even when ranked picks skew to puts (the NBIS
    # 2026-07-16 case). Never touches quality_score or auto-log.
    if verbose and not picks.empty and mode not in ("Premium Selling", "Credit Spreads", "Iron Condor"):
        try:
            from src.squeeze.detector import SETUP as _SQ_SETUP
            from src.squeeze.detector import assess_squeeze_row as _sq_assess
            from src.squeeze.board import banner as _sq_banner
            from src.squeeze.board import call_board as _sq_call_board
            if "symbol" in picks.columns:
                _sq_symbols = list(dict.fromkeys(picks["symbol"].astype(str)))
            else:
                _sq_symbols = [str(tickers[0])] if len(tickers) == 1 else []
            for _sq_sym in _sq_symbols:
                _sq_rows = (picks[picks["symbol"].astype(str) == _sq_sym]
                            if "symbol" in picks.columns else picks)
                if _sq_rows.empty:
                    continue
                _sq_fields = _sq_rows.iloc[0].to_dict()
                # ctx spot is only trustworthy on single-ticker scans (rows carry
                # their own underlying elsewhere — see tearsheet spot bug).
                if len(tickers) == 1 and "underlying_price" not in _sq_fields:
                    _sq_fields.setdefault("spot", underlying_price)
                _sq_setup = _sq_assess(_sq_fields)
                _sq_text = _sq_banner(_sq_setup, _sq_sym, width=WIDTH)
                if not _sq_text:
                    continue
                print("\n" + _sq_text)
                if _sq_setup.grade == _SQ_SETUP:
                    # The stash, not _sq_rows: the ranked picks have already
                    # lost every near-ATM call to the delta band, which is
                    # where a squeeze is expressed.
                    _sq_src = squeeze_calls_map.get(_sq_sym)
                    _sq_from_stash = _sq_src is not None and not _sq_src.empty
                    if not _sq_from_stash:
                        _sq_src = _sq_rows
                    _sq_cb = _sq_call_board(_sq_src, _sq_sym, width=WIDTH, rfr=rfr)
                    if _sq_cb:
                        print(_sq_cb)
                        if _sq_from_stash:
                            print(ui.kv_line("Note", fmt.style(
                                "shown because the squeeze thesis is long — these sit outside the "
                                "scan's delta band and the scorer did not rank them",
                                "muted")))
                    else:
                        print(ui.kv_line("Note", fmt.style(
                            "no calls passed the scan filters — the squeeze long side needs a manual chain look",
                            "warn")))
        except Exception as _sq_exc:
            logging.getLogger(__name__).debug("squeeze read skipped: %s", _sq_exc)

    # Phase 4: Executive Summary
    if verbose and HAS_ENHANCED_CLI and not picks.empty:
        print_executive_summary(
            picks,
            config,
            mode=mode,
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            macro_risk=macro_risk_active,
            num_tickers=len(tickers)
        )

    # Phase 7: optional HTML tearsheet for one pick.
    # One hook covers every single-leg mode, because they all fall through here.
    if verbose and not picks.empty:
        try:
            _ts_ctx = {"mode": mode, "spot": underlying_price, "rfr": rfr,
                       "vix": get_vix_level(), "vix_regime": volatility_regime,
                       "config": config, "config_sha": _config_sha(config)}
            # The frame the CARDS were numbered from, not `picks`.
            # `picks` is pre-gate and pre-ordering: a tearsheet built from it
            # could show a contract the board had refused, at a different
            # strike and expiry from anything on screen. Observed 2026-08-10 on
            # an NVDA scan — terminal showed $222.5 08-19/08-21, the tearsheet
            # rendered $225 08-28.
            offer_tearsheet(_display_df if _display_df is not None else picks,
                            _ts_ctx, interactive=interactive,
                            preselect=tearsheet_pick)
        except Exception as _ts_exc:
            logging.getLogger(__name__).debug("tearsheet skipped: %s", _ts_exc)

    # Phase 5: News & Events digest — shown after picks so it doesn't interrupt the report flow
    if verbose and news_map and not picks.empty:
        print_news_panel(news_map, picks, width=WIDTH)

    # Phase 6: Macro overlay — scan-aware situational awareness + opt-in AI ranking
    if verbose and not picks.empty:
        _single = len(tickers) == 1
        if _single:
            _macro_universe = [tickers[0]]
            _macro_focus = tickers[0]
        else:
            _macro_universe = (list(dict.fromkeys(picks["symbol"].astype(str).tolist()))
                               if "symbol" in picks.columns else list(tickers))
            _macro_focus = None
        _macro_scan_section(_macro_universe, focus_symbol=_macro_focus)

    # The pick that feeds the tearsheet, the 3D surface and the visualizer.
    # It used to be whatever `quality_score` (or `return_on_risk` on condors)
    # liked most — both measured negative against outcome. Nothing ranks, so
    # this is now the first surviving row on the already-gated board rather
    # than a claim about which candidate is best.
    top_pick = None
    _gated = gate_and_report(picks, "TOP PICK", verbose=False) if not picks.empty else picks
    if _gated is not None and not _gated.empty:
        picks["overall_score"] = picks["quality_score"]
        top_pick = _gated.iloc[0]
    elif not credit_spreads_df.empty:
        _gs = gate_and_report(credit_spreads_df, "TOP PICK",
                              label_structures=True, verbose=False)
        top_pick = _gs.iloc[0] if not _gs.empty else None
    elif not iron_condors_df.empty:
        _gc = gate_and_report(iron_condors_df, "TOP PICK",
                              label_structures=True, verbose=False)
        top_pick = _gc.iloc[0] if not _gc.empty else None

    chain_iv_median = 0.0
    if not picks.empty and "impliedVolatility" in picks.columns:
        chain_iv_median = picks["impliedVolatility"].median()

    # Post-run warning summary
    n_scored = len(picks) if not picks.empty else 0
    if _SCAN_WARNINGS[0] > 0:
        _warn_msg = f"Scan complete: {n_scored} contracts scored, {_SCAN_WARNINGS[0]} warnings logged (see DEBUG log for details)"
    else:
        _warn_msg = f"Scan complete: {n_scored} contracts scored"
    if verbose:
        print(f"\n  {_warn_msg}")
    _SCAN_WARNINGS[0] = 0  # reset for next scan

    return ScanResult(
        picks=picks,
        spreads=pd.DataFrame(),
        credit_spreads=credit_spreads_df,
        iron_condors=iron_condors_df,
        squeeze_calls=squeeze_calls_map,
        top_pick=top_pick,
        underlying_price=underlying_price,
        rfr=rfr,
        chain_iv_median=chain_iv_median,
        timestamp=datetime.now().isoformat(),
        ticker_contexts=ticker_contexts,
        market_context={
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'market_trend': market_trend,
            'volatility_regime': volatility_regime,
            'macro_risk_active': macro_risk_active,
            'tnx_change_pct': tnx_change_pct,
            'sector_ctx': sector_ctx,
        },
    )

def select_trades_to_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive helper to let the user select specific trades to log.
    Returns a DataFrame containing only the selected rows.
    """
    if df.empty:
        print("No trades to select.")
        return pd.DataFrame()

    # Presented in board order. Sorting this menu by `quality_score` put the
    # losing top quintile at positions 1-10, which is where a human picking
    # from a list actually picks.
    df_sorted = df.reset_index(drop=True)

    top_n = df_sorted.head(50)

    print("\n" + "="*60)
    print("  SELECT TRADES TO LOG")
    print("="*60)
    
    for i, row in top_n.iterrows():
        symbol = row.get('symbol', 'N/A')
        type_ = str(row.get('type', 'N/A')).upper()
        
        # Determine strike display (Single vs Spread vs Condor)
        if 'short_strike' in row and 'long_strike' in row:
            # Credit Spread
            strike_val = f"{row['short_strike']:.0f}/{row['long_strike']:.0f}"
        elif 'short_put_strike' in row:
            # Iron Condor
            strike_val = f"{row['short_put_strike']:.0f}/{row['short_call_strike']:.0f}"
        else:
            # Single option
            strike_val = f"{row.get('strike', 0.0):.1f}"
            
        exp = row.get('expiration', 'N/A')
        if isinstance(exp, str):
            exp = exp.split("T")[0]
        
        # Determine premium display
        premium = row.get('premium') or row.get('net_credit') or row.get('total_credit') or 0.0
        quality = row.get('quality_score', 0.0)
        
        print(f"  [{i+1}] {symbol:<5} {type_:<12} {strike_val:>12} {exp} | Prem: ${premium:>6.2f} | Qual: {quality:.2f}")

    print("="*60)
    print("Enter the numbers of the trades you want to log, separated by commas.")
    print("Example: 1, 3, 5 (or 'all' for all listed, 'q' to cancel)")
    
    selection = prompt_input("Selection", "").strip().lower()
    
    if not selection or selection == 'q':
        print("Selection cancelled.")
        return pd.DataFrame()
    
    if selection == 'all':
        return top_n

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        valid_indices = [i for i in indices if 0 <= i < len(top_n)]
        
        if not valid_indices:
            print("No valid selections made.")
            return pd.DataFrame()
            
        selected_df = top_n.iloc[valid_indices].copy()
        print(f"Selected {len(selected_df)} trades.")
        return selected_df
        
    except Exception as e:
        print(f"Error parsing selection: {e}")
        return pd.DataFrame()


def _check_market_hours() -> tuple:
    """
    Returns (is_open: bool, message: str) in US Eastern time.

    Delegates to ``src.data_quality.check_market_hours`` so the market-hours
    logic has a single implementation shared with the freshness classifier.
    """
    from src.data_quality import check_market_hours
    return check_market_hours()


def _run_ai_pipeline(picks: "pd.DataFrame", volatility_regime: str, verbose: bool = True,
                     sector_ctx=None, ticker_contexts: "Optional[dict]" = None,
                     mode: str = "Scan") -> "Optional[pd.DataFrame]":
    """Thin wrapper: delegates to ai_rank pipeline so CLI and ai_rank.py share one code path.

    `ticker_contexts` is the per-symbol context the scan already computed
    (`ScanResults.ticker_contexts`). It has to be handed in: this used to be
    rebuilt here by looking symbols up in `data_fetching._CHAIN_CACHE`, but that
    cache is keyed by `(symbol, min_dte, max_dte)`, so a bare-symbol lookup never
    hit. Two-pass scoring degraded to single-pass on every run, and the
    "two-pass" label is gated on the same empty dict, so nothing reported it.
    """
    try:
        from ai_rank import score_and_rank
        from src.ranking import print_ranked_table
        from src.config_ai import AI_CONFIG
    except Exception as exc:
        msg = f"AI scoring unavailable — import failed: {type(exc).__name__}: {exc}"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        import traceback as _tb
        _tb.print_exc()
        return None

    # Ensure at least one OpenRouter / Anthropic key is set before trying any network call
    _key_env_vars = [
        "OPENROUTER_API_KEY", "OPENROUTER_LING_KEY",
        "OPENROUTER_NVIDIA_KEY", "OPENROUTER_POOLSIDE_KEY",
        "ANTHROPIC_API_KEY",
    ]
    if not any(os.environ.get(k) for k in _key_env_vars):
        msg = "AI scoring skipped — no API key set. Add OPENROUTER_API_KEY (or similar) to .env"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        return None

    try:
        vix_map = {"Low": "low", "Normal": "normal", "High": "high"}
        vix_regime = vix_map.get(str(volatility_regime), "normal")

        # Same ordering as the report, for the same reason as the concentration
        # warning above: ranking these by quality_score spent tokens scoring a
        # different 20 contracts than the ones on screen.
        candidates = rank_single_legs_by_verdict(picks, mode).head(20).copy()

        # Only the symbols that actually produced candidates: context for a
        # ticker with no picks is tokens spent on a row the model never ranks.
        _supplied = ticker_contexts or {}
        contexts: dict = {}
        if AI_CONFIG.get("two_pass_enabled", True):
            for sym in candidates["symbol"].unique():
                if sym in _supplied:
                    contexts[sym] = _supplied[sym]

        if verbose:
            _tp = "two-pass " if contexts else ""
            print(f"\n  Running AI {_tp}analysis on top {len(candidates)} picks...")

        ranked = score_and_rank(candidates, contexts, vix_regime, sector_ctx=sector_ctx)
        if verbose:
            print_ranked_table(ranked, top_n=10)
        return ranked
    except Exception as exc:
        msg = f"AI scoring failed: {type(exc).__name__}: {exc}"
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(msg))
        else:
            print(f"\n⚠  {msg}")
        import traceback as _tb
        _tb.print_exc()
        return None


def _macro_scan_section(symbols, focus_symbol=None) -> None:
    """After-scan macro overlay (interactive only — zero cost in headless runs).

    Shows a scan-aware situational-awareness panel: sector-focused when a single
    ticker is scanned ("how the tape affects AAPL / tech"), general market-wide
    for discovery scans. Then offers a [1] yes / [2] no opt-in for a single
    batched AI ranking of the scanned tickers. The deterministic panel reuses the
    once-a-day cached macro narrative, so default scans add no AI tokens; the AI
    ranking fires only on an explicit 'yes'.
    """
    if not sys.stdin.isatty():
        return  # headless / auto-log / maintenance → skip entirely (no fetch/AI)
    try:
        from src.macro_pulse import orchestrator as _mp
        from src.macro_pulse import rank as _mpr
        from src.macro_pulse import ticker as _mpt
    except Exception:
        return

    uniq: List[str] = []
    for s in (symbols or []):
        s = str(s).upper().strip()
        if s and s not in uniq:
            uniq.append(s)
        if len(uniq) >= 12:  # cap sector lookups / AI prompt size
            break

    try:
        # Deterministic narrative only: this panel is shown BEFORE the AI-ranking
        # prompt below, so building it must not fire an AI request. The opt-in
        # AI call is the ranking, gated on an explicit 'yes'.
        ctx = _mp.build_context(use_ai=False)
    except Exception as exc:
        logging.getLogger(__name__).debug("macro scan section skipped: %s", exc)
        return

    sectors = {s: _mp._lookup_sector(s) for s in uniq}
    focus = (focus_symbol or "").upper() or None
    focus_sector = sectors.get(focus) if focus else None
    try:
        print("\n" + _mpt.render_ticker(ctx, focus, focus_sector))
    except Exception as exc:
        logging.getLogger(__name__).debug("macro panel render failed: %s", exc)
        return

    if not uniq:
        return
    ans = prompt_input(
        "Run AI macro ranking of these tickers? [1] yes  [2] no", "2"
    ).strip().lower()
    if ans in ("1", "y", "yes"):
        try:
            rows = _mpr.rank_tickers(uniq, ctx, sectors=sectors, use_ai=True)
            print("\n" + _mpr.render_ranking(rows))
        except Exception as exc:
            logging.getLogger(__name__).debug("macro ranking failed: %s", exc)


def run_top_scan(
    tickers: List[str],
    top_n: int = 10,
    mode: str = "Discovery scan",
    export_csv: bool = False,
    min_dte: int = 7,
    max_dte: int = 45,
    max_expiries: int = 4,
    tearsheet_pick: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Fetch and score contracts across all tickers, return the top_n by verdict.

    Ordered by `rank_by_verdict` — what survives its own costs — not by
    `quality_score`, which this docstring claimed until 2026-08-07 and which
    measures -0.10 against friction-adjusted return on the long-premium book.

    Groups results into DTE buckets: Short (7-14), Standard (15-30), Swing (31-45).
    Prints a ranked table and optionally saves a CSV. With `tearsheet_pick` set,
    writes an HTML tearsheet for that rank (1-based) after the table.
    """
    from .cli_display import print_top_n_table

    _logger = setup_logging()
    config = load_config("config.json")
    rfr = get_risk_free_rate()
    vix_level = get_vix_level()
    vix_regime, vix_weights = determine_vix_regime(vix_level, config)
    market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()

    all_rows = []
    for sym in tickers:
        try:
            # The bounds decide WHICH expiries are pulled, not just which rows
            # survive scoring: fetch_options_yfinance skips out-of-range
            # expirations BEFORE the max_expiries slice. Omitting them here
            # took the front 4 expiries regardless of what was asked for, so a
            # 25-70 DTE request came back with 7-23 DTE contracts.
            data = fetch_options_yfinance(sym, max_expiries,
                                          min_dte=min_dte, max_dte=max_dte)
            # Same guard the parallel scan path applies before scoring. Without
            # it a fetch result with no chain reached _score_fetched_data, which
            # raised KeyError('df') into its own handler and logged the ticker
            # to scan_errors.log as `error: 'df'` — naming the missing key
            # rather than the fetch that failed.
            if not isinstance(data, dict) or "error" in data or "df" not in data:
                continue
            result = _score_fetched_data(
                sym, data, mode, min_dte, max_dte,
                rfr, config, vix_weights, "swing",
                None, macro_risk_active, tnx_change_pct,
            )
            if result.get("success"):
                for picks_df in result.get("picks", []):
                    if not picks_df.empty:
                        all_rows.append(picks_df)
        except Exception:
            continue

    if not all_rows:
        print("No results from top scan.")
        return None

    combined = pd.concat(all_rows, ignore_index=True)
    # Labelled first so _legs_of reads the right side: this path defaults to a
    # buyer mode, where `buy` happens to be correct, but a short-premium mode
    # would have been priced as a debit. See rank_single_legs_by_verdict.
    combined = rank_single_legs_by_verdict(combined, mode)
    combined = gate_and_report(combined, f"TOP {top_n}")
    if combined.empty:
        return None
    top = combined.head(top_n).copy()

    print_top_n_table(top, top_n)

    # Scan-aware macro overlay (general / discovery) + opt-in AI ranking
    _top_syms = (list(dict.fromkeys(top["symbol"].astype(str).tolist()))
                 if "symbol" in top.columns else list(tickers))
    _macro_scan_section(_top_syms, focus_symbol=None)

    if export_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        fname = _export_path(f"scan_results_{ts}.csv")
        export_cols = [
            "symbol", "type", "strike", "expiration", "T_years",
            "bid", "ask", "premium", "delta", "impliedVolatility",
            "iv_rank_30", "prob_profit", "ev_per_contract",
            "ev_gross_per_contract", "ev_cost_per_contract",
            "vega_dollar", "iv_confidence",
            "quality_score", "score_drivers",
        ]
        export_df = top[[c for c in export_cols if c in top.columns]].copy()
        if "T_years" in export_df.columns:
            export_df["DTE"] = (export_df["T_years"] * 365.0).round(0).astype(int)
            export_df.drop(columns=["T_years"], inplace=True, errors="ignore")
        export_df.to_csv(fname, index=False)
        print(f"\nExported {len(export_df)} rows to {fname}")

    if tearsheet_pick is not None and 1 <= tearsheet_pick <= len(top):
        try:
            from src.tearsheet import build, write_tearsheet
            row = top.iloc[tearsheet_pick - 1].to_dict()
            ctx = {"mode": mode, "rank": tearsheet_pick, "n_picks": len(top),
                   "spot": row.get("underlying"), "rfr": rfr, "vix": vix_level,
                   "vix_regime": vix_regime, "config": config,
                   "config_sha": _config_sha(config),
                   "sibling_rows": top.to_dict("records")}
            data = build(row, ctx)
            html_path, _ = write_tearsheet(data)
            print(f"\n  tearsheet: {html_path}")
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception as _ts_exc:
            logging.getLogger(__name__).debug("top-scan tearsheet skipped: %s", _ts_exc)

    return top


def main():
    # ── Raise soft file-descriptor limit ────────────────────────────────────
    # macOS defaults RLIMIT_NOFILE soft to 256, which is too low once curl_cffi
    # sessions, sqlite caches, and the fetch ThreadPool all hold sockets at once
    # — symptom is "[Errno 24] Too many open files" cascading across every
    # ticker mid-scan. Hard limit is unlimited, so bump soft to 8192.
    try:
        import resource
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _target = 8192 if _hard == resource.RLIM_INFINITY else min(_hard, 8192)
        if _soft < _target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
    except (ImportError, ValueError, OSError):
        pass

    # ── CLI argument parsing (Phase 7) ───────────────────────────────────────
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--help", "-h", action="store_true", help="Show help and exit")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument("--close-trades", action="store_true", help="Update trade log with closing P/L")
    parser.add_argument("--enforce-exits", action="store_true", help="Run exit-rule enforcement on paper_trades.db and exit (suitable for cron)")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI analysis after scan")
    parser.add_argument("--tearsheet", type=int, metavar="N", default=None,
                        help="render an HTML tearsheet for pick N and skip the prompt")
    parser.add_argument("--no-tearsheet", action="store_true",
                        help="never offer the tearsheet")
    parser.add_argument("--top", type=int, default=None, metavar="N", help="Run top-N cross-ticker scan and exit")
    parser.add_argument("--export", type=str, default=None, metavar="FORMAT", help="Export results to file (csv)")
    parser.add_argument("--watchlist", type=str, default=None, metavar="NAME", help="Use named watchlist from config as ticker input")
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching (requests, AI scores, IV history)")
    parser.add_argument("--surface", action="store_true", help="Show 3D P&L risk surface for top pick")
    parser.add_argument("--surface-mode", choices=["ascii", "braille"], default="braille", help="Surface render mode (default: braille)")
    parser.add_argument("--surface-greek", choices=["delta", "gamma", "vega", "theta"], default=None, help="Show greek sensitivity surface (implies --surface)")
    parser.add_argument("--no-contours", action="store_true", help="Disable contour lines on surface")
    parser.add_argument("--viz", action="store_true", help="Launch interactive 3D visualizer in browser after scan")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompts, use config defaults")
    parser.add_argument("--compact", action="store_true", help="Compact per-pick output (3 lines per pick)")
    parser.add_argument("--mode", type=str, default=None, choices=["ticker", "all", "discover", "sell", "spreads", "iron", "portfolio", "mylist", "lottery", "squeeze"], help="Scan mode (skip mode menu)")
    parser.add_argument("--ticker", type=str, default=None, metavar="SYM", help="Ticker symbol (implies --mode ticker)")
    parser.add_argument("--weights", type=str, default=None, metavar="NAME", help="Weight profile name (in configs/weights/) or path to a JSON file; tags logged trades with the profile id")
    parser.add_argument("--auto-log", action="store_true", help="Skip the save-menu prompt and automatically log top-N picks to paper_trades.db")
    parser.add_argument("--log-top", type=int, default=5, metavar="N", help="With --auto-log: number of top-ranked picks to log (default 5)")
    parser.add_argument("--min-dte", type=int, default=None, metavar="N", help="Override minimum days-to-expiration for this scan (e.g. 30 to feed the validation cohort)")
    parser.add_argument("--max-dte", type=int, default=None, metavar="N", help="Override maximum days-to-expiration for this scan")
    parser.add_argument("--list-profiles", action="store_true", help="List available weight profiles and exit")
    args, _ = parser.parse_known_args()

    if args.no_cache:
        try:
            import requests_cache as _rc
            _rc.uninstall_cache()
        except Exception:
            pass
        # Disable IV history DB reads (bypass_cache flag on data_fetching)
        try:
            from . import data_fetching as _df_mod
            _df_mod._NO_CACHE = True
        except Exception:
            pass
        logging.getLogger(__name__).info("All caching disabled via --no-cache")

    if args.no_color and HAS_ENHANCED_CLI:
        fmt.set_color_enabled(False)

    if HAS_ENHANCED_CLI:
        settings.apply_saved_theme()

    if args.version:
        print("Options Screener v1.0.0")
        sys.exit(0)

    if args.list_profiles:
        from .weight_profiles import list_profiles as _lp
        _names = _lp()
        if _names:
            print("Available weight profiles:")
            for n in _names:
                print(f"  {n}")
        else:
            print("No weight profiles found in configs/weights/")
        sys.exit(0)

    # ── Load weight profile if requested (used below by run_scan + trade logging) ──
    _weight_profile_id: Optional[str] = None
    _custom_weights: Optional[Dict] = None
    if args.weights:
        try:
            from .weight_profiles import load_weight_profile as _lwp
            _weight_profile_id, _custom_weights = _lwp(args.weights)
            print(f"Weight profile: {_weight_profile_id} ({len(_custom_weights)} weights)")
        except (FileNotFoundError, ValueError) as _wp_exc:
            print(f"Error loading --weights {args.weights}: {_wp_exc}")
            sys.exit(2)

    WIDTH = get_display_width()

    if args.help:
        if HAS_ENHANCED_CLI:
            print(fmt.draw_box("OPTIONS SCREENER  \u2014  HELP", WIDTH, double=True))
            print(fmt.colorize("\nUsage:", fmt.Colors.BRIGHT_CYAN, bold=True))
            print("  python -m src.options_screener [OPTIONS]\n")
            print(fmt.colorize("Options:", fmt.Colors.BRIGHT_CYAN, bold=True))
            for flag, desc in [
                ("--no-color",     "Disable colored output"),
                ("-h, --help",     "Show this help and exit"),
                ("--version",      "Show version string and exit"),
                ("--close-trades", "Update trade log with closing P/L"),
                ("--ui",           "Launch the Streamlit dashboard"),
                ("--top N",        "Cross-ticker top-N scan (default 10), grouped by DTE bucket"),
                ("--export csv",   "Export top scan results to exports/scan_results_YYYYMMDD_HHMM.csv"),
                ("--watchlist N",  "Use named watchlist from config (liquid_large_cap, sector_etfs, high_iv, income)"),
                ("--no-cache",    "Disable all caching (requests, AI scores, IV history)"),
                ("--tearsheet N",  "Open an HTML tearsheet for pick N (skips the prompt)"),
                ("--no-tearsheet", "Never offer the HTML tearsheet"),
                ("--surface",      "Show 3D P&L risk surface for top pick (braille hi-res by default)"),
                ("--surface-mode", "Surface render: ascii or braille (default: braille)"),
                ("--surface-greek","Show greek surface: delta, gamma, vega, theta"),
                ("--no-contours",  "Disable contour lines on surface"),
                ("--viz",          "Open interactive 3D visualizer (Plotly, opens in browser)"),
                ("--weights NAME", "Weight profile (configs/weights/<name>.json) — tags logged trades"),
                ("--auto-log",     "Skip save-menu and auto-log top-N picks to paper_trades.db"),
                ("--log-top N",    "With --auto-log: how many top picks to log (default 5)"),
                ("--min-dte N",    "Override minimum DTE for this scan (30 = cohort-eligible calls)"),
                ("--max-dte N",    "Override maximum DTE for this scan"),
                ("--list-profiles","List available weight profiles and exit"),
            ]:
                print(f"  {fmt.colorize(f'{flag:<18}', fmt.Colors.BRIGHT_YELLOW)} {desc}")
        else:
            print("Options Screener v1.0.0")
            print("Usage: python -m src.options_screener [--no-color] [-h/--help] [--version] [--close-trades] [--ui]")
        sys.exit(0)

    if args.close_trades:
        close_trades()
        sys.exit(0)

    if args.enforce_exits:
        print("Enforcing exit rules on paper_trades.db...")
        pm = PaperManager(db_path="paper_trades.db", config_path="config.json")
        pm.update_positions()
        print("Done.")
        sys.exit(0)

    if args.ui:
        import subprocess
        print("Launching Streamlit dashboard...")
        subprocess.run(["streamlit", "run", "src/dashboard.py"])
        sys.exit(0)

    config = load_config("config.json")

    # Config-level AI kill switch: honor ai_scoring.enabled=false as if --no-ai
    # were passed, so the AI ranking stays off in interactive runs too.
    if ai_scoring_disabled(config):
        args.no_ai = True

    # ── Config validation ─────────────────────────────────────────────────────
    _cw = config.get("composite_weights", {})
    _cw_sum = sum(_cw.values()) if _cw else 0
    if _cw and (_cw_sum > 1.1 or _cw_sum < 0.9):
        _warn = f"composite_weights sum to {_cw_sum:.2f} (expected ~1.0)"
        print(fmt.colorize(f"  \u26a0 Config: {_warn}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0 Config: {_warn}")

    # ── Calibration banner ────────────────────────────────────────────────────
    try:
        from .backtester import get_calibration_status
        _cal_n, _cal_status = get_calibration_status()
        _cal_line = f"  Paper-trade calibration: {_cal_n} closed — {_cal_status}"
        if HAS_ENHANCED_CLI:
            _color = fmt.Colors.BRIGHT_GREEN if "available" in _cal_status else fmt.Colors.DIM
            print(fmt.colorize(_cal_line, _color))
        else:
            print(_cal_line)
    except Exception:
        pass

    if not getattr(args, 'no_ai', False):
        # Ensure .env is loaded before checking for keys
        _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if os.path.isfile(_env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=_env_path, override=False)
            except ImportError:
                # Manual fallback: parse KEY=VALUE lines
                with open(_env_path) as _ef:
                    for _line in _ef:
                        _line = _line.strip()
                        if _line and not _line.startswith("#") and "=" in _line:
                            _k, _, _v = _line.partition("=")
                            _k, _v = _k.strip(), _v.strip().strip("'\"")
                            if _k and _v:
                                os.environ.setdefault(_k, _v)
        _has_any_key = any(
            os.environ.get(k) for k in [
                "OPENROUTER_API_KEY", "OPENROUTER_LING_KEY",
                "OPENROUTER_NVIDIA_KEY", "OPENROUTER_POOLSIDE_KEY",
                "ANTHROPIC_API_KEY",
            ]
        )
        if not _has_any_key:
            _warn = "No AI API keys found in environment. AI scoring will be skipped. Set keys in .env file."
            print(fmt.colorize(f"  \u26a0 {_warn}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI else f"  \u26a0 {_warn}")

    # ── --watchlist: resolve named watchlist tickers from config ─────────────
    _watchlist_tickers = None
    if args.watchlist:
        _wl_name = args.watchlist.lower().replace("-", "_")
        _wls = config.get("watchlists", {})
        if _wl_name in _wls:
            _watchlist_tickers = _wls[_wl_name]
            print(f"Using watchlist '{_wl_name}': {len(_watchlist_tickers)} tickers")
        else:
            _available = list(_wls.keys())
            print(f"Unknown watchlist '{_wl_name}'. Available: {', '.join(_available)}")
            sys.exit(1)

    # ── --top N: run cross-ticker top-N scan and exit ─────────────────────────
    if args.top is not None:
        _top_n = max(1, args.top)
        _top_tickers = _watchlist_tickers or config.get("watchlists", {}).get("liquid_large_cap", [
            "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
            "JPM", "BAC", "GS", "V", "MA", "AMD", "XOM", "CVX",
        ])
        _do_export = (args.export or "").lower() == "csv"
        _ts_pick = None if getattr(args, "no_tearsheet", False) else getattr(args, "tearsheet", None)
        print(f"\nRunning top-{_top_n} scan across {len(_top_tickers)} tickers...")
        # --min-dte / --max-dte parse fine but were dropped here, so the scan
        # silently used run_top_scan's own 7/45 defaults. Passed through only
        # when set, so omitting the flags keeps those defaults.
        _dte_kw = {}
        if getattr(args, "min_dte", None) is not None:
            _dte_kw["min_dte"] = args.min_dte
        if getattr(args, "max_dte", None) is not None:
            _dte_kw["max_dte"] = args.max_dte
        run_top_scan(_top_tickers, top_n=_top_n, export_csv=_do_export,
                     tearsheet_pick=_ts_pick, **_dte_kw)
        sys.exit(0)

    # ── Startup Banner (Phase 1) ─────────────────────────────────────────────
    now_str = datetime.now().strftime("%a %d %b %Y  %H:%M")
    if HAS_ENHANCED_CLI:
        print("\n" + fmt.draw_box("OPTIONS SCREENER  \u2022  Pro Edition", WIDTH, double=True))
        print(fmt.colorize(f"  {now_str}", fmt.Colors.DIM))
    else:
        print("\n" + "=" * WIDTH)
        print("  OPTIONS SCREENER  \u2022  Pro Edition")
        print(f"  {now_str}")
        print("=" * WIDTH)

    print(fmt.colorize("  Note: For personal/informational use only. Review data provider terms.", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "  Note: For personal/informational use only. Review data provider terms.")

    # ── Config Validation ──────────────────────────────────────────────────
    try:
        from .config_validator import validate_core_config
        _cfg_warnings = validate_core_config(config)
        if _cfg_warnings:
            for _cw in _cfg_warnings:
                print(fmt.format_warning(f"Config: {_cw}") if HAS_ENHANCED_CLI else f"  WARNING: Config: {_cw}")
    except Exception:
        pass

    # ── Interactivity, decided once up front ────────────────────────────────
    # Same flag the mode-menu loop below guards on: a real TTY with no
    # --auto/--mode/--ticker/--auto-log automation flag. Computed here (rather
    # than only later, next to the loop) so the dead-scheduler hard-confirm —
    # which must fire in the startup maintenance block below — can gate on
    # the exact same notion of "interactive" as everything else, instead of a
    # second, possibly-divergent check.
    _interactive = (sys.stdin.isatty() and not args.auto and not args.mode
                    and not args.ticker and not getattr(args, "auto_log", False))

    # ── Startup maintenance (replaces retired cron) ─────────────────────────
    try:
        import json as _json
        from .maintenance import run_startup_maintenance
        with open("config.json") as _cf:
            _p1 = (_json.load(_cf).get("auto_log") or {}).get("phase1_start_date")
        _maint = run_startup_maintenance(db_path="paper_trades.db", phase1_start=_p1,
                                         background=True)
        if _maint.get("cohort"):
            print(fmt.colorize(f"  {_maint['cohort']}", fmt.Colors.BRIGHT_WHITE)
                  if HAS_ENHANCED_CLI else f"  {_maint['cohort']}")
        if _maint.get("ran"):
            print(fmt.colorize(f"    (ran: {', '.join(_maint['ran'])})", fmt.Colors.DIM)
                  if HAS_ENHANCED_CLI else f"    (ran: {', '.join(_maint['ran'])})")
        # Staleness guard: loud, escalating banner when maintenance has fallen
        # behind (silent when fresh). Catches the machine-was-asleep case that
        # otherwise stalls the gate invisibly.
        try:
            from .maintenance import load_state, save_state, DEFAULT_STATE_PATH
            from .maintenance_health import (compute_health, health_banner,
                                             read_launchd_status, launchd_dead_days,
                                             next_launchd_dead_state,
                                             launchd_silence_days,
                                             seed_dead_since_date)
            _mh_state = load_state(DEFAULT_STATE_PATH)
            _launchd_jobs = read_launchd_status()
            # Exit status alone misses the outage: loaded jobs report 0 whether
            # they ran or not. The agents' own log is the only record of an
            # actual firing — see launchd_silence_days.
            _silence = launchd_silence_days()
            _banner = health_banner(compute_health(_mh_state, datetime.now()),
                                    launchd_jobs=_launchd_jobs,
                                    silence_days=_silence)
            if _banner:
                print(_banner)
            # Escalated hard-confirm (interactive only): launchctl carries no
            # timestamps, so "how long has it been dead" is tracked by
            # stamping first-observed-dead into the maintenance state and
            # diffing it on every later run. A *first* observation is seeded
            # from the LaunchAgent-only log rather than today, so a scheduler
            # that has genuinely been dead for weeks doesn't get handed a
            # fresh one-day-old marker (and a silent extra week before the
            # ack fires) the first time this code runs on it.
            _today = datetime.now().date()
            _next_mh_state = next_launchd_dead_state(
                _launchd_jobs, _mh_state, _today, seed_date=seed_dead_since_date(),
                silence_days=_silence)
            if _next_mh_state.get("launchd_dead_since") != _mh_state.get("launchd_dead_since"):
                save_state(DEFAULT_STATE_PATH, _next_mh_state)
            _dead_days = launchd_dead_days(_launchd_jobs, _next_mh_state, _today,
                                           silence_days=_silence)
            _dead_scheduler_ack(_dead_days, _interactive, WIDTH)
        except Exception:
            pass
    except Exception as _e:
        print(f"  (startup maintenance skipped: {_e})")

    # ── Automation health (catch silent cron death) ────────────────────────
    try:
        from .health import automation_health_warnings
        _health = automation_health_warnings(db_path="paper_trades.db")
        if _health:
            hdr = "Automation stale — scheduled jobs may have stopped:"
            print(fmt.format_warning(hdr) if HAS_ENHANCED_CLI else f"  WARNING: {hdr}")
            for _hw in _health:
                print(fmt.colorize(f"    • {_hw}", fmt.Colors.YELLOW) if HAS_ENHANCED_CLI
                      else f"    - {_hw}")
    except Exception:
        pass

    # ── Long-term buy zones (HOLDINGS desk) ────────────────────────────────
    # One quiet line per name near/in its accumulation zone. The desk itself
    # is launcher [5]; the banner lives here because this is the surface used
    # daily. Silent when the plan is empty or nothing is triggered.
    # Automation (cron via --auto/--mode/--auto-log) must never run this —
    # it does a real, untimed yf.download() fetch per name. Gate on the same
    # flags the later `_interactive` check uses so cron never blocks here.
    # See project_automation_staleness_guard / project_portfolio_spot_cache
    # for this exact failure class (silent blocking fetch on the cron path).
    _lt_automated = bool(args.auto or args.mode or getattr(args, "auto_log", False))
    if not _lt_automated:
        try:
            from .longterm.plan import DEFAULT_PATH as _LT_PLAN_PATH
            from .longterm.plan import load_plan as _lt_load_plan
            if os.path.exists(_LT_PLAN_PATH):
                _lt_plan = _lt_load_plan(_LT_PLAN_PATH)
                if _lt_plan.names:
                    from .longterm.board import _earnings_flags as _lt_flags
                    from .longterm.board import _gather as _lt_gather
                    from .longterm.board import banner as _lt_banner
                    from .longterm.zones import IN_ZONE as _LT_IN
                    from .longterm.zones import NEAR as _LT_NEAR
                    with ui.spinner("checking buy zones…"):
                        _snaps, _lt_reads, _lt_book, _lt_remaining = _lt_gather(_lt_plan)
                        _lt_earn = _lt_flags([r.ticker for r in _lt_reads
                                              if r.state in (_LT_IN, _LT_NEAR)])
                    _lt_text = _lt_banner(_lt_reads, _lt_plan, _lt_remaining,
                                          earnings=_lt_earn, width=WIDTH)
                    if _lt_text:
                        print("\n" + _lt_text)
        except Exception as _lt_exc:
            logging.getLogger(__name__).debug("longterm banner skipped: %s", _lt_exc)

    # ── Regime Dashboard + Portfolio Update (overlapped, race-free) ────────
    # update_positions() does no printing → it runs in the daemon thread, while
    # the regime dashboard renders synchronously here so the global sys.stdout is
    # always restored before the mode menu prints. See the helper for the race
    # the old thread-renders-and-redirects-stdout design caused (blank UI).
    pm = PaperManager(db_path="paper_trades.db", config_path="config.json")
    _dash = _render_regime_with_exit_enforcement(pm, WIDTH)
    if _dash:
        print(_dash, end="")

    # ── Sector/asset outlook (cache-first, instant; refreshes in background) ──
    try:
        from .outlook.display import print_outlook_box
        print_outlook_box(WIDTH)
    except Exception:
        pass

    # ── Market Hours Check ───────────────────────────────────────────────────
    is_open, mkt_msg = _check_market_hours()
    if not is_open:
        if HAS_ENHANCED_CLI:
            print(fmt.format_warning(mkt_msg))
            print(fmt.colorize("  Quotes are 15+ min delayed. Use results for planning, not live execution.", fmt.Colors.DIM))
        else:
            print(f"⚠  {mkt_msg}")
            print("  Quotes are 15+ min delayed. Use results for planning, not live execution.")
    print()

    # ── Interactive session loop ────────────────────────────────────────────
    # Re-show the mode menu after each action so a scan / intel / portfolio /
    # ticker returns the user here instead of exiting. Automation (cron, --auto,
    # --mode, --ticker, --auto-log) runs exactly one cycle and exits unchanged.
    # (`_interactive` itself is computed once, up in the startup-maintenance
    # section above, so the dead-scheduler hard-confirm can gate on it too.)
    # Let prompt_input resolve to defaults too, so a hand-run `run.py -ds` does
    # not stall on the ticker-source prompt with --auto already in its expansion.
    # Deliberately narrower than _interactive — see suppress_prompts_for.
    set_auto_mode(suppress_prompts_for(args))
    while True:
        # ── Mode Menu (Phase 1) ──────────────────────────────────────────────────
        _wl = load_watchlist()
        _wl_desc = f"Scan your {len(_wl)} saved ticker(s)" if _wl else "(empty \u2014 type ADD AAPL to begin)"
        if HAS_ENHANCED_CLI:
            from . import ui as _menu_ui
            print()
            print(_menu_ui.rule(WIDTH, "MODES"))
            # Thirteen modes, several tagged "no edge" or "read-only", and
            # nothing saying where to begin. One line is the whole fix; the
            # order below is the flow that actually works.
            print(fmt.style(_START_HINT, 'muted'))
            modes = [
                ("1", "TICKER",    "Single-stock deep analysis (e.g. AAPL)"),
                ("2", "ALL",       "Budget-based multi-stock scan"),
                ("3", "DISCOVER",  "Top 100 most-traded tickers \u2014 no budget limit"),
                ("4", "SELL",      "Premium Selling \u2014 income via short puts"),
                ("5", "SPREADS",   "Credit Spread analysis"),
                ("6", "IRON",      "Iron Condor analysis \u2014 range-bound"),
                ("7", "PORTFOLIO", "View open position P/L"),
                ("8", "MY LIST",   _wl_desc),
                ("9", "LOTTERY",   "Lottery Ticket \u2014 far-OTM plays on extreme moves"),
                ("10", "INTEL",    "Intel Briefing \u2014 everything before you buy + what to do"),
                ("11", "SQUEEZE",  "Short-squeeze setups \u2014 high-short-float candidates"),
                ("12", "PROB LAB", "Risk-neutral density + your-view structure ranking"),
                ("13", "STRUCTURE", "View → structure expression sized to your account"),
                ("Q", "QUIT",      "Exit the screener"),
            ]
            for num, cmd, desc in modes:
                n = fmt.style(f"[{num:>2}]", 'accent')
                c = fmt.style(f"{cmd:<10}", 'emph', bold=True)
                d = fmt.style(f"\u2014 {desc}", 'muted')
                print(f"  {n} {c} {d}")
            print(_menu_ui.rule(WIDTH))
        else:
            print("\nModes:")
            print(_START_HINT)
            print("  [1] TICKER     \u2014 Single-stock deep analysis (e.g. AAPL)")
            print("  [2] ALL        \u2014 Budget-based multi-stock scan")
            print("  [3] DISCOVER   \u2014 Top 100 most-traded tickers (no budget limit)")
            print("  [4] SELL       \u2014 Premium Selling analysis (short puts)")
            print("  [5] SPREADS    \u2014 Credit Spread analysis")
            print("  [6] IRON       \u2014 Iron Condor analysis")
            print("  [7] PORTFOLIO  \u2014 View open position P/L")
            print(f"  [8] MY LIST    \u2014 {_wl_desc}")
            print("  [9] LOTTERY    \u2014 Lottery Ticket: far-OTM plays on extreme moves")
            print("  [10] INTEL     \u2014 Intel Briefing: everything before you buy + what to do")
            print("  [11] SQUEEZE   \u2014 Short-squeeze setups (high short interest)")
            print("  [12] PROB LAB  \u2014 Risk-neutral density + your-view structure ranking")
            print("  [13] STRUCTURE \u2014 View \u2192 structure expression sized to your account")
            print("  [Q] QUIT       \u2014 Exit the screener")
        print()

        # ── --mode / --ticker CLI bypass ──────────────────────────────────────────
        _mode_map_cli = {
            "ticker": "TICKER", "all": "ALL", "discover": "DISCOVER",
            "sell": "SELL", "spreads": "SPREADS", "iron": "IRON",
            "portfolio": "PORTFOLIO", "mylist": "MY LIST",
            "lottery": "LOTTERY", "intel": "INTEL", "squeeze": "SQUEEZE",
            "structure": "STRUCTURE",
        }
        if args.ticker:
            symbol_input = args.ticker.upper()
        elif args.mode:
            symbol_input = _mode_map_cli.get(args.mode.lower(), "DISCOVER")
        else:
            try:
                symbol_input = prompt_input(
                    "Enter number, ticker, command, or Q to quit (default: 3)",
                    "3").upper()
            except (EOFError, KeyboardInterrupt):
                print()
                break

        # ── Quit ──────────────────────────────────────────────────────────────────
        if symbol_input in ("Q", "QUIT", "EXIT"):
            break

        # ── Watchlist commands ────────────────────────────────────────────────────
        if symbol_input.startswith("ADD "):
            add_to_watchlist(symbol_input[4:].strip())
            if _interactive:
                continue
            return
        if symbol_input.startswith("REMOVE "):
            remove_from_watchlist(symbol_input[7:].strip())
            if _interactive:
                continue
            return
        if symbol_input in ("SHOW LIST", "SHOW"):
            wl_cur = load_watchlist()
            if wl_cur:
                print(f"  Your watchlist ({len(wl_cur)} tickers): " + ", ".join(wl_cur))
            else:
                print("  Watchlist is empty. Type ADD AAPL to begin.")
            if _interactive:
                continue
            return

        # ── Number → command mapping ──────────────────────────────────────────────
        _num_map = {"1": "TICKER", "2": "ALL", "3": "DISCOVER", "4": "SELL",
                    "5": "SPREADS", "6": "IRON", "7": "PORTFOLIO", "8": "MY LIST",
                    "9": "LOTTERY", "10": "INTEL", "11": "SQUEEZE", "12": "PROBLAB",
                    "13": "STRUCTURE"}
        if symbol_input in _num_map:
            symbol_input = _num_map[symbol_input]
        elif symbol_input in ("PROB LAB", "PROB", "PROBABILITY LAB", "RND"):
            symbol_input = "PROBLAB"

        # ── The manual, one keystroke from the mode menu ──────────────────────
        # Chapter 5 is exactly the list of modes printed above, so this is where
        # the question gets asked. Crash-isolated: help never breaks a scan.
        if symbol_input in ("?", "HELP"):
            try:
                from .help_desk import run_menu as _help_menu
                _help_menu()
            except Exception as exc:  # noqa: BLE001
                print(f"  Help unavailable: {exc}")
            if _interactive:
                continue
            return

        # ── STRUCTURE mode: view → structure expression, sized to the account ─────
        if symbol_input == "STRUCTURE":
            _run_structure_menu()
            if _interactive:
                continue
            return

        # ── PROBLAB mode: risk-neutral density + view-based structure ranking ──────
        if symbol_input == "PROBLAB":
            _run_probability_lab_menu()
            if _interactive:
                continue
            return

        # ── INTEL mode: pre-trade briefing (market overview or single-ticker) ──────
        if symbol_input == "INTEL":
            _run_intel_menu()
            if _interactive:
                continue
            return

        if symbol_input == "PORTFOLIO":
            from .check_pnl import view_portfolio_menu
            view_portfolio_menu()
            if _interactive:
                continue
            return

        # ── MY LIST mode ──────────────────────────────────────────────────────────
        is_my_list_mode = (symbol_input in ("MY LIST", "MYLIST"))
        if is_my_list_mode:
            _wl_tickers = load_watchlist()
            if not _wl_tickers:
                print("  Your watchlist is empty. Type ADD AAPL to add a ticker first.")
                if _interactive:
                    continue
                return
            symbol_input = "DISCOVER"  # reuse discovery flow with custom ticker list

        is_budget_mode = (symbol_input == "ALL")
        # Capital at risk a single position may tie up on THIS scan. A distinct
        # quantity from `budget` below, which is the Budget-scan mode's cost of
        # one CONTRACT. Defined here so every mode has the name, including the
        # ones that never reach the prompt; None means no limit.
        session_budget: Optional[float] = None
        # Whether the prompt was actually reached. Separate from the value
        # because None means "chose no limit", which is not the same claim as
        # "was never asked" — see _with_session_budget.
        budget_was_chosen = False
        is_discovery_mode = (symbol_input in ("DISCOVER", "")) or is_my_list_mode
        is_ticker_mode = (symbol_input == "TICKER")  # user chose [1] — will prompt for symbol
        is_premium_selling_mode = (symbol_input == "SELL")
        is_credit_spread_mode = (symbol_input == "SPREADS")
        is_iron_condor_mode = (symbol_input == "IRON")
        is_lottery_mode = (symbol_input == "LOTTERY")
        is_squeeze_mode = (symbol_input == "SQUEEZE")

        # CBOE failover is gated on how many symbols the scan will touch, not on
        # the mode's name — MY LIST runs through the discovery flow but over a
        # bounded watchlist, so it qualifies while a top-100 sweep does not.
        # CBOE is a free unauthenticated endpoint; staying polite is the point.
        _broad_sweep = (is_discovery_mode and not is_my_list_mode) or is_budget_mode
        try:
            from .data_fetching import set_cboe_fallback
            set_cboe_fallback(not _broad_sweep)
        except Exception:
            pass

        if is_my_list_mode:
            mode = "Discovery scan"
        elif is_discovery_mode:
            mode = "Discovery scan"
        elif is_budget_mode:
            mode = "Budget scan"
        elif is_premium_selling_mode:
            mode = "Premium Selling"
        elif is_credit_spread_mode:
            mode = "Credit Spreads"
        elif is_iron_condor_mode:
            mode = "Iron Condor"
        elif is_lottery_mode:
            mode = "Lottery Ticket"
        elif is_squeeze_mode:
            mode = "Squeeze Hunt"
        else:
            mode = "Single-stock"

        budget = None
        tickers = []

        if _watchlist_tickers and not is_my_list_mode:
            tickers = _watchlist_tickers
            print(f"  Using --watchlist tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        elif is_my_list_mode:
            tickers = _wl_tickers
            print(f"  Scanning your watchlist: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        elif is_lottery_mode:
            tickers = prompt_for_tickers()
            print(f"  Scanning {len(tickers)} tickers for lottery ticket setups: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        elif is_squeeze_mode:
            from src.squeeze.board import sourcing_lines
            from src.squeeze.universe import get_squeeze_universe_detailed
            print("  Sourcing from Finviz (Float Short > 20%, +10% week first, ranked by short interest)...")
            _uni = get_squeeze_universe_detailed(max_tickers=15)
            tickers = _uni.tickers
            print(f"  Scanning {len(tickers)} squeeze candidates: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
            for _line in sourcing_lines(_uni):
                print(f"  {_line}")
        elif is_discovery_mode or is_premium_selling_mode or is_credit_spread_mode or is_iron_condor_mode:
            tickers = prompt_for_tickers()
            print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        elif is_budget_mode:
            try:
                budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
                if budget <= 0:
                    print("Budget must be greater than 0.")
                    sys.exit(1)
            except Exception:
                print("Invalid budget amount.")
                sys.exit(1)
            scan_type = prompt_input("Enter 1 for TARGETED or 2 for DISCOVERY", "1")
            if scan_type == "2":
                tickers = prompt_for_tickers()
            else:
                default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
                tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
                tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        elif is_ticker_mode:
            ticker_sym = prompt_input("Enter stock ticker symbol", "AAPL").upper()
            if not ticker_sym.isalnum():
                print("Please enter a valid alphanumeric ticker.")
                sys.exit(1)
            tickers = [ticker_sym]
        else:
            if not symbol_input.isalnum():
                print("Please enter a valid alphanumeric ticker.")
                sys.exit(1)
            tickers = [symbol_input]

        # ── Per-scan budget ──────────────────────────────────────────────────
        # ONE call, after the ticker source is settled, keyed off the MODE
        # rather than off whichever branch above happened to resolve the
        # tickers. Asking inside those branches is what left MY LIST and
        # TICKER without a prompt: `elif is_my_list_mode` catches MY LIST
        # before the discovery branch, and `elif is_ticker_mode` sits after
        # it. Because there is exactly one call site, no mode can be prompted
        # twice however the branches are later rearranged.
        #
        # Excluded on purpose: the Budget scan (ALL) already asked for a
        # per-CONTRACT budget, which is a different quantity; the Lottery and
        # Squeeze sleeves are display/tracking boards, not sized entries.
        if (is_discovery_mode or is_my_list_mode or is_ticker_mode
                or is_premium_selling_mode or is_credit_spread_mode
                or is_iron_condor_mode):
            session_budget = prompt_for_budget()
            budget_was_chosen = True
            if session_budget is not None:
                print(f"Budget: ${session_budget:,.0f} capital at risk per position")

        logger = setup_logging()
        print("\nFetching market context (SPY/VIX)...")
        market_trend, volatility_regime, macro_risk_active, tnx_change_pct = get_market_context()
        if HAS_ENHANCED_CLI:
            trend_color = fmt.Colors.GREEN if market_trend == "Bullish" else (fmt.Colors.RED if market_trend == "Bearish" else fmt.Colors.YELLOW)
            vix_color = fmt.Colors.GREEN if volatility_regime == "Low" else (fmt.Colors.RED if volatility_regime == "High" else fmt.Colors.YELLOW)
            trend_str = fmt.colorize(market_trend, trend_color, bold=True)
            vol_str = fmt.colorize(volatility_regime, vix_color)
            print(f"\u2713 Market Trend: {trend_str} | Volatility: {vol_str}")
            if macro_risk_active:
                print(fmt.format_warning("Macro risk active \u2014 elevated market uncertainty"))
        else:
            print(f"\u2713 Market Trend: {market_trend} | Volatility: {volatility_regime}")

        f_config = config.get("filters", {})
        if is_iron_condor_mode:
            default_min_dte = str(f_config.get("min_days_to_expiration_iron", 30))
            default_max_dte = str(f_config.get("max_days_to_expiration_iron", 60))
        elif is_squeeze_mode:
            # The calls board floors at SQUEEZE_MIN_DTE so its multiples mean
            # what they say. Discovery's 45-day window never reaches that, so
            # the mode would warn on every run instead of selecting anything.
            default_min_dte = str(f_config.get("min_days_to_expiration", 7))
            default_max_dte = str(f_config.get("max_days_to_expiration_squeeze",
                                              SQUEEZE_MAX_DTE))
        else:
            default_min_dte = str(f_config.get("min_days_to_expiration", 7))
            default_max_dte = str(f_config.get("max_days_to_expiration", 45))
        # CLI overrides win everywhere (incl. the interactive prompt defaults) —
        # the maintenance cohort feeder relies on --min-dte 30.
        if getattr(args, "min_dte", None) is not None:
            default_min_dte = str(args.min_dte)
        if getattr(args, "max_dte", None) is not None:
            default_max_dte = str(args.max_dte)

        # Squeeze names are weekly-heavy: the nearest 4 expirations are all
        # inside a month, so the count has to rise with the DTE window or the
        # far expiries never get fetched to be floored on.
        _default_expiries = str(
            config.get("max_expirations_squeeze", SQUEEZE_MAX_EXPIRIES)
            if is_squeeze_mode else config.get("max_expirations", 4)
        )
        if args.auto:
            max_expiries = int(_default_expiries)
            min_dte = int(default_min_dte)
            max_dte = int(default_max_dte)
            trader_profile = "swing"
        else:
            try:
                max_expiries = int(prompt_input("How many nearest expirations to scan",
                                                _default_expiries))
            except Exception:
                print("Invalid number for expirations.")
                sys.exit(1)
            try:
                min_dte = int(prompt_input("Minimum days to expiration (DTE)", default_min_dte))
                max_dte = int(prompt_input("Maximum days to expiration (DTE)", default_max_dte))
            except Exception:
                print("Invalid DTE inputs.")
                sys.exit(1)
            profile_choice = prompt_input("Enter 1 for Swing or 2 for Day trader", "1").strip()
            trader_profile = "day" if profile_choice == "2" else "swing"

        # Account size for position sizing in order tickets
        if not args.auto:
            _acct_input = prompt_input("Account size in USD for position sizing (Enter to skip)", "").strip()
            if _acct_input:
                try:
                    config["_account_size"] = float(_acct_input)
                except ValueError:
                    pass

        _is_single_stock = (mode == "Single-stock")
        _repeat_count = 0

        try:
            while True:
                show_surface = getattr(args, 'surface', False) or getattr(args, 'surface_greek', None) is not None
                surface_mode = getattr(args, 'surface_mode', 'braille')
                surface_greek = getattr(args, 'surface_greek', None)
                surface_type = surface_greek if surface_greek else 'pnl'
                show_contours = not getattr(args, 'no_contours', False)
                scan_results = run_scan(mode=mode, tickers=tickers, budget=budget, max_expiries=max_expiries, min_dte=min_dte, max_dte=max_dte, trader_profile=trader_profile, logger=logger, market_trend=market_trend, volatility_regime=volatility_regime, macro_risk_active=macro_risk_active, tnx_change_pct=tnx_change_pct, custom_weights=_custom_weights, show_surface=show_surface, surface_mode=surface_mode, surface_type=surface_type, show_contours=show_contours, compact=getattr(args, 'compact', False), interactive=(_interactive and not getattr(args, 'no_tearsheet', False)), tearsheet_pick=(None if getattr(args, 'no_tearsheet', False) else getattr(args, 'tearsheet', None)), session_budget=session_budget)
                if scan_results is None:
                    sys.exit(0)

                picks = scan_results.picks

                # ── Squeeze Hunt summary board (display-only) ─────────────────
                if mode == "Squeeze Hunt" and not picks.empty and "symbol" in picks.columns:
                    try:
                        from src.squeeze.board import squeeze_scan_board as _sq_scan_board
                        from src.squeeze.board import best_call_label as _sq_best_call
                        from src.squeeze.detector import assess_squeeze_row as _sq_assess_row
                        _sq_stash_map = getattr(scan_results, "squeeze_calls", {}) or {}
                        _sq_per = []
                        for _sq_sym, _sq_grp in picks.groupby(picks["symbol"].astype(str)):
                            _sq_s = _sq_assess_row(_sq_grp.iloc[0].to_dict())
                            # Prefer the stash: every "Best call: —" on the
                            # 2026-08-07 board came from reading _sq_grp, which
                            # the delta band had already emptied of calls.
                            _sq_calls = _sq_stash_map.get(_sq_sym)
                            if _sq_calls is None or _sq_calls.empty:
                                _sq_calls = _sq_grp
                            _sq_per.append({
                                "ticker": _sq_sym, "setup": _sq_s,
                                "best_call": _sq_best_call(
                                    _sq_calls, rfr=getattr(scan_results, "rfr", 0.045)),
                            })
                        if _sq_per:
                            print("\n" + _sq_scan_board(_sq_per, width=WIDTH))
                    except Exception as _sqb_exc:
                        logging.getLogger(__name__).debug("squeeze board skipped: %s", _sqb_exc)

                # ── AI Analysis ────────────────────────────────────────────────
                _ai_ranked = None
                if not picks.empty and not getattr(args, 'no_ai', False):
                    _ai_ranked = _run_ai_pipeline(picks, volatility_regime, verbose=True, mode=mode,
                                                   sector_ctx=scan_results.market_context.get("sector_ctx"),
                                                   ticker_contexts=scan_results.ticker_contexts)

                # Pull spread/condor results for the save menu
                _credit_spreads = scan_results.credit_spreads
                _iron_condors   = scan_results.iron_condors
                _has_results = (
                    not picks.empty
                    or (isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty)
                    or (isinstance(_iron_condors,   pd.DataFrame) and not _iron_condors.empty)
                )

                # ── Auto-export if --export csv was passed ──────────────────────
                if getattr(args, "export", None) and str(args.export).lower() == "csv" and not picks.empty:
                    _ts = datetime.now().strftime("%Y%m%d_%H%M")
                    _auto_fname = _export_path(f"scan_results_{_ts}.csv")
                    _export_cols = [
                        "symbol", "type", "strike", "expiration",
                        "bid", "ask", "premium", "delta", "impliedVolatility",
                        "iv_rank_30", "prob_profit", "ev_per_contract",
                        "quality_score", "score_drivers",
                    ]
                    _auto_df = picks[[c for c in _export_cols if c in picks.columns]].copy()
                    _auto_df.to_csv(_auto_fname, index=False)
                    _msg = f"Auto-exported {len(_auto_df)} rows to {_auto_fname}"
                    print(fmt.format_success(_msg) if HAS_ENHANCED_CLI else f"  \u2713 {_msg}")

                # ── Auto-launch 3D visualizer if --viz was passed ─────────────
                if getattr(args, 'viz', False) and _has_results and not picks.empty:
                    try:
                        from .visualizer_3d import OptionsVisualizer
                        _viz_cfg = load_config("config.json")
                        _viz = OptionsVisualizer(scan_results, config=_viz_cfg)
                        _viz.show()
                        msg = "3D Visualizer opened in browser"
                        print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                    except ImportError:
                        print(fmt.format_warning("Plotly required for --viz: pip install plotly") if HAS_ENHANCED_CLI else "  plotly required for --viz")
                    except Exception as _viz_exc:
                        logger.warning("Visualizer failed: %s", _viz_exc)

                # ── Lottery sleeve auto-log (dedicated path; never touches the ──────
                # shared allowlist, so no other mode's logging behaviour changes) ──
                if _has_results and getattr(args, "auto_log", False) and mode == "Lottery Ticket":
                    try:
                        from src.lottery.sleeve import autolog_lottery_sleeve
                        from src.paper_manager import PaperManager as _LotPM
                        _lot_pm = _LotPM(db_path="paper_trades.db", config_path="config.json")
                        _lot_logged = autolog_lottery_sleeve(
                            picks, _lot_pm, config_path="config.json",
                            top_n=int(getattr(args, "log_top", 5) or 5),
                        )
                        if _lot_logged:
                            print()
                            _hdr = f"  Logged {len(_lot_logged)} to the lottery sleeve:"
                            print(fmt.colorize(_hdr, fmt.Colors.BRIGHT_GREEN) if HAS_ENHANCED_CLI else _hdr)
                            for _line in _lot_logged:
                                print(f"    {_line}")
                            print(fmt.colorize("    Track it: python3 -m src.check_pnl", fmt.Colors.DIM) if HAS_ENHANCED_CLI else "    Track it: python3 -m src.check_pnl")
                        else:
                            print("  Lottery sleeve: nothing logged (traps skipped or exposure cap reached).")
                    except Exception as _lot_exc:
                        logger.warning("lottery sleeve auto-log failed: %s", _lot_exc)

                # ── Auto-log mode: bypass interactive save menu ──────────────────
                if _has_results and getattr(args, "auto_log", False) and mode not in ("Lottery Ticket", "Squeeze Hunt"):
                    # Pick exactly one result source — prefer single-leg picks, otherwise
                    # the first non-empty spread/condor DF that the scan produced.
                    _log_src = picks if not picks.empty else (
                        _credit_spreads if isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty
                        else _iron_condors
                    )
                    _is_spread_src = (
                        isinstance(_log_src, pd.DataFrame) and not _log_src.empty
                        and ("short_strike" in _log_src.columns or "net_credit" in _log_src.columns or "total_credit" in _log_src.columns)
                    )

                    if _is_spread_src:
                        # ── Spreads / iron condors path ─────────────────────────
                        _spreads = _log_src.copy()
                        # Order by what survives its costs, not by the composite.
                        # Condors carry per-leg quotes now, so they can be priced
                        # rather than refused wholesale. See
                        # rank_structures_by_verdict.
                        #
                        # DELIBERATELY NOT GATED, and this asymmetry is the point.
                        # `pick_ranking` refuses off-index condors on the BOARD
                        # (G5: +9.5% mean return on capital on broad index against
                        # -11.8% elsewhere, n=139, p < 1e-5). The auto-logger keeps
                        # writing them, as `paper_only=1` research rows that never
                        # counted as book trades.
                        #
                        # Gating here would freeze the off-index sample at the
                        # n=139 that produced the rule, and a rule that can only
                        # ever be confirmed by its own training set is not a
                        # finding. Logging continues so the sample grows and
                        # `scripts/validate_gates.py` can overturn G5 if the edge
                        # was a three-month artifact. Ruled 2026-08-10.
                        _spreads = rank_structures_by_verdict(_spreads)
                        # One row per ticker — keep highest-scored structure per symbol
                        if "symbol" in _spreads.columns:
                            _spreads = _spreads.drop_duplicates(subset=["symbol"], keep="first")
                        _top_n = max(1, int(getattr(args, "log_top", 5) or 5))

                        # Budget pre-filter — see the single-leg path for why this must run
                        # BEFORE the top-N cut rather than at the ledger door.
                        _budget_cap = auto_log_budget_cap("config.json")
                        _displaced = 0
                        if _budget_cap and not _spreads.empty:
                            _afford_mask = _spreads.apply(
                                lambda r: pick_within_budget(
                                    r, structure_strategy_name(r), _budget_cap
                                ),
                                axis=1,
                            )
                            _displaced = int((~_afford_mask).head(_top_n).sum())
                            _spreads = _spreads[_afford_mask]
                        _candidates = _spreads.head(_top_n)
                        _today_str = datetime.now().strftime("%Y-%m-%d")
                        _inserted = 0
                        _skipped = 0
                        _skipped_bear_calls = 0
                        # Split budget refusals out of the duplicate count below.
                        _rejected_before = getattr(pm, "unaffordable_rejected", 0)
                        # Same for near-duplicates of an entry from the last few days —
                        # the catch-up replay case the same-day dedup cannot see.
                        _dup_before = getattr(pm, "duplicate_rejected", 0)

                        # Component-score fields carried over from the spread enrichment
                        _spread_score_keys = (
                            "pop_score", "ev_score", "rr_score", "liquidity_score",
                            "momentum_score", "iv_rank_score", "theta_score",
                            "iv_advantage_score", "vrp_score", "iv_mispricing_score",
                            "skew_align_score", "vega_risk_score", "term_structure_score",
                            "catalyst_score", "em_realism_score", "gamma_theta_score",
                            "gex_score", "gamma_magnitude_score", "gamma_pin_score",
                            "iv_velocity_score", "max_pain_score", "oi_change_score",
                            "option_rvol_score", "pcr_score", "sentiment_score_norm",
                            "spread_score", "trader_pref_score",
                            "entry_iv", "entry_delta", "entry_gamma", "entry_vega", "entry_theta",
                        )
                        # iv_edge_score has the source name iv_advantage_score in scoring, but the
                        # paper_trades column is iv_edge_score. Map at insert time below.
                        for _, row in _candidates.iterrows():
                            _sym = str(row.get("symbol", "")).upper()
                            try:
                                # Derive strategy name to feed the allowlist helper.
                                _strat_name = structure_strategy_name(row)
                                _is_condor = _strat_name == "Iron Condor"
                                _decision, _paper_only_flag = apply_auto_log_allowlist(
                                    {"strategy_name": _strat_name}, cfg_path="config.json"
                                )
                                if _decision == "drop":
                                    _skipped_bear_calls += 1  # reuse counter for the summary
                                    continue
                                _common_scores = {k: row.get(k) for k in _spread_score_keys if k in row.index}
                                _common_scores["iv_edge_score"] = row.get("iv_advantage_score")
                                _common_scores["weight_profile"] = _weight_profile_id

                                if _is_condor:
                                    _payload = dict(_common_scores)
                                    _payload.update({
                                        "date": _today_str,
                                        "ticker": _sym,
                                        "expiration": row["expiration"],
                                        "short_put_strike": row.get("short_put_strike", 0),
                                        "long_put_strike":  row.get("long_put_strike", 0),
                                        "short_call_strike": row.get("short_call_strike", 0),
                                        "long_call_strike":  row.get("long_call_strike", 0),
                                        "total_credit": row.get("total_credit", 0),
                                        "max_profit":   row.get("max_profit", 0),
                                        "max_risk":     row.get("max_risk", 0),
                                        "net_delta":    row.get("net_delta"),
                                        "quality_score": row.get("quality_score", 0.5),
                                        "paper_only": _paper_only_flag,
                                    })
                                    if pm.log_iron_condor_if_new(_payload, auto_log=True):
                                        _inserted += 1
                                    else:
                                        _skipped += 1
                                else:
                                    _payload = dict(_common_scores)
                                    _payload.update({
                                        "date": _today_str,
                                        "ticker": _sym,
                                        "expiration": row["expiration"],
                                        "short_strike": row.get("short_strike", 0),
                                        "long_strike":  row.get("long_strike", 0),
                                        "type": row.get("type", "Spread"),
                                        "net_credit": row.get("net_credit", 0),
                                        "max_profit": row.get("max_profit", 0),
                                        "max_loss":   row.get("max_loss", 0),
                                        "quality_score": row.get("quality_score", 0.5),
                                        "paper_only": _paper_only_flag,
                                    })
                                    if pm.log_spread_if_new(_payload, auto_log=True):
                                        _inserted += 1
                                    else:
                                        _skipped += 1
                            except Exception as _log_exc:
                                print(f"  Error auto-logging {_sym}: {_log_exc}")
                        _tag = _weight_profile_id or "untagged"
                        _bc_suffix = f", filtered {_skipped_bear_calls} disallowed structure(s)" if _skipped_bear_calls else ""
                        # A budget refusal is not a duplicate — see the single-leg path.
                        _refused = getattr(pm, "unaffordable_rejected", 0) - _rejected_before
                        _near_dupes = getattr(pm, "duplicate_rejected", 0) - _dup_before
                        _dupes = max(0, _skipped - _refused - _near_dupes)
                        _summary = (
                            f"Auto-logged {_inserted} spreads/condors, "
                            f"skipped {_dupes} duplicates{_bc_suffix} (profile: {_tag})"
                        )
                        if _near_dupes:
                            _summary += f", refused {_near_dupes} as re-logs of a recent entry"
                        if _refused:
                            _summary += f", refused {_refused} over budget"
                        if _displaced:
                            _summary += (
                                f", {_displaced} of the top {_top_n} exceeded the "
                                f"${_budget_cap:,.0f} budget"
                            )
                        print(fmt.format_success(_summary) if HAS_ENHANCED_CLI else f"  ✓ {_summary}")
                        _has_results = False

                    # ── Single-leg path (original) ──────────────────────────────
                    elif isinstance(_log_src, pd.DataFrame) and not _log_src.empty and "symbol" in _log_src.columns:
                        _single_legs = _log_src.copy()
                        # Order by what survives its costs, not by the composite.
                        # This decides BOTH which leg per symbol survives the dedup
                        # below and which symbols reach the top-N, so it selected
                        # every row in the ledger. See rank_single_legs_by_verdict.
                        _single_legs = rank_single_legs_by_verdict(_single_legs, mode)
                        # ...then refuse what the BOARD refused. Ordering alone
                        # let the logger write single legs the reader was never
                        # shown — these are `paper_only=0` book trades, unlike
                        # the condor research rows, so board and book must agree.
                        # Audited 2026-08-10: 0 of 5 top picks were board-refused
                        # on a 78-contract sample, because this ordering is
                        # EV-descending and the main gate is EV-based. It bites
                        # when the top-EV pick is refused for another reason.
                        _single_legs = gate_and_report(_single_legs, "AUTO-LOG",
                                                       verbose=False)
                        # One row per ticker — keep the highest-scored leg per symbol to avoid
                        # concentration (e.g. ORCL×6 from a single scan).
                        if "symbol" in _single_legs.columns:
                            _single_legs = _single_legs.drop_duplicates(subset=["symbol"], keep="first")
                        _today_str = datetime.now().strftime("%Y-%m-%d")
                        # Drop rows the allowlist would reject entirely (e.g. Long Puts once
                        # removed from paper_only_strategies) BEFORE taking the top-N. Without
                        # this, a scan whose top-scored legs are Long Puts logs almost nothing
                        # and the forward cohort starves — the dropped strategies would silently
                        # consume the top-N slots. Quarantined (paper_only) strategies are kept.
                        if not _single_legs.empty and "type" in _single_legs.columns:
                            def _allowlist_keeps(_row):
                                _sn = _strategy_label_for_mode(mode, _row.get("type"))
                                _dec, _ = apply_auto_log_allowlist(
                                    {"strategy_name": _sn,
                                     "expiration": _row.get("expiration"),
                                     "date": _today_str},
                                    cfg_path="config.json",
                                )
                                return _dec != "drop"
                            _single_legs = _single_legs[_single_legs.apply(_allowlist_keeps, axis=1)]
                        _top_n = max(1, int(getattr(args, "log_top", 5) or 5))
                        # Budget pre-filter — same reasoning as the allowlist filter above.
                        # An unaffordable pick IS refused by the ledger, but only after it has
                        # already consumed a top-N slot. On 2026-07-30 the short-premium window
                        # scored 1,109 contracts, gave all five slots to $13k-$74k cash-secured
                        # puts, and logged nothing. Rank what the account can actually take.
                        _budget_cap = auto_log_budget_cap("config.json")
                        _displaced = 0
                        if _budget_cap and not _single_legs.empty and "type" in _single_legs.columns:
                            def _affordable(_row):
                                return pick_within_budget(
                                    _row,
                                    _strategy_label_for_mode(mode, _row.get("type")),
                                    _budget_cap,
                                )
                            _afford_mask = _single_legs.apply(_affordable, axis=1)
                            # Report only how many of the WOULD-BE top-N were unaffordable; the
                            # pool-wide count is noise (most of 1,109 were never in contention).
                            _displaced = int((~_afford_mask).head(_top_n).sum())
                            _single_legs = _single_legs[_afford_mask]
                        _candidates = _single_legs.head(_top_n)

                        _inserted = 0
                        _skipped = 0
                        _skipped_long_puts = 0
                        # Split budget refusals out of the duplicate count below.
                        _rejected_before = getattr(pm, "unaffordable_rejected", 0)
                        # Same for near-duplicates of an entry from the last few days —
                        # the catch-up replay case the same-day dedup cannot see.
                        _dup_before = getattr(pm, "duplicate_rejected", 0)
                        # AI-score lookup keyed on (symbol, strike, expiration, type) — index-based
                        # lookups are unsafe because _ai_ranked is reset_index'd inside ranking.combine_scores
                        # and re-sorted, so positional alignment with picks is not preserved.
                        _ai_lookup = {}
                        if isinstance(_ai_ranked, pd.DataFrame) and not _ai_ranked.empty:
                            _ai_cols_present = [c for c in ("ai_score", "ai_confidence") if c in _ai_ranked.columns]
                            if _ai_cols_present:
                                for _r in _ai_ranked.itertuples(index=False):
                                    _r_dict = _r._asdict() if hasattr(_r, "_asdict") else {}
                                    try:
                                        _key = (
                                            str(_r_dict.get("symbol", "")).upper(),
                                            float(_r_dict.get("strike", 0)),
                                            str(_r_dict.get("expiration", "")),
                                            str(_r_dict.get("type", "")).lower(),
                                        )
                                    except (TypeError, ValueError):
                                        continue
                                    _ai_lookup[_key] = {c: _r_dict.get(c) for c in _ai_cols_present}
                        for _idx, row in _candidates.iterrows():
                            _entry_price = (
                                safe_float(row.get("ask") or None)
                                or safe_float(row.get("lastPrice"))
                                or safe_float(row.get("premium"), 0.0)
                            )
                            if not _entry_price or _entry_price <= 0:
                                continue
                            _strat_name = _strategy_label_for_mode(mode, row['type'])
                            # Phase 1 allowlist (supersedes the legacy auto_log_skip_long_puts flag).
                            # Pass expiration so the cohort DTE floor can quarantine
                            # short-horizon Long Calls (else they slip into the gate).
                            _decision, _paper_only_flag = apply_auto_log_allowlist(
                                {"strategy_name": _strat_name,
                                 "expiration": row["expiration"], "date": _today_str},
                                cfg_path="config.json",
                            )
                            if _decision == "drop":
                                _skipped_long_puts += 1  # reuse counter for the summary line
                                continue
                            _trade = {
                                "date": _today_str,
                                "ticker": row["symbol"],
                                "expiration": row["expiration"],
                                "strike": row["strike"],
                                "type": str(row["type"]).capitalize(),
                                "entry_price": _entry_price,
                                "quality_score": row.get("quality_score", 0.5),
                                "strategy_name": _strat_name,
                                "entry_iv": row.get("impliedVolatility"),
                                "entry_delta": row.get("delta"),
                                "entry_gamma": row.get("gamma"),
                                "entry_vega": row.get("vega"),
                                "entry_theta": row.get("theta"),
                                "dividend_yield": row.get("dividend_yield"),
                                "pop_score": row.get("pop_score"),
                                "ev_score": row.get("ev_score"),
                                # Levels, not the within-scan rank beside them:
                                # schema 21 makes net_ev/noise reconstructable.
                                "ev_per_contract": row.get("ev_per_contract"),
                                "ev_gross_per_contract": row.get("ev_gross_per_contract"),
                                "ev_cost_per_contract": row.get("ev_cost_per_contract"),
                                "ev_noise": row.get("ev_noise"),
                                "rr_score": row.get("rr_score"),
                                "liquidity_score": row.get("liquidity_score"),
                                "momentum_score": row.get("momentum_score"),
                                "iv_rank_score": row.get("iv_rank_score"),
                                "theta_score": row.get("theta_score"),
                                "iv_edge_score": row.get("iv_advantage_score"),
                                "vrp_score": row.get("vrp_score"),
                                "iv_mispricing_score": row.get("iv_mispricing_score"),
                                "skew_align_score": row.get("skew_align_score"),
                                "vega_risk_score": row.get("vega_risk_score"),
                                "term_structure_score": row.get("term_structure_score"),
                                "catalyst_score": row.get("catalyst_score"),
                                "em_realism_score": row.get("em_realism_score"),
                                "gamma_theta_score": row.get("gamma_theta_score"),
                                "gex_score": row.get("gex_score"),
                                "gamma_magnitude_score": row.get("gamma_magnitude_score"),
                                "gamma_pin_score": row.get("gamma_pin_score"),
                                "iv_velocity_score": row.get("iv_velocity_score"),
                                "max_pain_score": row.get("max_pain_score"),
                                "oi_change_score": row.get("oi_change_score"),
                                "option_rvol_score": row.get("option_rvol_score"),
                                "pcr_score": row.get("pcr_score"),
                                "sentiment_score_norm": row.get("sentiment_score_norm"),
                                "spread_score": row.get("spread_score"),
                                "trader_pref_score": row.get("trader_pref_score"),
                                "score_adjustments": row.get("score_adjustments"),
                                "weight_profile": _weight_profile_id,
                                "paper_only": _paper_only_flag,
                            }
                            _row_key = (
                                str(row.get("symbol", "")).upper(),
                                float(row.get("strike", 0) or 0),
                                str(row.get("expiration", "")),
                                str(row.get("type", "")).lower(),
                            )
                            _row_ai = _ai_lookup.get(_row_key, {})
                            _trade["ai_score"] = _row_ai.get("ai_score")
                            _trade["ai_confidence"] = _row_ai.get("ai_confidence")
                            try:
                                if pm.log_trade_if_new(_trade, auto_log=True):
                                    _inserted += 1
                                else:
                                    _skipped += 1
                            except Exception as _log_exc:
                                print(f"  Error auto-logging {row.get('symbol')}: {_log_exc}")

                        _tag = _weight_profile_id or "untagged"
                        # A budget refusal is not a duplicate. Counting them together reported
                        # "skipped 5 duplicates" for a window that logged nothing because every
                        # pick was over budget — the one line that would have shown the problem.
                        _refused = getattr(pm, "unaffordable_rejected", 0) - _rejected_before
                        _near_dupes = getattr(pm, "duplicate_rejected", 0) - _dup_before
                        _dupes = max(0, _skipped - _refused - _near_dupes)
                        _summary = f"Auto-logged {_inserted} new, skipped {_dupes} duplicates (profile: {_tag})"
                        if _near_dupes:
                            _summary += f", refused {_near_dupes} as re-logs of a recent entry"
                        if _refused:
                            _summary += f", refused {_refused} over budget"
                        if _skipped_long_puts:
                            _summary += f", filtered {_skipped_long_puts} disallowed pick(s)"
                        if _displaced:
                            _summary += (
                                f", {_displaced} of the top {_top_n} exceeded the "
                                f"${_budget_cap:,.0f} budget"
                            )
                        print(fmt.format_success(_summary) if HAS_ENHANCED_CLI else f"  \u2713 {_summary}")
                    # Skip the interactive save-menu loop below; continue to scan-another prompt
                    _has_results = False

                # ── Collapsed post-scan prompt (loops so V → P → L all work in one sitting) ──
                if _has_results:
                  while True:
                    if HAS_ENHANCED_CLI:
                        save_label = fmt.colorize("Save/Export?", fmt.Colors.BRIGHT_CYAN)
                        p_opt = fmt.colorize("[P]", fmt.Colors.BRIGHT_YELLOW) + " Paper trade top pick"
                        c_opt = fmt.colorize("[C]", fmt.Colors.BRIGHT_YELLOW) + " CSV"
                        l_opt = fmt.colorize("[L]", fmt.Colors.BRIGHT_YELLOW) + " Log trades"
                        v_opt = fmt.colorize("[V]", fmt.Colors.BRIGHT_YELLOW) + " 3D Visualizer"
                        skip_opt = fmt.colorize("[Enter]", fmt.Colors.DIM) + " Skip"
                        print(f"\n  {save_label}  {p_opt}  \u00b7  {c_opt}  \u00b7  {l_opt}  \u00b7  {v_opt}  \u00b7  {skip_opt}")
                    else:
                        print("\n  Save/Export?  [P] Paper trade top pick  [C] CSV  [L] Log trades  [V] 3D Visualizer  [Enter] Skip")
                    save_choice = prompt_input("Choice", "").strip().upper()

                    if save_choice == "":
                        break  # Enter → done with save menu

                    if save_choice == "P":
                        if mode in ("Credit Spreads", "Iron Condor"):
                            msg = "Paper trading for spreads/condors is not supported — use [L] Log trades instead."
                            print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  \u26a0  {msg}")
                        elif not picks.empty:
                            # Use AI-ranked top pick when available, otherwise fall back to quality_score.
                            # Match on (symbol, strike, expiration, type) — _ai_ranked indices are not
                            # aligned with picks indices after reset_index inside combine_scores.
                            top_pick_row = None
                            if _ai_ranked is not None and not _ai_ranked.empty and "final_score" in _ai_ranked.columns:
                                _best = _ai_ranked.sort_values("final_score", ascending=False).iloc[0]
                                try:
                                    _match = picks[
                                        (picks["symbol"].astype(str).str.upper() == str(_best.get("symbol", "")).upper())
                                        & (picks["strike"].astype(float) == float(_best.get("strike", 0)))
                                        & (picks["expiration"].astype(str) == str(_best.get("expiration", "")))
                                        & (picks["type"].astype(str).str.lower() == str(_best.get("type", "")).lower())
                                    ]
                                    if not _match.empty:
                                        top_pick_row = _match.iloc[0]
                                except (KeyError, ValueError, TypeError):
                                    top_pick_row = None
                            if top_pick_row is None:
                                # Same ordering as the bulk auto-log path: the
                                # composite selected every ledger row until now.
                                _ranked_one = rank_single_legs_by_verdict(picks, mode)
                                # Gated: [P] "paper trade top pick" must mean
                                # the pick the reader was shown, not the top of
                                # an ungated list they never saw.
                                _ranked_one = gate_and_report(_ranked_one,
                                                              "PAPER TRADE",
                                                              verbose=False)
                                if _ranked_one is None or _ranked_one.empty:
                                    print(fmt.format_warning(
                                        "no candidate cleared the gates — nothing logged")
                                        if HAS_ENHANCED_CLI else
                                        "  no candidate cleared the gates — nothing logged")
                                    top_pick_row = None
                                else:
                                    top_pick_row = _ranked_one.iloc[0]
                            if top_pick_row is not None:
                                today_str = datetime.now().strftime("%Y-%m-%d")
                                trade_dict = {
                                    "date": today_str,
                                    "ticker": top_pick_row["symbol"],
                                    "expiration": top_pick_row["expiration"],
                                    "strike": top_pick_row["strike"],
                                    "type": str(top_pick_row["type"]).capitalize(),
                                    "entry_price": (
                                        safe_float(top_pick_row.get("ask") or None)
                                        or safe_float(top_pick_row.get("lastPrice"))
                                        or safe_float(top_pick_row.get("premium"), 0.0)
                                    ),
                                    "quality_score": top_pick_row["quality_score"],
                                    "strategy_name": _strategy_label_for_mode(mode, top_pick_row['type']),
                                    "entry_iv": top_pick_row.get("impliedVolatility"),
                                    "entry_delta": top_pick_row.get("delta"),
                                    "entry_gamma": top_pick_row.get("gamma"),
                                    "entry_vega": top_pick_row.get("vega"),
                                    "entry_theta": top_pick_row.get("theta"),
                                    "dividend_yield": top_pick_row.get("dividend_yield"),
                                    # Component scores — enable backtester IC analysis once 30+ trades close
                                    "pop_score": top_pick_row.get("pop_score"),
                                    "ev_score": top_pick_row.get("ev_score"),
                                    # Levels, not the within-scan rank beside them.
                                    # Schema 21 makes net_ev/noise reconstructable —
                                    # `ev_score` is a rank and cannot say how large
                                    # an edge was. This site was missed when the
                                    # columns shipped: it builds from `top_pick_row`
                                    # rather than `row`, so a patch written against
                                    # the other two dicts skipped it, and every trade
                                    # logged on 2026-08-10 stored NULL.
                                    "ev_per_contract": top_pick_row.get("ev_per_contract"),
                                    "ev_gross_per_contract": top_pick_row.get("ev_gross_per_contract"),
                                    "ev_cost_per_contract": top_pick_row.get("ev_cost_per_contract"),
                                    "ev_noise": top_pick_row.get("ev_noise"),
                                    "rr_score": top_pick_row.get("rr_score"),
                                    "liquidity_score": top_pick_row.get("liquidity_score"),
                                    "momentum_score": top_pick_row.get("momentum_score"),
                                    "iv_rank_score": top_pick_row.get("iv_rank_score"),
                                    "theta_score": top_pick_row.get("theta_score"),
                                    "iv_edge_score": top_pick_row.get("iv_advantage_score"),
                                    "vrp_score": top_pick_row.get("vrp_score"),
                                    "iv_mispricing_score": top_pick_row.get("iv_mispricing_score"),
                                    "skew_align_score": top_pick_row.get("skew_align_score"),
                                    "vega_risk_score": top_pick_row.get("vega_risk_score"),
                                    "term_structure_score": top_pick_row.get("term_structure_score"),
                                    # v7: remaining 14 components — full IC coverage
                                    "catalyst_score": top_pick_row.get("catalyst_score"),
                                    "em_realism_score": top_pick_row.get("em_realism_score"),
                                    "gamma_theta_score": top_pick_row.get("gamma_theta_score"),
                                    "gex_score": top_pick_row.get("gex_score"),
                                    "gamma_magnitude_score": top_pick_row.get("gamma_magnitude_score"),
                                    "gamma_pin_score": top_pick_row.get("gamma_pin_score"),
                                    "iv_velocity_score": top_pick_row.get("iv_velocity_score"),
                                    "max_pain_score": top_pick_row.get("max_pain_score"),
                                    "oi_change_score": top_pick_row.get("oi_change_score"),
                                    "option_rvol_score": top_pick_row.get("option_rvol_score"),
                                    "pcr_score": top_pick_row.get("pcr_score"),
                                    "sentiment_score_norm": top_pick_row.get("sentiment_score_norm"),
                                    "spread_score": top_pick_row.get("spread_score"),
                                    "trader_pref_score": top_pick_row.get("trader_pref_score"),
                                    "score_adjustments": top_pick_row.get("score_adjustments"),
                                    "weight_profile": _weight_profile_id,
                                }
                                # The budget key rides along only if this mode
                                # actually reached the prompt; the --auto-log
                                # paths above never call this, so they keep
                                # falling back to config.
                                _with_session_budget(trade_dict,
                                                     budget_was_chosen,
                                                     session_budget)
                                # AI-score lookup via stable key (see auto-log path comment).
                                if _ai_ranked is not None and not _ai_ranked.empty:
                                    try:
                                        _m = _ai_ranked[
                                            (_ai_ranked["symbol"].astype(str).str.upper() == str(top_pick_row.get("symbol", "")).upper())
                                            & (_ai_ranked["strike"].astype(float) == float(top_pick_row.get("strike", 0)))
                                            & (_ai_ranked["expiration"].astype(str) == str(top_pick_row.get("expiration", "")))
                                            & (_ai_ranked["type"].astype(str).str.lower() == str(top_pick_row.get("type", "")).lower())
                                        ]
                                        if not _m.empty:
                                            if "ai_score" in _m.columns:
                                                trade_dict["ai_score"] = _m["ai_score"].iloc[0]
                                            if "ai_confidence" in _m.columns:
                                                trade_dict["ai_confidence"] = _m["ai_confidence"].iloc[0]
                                    except (KeyError, ValueError, TypeError):
                                        pass
                                pm.log_trade(trade_dict)
                                msg = f"Paper trade logged: {top_pick_row['symbol']} {str(top_pick_row['type']).upper()} ${top_pick_row['strike']:.0f}"
                                print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                                # Offer inline portfolio view
                                _view = prompt_input("View portfolio? (y/n)", "n").strip().lower()
                                if _view in ("y", "yes"):
                                    try:
                                        from .check_pnl import view_portfolio
                                        view_portfolio()
                                    except Exception as _pnl_exc:
                                        print(f"  Could not load portfolio: {_pnl_exc}")

                    elif save_choice == "C":
                        # Export best available data: AI-ranked picks > raw picks > spreads > condors
                        if _ai_ranked is not None and not _ai_ranked.empty:
                            export_df = _ai_ranked
                        elif not picks.empty:
                            export_df = picks
                        elif isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty:
                            export_df = _credit_spreads
                        else:
                            export_df = _iron_condors
                        csv_file = export_to_csv(export_df, mode, budget)
                        if csv_file:
                            msg = f"Results exported to: {csv_file}"
                            print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"\n  \U0001f4c4 {msg}")

                    elif save_choice == "V":
                        try:
                            from .visualizer_3d import OptionsVisualizer
                            _viz_cfg = load_config("config.json")
                            _viz = OptionsVisualizer(scan_results, config=_viz_cfg)
                            _viz.show()
                            msg = "3D Visualizer opened in browser"
                            print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2713 {msg}")
                        except ImportError:
                            msg = "Plotly required for visualizer: pip install plotly"
                            print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  {msg}")
                        except Exception as _viz_exc:
                            msg = f"Visualizer error: {_viz_exc}"
                            print(fmt.format_warning(msg) if HAS_ENHANCED_CLI else f"  {msg}")

                    elif save_choice == "L":
                        log_src = picks if not picks.empty else (
                            _credit_spreads if isinstance(_credit_spreads, pd.DataFrame) and not _credit_spreads.empty
                            else _iron_condors
                        )
                        if isinstance(log_src, pd.DataFrame) and not log_src.empty:
                            picks_to_log = select_trades_to_log(log_src)
                            if not picks_to_log.empty:
                                log_trade_entry(picks_to_log, mode)
                            
                                # Also log to PaperManager for portfolio visibility
                                today_str = datetime.now().strftime("%Y-%m-%d")
                                # Stable-key AI lookup (see auto-log path).
                                _ai_lookup_l = {}
                                if isinstance(_ai_ranked, pd.DataFrame) and not _ai_ranked.empty:
                                    _ai_cols_l = [c for c in ("ai_score", "ai_confidence") if c in _ai_ranked.columns]
                                    if _ai_cols_l:
                                        for _r in _ai_ranked.itertuples(index=False):
                                            _r_dict = _r._asdict() if hasattr(_r, "_asdict") else {}
                                            try:
                                                _key = (
                                                    str(_r_dict.get("symbol", "")).upper(),
                                                    float(_r_dict.get("strike", 0)),
                                                    str(_r_dict.get("expiration", "")),
                                                    str(_r_dict.get("type", "")).lower(),
                                                )
                                            except (TypeError, ValueError):
                                                continue
                                            _ai_lookup_l[_key] = {c: _r_dict.get(c) for c in _ai_cols_l}
                                for _idx_l, row in picks_to_log.iterrows():
                                    try:
                                        if "short_strike" in row or "net_credit" in row:
                                            # It's a Credit Spread
                                            # log_spread routes through log_trade,
                                            # so the budget gate already applies
                                            # here — but without the key it
                                            # applied at CONFIG's $4,000, so a
                                            # spread the board hid at a smaller
                                            # session budget was still logged.
                                            pm.log_spread(_with_session_budget({
                                                "date": today_str,
                                                "ticker": row["symbol"],
                                                "expiration": row["expiration"],
                                                "short_strike": row["short_strike"],
                                                "long_strike": row["long_strike"],
                                                "type": row["type"],
                                                "net_credit": row["net_credit"],
                                                "max_profit": row.get("max_profit", 0),
                                                "max_loss": row.get("max_loss", 0),
                                                "quality_score": row.get("quality_score", 0.5),
                                            }, budget_was_chosen, session_budget))
                                        elif "total_credit" in row:
                                            # Iron Condor — persist all four legs so the
                                            # portfolio viewer can render strikes and mark
                                            # to market. log_iron_condor_if_new dedups on
                                            # the (ticker, exp, 4 strikes) tuple.
                                            # Budget key: see the spread path above.
                                            pm.log_iron_condor_if_new(_with_session_budget({
                                                "date": today_str,
                                                "ticker": row["symbol"],
                                                "expiration": row["expiration"],
                                                "short_put_strike": row["short_put_strike"],
                                                "long_put_strike":  row["long_put_strike"],
                                                "short_call_strike": row["short_call_strike"],
                                                "long_call_strike":  row["long_call_strike"],
                                                "total_credit": row["total_credit"],
                                                "max_profit": row.get("max_profit", 0),
                                                "max_risk":   row.get("max_risk", 0),
                                                "net_delta":  row.get("net_delta"),
                                                "quality_score": row.get("quality_score", 0.5),
                                            }, budget_was_chosen, session_budget))
                                        else:
                                            # It's a single option
                                            trade_dict = {
                                                "date": today_str,
                                                "ticker": row["symbol"],
                                                "expiration": row["expiration"],
                                                "strike": row["strike"],
                                                "type": str(row["type"]).capitalize(),
                                                "entry_price": (
                                                    safe_float(row.get("ask") or None)
                                                    or safe_float(row.get("lastPrice"))
                                                    or safe_float(row.get("premium"), 0.0)
                                                ),
                                                "quality_score": row.get("quality_score", 0.5),
                                                "strategy_name": _strategy_label_for_mode(mode, row['type']),
                                                "entry_iv": row.get("impliedVolatility"),
                                                "entry_delta": row.get("delta"),
                                                "entry_gamma": row.get("gamma"),
                                                "entry_vega": row.get("vega"),
                                                "entry_theta": row.get("theta"),
                                                "pop_score": row.get("pop_score"),
                                                "ev_score": row.get("ev_score"),
                                                "ev_per_contract": row.get("ev_per_contract"),
                                                "ev_gross_per_contract": row.get("ev_gross_per_contract"),
                                                "ev_cost_per_contract": row.get("ev_cost_per_contract"),
                                                "ev_noise": row.get("ev_noise"),
                                                "rr_score": row.get("rr_score"),
                                                "liquidity_score": row.get("liquidity_score"),
                                                "momentum_score": row.get("momentum_score"),
                                                "iv_rank_score": row.get("iv_rank_score"),
                                                "theta_score": row.get("theta_score"),
                                                "iv_edge_score": row.get("iv_advantage_score"),
                                                "vrp_score": row.get("vrp_score"),
                                                "iv_mispricing_score": row.get("iv_mispricing_score"),
                                                "skew_align_score": row.get("skew_align_score"),
                                                "vega_risk_score": row.get("vega_risk_score"),
                                                "term_structure_score": row.get("term_structure_score"),
                                                "catalyst_score": row.get("catalyst_score"),
                                                "em_realism_score": row.get("em_realism_score"),
                                                "gamma_theta_score": row.get("gamma_theta_score"),
                                                "gex_score": row.get("gex_score"),
                                                "gamma_magnitude_score": row.get("gamma_magnitude_score"),
                                                "gamma_pin_score": row.get("gamma_pin_score"),
                                                "iv_velocity_score": row.get("iv_velocity_score"),
                                                "max_pain_score": row.get("max_pain_score"),
                                                "oi_change_score": row.get("oi_change_score"),
                                                "option_rvol_score": row.get("option_rvol_score"),
                                                "pcr_score": row.get("pcr_score"),
                                                "sentiment_score_norm": row.get("sentiment_score_norm"),
                                                "spread_score": row.get("spread_score"),
                                                "trader_pref_score": row.get("trader_pref_score"),
                                                "score_adjustments": row.get("score_adjustments"),
                                                "weight_profile": _weight_profile_id,
                                            }
                                            # See the [P] path: key presence is
                                            # the signal, and --auto-log must
                                            # never carry it.
                                            _with_session_budget(trade_dict,
                                                                 budget_was_chosen,
                                                                 session_budget)
                                            _row_key_l = (
                                                str(row.get("symbol", "")).upper(),
                                                float(row.get("strike", 0) or 0),
                                                str(row.get("expiration", "")),
                                                str(row.get("type", "")).lower(),
                                            )
                                            _row_ai_l = _ai_lookup_l.get(_row_key_l, {})
                                            trade_dict["ai_score"] = _row_ai_l.get("ai_score")
                                            trade_dict["ai_confidence"] = _row_ai_l.get("ai_confidence")
                                            pm.log_trade(trade_dict)
                                    except Exception as _log_exc:
                                        print(f"  Error logging to DB: {_log_exc}")

                                msg = f"Logged {len(picks_to_log)} trades."
                                print(fmt.format_success(msg) if HAS_ENHANCED_CLI else f"  \u2705 {msg}")

                    else:
                        _msg = "Unknown choice — press P / C / L / V or Enter to skip"
                        print(fmt.format_warning(_msg) if HAS_ENHANCED_CLI else f"  {_msg}")
                    # Loop back and re-prompt so V → P → L all work in one sitting

                # ── Scan-another shortcut (single-stock only, AFTER save menu) ──
                if _is_single_stock and _repeat_count < 5:
                    _another = prompt_input("Scan another ticker? (enter symbol or Enter to quit)", "").upper().strip()
                    if _another and _another.isalnum() and 1 <= len(_another) <= 6:
                        tickers = [_another]
                        _repeat_count += 1
                        continue  # loop back

                # Done message
                if HAS_ENHANCED_CLI:
                    WIDTH = get_display_width()
                    print("\n" + fmt.draw_separator(WIDTH, fmt.BoxChars.D_HORIZONTAL))
                    print(fmt.style("  \u2713  Done! Happy trading!", 'good', bold=True))
                    print(fmt.draw_separator(WIDTH, fmt.BoxChars.D_HORIZONTAL) + "\n")
                else:
                    print("\n\u2713 Done! Happy trading!\n")
                break

        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Automation (cron / --auto / --mode / --ticker) runs exactly one cycle.
        # Interactive sessions fall through and re-display the mode menu.
        if not _interactive:
            break


if __name__ == "__main__":
    main()
