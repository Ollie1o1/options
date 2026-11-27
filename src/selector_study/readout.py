"""Readout for the selector-vs-liquidity paper study.

The rules are frozen in docs/PREREG_SELECTOR_LIQUIDITY_BASELINE_20261126.md and in the
study database's settings. Before the read date a real study prints counts only; outcome
returns stay hidden. A synthetic study database is never locked and is labelled as such.

Uncertainty: one observation is one ISO week of entry sessions (the equal-weight mean of
that week's paired differences), because candidates from the same week share the same
market path. The interval is a moving-block bootstrap over consecutive weeks.
"""
from __future__ import annotations

import datetime as dt
import math
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.selector_study.capture import RETRYABLE, STATUSES, _parse_ts, _to_et
from src.selector_study.db import is_synthetic, load_settings
from src.selector_study.marks import entry_marks
from src.selector_study.outcomes import CASES, resolve_all

INSUFFICIENT = "Insufficient data"


def iso_week(day: str) -> str:
    y, w, _ = dt.date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def block_bootstrap_ci(values: Sequence[float], *, draws: int, block: int, seed: int,
                       level: float = 0.95) -> Optional[tuple[float, float]]:
    """Percentile CI for the mean of an ordered series, by moving-block resampling."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 2:
        return None
    b = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(draws, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(draws, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo = float(np.quantile(means, (1 - level) / 2))
    hi = float(np.quantile(means, 1 - (1 - level) / 2))
    return lo, hi


def build(conn: sqlite3.Connection, *, today: Optional[str] = None) -> Dict[str, Any]:
    settings = load_settings(conn)
    rd = settings["readout"]
    today = today or dt.date.today().isoformat()
    synthetic = is_synthetic(conn)
    locked = (not synthetic) and today < rd["read_after"]

    obs = [dict(r) for r in conn.execute("SELECT * FROM observations")]
    runs = [dict(r) for r in conn.execute("SELECT run_id, session, status FROM capture_runs")]
    status_counts = Counter(o["status"] for o in obs)
    times = [_to_et(t) for t in (_parse_ts(o["observed_at"]) for o in obs if o["status"] == "captured") if t]
    sessions = sorted({o["session"] for o in obs})
    coverage = {
        "sessions": len(sessions),
        "first_session": sessions[0] if sessions else None,
        "last_session": sessions[-1] if sessions else None,
        "agreed_symbols": len(settings["symbols"]),
        "observations": len(obs),
        "status": {s: status_counts.get(s, 0) for s in STATUSES},
        "retryable_open": sum(status_counts.get(s, 0) for s in RETRYABLE),
        "sessions_with_gaps": sorted({o["session"] for o in obs if o["status"] != "captured"}),
        "captured_with_no_selection": sum(o["status"] == "captured" and not o["selector_candidates"] for o in obs),
        "runs": dict(Counter(r["status"] for r in runs)),
        "capture_time_et": ([min(times).strftime("%H:%M"), max(times).strftime("%H:%M")] if times else None),
        "after_close": sum(t.time() > dt.time(16, 0) for t in times),
    }

    pairs = [dict(r) for r in conn.execute("SELECT * FROM pairs ORDER BY session, pair_id")]
    entries = {r["entry_id"]: dict(r) for r in conn.execute("SELECT * FROM entries")}
    pstat = Counter(p["status"] for p in pairs)
    matching = {
        "selector_entries": len(pairs),
        "matched": pstat.get("matched", 0),
        "identical": pstat.get("identical", 0),
        "no_match": pstat.get("no_match", 0),
        "no_match_reasons": dict(Counter(p["reason"] for p in pairs if p["status"] == "no_match")),
        "comparisons_possible": pstat.get("matched", 0) + pstat.get("identical", 0),
    }

    outcomes = {eid: resolve_all(e, entry_marks(conn, eid), settings) for eid, e in entries.items()}
    by_arm: Dict[str, Dict[str, Dict[str, int]]] = {}
    for eid, e in entries.items():
        for case in CASES:
            slot = by_arm.setdefault(e["arm"], {}).setdefault(case, {})
            st = outcomes[eid][case]["status"]
            slot[st] = slot.get(st, 0) + 1

    paired: Dict[str, Any] = {}
    for case in CASES:
        excluded: Counter = Counter()
        diffs: List[Dict[str, Any]] = []
        for p in pairs:
            if p["status"] == "no_match" or not p["baseline_entry"]:
                continue
            a, b = outcomes[p["selector_entry"]][case], outcomes[p["baseline_entry"]][case]
            sts = {a["status"], b["status"]}
            if sts == {"closed"}:
                diffs.append({"session": p["session"], "week": iso_week(p["session"]),
                              "selector": a["ror"], "baseline": b["ror"], "diff": a["ror"] - b["ror"]})
            else:
                key = ("not_filled" if "not_filled" in sts else "missing" if "missing" in sts else "open")
                excluded[key] += 1
        weeks: Dict[str, List[float]] = defaultdict(list)
        for d in diffs:
            weeks[d["week"]].append(d["diff"])
        week_keys = sorted(weeks)
        week_means = [float(np.mean(weeks[w])) for w in week_keys]
        res: Dict[str, Any] = {
            "resolved_pairs": len(diffs),
            "independent_sessions": len({d["session"] for d in diffs}),
            "weeks": len(week_keys),
            "excluded": dict(excluded),
            "identical_resolved": sum(1 for d in diffs if abs(d["diff"]) < 1e-12),
        }
        if not locked and diffs:
            ci = block_bootstrap_ci(week_means, draws=int(rd["bootstrap_draws"]),
                                    block=int(rd["block_weeks"]), seed=int(rd["seed"]))
            res.update({
                "mean_diff_weekly": float(np.mean(week_means)),
                "ci95": list(ci) if ci else None,
                "mean_ror_selector": float(np.mean([d["selector"] for d in diffs])),
                "mean_ror_baseline": float(np.mean([d["baseline"] for d in diffs])),
                "week_means": dict(zip(week_keys, week_means)),
            })
        paired[case] = res

    primary = paired["cross"]
    enough = primary["weeks"] >= int(rd["min_weeks"]) and primary["resolved_pairs"] >= int(rd["min_pairs"])
    if locked:
        verdict = f"Locked until {rd['read_after']} (counts only). " + (
            "Evidence threshold met." if enough else INSUFFICIENT + ".")
    elif not enough or primary.get("ci95") is None:
        verdict = INSUFFICIENT
    else:
        lo, hi = primary["ci95"]
        verdict = ("Selector ahead of the liquidity baseline" if lo > 0 else
                   "Liquidity baseline ahead of the selector" if hi < 0 else
                   "No difference established")
    mid = paired["mid"]
    cost_reversal = (not locked and "mean_diff_weekly" in primary and "mean_diff_weekly" in mid
                     and np.sign(primary["mean_diff_weekly"]) != np.sign(mid["mean_diff_weekly"])
                     and abs(mid["mean_diff_weekly"]) > 1e-12)
    return {
        "study_id": settings["study_id"],
        "source": settings.get("source", "chain_archive"),
        "exit_mode": settings["exits"].get("mode", "book_rules"),
        "synthetic": synthetic,
        "today": today,
        "read_after": rd["read_after"],
        "locked": locked,
        "threshold": {"min_weeks": rd["min_weeks"], "min_pairs": rd["min_pairs"]},
        "coverage": coverage,
        "matching": matching,
        "outcomes_by_arm": by_arm,
        "paired": paired,
        "cost_reversal": bool(cost_reversal),
        "verdict": verdict,
        "scope": ("Paper research. Quote snapshots only: crossed bid/ask plus fees is the primary case; "
                  "midpoint is a sensitivity with an assumed, unverifiable fill. Not a trading signal."),
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def render(r: Dict[str, Any]) -> str:
    tag = "SYNTHETIC DATA · " if r["synthetic"] else ""
    c, m = r["coverage"], r["matching"]
    lines = [
        f"{tag}Selector vs liquidity-first baseline · bull put · {r['study_id']}",
        r["scope"],
        f"Source {r['source']} · exits {r['exit_mode']}",
        "",
        "Collection",
        f"  sessions {c['sessions']} ({c['first_session']} .. {c['last_session']}) · "
        f"{c['observations']} observations over {c['agreed_symbols']} agreed symbols",
        "  " + " · ".join(f"{k} {v}" for k, v in c["status"].items()),
        f"  runs {c['runs']} · retryable still open {c['retryable_open']} · "
        f"captured with no selector pick {c['captured_with_no_selection']}",
        (f"  capture time ET {c['capture_time_et'][0]}-{c['capture_time_et'][1]} · "
         f"{c['after_close']} of {c['status']['captured']} captured after 16:00 (post-close quotes)")
        if c["capture_time_et"] else "  capture time ET n/a",
        "",
        "Matching (same ticker, session, expiration; short |delta| and width within tolerance)",
        f"  selector entries {m['selector_entries']} · comparisons possible {m['comparisons_possible']} "
        f"(matched {m['matched']}, identical legs {m['identical']}) · no match {m['no_match']}",
    ]
    for reason, n in m["no_match_reasons"].items():
        lines.append(f"    no match: {reason} ({n})")
    lines += ["", "Outcomes by arm (closed / open / missing / not filled)"]
    for arm, cases in sorted(r["outcomes_by_arm"].items()):
        for case, st in cases.items():
            lines.append(f"  {arm:<9} {case:<5} {st.get('closed', 0)} / {st.get('open', 0)} / "
                         f"{st.get('missing', 0)} / {st.get('not_filled', 0)}")
    lines += ["", "Paired difference in net return on capital at risk (selector - baseline)"]
    for case, label in (("cross", "crossed bid/ask + fees (primary)"), ("mid", "midpoint (sensitivity)")):
        p = r["paired"][case]
        head = (f"  {label}: {p['resolved_pairs']} resolved pairs · {p['independent_sessions']} sessions · "
                f"{p['weeks']} weeks · excluded {p['excluded'] or {}}")
        lines.append(head)
        if r["locked"]:
            lines.append("    returns hidden until the read date")
        elif "mean_diff_weekly" in p:
            ci = p["ci95"]
            ci_s = f"[{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "[n/a: fewer than 2 weeks]"
            lines.append(f"    weekly mean {_pct(p['mean_diff_weekly'])} 95% CI {ci_s} · "
                         f"selector {_pct(p['mean_ror_selector'])} vs baseline {_pct(p['mean_ror_baseline'])}")
    if r["cost_reversal"]:
        lines.append("  Costs reverse the sign: the midpoint advantage does not survive crossing the spread.")
    th = r["threshold"]
    lines += ["", f"Evidence threshold: >= {th['min_weeks']} weeks and >= {th['min_pairs']} resolved pairs "
                  f"(crossed); read date {r['read_after']}.",
              f"Verdict: {r['verdict']}"]
    return "\n".join(lines)
