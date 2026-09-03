"""Which structure the book takes next.

THE PROBLEM WITH A NAME LIST. `auto_log.allowed_strategies` is self-sealing.
Under `["Bull Put"]` no other structure can enter the book, so no other
structure can accumulate evidence, so a rule that cited absence of evidence
guarantees that absence permanently. Bear Call, Short Put and Iron Condor last
entered on 2026-07-30 and 2026-07-31; Long Put on 2026-07-13. Under the list
as it stands, none of them ever will again, whatever the market does.

WHAT THE MEASUREMENT ACTUALLY SAYS. Bull Put is not an arbitrary pick.
Bootstrap posteriors on mean return on CAPITAL AT RISK, 2026-08-24:

    Bull Put    +16.80%  [ +6.85%, +26.95%]   P(best) 99.0%
    Short Put    +0.59%  [ -0.41%,  +2.36%]   P(best)  0.0%
    Long Call    -0.18%  [ -8.45%,  +8.31%]   P(best)  0.6%
    Bear Call    -4.73%  [-13.85%,  +4.36%]   P(best)  0.1%
    Long Put     -4.96%  [-16.37%,  +7.27%]   P(best)  0.4%
    Iron Condor  -5.33%  [ -9.60%,  -1.08%]   P(best)  0.0%

So Thompson sampling alone does NOT fix the problem: it allocates 99% to Bull
Put, which at two entries a day is one exploratory trade every seven weeks.
Time-decaying the evidence barely moves it (99.3% at a 45-day half-life),
because the entire book is four months old.

EXPLORATION IS THEREFORE A PURCHASE, NOT A FREE LUNCH. It buys information
about structures whose evidence is going stale, and it pays for that in
expected return. `information_cost` prices it in return on capital at risk per
entry so the rate is chosen against its cost rather than by taste. The weights
are `(1 - rate)` on the posterior and `rate` spread uniformly over everything
eligible.

WHAT THIS DOES NOT DO. It does not rank contracts — `expected_return` tried
that and its guard refused it (walk-forward slope 0.442, 95% CI [-0.891,
1.774]). It does not remove a safety rail: `eligible` still bounds what the
book may ever take, and every existing gate (friction, EV, earnings, sizing,
the DTE floor on long premium) still runs first and still refuses. It only
decides which structures are in play, and it never closes a door permanently.

The draw is deterministic in the candidate's key, seeded through blake2b
rather than the builtin `hash`, which Python randomises per process. A
decision that cannot be replayed cannot be audited.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_HALF_LIFE_DAYS = 45.0
DEFAULT_DRAWS = 4000
#: A structure with fewer closed trades than this has no posterior worth
#: sampling; it relies on the exploration share to earn one.
MIN_ROWS_FOR_POSTERIOR = 20


@dataclass
class Allocation:
    """Target share of entries per structure, and the evidence behind it."""
    weights: Dict[str, float]
    posterior: Dict[str, float] = field(default_factory=dict)
    n_eff: Dict[str, float] = field(default_factory=dict)
    explore_rate: float = 0.0
    as_of: str = ""

    def share(self, strategy: Optional[str]) -> float:
        return float(self.weights.get(str(strategy or ""), 0.0))


def _ages(df: pd.DataFrame, as_of: str) -> np.ndarray:
    when = pd.to_datetime(df["entry_date"], errors="coerce")
    days = (pd.Timestamp(as_of) - when).dt.days
    return days.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)


def _weights(df: pd.DataFrame, as_of: str, half_life_days: float) -> np.ndarray:
    """Exponential decay by age. Recent evidence counts for more because the
    market it describes is closer to the one being traded."""
    if half_life_days is None or half_life_days <= 0:
        return np.ones(len(df))
    return np.asarray(0.5 ** (_ages(df, as_of) / float(half_life_days)))


def effective_n(df: pd.DataFrame, as_of: str,
                half_life_days: float = DEFAULT_HALF_LIFE_DAYS
                ) -> Dict[str, float]:
    """Kish effective sample size per structure once evidence is decayed.

    Stale evidence must WIDEN a posterior, never narrow it. This is the
    quantity that makes that happen.
    """
    out: Dict[str, float] = {}
    if df is None or len(df) == 0 or "strategy" not in df:
        return out
    for name, g in df.groupby("strategy"):
        w = _weights(g, as_of, half_life_days)
        total = float(w.sum())
        out[str(name)] = (total * total / float((w * w).sum())
                          if total > 0 else 0.0)
    return out


def posteriors(df: pd.DataFrame, as_of: str,
               half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
               n_draws: int = DEFAULT_DRAWS,
               seed: int = 20260824) -> Dict[str, np.ndarray]:
    """Bootstrap draws of each structure's mean return on capital at risk.

    Return on capital at risk, not dollars: the 2026-08-20 position-sizing
    change makes dollar figures incomparable across the split, and this metric
    is scale-free so it reads across it cleanly.
    """
    out: Dict[str, np.ndarray] = {}
    if df is None or len(df) == 0 or "ret_on_risk" not in df:
        return out
    rng = np.random.default_rng(seed)
    for name, g in df.groupby("strategy"):
        r = pd.to_numeric(g["ret_on_risk"], errors="coerce")
        keep = r.notna()
        r = r[keep].to_numpy(dtype=float)
        if len(r) < MIN_ROWS_FOR_POSTERIOR:
            continue
        w = _weights(g[keep.to_numpy()], as_of, half_life_days)
        if w.sum() <= 0:
            continue
        p = w / w.sum()
        m = max(2, int(round(float(p.sum() ** 2 / (p * p).sum()))))
        idx = rng.choice(len(r), size=(n_draws, m), replace=True, p=p)
        out[str(name)] = r[idx].mean(axis=1)
    return out


def p_best(draws: Dict[str, np.ndarray]) -> Dict[str, float]:
    """P(this structure has the highest true mean return on risk)."""
    names = [k for k, v in draws.items() if len(v)]
    if not names:
        return {}
    n = min(len(draws[k]) for k in names)
    matrix = np.column_stack([draws[k][:n] for k in names])
    winner = matrix.argmax(axis=1)
    return {k: float((winner == i).mean()) for i, k in enumerate(names)}


def allocate(df: pd.DataFrame, eligible: Sequence[str],
             explore_rate: float = 0.25, as_of: str = "",
             half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
             seed: int = 20260824,
             n_draws: int = DEFAULT_DRAWS) -> Allocation:
    """Target share of entries per eligible structure.

    `(1 - explore_rate)` follows the posterior; `explore_rate` is spread
    uniformly so that nothing is ever locked out. A structure with no history
    at all draws only the exploration share — which is the whole point, since
    that is the only way it can ever earn a posterior.
    """
    eligible = [str(s) for s in (eligible or [])]
    if not eligible:
        return Allocation({}, {}, {}, float(explore_rate), str(as_of))

    as_of = str(as_of or pd.Timestamp.today().strftime("%Y-%m-%d"))
    rate = float(min(max(explore_rate, 0.0), 1.0))

    draws = posteriors(df, as_of, half_life_days, n_draws=n_draws, seed=seed)
    draws = {k: v for k, v in draws.items() if k in eligible}
    post = p_best(draws)

    if not post:
        # NO EVIDENCE MEANS NO ALLOCATION, never uniform. Uniform is the most
        # dangerous default available here: maximum exposure justified by zero
        # information, spread equally over structures that include measured
        # losers. Caught by CI on PR #63, which has no `paper_trades.db` — the
        # weights fell through to 1/4 each and Long Call was admitted at 25%.
        # Returning nothing lets the caller fall back to the allowlist, which
        # can only narrow what the book takes.
        log.warning("strategy allocation: no structure has %d closed trades "
                    "yet — no allocation, the allowlist stands",
                    MIN_ROWS_FOR_POSTERIOR)
        return Allocation({}, {}, {}, rate, as_of)

    uniform = 1.0 / len(eligible)
    weights: Dict[str, float] = {}
    for s in eligible:
        weights[s] = (1.0 - rate) * float(post.get(s, 0.0)) + rate * uniform

    total = sum(weights.values())
    if total <= 0:
        return Allocation({}, post, {}, rate, as_of)
    weights = {s: v / total for s, v in weights.items()}

    n_eff = {k: v for k, v in effective_n(df, as_of, half_life_days).items()
             if k in eligible}
    return Allocation(weights, post, n_eff, rate, as_of)


def admits(alloc: Allocation, strategy: Optional[str], key: str) -> bool:
    """Is this candidate the one that fills its structure's share?

    Deterministic in `key`, via blake2b rather than the builtin `hash`, which
    Python randomises per process — the same defect that once made every PoP
    and every quality_score differ between interpreters on identical input.
    """
    share = alloc.share(strategy)
    if share <= 0.0:
        return False
    if share >= 1.0:
        return True
    digest = hashlib.blake2b(
        f"{alloc.as_of}|{strategy}|{key}".encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / float(1 << 64)
    return draw < share


def information_cost(alloc: Allocation, df: pd.DataFrame) -> float:
    """Expected return on risk given up per entry, versus pure exploitation.

    The price of the exploration budget, in the same units the book is
    measured in. A rate chosen without this number is a rate chosen by taste.
    """
    if not alloc.weights or df is None or len(df) == 0:
        return 0.0
    means = df.groupby("strategy")["ret_on_risk"].mean()
    known = {s: float(means[s]) for s in alloc.weights if s in means.index}
    if not known:
        return 0.0

    best = max(known.values())
    # Structures with no history are priced at the worst known mean rather
    # than at zero: an unmeasured structure is not a free one.
    floor = min(known.values())
    spent = sum(w * (best - known.get(s, floor))
                for s, w in alloc.weights.items())
    return float(max(0.0, spent))


@dataclass
class Replay:
    """What the policy would have taken, and what it would have earned."""
    taken: pd.DataFrame
    decisions: pd.DataFrame
    mean_return: float = 0.0
    explore_rate: float = 0.0

    @property
    def mix(self) -> Dict[str, int]:
        if len(self.taken) == 0:
            return {}
        return {str(k): int(v) for k, v
                in self.taken["strategy"].value_counts().items()}


def entries_per_day(df: pd.DataFrame) -> float:
    """The book's actual entry cadence, for sizing a replay's slots."""
    if df is None or len(df) == 0:
        return 1.0
    per = df.groupby("entry_date").size()
    return float(max(1.0, round(per.mean())))


def replay(df: pd.DataFrame, eligible: Sequence[str],
           explore_rate: float = 0.25, warmup: int = 200,
           half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
           seed: int = 20260824,
           per_day: Optional[float] = None) -> Replay:
    """Walk the policy forward over trades that actually happened.

    One pass per DATE, taking at most `per_day` entries — the book's real
    cadence. Sizing the replay by the number of historical trades instead
    would have it take fifteen entries a day against a book that takes two:
    it exhausts the day's supply of its first choice and then spends the
    remaining slots on whatever is left, so AVAILABILITY rather than the
    policy decides the realised mix. Measured at 70.6% and then 30.1% Bull
    Put under pure exploitation, against a target weight of 99.5%.

    At each slot the allocation is rebuilt from trades entered STRICTLY
    BEFORE that date, restricted to the structures the board actually offered
    that day, and one unused real trade of the drawn structure is taken.
    Every return recorded really occurred; the only thing simulated is which
    of the available trades the policy would have chosen.

    The honest limit: it can only take what was on the board. After
    2026-07-31 the book produced nothing but Bull Put, so no policy can show
    anything else there — which is the self-sealing problem, visible.
    """
    cols = ["entry_date", "strategy", "ret_on_risk"]
    dcols = ["entry_date", "wanted", "got", "evidence_through"]
    empty = Replay(pd.DataFrame(columns=cols), pd.DataFrame(columns=dcols),
                   0.0, float(explore_rate))
    if df is None or len(df) <= warmup:
        return empty

    book = (df.dropna(subset=["ret_on_risk"])
              .sort_values("entry_date", kind="mergesort")
              .reset_index(drop=True))
    if len(book) <= warmup:
        return empty

    slots = int(per_day if per_day else entries_per_day(book))
    rng = np.random.default_rng(seed)
    names = [str(s) for s in eligible]
    dates = book["entry_date"].astype(str)
    start_date = str(dates.iloc[warmup])

    used: set = set()
    taken: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    for slot_date in sorted(d for d in dates.unique() if d >= start_date):
        past = book[dates < slot_date]
        if len(past) == 0:
            continue
        evidence_through = str(past["entry_date"].max())
        alloc = allocate(past, names, explore_rate, as_of=slot_date,
                         half_life_days=half_life_days, seed=seed,
                         n_draws=1000)
        if not alloc.weights:
            continue

        for _ in range(slots):
            today = book[(dates == slot_date) & (~book.index.isin(used))]
            offered = [k for k in alloc.weights
                       if (today["strategy"] == k).any()]
            if not offered:
                decisions.append({"entry_date": slot_date, "wanted": None,
                                  "got": None,
                                  "evidence_through": evidence_through})
                break
            probs = np.array([alloc.weights[k] for k in offered], dtype=float)
            if probs.sum() <= 0:
                break
            wanted = str(rng.choice(offered, p=probs / probs.sum()))
            pool = today[today["strategy"] == wanted]
            pick = pool.index[0]
            used.add(pick)
            row = book.loc[pick]
            taken.append({"entry_date": str(row["entry_date"]),
                          "strategy": str(row["strategy"]),
                          "ret_on_risk": float(row["ret_on_risk"])})
            decisions.append({"entry_date": slot_date, "wanted": wanted,
                              "got": wanted,
                              "evidence_through": evidence_through})

    taken_df = pd.DataFrame(taken, columns=cols)
    mean = float(taken_df["ret_on_risk"].mean()) if len(taken_df) else 0.0
    return Replay(taken_df, pd.DataFrame(decisions, columns=dcols), mean,
                  float(explore_rate))


def describe(alloc: Allocation, df: pd.DataFrame) -> List[str]:
    """Human-readable allocation table, for the report and the scan header."""
    if not alloc.weights:
        return ["no eligible structures — nothing can be logged"]
    means = (df.groupby("strategy")["ret_on_risk"].mean()
             if df is not None and len(df) else pd.Series(dtype=float))
    lines = [f"Allocation as of {alloc.as_of} — explore rate "
             f"{alloc.explore_rate:.0%}, information cost "
             f"{information_cost(alloc, df):+.2%} of return on risk per entry",
             f"  {'structure':<14}{'share':>8}{'P(best)':>9}"
             f"{'n_eff':>8}{'mean ret':>10}"]
    for s, w in sorted(alloc.weights.items(), key=lambda kv: -kv[1]):
        mean = f"{means[s]:+.2%}" if s in getattr(means, "index", []) else "  no data"
        lines.append(f"  {s:<14}{w:>8.1%}{alloc.posterior.get(s, 0.0):>9.1%}"
                     f"{alloc.n_eff.get(s, 0.0):>8.0f}{mean:>10}")
    return lines


#: Below this many eligible entries in the window, a Clopper-Pearson interval
#: is so wide it cannot distinguish drift from ordinary sampling noise — the
#: check would either always pass or flap on every recomputation. Silence
#: instead of a manufactured verdict.
MIN_WINDOW_FOR_DRIFT = 10


def _entered_mix(db_path: str, eligible: Sequence[str],
                 window: int) -> Dict[str, int]:
    """Strategy counts over the most recent `window` allocation-eligible
    entries, open or closed — a still-open position was still a pick, and
    excluding it would make the check blind to the days closest to now."""
    eligible = [str(s) for s in eligible]
    if not eligible:
        return {}
    placeholders = ",".join("?" * len(eligible))
    sql = (f"SELECT strategy_name FROM trades "
          f"WHERE strategy_name IN ({placeholders}) "
          f"ORDER BY entry_id DESC LIMIT ?")
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(sql, (*eligible, window)).fetchall()
    except Exception:
        log.warning("strategy allocation drift: ledger unreadable at %s",
                   db_path, exc_info=True)
        return {}
    finally:
        if conn is not None:
            conn.close()
    counts: Dict[str, int] = {}
    for (name,) in rows:
        counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def drift_severity(weights: Dict[str, float], counts: Dict[str, int],
                   confidence: float = 0.95) -> Tuple[str, List[str]]:
    """Has the entered mix actually tracked the allocation's target weights?

    `allocate` sets a SHARE per eligible structure and `admits` gates each
    candidate independently against it — a per-candidate mechanism, with
    nothing that enforces the target over any particular stretch of real
    entries. A downstream defect (the per-symbol dedup picking a structure by
    raw ROW COUNT before the weight was ever checked — many strikes on a few
    tickers beat few strikes on many, regardless of weight) went unnoticed
    for two weeks because nothing compared the entered mix to what it was
    supposed to track. Fixed 2026-09-03; this is the comparison that should
    make the next such drift show up in days, not a manual audit.

    A 95% Clopper-Pearson interval on each structure's OBSERVED share, not a
    point comparison, so a small window does not flare on ordinary sampling
    noise. The severity is deliberately asymmetric, because the two
    directions are not equally bad:

    - CRITICAL only when the SINGLE highest-weighted structure's interval
      sits entirely below its target — the book is under-taking the one
      thing with a measured edge, which is the failure that motivated this.
    - WARN when any OTHER (lower-weighted) structure's interval sits
      entirely above its target — a structure being explored, not exploited,
      is taking more of the book than its own evidence earned it.

    Being UNDER its target on a low-weight structure, or OVER on the
    dominant one, is not flagged: the exploration budget existing to buy
    information means low-weight structures are EXPECTED to come up short
    most of the time, and taking more than intended of the structure with
    the actual edge is not the failure this check exists to catch.
    """
    n = sum(counts.values())
    if not weights or n == 0:
        return "OK", []
    from scipy.stats import binomtest

    dominant = max(weights, key=weights.get)
    sev = "OK"
    lines: List[str] = []
    for strat, target in sorted(weights.items(), key=lambda kv: -kv[1]):
        k = counts.get(strat, 0)
        lo, hi = binomtest(k, n, target).proportion_ci(
            confidence_level=confidence)
        share = k / n
        pct = f"{int(round(confidence * 100))}%"
        if strat == dominant and hi < target:
            sev = "CRITICAL"
            lines.append(
                f"     {strat}: {k}/{n} = {share:.0%} entered vs "
                f"{target:.0%} target, {pct} CI [{lo:.0%}, {hi:.0%}] sits "
                "BELOW target — the one measured-edge structure is "
                "under-represented")
        elif strat != dominant and lo > target:
            if sev == "OK":
                sev = "WARN"
            lines.append(
                f"     {strat}: {k}/{n} = {share:.0%} entered vs "
                f"{target:.0%} target, {pct} CI [{lo:.0%}, {hi:.0%}] sits "
                "ABOVE target — over-represented against its own evidence")
    return sev, lines


def drift_health_lines(db_path: str = "paper_trades.db",
                       cfg_path: str = "config.json",
                       window: int = 30) -> List[str]:
    """`--health` line: has the entered mix tracked the allocation's target?

    Same pattern as `candidate_marks.health_lines` — one summary line plus
    detail lines on drift. Failure-safe throughout: a missing config, a
    missing ledger, or a disabled allocation all report OK rather than
    raising, because this is a smoke alarm running inside a health check,
    not a gate that should ever stop a scan.
    """
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    alloc_cfg = (cfg.get("auto_log") or {}).get("allocation") or {}
    label = "alloc drift"
    if not alloc_cfg.get("enabled"):
        return [f"  {label:<14} allocation disabled — allowlist rules"
               f"{'':<10}[OK]"]

    eligible = [str(s) for s in (alloc_cfg.get("eligible_strategies") or [])]
    if not eligible:
        return [f"  {label:<14} no eligible structures configured"
               f"{'':<10}[OK]"]

    from . import pop_calibration as _pc
    book = _pc.load_training_set(db_path)
    book = book.dropna(subset=["ret_on_risk"]) if len(book) else book
    alloc = allocate(book, eligible,
                     explore_rate=float(alloc_cfg.get("explore_rate", 0.25)),
                     half_life_days=float(alloc_cfg.get("half_life_days", 45.0)))
    if not alloc.weights:
        return [f"  {label:<14} no posterior yet — allowlist stands"
               f"{'':<10}[OK]"]

    counts = _entered_mix(db_path, eligible, window)
    n = sum(counts.values())
    if n < MIN_WINDOW_FOR_DRIFT:
        return [f"  {label:<14} only {n} of the last {window} entries are "
               f"allocation-eligible — too few to read{'':<3}[OK]"]

    sev, detail = drift_severity(alloc.weights, counts)
    header = (f"  {label:<14} {n} of last {window} eligible entries vs "
             f"target weights{'':<5}[{sev}]")
    return [header] + detail
