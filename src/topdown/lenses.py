"""Discovery lenses — "why look here?" — each labelled with what tested it.

A lens is a transparent question with an evidence label, never a hidden score:

  PROVEN      passed a preregistered out-of-sample test (configs/playbooks.json VALIDATED)
  PROMISING   passed part of one; read its caveat (LEAD)
  TESTING     registered test, readout not yet due — a watchlist until then
  WATCHLIST   descriptive overlay with no test yet (CONTEXT playbooks)
  CONTEXT     plain description; nothing in this repo tested it as a signal
  AVOID       a measured negative, or event risk you should not hold through

`NULL_LEDGER` lists ideas that WERE tested here and failed, so they can't be
re-presented as discovery lenses by accident.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path
from src.radar.schema import RadarSnapshot
from src.radar.scanners import get_sector_etf
from .sectors import SectorBoard, _is_fund

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

LABEL_OF_TIER = {
    "VALIDATED": "PROVEN", "LEAD": "PROMISING", "CONTEXT": "WATCHLIST",
    "RESEARCH": "TESTING", "CLOSED": "RETIRED",
}


@dataclass(frozen=True)
class Lens:
    id: str
    title: str
    question: str
    label: str          # PROVEN | PROMISING | TESTING | WATCHLIST | CONTEXT | AVOID
    evidence: str       # one sourced sentence: what was measured, and the caveat
    source: str         # doc / file that backs `evidence`
    sort_note: str      # how rows are ordered, in words


@dataclass
class LensRow:
    ticker: str
    sector_etf: Optional[str]
    why: str
    metric_label: str = ""
    metric: Optional[float] = None
    flags: Tuple[str, ...] = ()
    as_of: Optional[str] = None
    playbook_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["flags"] = list(self.flags)
        return d


@dataclass
class LensResult:
    lens: Lens
    rows: List[LensRow] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"lens": asdict(self.lens), "rows": [r.to_dict() for r in self.rows], "note": self.note}


@dataclass
class LensContext:
    ohlcv: OhlcvMap
    board: SectorBoard
    snapshot: Optional[RadarSnapshot] = None
    earnings_days: Dict[str, int] = field(default_factory=dict)
    crash_flags: Dict[str, str] = field(default_factory=dict)
    rev_breadth: Dict[str, float] = field(default_factory=dict)
    rev_as_of: Optional[str] = None

    @property
    def spy_above_200d(self) -> Optional[bool]:
        if "SPY" not in self.ohlcv or len(self.ohlcv["SPY"][1]) < 200:
            return None
        c = np.asarray(self.ohlcv["SPY"][1], dtype=float)
        return bool(c[-1] > float(np.mean(c[-200:])))


LENSES: Tuple[Lens, ...] = (
    Lens("core_rotation", "Core ETF rotation (TFSA sleeve)",
         "Which sector/asset ETFs does the validated rotation hold this month?", "PROVEN",
         "Top-3 Outlook ETFs, vol-scaled: Sharpe 0.82 vs SPY 0.69 and MaxDD -17% vs -55% (2006-26). "
         "The drawdown cut is robust; the Sharpe edge's CI still spans zero.",
         "docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md", "Outlook rank"),
    Lens("pullbacks", "Uptrend pullbacks (shares)",
         "Which stocks in long-term uptrends just had a sharp short-term dip?", "PROMISING",
         "Holdout PF 1.23-1.43 at 10 bps, but no excess over SPY and it LOST in 2018 and 2022. "
         "Shares only; fails at 20 bps.",
         "docs/RESULT_EQUITY_PULLBACKS_20261124.md", "Radar order (tier, confluence, friction)"),
    Lens("etf_spreads", "ETF option spreads (limit orders only)",
         "Which 60-120 DTE ETF put spreads and ETF bear calls are live?", "PROMISING",
         "Wide SPY Bull Put passed only at k<=0.35 limit fills (PF 2.41); never at market. Bear Calls' "
         "profit was a mid-exit figure (crossed PF 1.19) — forward readout >= 2027-01-20.",
         "docs/RESULT_ODX_WIDE_60_120_20261124.md", "Radar order"),
    Lens("leaders", "Sector leaders (12-1 momentum)",
         "Inside the top-ranked sectors, which stocks have the strongest 12-1 month trend?", "CONTEXT",
         "Untested here for share returns. 12-1 momentum is a documented anomaly (Jegadeesh-Titman 1993) "
         "that crashes after bear markets (Daniel-Moskowitz 2016); 1-month strength tends to reverse, "
         "so this never sorts on 21-day returns.",
         "src/outlook/factors.py (factor definitions)", "12-1 month return, high to low"),
    Lens("revisions", "Analyst estimate revisions",
         "Where are analysts raising estimates across firms (30-day breadth)?", "TESTING",
         "Registered test (Arm B) reads out >= 2027-01-15; until then this is a watchlist, not a signal. "
         "Upgrade/downgrade actions alone tested NULL (+0.02% [-0.16, +0.17]).",
         "docs/PREREG_PEAD_AND_REVISIONS_20261124.md", "30d revision breadth, high to low"),
    Lens("insiders", "Insider cluster buys",
         "Where are several insiders buying on the open market?", "WATCHLIST",
         "Descriptive SEC Form 4 overlay; the backtest corpus is still pending.",
         "configs/playbooks.json (insider_cluster)", "Radar order"),
    Lens("holdings", "Your holdings: buy zones & covered calls",
         "Which of your holdings are near your buy zone, or can write a covered call?", "WATCHLIST",
         "Deterministic proximity to the zones in longterm_plan.json; not a forecast.",
         "longterm_plan.json", "Radar order"),
    Lens("landmines", "Landmines — avoid or size down",
         "Which names carry a measured negative or an imminent event?", "AVOID",
         "Post-earnings breakdowns (-5% on 2x volume) lagged SPY by -1.16% [-2.19, -0.19] over 20-40d in both "
         "halves. Earnings within 7 days is gap risk, not a forecast.",
         "docs/RESULT_PEAD_EQUITY_20261124.md", "Crash flags first, then days to earnings"),
)

LENS_BY_ID: Dict[str, Lens] = {lz.id: lz for lz in LENSES}

# Short tag shown on a ticker that a lens lists (coloured by the lens's label).
LENS_TAG: Dict[str, str] = {
    "core_rotation": "CORE", "pullbacks": "PULL", "etf_spreads": "SPRD", "leaders": "LEAD",
    "revisions": "REV", "insiders": "INS", "holdings": "HOLD", "landmines": "MINE",
}
LENS_TAG_MEANING: Dict[str, str] = {
    "CORE": "core rotation", "PULL": "uptrend pullback", "SPRD": "ETF option spread", "LEAD": "sector leader",
    "REV": "estimate revisions", "INS": "insider cluster", "HOLD": "your holding", "MINE": "landmine",
}

# Tested here and failed. (idea, result, date, source)
NULL_LEDGER: Tuple[Tuple[str, str, str, str], ...] = (
    ("Analyst upgrades/downgrades -> next 2 sessions", "NULL: +0.02% [-0.16, +0.17], n=6,366", "2026-10-14",
     "docs/RESEARCH_FINDINGS.md"),
    ("News event types -> direction", "NULL: +0.15% [-0.41, +0.67] at 2 sessions", "2026-10-14",
     "docs/RESEARCH_FINDINGS.md"),
    ("Earnings-surprise drift", "Real (+0.14%) but ~5x below the options cost hurdle", "2026-10-14",
     "docs/RESEARCH_FINDINGS.md"),
    ("Post-earnings drift (PEAD), 20-40d", "+0.04% excess vs SPY, n=614", "2026-11-24",
     "docs/RESULT_PEAD_EQUITY_20261124.md"),
    ("Non-earnings volume breakouts", "Failed Bonferroni at 5/10/20d on both universes", "2026-11-24",
     "docs/RESULT_NON_EARNINGS_BREAKOUT_20261124.md"),
    ("Pre-earnings run-up (T-10 to T-1)", "-0.01% excess vs SPY (shares)", "2026-11-24",
     "docs/RESULT_PRE_EARNINGS_RUNUP_20261124.md"),
    ("Funded-biotech pre-catalyst run-up", "-0.32% excess vs XBI", "2026-11-24",
     "docs/RESULT_BIOTECH_RUNUP_20261124.md"),
    ("Squeeze momentum (path-managed)", "All arms' CIs span zero", "2026-11-24",
     "docs/RESULT_SQUEEZE_PATH_EQUITY_20261124.md"),
    ("Peer moves -> next-week direction", "NOT_ESTABLISHED", "2026-11-24", "docs/discovery/research_20261124.json"),
    ("VIX -> next-week market direction", "NOT_ESTABLISHED", "2026-11-24", "docs/discovery/research_20261124.json"),
    ("Outlook composite on single names", "NULL: residualized holdout t=0.45", "2026-09-05",
     "docs/OUTLOOK_FEATURE_RESULT_20260905.md"),
    ("Buying options for a 1-3 day swing", "Loses ~8% of premium in 2 sessions, mostly the spread", "2026-10-13",
     "docs/RESEARCH_FINDINGS.md"),
)


# ── helpers ──────────────────────────────────────────────────────────────────


def latest_estimates_snapshot(db_path: str = "data/estimates_archive.db") -> Optional[str]:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return None
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(snap_date) FROM estimate_snapshots WHERE dataset = 'eps_revisions'"
            ).fetchone()
        finally:
            con.close()
        return str(row[0])[:10] if row and row[0] else None
    except sqlite3.Error:
        return None


def _opp_rows(ctx: LensContext, playbooks: Sequence[str]) -> List[LensRow]:
    snap = ctx.snapshot
    if snap is None:
        return []
    out: List[LensRow] = []
    for o in snap.opportunities:
        if o.playbook_id not in playbooks:
            continue
        out.append(LensRow(
            ticker=o.ticker, sector_etf=get_sector_etf(o.ticker),
            why=o.setup_summary, metric_label="", flags=tuple(o.risk_flags),
            as_of=(o.as_of or snap.generated_at)[:10], playbook_id=o.playbook_id,
        ))
    return out


def _posture(ctx: LensContext, sec: Optional[str]) -> Optional[str]:
    if not sec:
        return None
    row = ctx.board.row(sec)
    return row.posture if row else None


# ── lens runners ─────────────────────────────────────────────────────────────


def _core_rotation(ctx: LensContext) -> LensResult:
    res = LensResult(LENS_BY_ID["core_rotation"])
    for r in ctx.board.rows:
        if r.weight > 0:
            res.rows.append(LensRow(
                ticker=r.etf, sector_etf=r.etf,
                why=f"Outlook #{r.rank} · weight {r.weight*100:.0f}% · {r.drivers or 'composite ' + str(r.score)}",
                metric_label="weight %", metric=round(r.weight * 100.0, 1), as_of=ctx.board.as_of,
                playbook_id="vol_managed_outlook",
            ))
    if ctx.board.sleeve_cash is not None:
        res.note = (f"Cash/T-bills {ctx.board.sleeve_cash*100:.0f}% · vol scale {ctx.board.sleeve_scale:.2f} "
                    f"(basket RV20 {(ctx.board.basket_rv20 or 0)*100:.1f}%). Rebalance monthly.")
    return res


def _pullbacks(ctx: LensContext) -> LensResult:
    res = LensResult(LENS_BY_ID["pullbacks"])
    for row in _opp_rows(ctx, ("equity_pullback",)):
        post = _posture(ctx, row.sector_etf)
        flags = list(row.flags)
        if post == "FAVOR":
            row.why = f"[sector {row.sector_etf} held by sleeve] " + row.why
        elif post == "AVOID":
            flags.append(f"AVOID_SEC:{row.sector_etf}")
        row.flags = tuple(dict.fromkeys(flags))
        res.rows.append(row)
    if ctx.spy_above_200d is False:
        res.note = "SPY is below its 200-day average: this playbook lost money in the 2018 and 2022 bear years."
    if ctx.snapshot is None:
        res.note = "No Radar snapshot cached — press Refresh."
    return res


def _etf_spreads(ctx: LensContext) -> LensResult:
    res = LensResult(LENS_BY_ID["etf_spreads"])
    res.rows = _opp_rows(ctx, ("etf_wide_spread", "etf_bear_call"))
    if not res.rows:
        res.note = "None cached. Live quotes: [o] in the TUI or `opt radar --live-options` (market hours)."
    return res


def _leaders(ctx: LensContext, *, top_sectors: int = 5, limit: int = 25, extended_pct: float = 15.0) -> LensResult:
    res = LensResult(LENS_BY_ID["leaders"])
    from .sectors import sector_members

    ranked = [r for r in ctx.board.rows if r.rank is not None and r.etf in _sector_etfs()]
    chosen = [r.etf for r in sorted(ranked, key=lambda r: r.rank or 99)[:top_sectors]]
    seen: Dict[str, LensRow] = {}
    for sec in chosen:
        sec_row = ctx.board.row(sec)
        sec_rank = sec_row.rank if sec_row is not None else "?"
        for sym in sector_members(sec, ctx.ohlcv):
            if sym in seen:
                continue
            ds, cs, _ = ctx.ohlcv[sym]
            c = np.asarray(cs, dtype=float)
            if len(c) < 253:
                continue
            m12 = float(c[-22] / c[-253] - 1.0)
            s200 = float(np.mean(c[-200:]))
            s50 = float(np.mean(c[-50:]))
            if m12 <= 0 or c[-1] <= s200:
                continue
            ext = (c[-1] / s50 - 1.0) * 100.0
            flags: List[str] = []
            if ext > extended_pct:
                flags.append(f"EXTENDED:+{ext:.0f}%>50d")
            if sym in ctx.crash_flags:
                flags.append(ctx.crash_flags[sym])
            if ctx.earnings_days.get(sym, 99) <= 7:
                flags.append(f"EARN:{ctx.earnings_days[sym]}d")
            seen[sym] = LensRow(
                ticker=sym, sector_etf=sec,
                why=f"12-1m {m12*100:+.0f}% · {((c[-1]/s200)-1)*100:+.0f}% vs 200d · in #{sec_rank} {sec}",
                metric_label="12-1m %", metric=round(m12 * 100.0, 1), flags=tuple(flags), as_of=str(ds[-1]),
            )
    res.rows = sorted(seen.values(), key=lambda r: -(r.metric or 0.0))[:limit]
    res.note = f"Universe: archived names in the top-{top_sectors} Outlook sectors ({', '.join(chosen)}), above their 200d, 12-1m > 0."
    if ctx.spy_above_200d is False:
        res.note += " SPY is below its 200d — momentum has historically crashed in rebounds after bear markets."
    return res


def _sector_etfs() -> frozenset:
    from .sectors import _SECTOR_ETFS

    return _SECTOR_ETFS


def _revisions(ctx: LensContext, *, threshold: float = 0.35, limit: int = 25) -> LensResult:
    res = LensResult(LENS_BY_ID["revisions"])
    for sym, b in sorted(ctx.rev_breadth.items(), key=lambda kv: -kv[1]):
        if b < threshold or _is_fund(sym):
            continue
        res.rows.append(LensRow(
            ticker=sym, sector_etf=get_sector_etf(sym),
            why=f"30d estimate-revision breadth {b:+.2f} (share of up-minus-down revisions)",
            metric_label="breadth", metric=round(b, 3), as_of=ctx.rev_as_of,
        ))
        if len(res.rows) >= limit:
            break
    if not ctx.rev_breadth:
        res.note = "No revision breadth available from data/estimates_archive.db."
    return res


def _insiders(ctx: LensContext) -> LensResult:
    res = LensResult(LENS_BY_ID["insiders"])
    res.rows = _opp_rows(ctx, ("insider_cluster",))
    return res


def _holdings(ctx: LensContext) -> LensResult:
    res = LensResult(LENS_BY_ID["holdings"])
    res.rows = _opp_rows(ctx, ("holdings_buy_zone", "covered_call_holdings"))
    if not res.rows:
        res.note = "Nothing near a buy zone (zones live in longterm_plan.json)."
    return res


def _landmines(ctx: LensContext, *, earn_days: int = 7) -> LensResult:
    res = LensResult(LENS_BY_ID["landmines"])
    rows: List[LensRow] = []
    for sym, flag in ctx.crash_flags.items():
        rows.append(LensRow(ticker=sym, sector_etf=get_sector_etf(sym),
                            why=f"{flag}: post-earnings breakdown (measured -1.16% vs SPY after)",
                            metric_label="kind", metric=0.0, flags=(flag,)))
    for sym, d in ctx.earnings_days.items():
        if d <= earn_days and sym in ctx.ohlcv and sym not in ctx.crash_flags:
            rows.append(LensRow(ticker=sym, sector_etf=get_sector_etf(sym),
                                why=f"Earnings in {d} day(s): gap risk — don't hold a swing or short premium through it",
                                metric_label="days", metric=float(d), flags=(f"EARN:{d}d",)))
    res.rows = sorted(rows, key=lambda r: (r.metric_label != "kind", r.metric or 0.0, r.ticker))
    return res


_RUNNERS: Dict[str, Callable[[LensContext], LensResult]] = {
    "core_rotation": _core_rotation, "pullbacks": _pullbacks, "etf_spreads": _etf_spreads,
    "leaders": _leaders, "revisions": _revisions, "insiders": _insiders,
    "holdings": _holdings, "landmines": _landmines,
}


def build_context(ohlcv: OhlcvMap, board: SectorBoard, snapshot: Optional[RadarSnapshot] = None,
                  *, estimates_db: str = "data/estimates_archive.db") -> LensContext:
    """Collect the shared inputs once (earnings, crash flags, revision breadth). Failure-isolated."""
    from src.radar.scanners import (
        _load_latest_rev_breadth,
        _load_recent_post_earnings_crashes,
        _load_upcoming_earnings_map,
    )

    ctx = LensContext(ohlcv=ohlcv, board=board, snapshot=snapshot)
    try:
        ctx.earnings_days = {k: int(v) for k, v in (_load_upcoming_earnings_map() or {}).items()}
    except Exception:
        pass
    try:
        ctx.crash_flags = dict(_load_recent_post_earnings_crashes(ohlcv) or {})
    except Exception:
        pass
    try:
        ctx.rev_breadth = {k: float(v) for k, v in (_load_latest_rev_breadth(db_path=estimates_db) or {}).items()}
        ctx.rev_as_of = latest_estimates_snapshot(estimates_db)
    except Exception:
        pass
    return ctx


def run_lens(lens_id: str, ctx: LensContext) -> LensResult:
    fn = _RUNNERS.get(lens_id)
    if fn is None:
        raise KeyError(f"unknown lens {lens_id!r}; choose from {', '.join(_RUNNERS)}")
    try:
        return fn(ctx)
    except Exception as exc:  # one broken input must not blank the whole hub
        return LensResult(LENS_BY_ID[lens_id], note=f"unavailable: {str(exc)[:80]}")


def run_lenses(ctx: LensContext, ids: Optional[Sequence[str]] = None) -> List[LensResult]:
    return [run_lens(i, ctx) for i in (ids or [lz.id for lz in LENSES])]


def lens_hits(ticker: str, results: Sequence[LensResult]) -> List[Tuple[str, str]]:
    """[(lens title, label)] for every lens that lists `ticker`."""
    sym = ticker.upper()
    return [(r.lens.title, r.lens.label) for r in results if any(x.ticker == sym for x in r.rows)]


def tags_by_ticker(results: Sequence[LensResult]) -> Dict[str, List[Tuple[str, str]]]:
    """{ticker: [(short tag, label), ...]} across all lens results, in lens order."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for r in results:
        tag = LENS_TAG.get(r.lens.id, r.lens.id.upper())
        for x in r.rows:
            lst = out.setdefault(x.ticker, [])
            if (tag, r.lens.label) not in lst:
                lst.append((tag, r.lens.label))
    return out


__all__ = [
    "LABEL_OF_TIER", "LENS_TAG", "LENS_TAG_MEANING", "tags_by_ticker", "Lens", "LensRow", "LensResult", "LensContext", "LENSES", "LENS_BY_ID", "NULL_LEDGER",
    "build_context", "run_lens", "run_lenses", "lens_hits", "latest_estimates_snapshot",
]
