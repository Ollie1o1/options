"""Portfolio fit: does this ticker add a new risk, or more of one you already carry?

Descriptive, never a gate. Holdings come only from real-money sources, each labelled:

  own picks       data/own_picks.db, status OPEN
  long-term plan  data/longterm.db fills (tranches bought)
  core sleeve     the PROVEN rotation's target ETFs this month (held if you follow it)

The paper book is deliberately excluded — it is research, not exposure.

Checks (thresholds are rules of thumb, not tested edges — the card says so):
  same sector     >= 2 holdings in the candidate's sector ETF
  co-movement     126-session log-return correlation >= 0.70 with a holding
  sleeve overlap  the candidate's sector ETF is a core-sleeve holding this month
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.paths import repo_path

from .peers import CORR_WINDOW, _corr_beta, _logret_by_date

SAME_SECTOR_MIN = 2
HIGH_CORR = 0.70
METHOD = (f"126-session return correlation; same-sector count from the GICS sector map; "
          f"thresholds (>= {SAME_SECTOR_MIN} same-sector, rho >= {HIGH_CORR:.2f}) are rules of thumb, not tested")


@dataclass(frozen=True)
class Holding:
    ticker: str
    source: str                       # "own picks" | "long-term plan" | "core sleeve (target)"
    shares: Optional[float] = None
    weight: Optional[float] = None    # sleeve target weight


@dataclass
class PortfolioFit:
    ticker: str
    sector_etf: Optional[str]
    holdings: List[Holding] = field(default_factory=list)
    same_sector: List[str] = field(default_factory=list)
    correlations: List[Tuple[str, float]] = field(default_factory=list)   # (holding, rho), high to low
    sleeve_overlap: Optional[Tuple[str, float]] = None                    # (etf, target weight)
    diversifiers: List[Tuple[str, float]] = field(default_factory=list)   # sleeve-universe ETFs, low rho first
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    method: str = METHOD

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["holdings"] = [asdict(h) for h in self.holdings]
        return d


def load_holdings(*, own_picks_db: str = "data/own_picks.db", longterm_db: str = "data/longterm.db",
                  board: Any = None) -> List[Holding]:
    """Real-money holdings from every local source that has them (missing sources are skipped)."""
    out: List[Holding] = []
    p = repo_path(own_picks_db)
    if os.path.exists(p):
        try:
            from src.own_picks import list_equity_picks

            agg: Dict[str, float] = {}
            for row in list_equity_picks(status="OPEN", db_path=p):
                tk = str(row.get("ticker") or "").upper()
                if tk and str(row.get("side") or "LONG").upper() == "LONG":
                    agg[tk] = agg.get(tk, 0.0) + float(row.get("shares") or 0.0)
            out += [Holding(tk, "own picks", shares=sh) for tk, sh in sorted(agg.items())]
        except Exception:
            pass
    lt = repo_path(longterm_db)
    if os.path.exists(lt):
        try:
            from src.longterm.fills import book

            for tk, slot in sorted(book(lt).items()):
                if float(slot.get("shares") or 0.0) > 0:
                    out.append(Holding(tk.upper(), "long-term plan", shares=float(slot["shares"])))
        except Exception:
            pass
    if board is not None:
        for r in getattr(board, "rows", []):
            if getattr(r, "weight", 0) and r.weight > 0:
                out.append(Holding(r.etf, "core sleeve (target)", weight=float(r.weight)))
    return out


def portfolio_fit(ticker: str, *, holdings: Sequence[Holding], ohlcv: Dict[str, Any],
                  sector_of: Callable[[str], Optional[str]], universe: Sequence[str] = ()) -> PortfolioFit:
    sym = ticker.upper().strip()
    sec = sector_of(sym)
    fit = PortfolioFit(ticker=sym, sector_etf=sec, holdings=list(holdings))
    others = [h for h in holdings if h.ticker != sym]
    if not others:
        fit.notes.append("No holdings recorded (own picks, long-term fills, core sleeve) — nothing to overlap with.")
        return fit
    held_already = [h for h in holdings if h.ticker == sym]
    if held_already:
        fit.notes.append("Already held via " + ", ".join(sorted({h.source for h in held_already})) + ".")

    # same sector (sleeve ETFs count as their own sector)
    fit.same_sector = sorted({h.ticker for h in others
                              if (sector_of(h.ticker) or h.ticker) == sec and sec is not None})
    if len(fit.same_sector) >= SAME_SECTOR_MIN:
        fit.warnings.append(f"You already hold {len(fit.same_sector)} positions in {sec}: "
                            + ", ".join(fit.same_sector))

    sleeve = {h.ticker: float(h.weight or 0.0) for h in others if h.source == "core sleeve (target)"}
    if sec and sec in sleeve:
        fit.sleeve_overlap = (sec, sleeve[sec])
        fit.notes.append(f"Your core sleeve holds {sec} at {sleeve[sec] * 100:.0f}% this month — this adds to "
                         "that same exposure.")

    if sym in ohlcv:
        a = _logret_by_date(ohlcv, sym)
        cors: List[Tuple[str, float]] = []
        for tk in sorted({h.ticker for h in others}):
            if tk not in ohlcv:
                continue
            rho, _beta, _n = _corr_beta(a, _logret_by_date(ohlcv, tk))
            if rho is not None:
                cors.append((tk, round(float(rho), 2)))
        cors.sort(key=lambda x: -x[1])
        fit.correlations = cors
        hi = [f"{tk} (rho {r:.2f})" for tk, r in cors if r >= HIGH_CORR]
        if hi:
            fit.warnings.append(f"Moves with what you hold over {CORR_WINDOW} sessions: " + ", ".join(hi))
        # The lowest-correlated ETFs of the tested sleeve universe — information, not a recommendation.
        held = {h.ticker for h in holdings}
        div: List[Tuple[str, float]] = []
        for etf in universe:
            if etf in held or etf == sym or etf not in ohlcv:
                continue
            rho, _b, _n = _corr_beta(a, _logret_by_date(ohlcv, etf))
            if rho is not None:
                div.append((etf, round(float(rho), 2)))
        div.sort(key=lambda x: x[1])
        fit.diversifiers = div[:3]
    return fit


def max_holding_correlation(
    ticker: str,
    *,
    holdings: Sequence[Holding],
    ohlcv: Dict[str, Any],
) -> Tuple[Optional[float], Optional[str]]:
    """Return `(max_rho, matched_holding_ticker)` over 126 sessions against current real holdings/sleeve."""
    sym = ticker.upper().strip()
    if sym not in ohlcv or not holdings:
        return None, None
    others = sorted({h.ticker for h in holdings if h.ticker != sym and h.ticker in ohlcv})
    if not others:
        return None, None
    a = _logret_by_date(ohlcv, sym)
    best_rho: Optional[float] = None
    best_tk: Optional[str] = None
    for tk in others:
        rho, _b, _n = _corr_beta(a, _logret_by_date(ohlcv, tk))
        if rho is not None and (best_rho is None or float(rho) > best_rho):
            best_rho = round(float(rho), 2)
            best_tk = tk
    return best_rho, best_tk


def rank_rows_for_diversification(
    rows: Sequence[Any],
    *,
    holdings: Sequence[Holding],
    ohlcv: Dict[str, Any],
) -> List[Any]:
    """Annotate `LensRow`s with max 126d holding correlation and sort low-correlation diversifiers first."""
    from dataclasses import replace

    scored: List[Tuple[float, int, Any]] = []
    for idx, row in enumerate(rows):
        rho, match_tk = max_holding_correlation(row.ticker, holdings=holdings, ohlcv=ohlcv)
        flags = list(row.flags)
        why = str(row.why)
        if rho is not None and match_tk is not None:
            why = f"[max rho {rho:+.2f} vs {match_tk}] " + why
            if rho >= HIGH_CORR:
                flags.append(f"CORR_DUP:{match_tk}({rho:.2f})")
            elif rho <= 0.40:
                flags.append("LOW_CORR_DIV")
        updated = replace(row, why=why, flags=tuple(dict.fromkeys(flags)))
        sort_val = rho if rho is not None else 9.0
        scored.append((sort_val, idx, updated))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in scored]


__all__ = [
    "Holding",
    "PortfolioFit",
    "SAME_SECTOR_MIN",
    "HIGH_CORR",
    "load_holdings",
    "portfolio_fit",
    "max_holding_correlation",
    "rank_rows_for_diversification",
]
