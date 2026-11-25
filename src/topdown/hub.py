"""One call that assembles the whole top-down view, shared by the CLI, TUI and web cockpit."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.radar.schema import RadarSnapshot
from .card import EvidenceCard, build_evidence_card
from .lenses import NULL_LEDGER, LensContext, LensResult, build_context, run_lenses
from .macro import MacroSnapshot, build_macro_snapshot
from .sectors import SectorBoard, build_sector_board

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]


@dataclass
class DiscoveryHub:
    built_at: str
    macro: MacroSnapshot
    board: SectorBoard
    lenses: List[LensResult]
    ctx: LensContext
    ohlcv: OhlcvMap
    snapshot: Optional[RadarSnapshot] = None
    timings: Dict[str, float] = field(default_factory=dict)
    own_picks_db: str = "data/own_picks.db"
    longterm_db: str = "data/longterm.db"
    _holdings: Optional[List[Any]] = None

    def holdings(self) -> List[Any]:
        """Real-money holdings (own picks, long-term fills, core-sleeve targets), loaded once."""
        held = self._holdings
        if held is None:
            from .portfolio_fit import load_holdings

            held = load_holdings(own_picks_db=self.own_picks_db, longterm_db=self.longterm_db, board=self.board)
            self._holdings = held
        return held

    def freshness(self) -> Dict[str, Any]:
        """How current the price archive is: SPY's last bar and how many symbols lag it."""
        spy = self.ohlcv.get("SPY")
        spy_last = str(spy[0][-1]) if spy is not None and spy[0] else None
        lag = sorted(s for s, (ds, _c, _v) in self.ohlcv.items() if ds and spy_last and str(ds[-1]) < spy_last)
        return {"spy_last": spy_last, "symbols": len(self.ohlcv), "behind": len(lag), "behind_sample": lag[:5]}

    def lens(self, lens_id: str) -> Optional[LensResult]:
        return next((r for r in self.lenses if r.lens.id == lens_id), None)

    def card(self, ticker: str, **kw: Any) -> EvidenceCard:
        kw.setdefault("holdings", self.holdings())
        return build_evidence_card(
            ticker, ohlcv=self.ohlcv, board=self.board, lens_results=self.lenses,
            snapshot=self.snapshot, estimates_as_of=self.ctx.rev_as_of, **kw,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built_at": self.built_at,
            "freshness": self.freshness(),
            "macro": self.macro.to_dict(),
            "board": self.board.to_dict(),
            "lenses": [r.to_dict() for r in self.lenses],
            "null_ledger": [
                {"idea": a, "result": b, "date": c, "source": d} for a, b, c, d in NULL_LEDGER
            ],
            "timings_s": dict(self.timings),
        }


def build_hub(
    *,
    ohlcv: Optional[OhlcvMap] = None,
    snapshot: Optional[RadarSnapshot] = None,
    ohlcv_db: str = "data/equity_ohlcv.db",
    cache_path: str = "data/radar_cache.json",
    fred_path: str = "data/fred_cache.json",
    cboe_dir: str = "data/discovery",
    estimates_db: str = "data/estimates_archive.db",
    own_picks_db: str = "data/own_picks.db",
    longterm_db: str = "data/longterm.db",
) -> DiscoveryHub:
    """Macro → board → lenses, from local archives only (no network)."""
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    if ohlcv is None:
        from src.radar.scanners import load_local_ohlcv_safe

        ohlcv = load_local_ohlcv_safe(ohlcv_db)
    timings["prices"] = round(time.perf_counter() - t0, 3)
    if snapshot is None:
        from src.radar.engine import load_cached_radar

        snapshot = load_cached_radar(cache_path)
    t1 = time.perf_counter()
    macro = build_macro_snapshot(ohlcv, fred_path=fred_path, cboe_dir=cboe_dir)
    timings["macro"] = round(time.perf_counter() - t1, 3)
    t2 = time.perf_counter()
    board = build_sector_board(ohlcv)
    timings["board"] = round(time.perf_counter() - t2, 3)
    t3 = time.perf_counter()
    ctx = build_context(ohlcv, board, snapshot, estimates_db=estimates_db)
    lenses = run_lenses(ctx)
    timings["lenses"] = round(time.perf_counter() - t3, 3)
    return DiscoveryHub(
        built_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        macro=macro, board=board, lenses=lenses, ctx=ctx, ohlcv=ohlcv, snapshot=snapshot, timings=timings,
        own_picks_db=own_picks_db, longterm_db=longterm_db,
    )


__all__ = ["DiscoveryHub", "build_hub"]
