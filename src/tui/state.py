"""Data provider & state layer for the Textual Daily Trading Cockpit.

Supports seamless injection of an offline FakeProvider in tests so pilot tests
never touch production databases or network sockets.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

from src.fill_experiment import fill_rate_summary
from src.own_picks import build_equity_scorecard, list_equity_picks, log_equity_pick
from src.paths import repo_path
from src.radar import (
    RadarSnapshot,
    TickerDossier,
    build_dossier,
    load_cached_radar,
    run_radar,
)
from src.radar.playbooks import PlaybookMeta, list_playbooks
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView, VehicleComparison


class DeskDataProvider:
    """Default local/cached provider backed by src/radar, src/structure, and src/own_picks."""

    def __init__(
        self,
        *,
        ohlcv_db: str = "data/equity_ohlcv.db",
        cache_path: str = "data/radar_cache.json",
        own_picks_db: str = "data/own_picks.db",
        fills_db: str = "data/fills.db",
        playbooks_path: str = "configs/playbooks.json",
    ) -> None:
        self.ohlcv_db = ohlcv_db
        self.cache_path = cache_path
        self.own_picks_db = own_picks_db
        self.fills_db = fills_db
        self.playbooks_path = playbooks_path

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        snap = None if refresh else load_cached_radar(self.cache_path)
        if snap is None:
            snap = run_radar(
                ohlcv_db=self.ohlcv_db,
                playbooks_path=self.playbooks_path,
                cache_path=self.cache_path,
            )
        return snap

    def get_dossier(self, ticker: str, *, fetch_chain: bool = False) -> TickerDossier:
        return build_dossier(
            ticker,
            ohlcv_db=self.ohlcv_db,
            cache_path=self.cache_path,
            fetch_chain=fetch_chain,
        )

    def get_express(self, view: DetailedView, *, spot: float, iv: float = 0.25) -> List[VehicleComparison]:
        return compare_vehicles(view, spot=spot, iv=iv)

    def get_book(self) -> Dict[str, Any]:
        picks = list_equity_picks(db_path=self.own_picks_db)
        scorecard = build_equity_scorecard(db_path=self.own_picks_db, ohlcv_db=self.ohlcv_db)
        return {"equity_picks": picks, "scorecard": scorecard}

    def log_stock_pick(
        self,
        *,
        ticker: str,
        price: float,
        shares: float,
        playbook_id: str,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None,
        notes: str = "",
    ) -> int:
        return log_equity_pick(
            ticker,
            price,
            shares=shares,
            stop_price=stop_price,
            target_price=target_price,
            playbook_id=playbook_id,
            notes=notes,
            db_path=self.own_picks_db,
        )

    def get_playbooks(self) -> List[PlaybookMeta]:
        return list_playbooks(include_closed=True, path=self.playbooks_path)

    def get_health(self) -> Dict[str, Any]:
        free_gb = round(shutil.disk_usage(repo_path(".")).free / (1024 ** 3), 2)
        worst = "OK" if free_gb >= 15 else ("WARN" if free_gb >= 8 else "STALE")
        fills = fill_rate_summary(db_path=self.fills_db)
        return {
            "disk_free_gb": free_gb,
            "worst": worst,
            "live_execution": False,
            "fills": fills,
            "readouts": [
                "2027-01-15: Estimate Revisions Arm B + Bull Put §2",
                "2027-01-20: B1-F ETF Bear Call Forward",
                "2027-03-15: E2 Earnings Forward",
            ],
        }
