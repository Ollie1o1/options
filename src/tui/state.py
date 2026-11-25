"""Data provider & state layer for the Textual Daily Trading Cockpit & Web Dashboard.

Supports seamless injection of an offline FakeProvider in tests so pilot tests
never touch production databases or network sockets.
"""
from __future__ import annotations

import datetime as dt
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.equity_playbooks.indicators import sma
from src.fill_experiment import fill_rate_summary, log_real_fill
from src.own_picks import build_equity_scorecard, list_equity_picks, log_equity_pick
from src.paths import repo_path
from src.radar import (
    Opportunity,
    RadarSnapshot,
    TickerDossier,
    build_dossier,
    load_cached_radar,
    run_radar,
)
from src.radar.live_options import refresh_local_prices, scan_live_etf_wide_spreads
from src.radar.playbooks import PlaybookMeta, list_playbooks
from src.radar.scanners import load_local_ohlcv_safe
from src.structure.express import compare_vehicles
from src.structure.types import DetailedView, VehicleComparison


class DeskDataProvider:
    """Default local/cached provider backed by src/radar, src/structure, src/own_picks, and fills."""

    def __init__(
        self,
        *,
        ohlcv_db: str = "data/equity_ohlcv.db",
        cache_path: str = "data/radar_cache.json",
        own_picks_db: str = "data/own_picks.db",
        fills_db: str = "data/fills.db",
        chain_db: str = "data/chain_archive.db",
        paper_db: str = "paper_trades.db",
        longterm_db: str = "data/longterm.db",
        plan_path: str = "longterm_plan.json",
        playbooks_path: str = "configs/playbooks.json",
    ) -> None:
        self.ohlcv_db = ohlcv_db
        self.cache_path = cache_path
        self.own_picks_db = own_picks_db
        self.fills_db = fills_db
        self.chain_db = chain_db
        self.paper_db = paper_db
        self.longterm_db = longterm_db
        self.plan_path = plan_path
        self.playbooks_path = playbooks_path
        self._dossier_cache: Dict[str, TickerDossier] = {}

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        if refresh:
            self._dossier_cache.clear()
        snap = None if refresh else load_cached_radar(self.cache_path)
        if snap is None:
            snap = run_radar(
                ohlcv_db=self.ohlcv_db,
                playbooks_path=self.playbooks_path,
                cache_path=self.cache_path,
                chain_db=self.chain_db,
                own_picks_db=self.own_picks_db,
                paper_db=self.paper_db,
                longterm_db=self.longterm_db,
            )
        return snap

    def get_live_etf_options(self) -> Tuple[List[Opportunity], List[str]]:
        return scan_live_etf_wide_spreads()

    def update_prices(self, *, expand_universe: bool = False) -> Dict[str, Any]:
        self._dossier_cache.clear()
        return refresh_local_prices(self.ohlcv_db, expand_universe=expand_universe)

    def get_cached_dossier(self, ticker: str) -> Optional[TickerDossier]:
        return self._dossier_cache.get(str(ticker).upper().strip())

    def get_dossier(self, ticker: str, *, fetch_chain: bool = False) -> TickerDossier:
        sym = str(ticker).upper().strip()
        if not fetch_chain and sym in self._dossier_cache:
            return self._dossier_cache[sym]
        dos = build_dossier(
            sym,
            ohlcv_db=self.ohlcv_db,
            cache_path=self.cache_path,
            chain_db=self.chain_db,
            fetch_chain=fetch_chain,
        )
        self._dossier_cache[sym] = dos
        return dos

    def get_express(
        self,
        view: DetailedView,
        *,
        spot: float,
        iv: float = 0.25,
        allow_live: bool = False,
    ) -> List[VehicleComparison]:
        return compare_vehicles(
            view,
            spot=spot,
            iv=iv,
            chain_db=self.chain_db,
            allow_live=allow_live,
        )

    def _load_open_paper_options(self, limit: int = 25) -> List[Dict[str, Any]]:
        full = repo_path(self.paper_db)
        if not os.path.exists(full) or os.path.getsize(full) == 0:
            return []
        try:
            con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            try:
                tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "trades" not in tables:
                    return []
                rows = con.execute(
                    "SELECT entry_id, date, ticker, strategy_name, expiration, strike, type, entry_price, quantity "
                    "FROM trades WHERE status = 'OPEN' ORDER BY entry_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                con.close()
        except Exception:
            return []

    def _load_longterm_summary(self, ohlcv: Dict[str, Any]) -> List[Dict[str, Any]]:
        full_plan = repo_path(self.plan_path)
        if not os.path.exists(full_plan):
            return []
        try:
            from src.longterm.fills import book as lt_book, filled_levels
            from src.longterm.plan import load_plan

            plan = load_plan(full_plan)
            full_lt = repo_path(self.longterm_db)
            held = lt_book(full_lt) if os.path.exists(full_lt) else {}
            out: List[Dict[str, Any]] = []
            for n in plan.names:
                sym = n.ticker.upper()
                fl = filled_levels(sym, db_path=full_lt) if os.path.exists(full_lt) else set()
                slot = held.get(sym) or {}
                spot = float(ohlcv[sym][1][-1]) if sym in ohlcv and len(ohlcv[sym][1]) else None
                next_lvl = max((t.level for t in n.tranches if t.level not in fl), default=None)
                out.append(
                    {
                        "ticker": sym,
                        "tranches_filled": len(fl),
                        "tranches_total": len(n.tranches),
                        "shares": float(slot.get("shares") or 0.0),
                        "avg_price": float(slot.get("avg_price") or 0.0),
                        "spot": round(spot, 2) if spot is not None else None,
                        "next_level": next_lvl,
                    }
                )
            return out
        except Exception:
            return []

    def get_book(self) -> Dict[str, Any]:
        picks = list_equity_picks(db_path=self.own_picks_db)
        scorecard = build_equity_scorecard(db_path=self.own_picks_db, ohlcv_db=self.ohlcv_db)
        needed_syms = {str(p.get("ticker") or "").upper() for p in picks if p.get("ticker")}
        full_plan = repo_path(self.plan_path)
        if os.path.exists(full_plan):
            try:
                from src.longterm.plan import load_plan

                for n in load_plan(full_plan).names:
                    needed_syms.add(n.ticker.upper())
            except Exception:
                pass
        ohlcv = (
            load_local_ohlcv_safe(self.ohlcv_db, symbols=sorted(needed_syms), lookback_days=45)
            if needed_syms
            else {}
        )
        today = dt.date.today()

        enriched_picks: List[Dict[str, Any]] = []
        exit_alerts: List[str] = []
        for p in picks:
            row = dict(p)
            sym = str(row.get("ticker") or "").upper()
            entry_px = float(row.get("entry_price") or 0.0)
            status = str(row.get("status") or "OPEN").upper()
            cur_px: Optional[float] = None
            unrl_pct: Optional[float] = None
            exit_sig = ""
            if sym in ohlcv and len(ohlcv[sym][1]) >= 6:
                cs = ohlcv[sym][1]
                cur_px = round(float(cs[-1]), 2)
                if entry_px > 0:
                    sign = -1.0 if str(row.get("side") or "LONG").upper() == "SHORT" else 1.0
                    unrl_pct = round(sign * (cur_px / entry_px - 1.0) * 100.0, 2)
                if status == "OPEN":
                    try:
                        d_ent = dt.date.fromisoformat(str(row.get("date") or "")[:10])
                        days_held = (today - d_ent).days
                    except ValueError:
                        days_held = 0
                    s5 = float(sma(cs, 5)[-1])
                    stop_p = row.get("stop_price")
                    tgt_p = row.get("target_price")
                    if stop_p is not None and cur_px <= float(stop_p):
                        exit_sig = "STOP HIT"
                    elif tgt_p is not None and cur_px >= float(tgt_p):
                        exit_sig = "TARGET HIT"
                    elif str(row.get("playbook_id") or "") == "equity_pullback":
                        if np.isfinite(s5) and cur_px > s5:
                            exit_sig = f"EXIT >SMA5 (${s5:.2f})"
                        elif days_held >= 14:
                            exit_sig = "EXIT 10td TIME"
                    if exit_sig:
                        exit_alerts.append(f"{sym} (#{row['pick_id']}): {exit_sig} @ ${cur_px:.2f}")
            row["current_price"] = cur_px
            row["unrealized_pct"] = unrl_pct
            row["exit_signal"] = exit_sig
            enriched_picks.append(row)

        return {
            "equity_picks": enriched_picks,
            "scorecard": scorecard,
            "exit_alerts": exit_alerts,
            "paper_options": self._load_open_paper_options(),
            "longterm_holdings": self._load_longterm_summary(ohlcv),
        }

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

    def log_fill_attempt(
        self,
        *,
        broker: str,
        ticker: str,
        structure: str,
        side: str,
        bid: float,
        ask: float,
        limit_price: float,
        filled: bool,
        fill_price: Optional[float] = None,
        notes: str = "",
    ) -> int:
        return log_real_fill(
            broker=broker,
            ticker=ticker,
            structure=structure,
            side=side,
            bid=bid,
            ask=ask,
            limit_posted=limit_price,
            filled=filled,
            fill_price=fill_price,
            notes=notes,
            db_path=self.fills_db,
        )

    def get_playbooks(self) -> List[PlaybookMeta]:
        return list_playbooks(include_closed=True, path=self.playbooks_path)

    def get_health(self) -> Dict[str, Any]:
        free_gb = round(shutil.disk_usage(repo_path(".")).free / (1024 ** 3), 2)
        worst = "OK" if free_gb >= 15 else ("WARN" if free_gb >= 8 else "STALE")
        fills = fill_rate_summary(db_path=self.fills_db)

        # Read real maintenance health + OHLCV freshness when available
        maint_summary = ""
        try:
            from src.maintenance import DEFAULT_STATE_PATH, load_state
            from src.maintenance_health import compute_health, launchd_silence_days

            st = load_state(DEFAULT_STATE_PATH)
            rep = compute_health(st, dt.datetime.now())
            silence = launchd_silence_days()
            if getattr(rep, "worst", None) in ("STALE", "WARN") and worst == "OK":
                worst = str(rep.worst)
            stale_n = sum(1 for j in getattr(rep, "jobs", []) if getattr(j, "status", "") == "STALE")
            maint_summary = f"Jobs stale: {stale_n}" + (f" · Launchd quiet: {silence}d" if silence else "")
        except Exception:
            pass

        ohlcv_syms = 0
        ohlcv_last = ""
        full_ohlcv = repo_path(self.ohlcv_db)
        if os.path.exists(full_ohlcv):
            try:
                con = sqlite3.connect(f"file:{full_ohlcv}?mode=ro", uri=True)
                try:
                    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if "price_state" in tables:
                        r_cnt = con.execute("SELECT COUNT(*) FROM price_state").fetchone()
                        r_dt = con.execute("SELECT MAX(date) FROM ohlcv WHERE ticker = 'SPY'").fetchone()
                        ohlcv_syms = int(r_cnt[0] or 0) if r_cnt else 0
                        ohlcv_last = str(r_dt[0] or "") if r_dt and r_dt[0] else ""
                    if ohlcv_syms == 0:
                        row = con.execute("SELECT COUNT(DISTINCT ticker), MAX(date) FROM ohlcv").fetchone()
                        if row:
                            ohlcv_syms = int(row[0] or 0)
                            ohlcv_last = str(row[1] or "")
                finally:
                    con.close()
            except Exception:
                pass

        return {
            "disk_free_gb": free_gb,
            "worst": worst,
            "live_execution": False,
            "fills": fills,
            "maintenance_summary": maint_summary,
            "ohlcv_symbols": ohlcv_syms,
            "ohlcv_last": ohlcv_last,
            "readouts": [
                "2027-01-15: Estimate Revisions Arm B + Bull Put §2",
                "2027-01-20: B1-F ETF Bear Call Forward",
                "2027-03-15: E2 Earnings Forward",
            ],
        }
