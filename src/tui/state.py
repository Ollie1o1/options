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
from src.own_picks import (
    build_equity_scorecard,
    build_mistake_cost_report,
    close_equity_pick,
    list_equity_picks,
    log_equity_pick,
)
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
        self._hub: Optional[Any] = None
        self.last_live_skipped: Dict[str, str] = {}

    def get_radar(self, *, refresh: bool = False) -> RadarSnapshot:
        if refresh:
            self._dossier_cache.clear()
            self._hub = None
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
        from src.radar.live_options import scan_live_etf_wide_spreads_detailed

        res = scan_live_etf_wide_spreads_detailed()
        # ticker -> plain-English reason it produced no spread (shown by the desk)
        self.last_live_skipped = dict(res.skipped)
        return res.opportunities, res.warnings

    def update_prices(self, *, expand_universe: bool = False) -> Dict[str, Any]:
        self._dossier_cache.clear()
        self._hub = None
        res = refresh_local_prices(self.ohlcv_db, expand_universe=expand_universe)
        try:  # FRED credit/curve context for ① MARKET; never blocks a price update
            from src.topdown.macro import refresh_fred_cache

            res["fred"] = refresh_fred_cache()
        except Exception as exc:
            res["fred"] = f"error: {exc}"
        return res

    def get_hub(self, *, refresh: bool = False) -> Any:
        """Top-down Discover bundle (① market, ② sector board, ③ lenses), built once per refresh."""
        if refresh or self._hub is None:
            from src.topdown.hub import build_hub

            self._hub = build_hub(ohlcv_db=self.ohlcv_db, cache_path=self.cache_path)
        return self._hub

    def get_card(self, ticker: str) -> Any:
        """④ Evidence Card for one ticker (reuses the cached Dossier when present)."""
        hub = self.get_hub()
        return hub.card(ticker, dossier=self.get_dossier(ticker, fetch_chain=False), chain_db=self.chain_db)

    def get_cached_dossier(self, ticker: str) -> Optional[TickerDossier]:
        return self._dossier_cache.get(str(ticker).upper().strip())

    def get_dossier(
        self,
        ticker: str,
        *,
        fetch_chain: bool = False,
        allow_live_spot: bool = False,
    ) -> TickerDossier:
        sym = str(ticker).upper().strip()
        if not fetch_chain and not allow_live_spot and sym in self._dossier_cache:
            return self._dossier_cache[sym]
        dos = build_dossier(
            sym,
            ohlcv_db=self.ohlcv_db,
            cache_path=self.cache_path,
            chain_db=self.chain_db,
            fetch_chain=fetch_chain,
            allow_live_spot=allow_live_spot,
        )
        self._dossier_cache[sym] = dos
        return dos

    def get_live_quotes(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Best-effort live last price & 1d % change for market-hours polling."""
        syms = sorted({str(t).upper().strip() for t in tickers if str(t).strip()})[:25]
        if not syms:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        try:
            import math
            import yfinance as yf

            for tk in syms:
                try:
                    h = yf.Ticker(tk).history(period="5d", timeout=8)
                    if h is None or h.empty:
                        continue
                    closes = [float(x) for x in h["Close"].dropna().tolist() if math.isfinite(float(x))]
                    if not closes:
                        continue
                    last = closes[-1]
                    chg = ((last / closes[-2] - 1.0) * 100.0) if len(closes) >= 2 and closes[-2] > 0 else None
                    out[tk] = {
                        "price": round(last, 2),
                        "chg_1d_pct": round(chg, 2) if chg is not None else None,
                    }
                except Exception:
                    continue
            try:
                from src.cache_release import release_yfinance_caches

                release_yfinance_caches()
            except Exception:
                pass
        except ImportError:
            pass
        return out

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
            "mistake_costs": build_mistake_cost_report(db_path=self.own_picks_db),
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

    def close_stock_pick(
        self,
        *,
        pick_id: int,
        exit_price: float,
        exit_date: Optional[str] = None,
        notes: Optional[str] = None,
        close_tag: Optional[str] = None,
    ) -> bool:
        return close_equity_pick(
            int(pick_id),
            float(exit_price),
            exit_date=exit_date,
            notes=notes,
            close_tag=close_tag,
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

    def get_ghost_race(self, *, lookback_days: int = 126) -> Optional[Any]:
        try:
            from src.equity_playbooks.ghost_race import compute_ghost_race
            from src.profile import load_profile

            prof = load_profile()
            return compute_ghost_race(
                ohlcv_db=self.ohlcv_db,
                own_picks_db=self.own_picks_db,
                lookback_days=lookback_days,
                account_usd=prof.tfsa_usd,
            )
        except Exception:
            return None

    def get_health(self) -> Dict[str, Any]:
        free_gb = round(shutil.disk_usage(repo_path(".")).free / (1024 ** 3), 2)
        worst = "OK" if free_gb >= 15 else ("WARN" if free_gb >= 8 else "STALE")
        fills = fill_rate_summary(db_path=self.fills_db)
        try:
            from src.fill_experiment import o1_fill_verdict

            o1 = o1_fill_verdict(self.fills_db)
        except Exception:
            o1 = {}

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

        pipes = self._compute_pipe_health(
            free_gb=free_gb,
            ohlcv_syms=ohlcv_syms,
            ohlcv_last=ohlcv_last,
        )
        for pinfo in pipes.values():
            st_p = pinfo.get("status")
            if st_p == "STALE":
                worst = "STALE"
            elif st_p == "WARN" and worst == "OK":
                worst = "WARN"

        return {
            "disk_free_gb": free_gb,
            "worst": worst,
            "live_execution": False,
            "fills": fills,
            "o1_fills": o1,
            "maintenance_summary": maint_summary,
            "ohlcv_symbols": ohlcv_syms,
            "ohlcv_last": ohlcv_last,
            "pipes": pipes,
            "readouts": [
                "2027-01-15: Estimate Revisions Arm B + Bull Put §2",
                "2027-01-20: B1-F ETF Bear Call Forward",
                "2027-03-15: E2 Earnings Forward",
                "2027-05-25: Discover lens forward read (26 weeks)",
            ],
        }

    def _compute_pipe_health(
        self,
        *,
        free_gb: float,
        ohlcv_syms: int,
        ohlcv_last: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Per-pipeline freshness LEDs so broken feeds never hide behind disk-only OK."""
        today = dt.date.today()

        def _age_days(d_str: str) -> Optional[int]:
            try:
                return (today - dt.date.fromisoformat(str(d_str)[:10])).days
            except Exception:
                return None

        pipes: Dict[str, Dict[str, Any]] = {}

        # 1. OHLCV
        o_age = _age_days(ohlcv_last)
        o_st = "OK" if (ohlcv_syms > 0 and o_age is not None and o_age <= 4) else ("WARN" if ohlcv_syms > 0 else "STALE")
        pipes["ohlcv"] = {"status": o_st, "detail": f"{ohlcv_syms} syms ({ohlcv_last or '—'})", "as_of": ohlcv_last}

        # 2. Chain Archive
        ch_last = ""
        ch_syms = 0
        full_ch = repo_path(self.chain_db)
        if os.path.exists(full_ch):
            try:
                con = sqlite3.connect(f"file:{full_ch}?mode=ro", uri=True)
                try:
                    row = con.execute("SELECT MAX(snap_date), COUNT(DISTINCT symbol) FROM chain_snapshots").fetchone()
                    if row:
                        ch_last = str(row[0] or "")
                        ch_syms = int(row[1] or 0)
                finally:
                    con.close()
            except Exception:
                pass
        c_age = _age_days(ch_last)
        c_st = "OK" if (c_age is not None and c_age <= 4) else ("WARN" if (c_age is not None and c_age <= 7) else "STALE")
        pipes["chains"] = {"status": c_st, "detail": f"{ch_syms} syms ({ch_last or 'missing'})", "as_of": ch_last}

        # 3. Earnings Calendar (future coverage)
        e_fut = 0
        full_dolt = repo_path("data/dolt_options.db")
        if os.path.exists(full_dolt):
            try:
                con = sqlite3.connect(f"file:{full_dolt}?mode=ro", uri=True)
                try:
                    r_e = con.execute(
                        "SELECT COUNT(DISTINCT symbol) FROM earnings_cal WHERE date >= ?",
                        (today.isoformat(),),
                    ).fetchone()
                    e_fut = int(r_e[0] or 0) if r_e else 0
                finally:
                    con.close()
            except Exception:
                pass
        e_st = "OK" if e_fut >= 25 else ("WARN" if e_fut >= 5 else "STALE")
        pipes["earnings"] = {"status": e_st, "detail": f"{e_fut} future syms", "as_of": today.isoformat()}

        # 4. Estimates Archive
        est_last = ""
        full_est = repo_path("data/estimates_archive.db")
        if os.path.exists(full_est):
            try:
                con = sqlite3.connect(f"file:{full_est}?mode=ro", uri=True)
                try:
                    r_es = con.execute("SELECT MAX(snap_date) FROM estimate_snapshots").fetchone()
                    est_last = str(r_es[0] or "") if r_es and r_es[0] else ""
                finally:
                    con.close()
            except Exception:
                pass
        es_age = _age_days(est_last)
        es_st = "OK" if (es_age is not None and es_age <= 7) else ("WARN" if (es_age is not None and es_age <= 14) else "STALE")
        pipes["estimates"] = {"status": es_st, "detail": est_last or "missing", "as_of": est_last}

        # 5. FRED Cache
        full_fred = repo_path("data/fred_cache.json")
        f_st = "STALE"
        f_det = "missing"
        if os.path.exists(full_fred):
            age_d = (today - dt.date.fromtimestamp(os.path.getmtime(full_fred))).days
            f_st = "OK" if age_d <= 7 else ("WARN" if age_d <= 14 else "STALE")
            f_det = f"{age_d}d old"
        pipes["fred"] = {"status": f_st, "detail": f_det, "as_of": ""}

        # 6. Kalshi Predmarkets
        pm_last = ""
        full_pm = repo_path("data/predmarkets.db")
        if os.path.exists(full_pm):
            try:
                con = sqlite3.connect(f"file:{full_pm}?mode=ro", uri=True)
                try:
                    r_pm = con.execute("SELECT MAX(archived_at) FROM pm_quotes").fetchone()
                    pm_last = str(r_pm[0] or "")[:10] if r_pm and r_pm[0] else ""
                finally:
                    con.close()
            except Exception:
                pass
        p_age = _age_days(pm_last)
        p_st = "OK" if (p_age is not None and p_age <= 4) else ("WARN" if (p_age is not None and p_age <= 7) else "STALE")
        pipes["kalshi"] = {"status": p_st, "detail": pm_last or "missing", "as_of": pm_last}

        # 7. Disk
        d_st = "OK" if free_gb >= 25 else ("WARN" if free_gb >= 12 else "STALE")
        pipes["disk"] = {"status": d_st, "detail": f"{free_gb:.1f} GB free", "as_of": ""}
        return pipes
