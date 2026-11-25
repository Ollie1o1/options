"""Real option-chain loader and SVI Risk-Neutral Density adapter (`src/structure/chain_data.py`).

Reads real per-contract bid/ask/IV/Greeks from `data/chain_archive.db` (zero network,
sub-second) with optional opt-in live Yahoo Finance chain fetch. Feeds both the
Shares-vs-Options Express Workbench (`src/structure/express.py`) and the 1-Screen
Ticker Dossier (`src/radar/dossier.py`) so vehicle comparisons use real market
spreads and Breeden-Litzenberger SVI RND instead of flat-vol synthetic proxies.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path


@dataclass(frozen=True)
class ChainContract:
    strike: float
    opt_type: str  # "call" | "put"
    bid: float
    ask: float
    mid: float
    iv: Optional[float] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    open_interest: int = 0
    volume: int = 0

    @property
    def half_spread(self) -> float:
        return max(0.0, 0.5 * (self.ask - self.bid))

    @property
    def half_spread_pct(self) -> float:
        return self.half_spread / self.mid if self.mid > 0 else 1.0


@dataclass
class ChainSlice:
    symbol: str
    expiration: str
    dte: int
    snap_date: str
    source: str  # "chain_archive" | "live"
    calls: List[ChainContract] = field(default_factory=list)
    puts: List[ChainContract] = field(default_factory=list)
    atm_iv: Optional[float] = None
    skew_25d_vp: Optional[float] = None
    median_half_spread_pct: Optional[float] = None
    rnd_density: Any = None
    rnd_source: str = "lognormal"

    def contracts(self, opt_type: str) -> List[ChainContract]:
        return self.calls if opt_type.lower().startswith("c") else self.puts

    def match_contract(
        self,
        opt_type: str,
        target_strike: float,
        *,
        max_rel_dist: float = 0.08,
        min_bid: float = 0.01,
    ) -> Optional[ChainContract]:
        """Find the closest liquid two-sided contract to `target_strike`."""
        pool = [
            c for c in self.contracts(opt_type)
            if c.bid >= min_bid and c.ask >= c.bid and c.mid > 0
        ]
        if not pool:
            return None
        best = min(pool, key=lambda c: abs(c.strike - target_strike))
        if target_strike > 0 and abs(best.strike - target_strike) / target_strike > max_rel_dist:
            return None
        return best

    def match_by_delta(
        self,
        opt_type: str,
        target_abs_delta: float,
        *,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
        min_bid: float = 0.05,
    ) -> Optional[ChainContract]:
        """Find the two-sided contract closest to `target_abs_delta`."""
        pool = [
            c for c in self.contracts(opt_type)
            if c.bid >= min_bid
            and c.ask >= c.bid
            and c.delta is not None
            and math.isfinite(c.delta)
            and (min_strike is None or c.strike >= min_strike)
            and (max_strike is None or c.strike <= max_strike)
        ]
        if not pool:
            return None
        return min(pool, key=lambda c: abs(abs(c.delta or 0.0) - target_abs_delta))


def _clean_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _summarize_slice(
    sl: ChainSlice,
    spot: Optional[float],
    *,
    r: float = 0.045,
    build_rnd: bool = True,
) -> ChainSlice:
    """Populate ATM IV, 25-delta skew, median half-spread %, and optional SVI RND."""
    all_c = sl.calls + sl.puts
    if not all_c:
        return sl

    # Median half-spread % on near-the-money liquid quotes
    spreads: List[float] = []
    for c in all_c:
        if c.bid > 0 and c.ask >= c.bid and c.mid > 0.05:
            if spot is None or abs(c.strike - spot) / max(spot, 1e-6) <= 0.15:
                spreads.append(c.half_spread_pct)
    if spreads:
        sl.median_half_spread_pct = round(float(np.median(spreads)), 4)

    # ATM IV from closest strikes with finite IV in (0.02, 4.0)
    if spot is not None and spot > 0:
        near_ivs = [
            (abs(c.strike - spot), c.iv)
            for c in all_c
            if c.iv is not None and 0.02 <= c.iv <= 4.0
        ]
        if near_ivs:
            near_ivs.sort(key=lambda x: x[0])
            top_k = [v for _, v in near_ivs[:4]]
            sl.atm_iv = float(np.median(top_k))

    # 25-delta skew (put 25d IV - call 25d IV) in vol points
    p25 = sl.match_by_delta("put", 0.25, min_bid=0.01)
    c25 = sl.match_by_delta("call", 0.25, min_bid=0.01)
    if p25 and c25 and p25.iv and c25.iv and 0.02 <= p25.iv <= 4.0 and 0.02 <= c25.iv <= 4.0:
        sl.skew_25d_vp = round((p25.iv - c25.iv) * 100.0, 2)

    # SVI Risk-Neutral Density via Breeden-Litzenberger (src/probability_lab/rnd.py)
    if build_rnd and spot is not None and spot > 0 and sl.dte >= 5:
        # Prefer OTM smile (puts <= spot, calls >= spot) for cleanest SVI fit
        smile_pts: Dict[float, List[float]] = {}
        for c in all_c:
            if c.iv is None or not (0.03 <= c.iv <= 3.5) or c.bid <= 0:
                continue
            if (c.opt_type == "put" and c.strike <= spot * 1.02) or (
                c.opt_type == "call" and c.strike >= spot * 0.98
            ):
                smile_pts.setdefault(c.strike, []).append(c.iv)
        if len(smile_pts) >= 5:
            ks = sorted(smile_pts.keys())
            ivs = [float(np.median(smile_pts[k])) for k in ks]
            try:
                from src.probability_lab.rnd import rnd_from_smile

                conf: Dict[str, Any] = {}
                dens = rnd_from_smile(ks, ivs, sl.dte / 365.0, spot, r, confidence_out=conf)
                sl.rnd_density = dens
                sl.rnd_source = str(conf.get("source", "lognormal"))
            except Exception:
                pass
    return sl


def load_archived_chain_slice(
    symbol: str,
    *,
    target_dte: int = 35,
    min_dte: int = 10,
    max_dte: int = 150,
    spot: Optional[float] = None,
    chain_db: str = "data/chain_archive.db",
    max_snap_age_days: int = 21,
    build_rnd: bool = True,
) -> Optional[ChainSlice]:
    """Load the nearest-horizon real option slice from `data/chain_archive.db`."""
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return None
    sym = str(symbol).upper().strip()
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(snap_date) FROM chain_snapshots WHERE symbol = ?",
                (sym,),
            ).fetchone()
            if not row or not row[0]:
                return None
            snap_date = str(row[0])
            snap_dt = dt.date.fromisoformat(snap_date[:10])
            if (dt.date.today() - snap_dt).days > max_snap_age_days:
                return None

            exps = [
                str(r[0])
                for r in con.execute(
                    "SELECT DISTINCT expiration FROM chain_snapshots WHERE symbol = ? AND snap_date = ?",
                    (sym, snap_date),
                ).fetchall()
                if r and r[0]
            ]
            scored_exps: List[Tuple[int, int, str]] = []
            for e in exps:
                try:
                    dte = (dt.date.fromisoformat(e[:10]) - snap_dt).days
                except ValueError:
                    continue
                if min_dte <= dte <= max_dte:
                    scored_exps.append((abs(dte - target_dte), dte, e))
            if not scored_exps:
                return None
            scored_exps.sort()
            _, chosen_dte, chosen_exp = scored_exps[0]

            rows = con.execute(
                """
                SELECT strike, type, bid, ask, iv, delta, theta, open_interest, volume
                FROM chain_snapshots
                WHERE symbol = ? AND snap_date = ? AND expiration = ?
                """,
                (sym, snap_date, chosen_exp),
            ).fetchall()
        finally:
            con.close()
    except Exception:
        return None

    calls: List[ChainContract] = []
    puts: List[ChainContract] = []
    for k_raw, t_raw, b_raw, a_raw, iv_raw, dl_raw, th_raw, oi_raw, vol_raw in rows:
        k = _clean_float(k_raw)
        b = _clean_float(b_raw)
        a = _clean_float(a_raw)
        if k is None or b is None or a is None or k <= 0 or a < b or a <= 0:
            continue
        mid = 0.5 * (b + a)
        ot = "call" if str(t_raw or "").lower().startswith("c") else "put"
        iv = _clean_float(iv_raw)
        if iv is not None and iv > 5.0:
            iv = iv / 100.0  # guard against percentage-encoded IVs
        c = ChainContract(
            strike=k,
            opt_type=ot,
            bid=b,
            ask=a,
            mid=mid,
            iv=iv,
            delta=_clean_float(dl_raw),
            theta=_clean_float(th_raw),
            open_interest=int(oi_raw or 0),
            volume=int(vol_raw or 0),
        )
        if ot == "call":
            calls.append(c)
        else:
            puts.append(c)

    if not calls and not puts:
        return None
    calls.sort(key=lambda x: x.strike)
    puts.sort(key=lambda x: x.strike)
    sl = ChainSlice(
        symbol=sym,
        expiration=chosen_exp,
        dte=chosen_dte,
        snap_date=snap_date,
        source="chain_archive",
        calls=calls,
        puts=puts,
    )
    return _summarize_slice(sl, spot=spot, build_rnd=build_rnd)


def load_live_chain_slice(
    symbol: str,
    *,
    target_dte: int = 35,
    min_dte: int = 10,
    max_dte: int = 150,
    spot: Optional[float] = None,
    build_rnd: bool = True,
) -> Optional[ChainSlice]:
    """Opt-in live Yahoo Finance option-chain fetch matching `target_dte`."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    sym = str(symbol).upper().strip()
    today = dt.date.today()
    try:
        tkr = yf.Ticker(sym)
        exps = list(tkr.options or ())
        if not exps:
            return None
        scored: List[Tuple[int, int, str]] = []
        for e in exps:
            try:
                dte = (dt.date.fromisoformat(e[:10]) - today).days
            except ValueError:
                continue
            if min_dte <= dte <= max_dte:
                scored.append((abs(dte - target_dte), dte, e))
        if not scored:
            return None
        scored.sort()
        _, chosen_dte, chosen_exp = scored[0]

        if spot is None:
            hist = tkr.history(period="5d")
            if hist is not None and not hist.empty:
                spot = float(hist["Close"].iloc[-1])

        ch = tkr.option_chain(chosen_exp)
        calls_raw = ch.calls.to_dict("records") if ch.calls is not None else []
        puts_raw = ch.puts.to_dict("records") if ch.puts is not None else []
    except Exception:
        return None

    def _parse(records: Sequence[Dict[str, Any]], ot: str) -> List[ChainContract]:
        out: List[ChainContract] = []
        for r in records:
            k = _clean_float(r.get("strike"))
            b = _clean_float(r.get("bid"))
            a = _clean_float(r.get("ask"))
            if k is None or b is None or a is None or k <= 0 or a < b or a <= 0:
                continue
            iv = _clean_float(r.get("impliedVolatility"))
            oi = _clean_float(r.get("openInterest")) or 0.0
            vol = _clean_float(r.get("volume")) or 0.0
            out.append(
                ChainContract(
                    strike=k,
                    opt_type=ot,
                    bid=b,
                    ask=a,
                    mid=0.5 * (b + a),
                    iv=iv,
                    open_interest=int(oi),
                    volume=int(vol),
                )
            )
        out.sort(key=lambda x: x.strike)
        return out

    calls = _parse(calls_raw, "call")
    puts = _parse(puts_raw, "put")
    if not calls and not puts:
        return None
    sl = ChainSlice(
        symbol=sym,
        expiration=chosen_exp,
        dte=chosen_dte,
        snap_date=today.isoformat(),
        source="live",
        calls=calls,
        puts=puts,
    )
    try:
        from src.cache_release import release_yfinance_caches

        release_yfinance_caches()
    except Exception:
        pass
    return _summarize_slice(sl, spot=spot, build_rnd=build_rnd)


def load_chain_for_horizon(
    symbol: str,
    *,
    target_dte: int = 35,
    min_dte: int = 10,
    max_dte: int = 150,
    spot: Optional[float] = None,
    chain_db: str = "data/chain_archive.db",
    allow_live: bool = False,
    build_rnd: bool = True,
) -> Optional[ChainSlice]:
    """Prefer live chain when `allow_live=True`, else read local `chain_archive.db` instantly."""
    if allow_live:
        live = load_live_chain_slice(
            symbol,
            target_dte=target_dte,
            min_dte=min_dte,
            max_dte=max_dte,
            spot=spot,
            build_rnd=build_rnd,
        )
        if live is not None:
            return live
    return load_archived_chain_slice(
        symbol,
        target_dte=target_dte,
        min_dte=min_dte,
        max_dte=max_dte,
        spot=spot,
        chain_db=chain_db,
        build_rnd=build_rnd,
    )


def load_latest_atm_iv_map(
    symbols: Sequence[str],
    spots: Dict[str, float],
    *,
    chain_db: str = "data/chain_archive.db",
    max_snap_age_days: int = 21,
) -> Dict[str, Dict[str, Any]]:
    """Fast batch lookup of latest ATM IV and snap_date from `chain_archive.db` for Radar badges."""
    full = repo_path(chain_db)
    if not os.path.exists(full) or not symbols:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        sp = spots.get(sym)
        if sp is None or sp <= 0:
            continue
        sl = load_archived_chain_slice(
            sym,
            target_dte=30,
            min_dte=14,
            max_dte=65,
            spot=sp,
            chain_db=chain_db,
            max_snap_age_days=max_snap_age_days,
            build_rnd=False,
        )
        if sl is not None and sl.atm_iv is not None:
            out[sym] = {
                "atm_iv": sl.atm_iv,
                "skew_25d_vp": sl.skew_25d_vp,
                "snap_date": sl.snap_date,
                "median_half_spread_pct": sl.median_half_spread_pct,
            }
    return out
