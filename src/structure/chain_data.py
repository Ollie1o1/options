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
    last_trade_as_of: Optional[str] = None
    snapshot_at: Optional[str] = None
    source: Optional[str] = None
    bid_size: int = 0
    ask_size: int = 0

    @property
    def half_spread(self) -> float:
        return max(0.0, 0.5 * (self.ask - self.bid))

    @property
    def half_spread_pct(self) -> float:
        return self.half_spread / self.mid if self.mid > 0 else 1.0

    @property
    def min_depth(self) -> Optional[int]:
        if self.bid_size > 0 and self.ask_size > 0:
            return min(self.bid_size, self.ask_size)
        return None


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
    spot: Optional[float] = None

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
    snap_date: Optional[str] = None,
) -> Optional[ChainSlice]:
    """Load the nearest-horizon real option slice from `data/chain_archive.db`.

    `snap_date` reads that archived session instead of the latest one; the
    age limit then does not apply, because the caller chose the session."""
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return None
    sym = str(symbol).upper().strip()
    try:
        con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
        try:
            if snap_date is None:
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
            else:
                snap_dt = dt.date.fromisoformat(snap_date[:10])

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

            columns = {str(row[1]) for row in con.execute("PRAGMA table_info(chain_snapshots)")}
            provenance = ", ".join(
                name if name in columns else "NULL"
                for name in ("last_trade_time", "snapshot_ts", "source", "bid_size", "ask_size")
            )
            rows = con.execute(
                f"""
                SELECT strike, type, bid, ask, iv, delta, theta, open_interest, volume, {provenance}
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
    for k_raw, t_raw, b_raw, a_raw, iv_raw, dl_raw, th_raw, oi_raw, vol_raw, trade_at, snapshot_at, source, bsz_raw, asz_raw in rows:
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
            last_trade_as_of=trade_at,
            snapshot_at=snapshot_at,
            source=source,
            bid_size=int(_clean_float(bsz_raw) or 0),
            ask_size=int(_clean_float(asz_raw) or 0),
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
    max_expiry_checks: int = 1,
    diagnostics: Optional[Dict[str, int]] = None,
) -> Optional[ChainSlice]:
    """Fetch eligible Yahoo expiries in order of proximity, within a request budget."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    sym = str(symbol).upper().strip()
    today = dt.date.today()
    try:
        tkr = yf.Ticker(sym)
        exps = list(tkr.options or ())
        if diagnostics is not None:
            diagnostics.update(listed=len(exps), eligible=0, checked=0, two_sided=0)
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
        if diagnostics is not None:
            diagnostics["eligible"] = len(scored)
        if not scored:
            return None
        scored.sort()

        if spot is None:
            hist = tkr.history(period="5d")
            if hist is not None and not hist.empty:
                spot = float(hist["Close"].iloc[-1])

    except Exception:
        return None

    def _parse(records: Sequence[Dict[str, Any]], ot: str, chosen_dte: int) -> List[ChainContract]:
        from src.data_quality import implied_vol_from_price
        from src.radar.wide_spread_rule import risk_free_rate
        from src.utils import bs_delta, bs_theta

        r_rate, _ = risk_free_rate()
        T = max(chosen_dte, 1) / 365.0
        out: List[ChainContract] = []
        for r in records:
            k = _clean_float(r.get("strike"))
            b = _clean_float(r.get("bid"))
            a = _clean_float(r.get("ask"))
            if k is None or b is None or a is None or k <= 0 or a < b or a <= 0:
                continue
            mid = 0.5 * (b + a)
            iv = None
            if spot is not None and spot > 0 and b > 0:
                iv = implied_vol_from_price(ot, spot, k, T, r_rate, mid)
            if iv is None:
                raw_iv = _clean_float(r.get("impliedVolatility"))
                if raw_iv is not None and 0.02 <= raw_iv <= 4.0:
                    iv = raw_iv
            dl = th = None
            if spot is not None and spot > 0 and iv is not None and 0.02 <= iv <= 4.0:
                try:
                    dl = float(bs_delta(ot, spot, k, T, r_rate, iv))
                    th = float(bs_theta(ot, spot, k, T, r_rate, iv))
                except Exception:
                    pass
            oi = _clean_float(r.get("openInterest")) or 0.0
            vol = _clean_float(r.get("volume")) or 0.0
            out.append(
                ChainContract(
                    strike=k,
                    opt_type=ot,
                    bid=b,
                    ask=a,
                    mid=mid,
                    iv=iv,
                    delta=dl,
                    theta=th,
                    open_interest=int(oi),
                    volume=int(vol),
                    last_trade_as_of=str(r["lastTradeDate"]) if r.get("lastTradeDate") is not None else None,
                    source="yfinance",
                )
            )
        out.sort(key=lambda x: x.strike)
        return out

    sl = None
    for _, chosen_dte, chosen_exp in scored[:max(1, max_expiry_checks)]:
        if diagnostics is not None:
            diagnostics["checked"] += 1
        try:
            chain = tkr.option_chain(chosen_exp)
            calls = _parse(chain.calls.to_dict("records"), "call", chosen_dte) if chain.calls is not None else []
            puts = _parse(chain.puts.to_dict("records"), "put", chosen_dte) if chain.puts is not None else []
        except Exception:
            continue
        two_sided = sum(c.bid > 0 for c in calls + puts)
        if diagnostics is not None:
            diagnostics["two_sided"] += two_sided
        if not (calls or puts) or (max_expiry_checks > 1 and not two_sided):
            continue
        sl = ChainSlice(
            symbol=sym, expiration=chosen_exp, dte=chosen_dte, snap_date=today.isoformat(),
            source="live", calls=calls, puts=puts, spot=spot,
        )
        break
    try:
        from src.cache_release import release_yfinance_caches

        release_yfinance_caches()
    except Exception:
        pass
    return _summarize_slice(sl, spot=spot, build_rnd=build_rnd) if sl is not None else None


def _slices_from_provider_rows(
    symbol: str,
    rows: Sequence[Dict[str, Any]],
    *,
    min_dte: int,
    max_dte: int,
    max_expirations: int,
    target_dte: Optional[int],
    spot: Optional[float],
    source_label: str,
    build_rnd: bool,
    prefer_monthly: bool = True,
    today: Optional[dt.date] = None,
) -> List[ChainSlice]:
    """Convert normalized CBOE / Alpaca snapshot rows into spaced, monthly-preferred `ChainSlice`s."""
    from src.data_quality import implied_vol_from_price
    from src.expiry_selection import select_expirations
    from src.radar.wide_spread_rule import risk_free_rate
    from src.utils import bs_delta, bs_theta

    if not rows:
        return []
    today = today or dt.date.today()
    by_exp: Dict[str, List[Dict[str, Any]]] = {}
    inferred_spot = spot
    for r in rows:
        exp = str(r.get("expiration") or "")[:10]
        if len(exp) != 10:
            continue
        by_exp.setdefault(exp, []).append(r)
        if inferred_spot is None:
            sp = _clean_float(r.get("spot"))
            if sp is not None and sp > 0:
                inferred_spot = sp

    if not by_exp:
        return []
    all_exps = sorted(by_exp.keys())
    if max_expirations == 1 and target_dte is not None:
        cands: List[Tuple[int, int, str]] = []
        for e in all_exps:
            try:
                dte = (dt.date.fromisoformat(e) - today).days
            except ValueError:
                continue
            if min_dte <= dte <= max_dte:
                cands.append((abs(dte - target_dte), dte, e))
        cands.sort()
        chosen_exps = [c[2] for c in cands[:1]]
    else:
        chosen_exps = select_expirations(
            all_exps,
            n=max(1, int(max_expirations)),
            today=today,
            min_dte=min_dte,
            max_dte=max_dte,
            method="spread",
            prefer_monthly=prefer_monthly,
        )

    r_rate, _ = risk_free_rate()
    slices: List[ChainSlice] = []
    for exp in chosen_exps:
        try:
            dte = (dt.date.fromisoformat(exp) - today).days
        except ValueError:
            continue
        if not (min_dte <= dte <= max_dte):
            continue
        T = max(dte, 1) / 365.0
        calls: List[ChainContract] = []
        puts: List[ChainContract] = []
        for rec in by_exp.get(exp) or []:
            k = _clean_float(rec.get("strike"))
            b = _clean_float(rec.get("bid"))
            a = _clean_float(rec.get("ask"))
            if k is None or b is None or a is None or k <= 0 or a < b or a <= 0:
                continue
            mid = 0.5 * (b + a)
            ot = "call" if str(rec.get("type") or "").lower().startswith("c") else "put"
            iv = _clean_float(rec.get("iv"))
            if iv is not None and iv > 5.0:
                iv = iv / 100.0
            if (iv is None or not (0.02 <= iv <= 4.0)) and inferred_spot and inferred_spot > 0 and b > 0:
                iv = implied_vol_from_price(ot, inferred_spot, k, T, r_rate, mid)
            dl = _clean_float(rec.get("delta"))
            th = _clean_float(rec.get("theta"))
            if dl is None and inferred_spot and inferred_spot > 0 and iv and 0.02 <= iv <= 4.0:
                try:
                    dl = float(bs_delta(ot, inferred_spot, k, T, r_rate, iv))
                    th = float(bs_theta(ot, inferred_spot, k, T, r_rate, iv))
                except Exception:
                    pass
            contract = ChainContract(
                strike=k,
                opt_type=ot,
                bid=b,
                ask=a,
                mid=mid,
                iv=iv,
                delta=dl,
                theta=th,
                open_interest=int(_clean_float(rec.get("open_interest")) or 0),
                volume=int(_clean_float(rec.get("volume")) or 0),
                last_trade_as_of=str(rec["last_trade_time"]) if rec.get("last_trade_time") else None,
                snapshot_at=str(rec["snapshot_ts"]) if rec.get("snapshot_ts") else None,
                source=str(rec.get("source") or source_label),
                bid_size=int(_clean_float(rec.get("bid_size")) or 0),
                ask_size=int(_clean_float(rec.get("ask_size")) or 0),
            )
            if ot == "call":
                calls.append(contract)
            else:
                puts.append(contract)
        if not calls and not puts:
            continue
        if sum(c.bid > 0 for c in calls + puts) == 0:
            continue
        calls.sort(key=lambda x: x.strike)
        puts.sort(key=lambda x: x.strike)
        sl = ChainSlice(
            symbol=symbol.upper(),
            expiration=exp,
            dte=dte,
            snap_date=today.isoformat(),
            source=source_label,
            calls=calls,
            puts=puts,
            spot=inferred_spot,
        )
        slices.append(_summarize_slice(sl, spot=inferred_spot, build_rnd=build_rnd))
    return slices


def load_live_chain_slices(
    symbol: str,
    *,
    min_dte: int = 14,
    max_dte: int = 125,
    max_expirations: int = 3,
    target_dte: Optional[int] = None,
    spot: Optional[float] = None,
    build_rnd: bool = False,
    prefer_monthly: bool = True,
    providers: Sequence[str] = ("cboe", "alpaca", "yfinance"),
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[ChainSlice]:
    """3-tier live multi-expiry option-chain router: CBOE -> Alpaca indicative -> Yahoo.

    Spreads `max_expirations` across `[min_dte, max_dte]` with 3rd-Friday monthly preference
    so live scans cover the full term structure without Yahoo's post-open 0/0 dead zone.
    """
    sym = str(symbol).upper().strip()
    today = dt.date.today()
    diag = diagnostics if diagnostics is not None else {}
    diag.setdefault("listed", 0)
    diag.setdefault("eligible", 0)
    diag.setdefault("checked", 0)
    diag.setdefault("two_sided", 0)
    diag.setdefault("provider", "none")

    # Resolve spot from local OHLCV archive (fast, offline) so providers without spot
    # fields (e.g. Alpaca option snapshots) still build complete deltas, moneyness & RND.
    if spot is None or spot <= 0:
        try:
            full_ohlcv = repo_path("data/equity_ohlcv.db")
            if os.path.exists(full_ohlcv):
                con = sqlite3.connect(f"file:{full_ohlcv}?mode=ro", uri=True)
                try:
                    row = con.execute(
                        "SELECT close FROM ohlcv WHERE ticker = ? ORDER BY date DESC LIMIT 1",
                        (sym,),
                    ).fetchone()
                    if row and row[0] is not None and float(row[0]) > 0:
                        spot = float(row[0])
                finally:
                    con.close()
        except Exception:
            pass

    for prov in providers:
        p_key = str(prov).lower().strip()
        if p_key == "cboe":
            try:
                from src.cboe_client import fetch_chain

                rows = fetch_chain(sym, timeout=10)
            except Exception:
                rows = []
            if rows:
                slices = _slices_from_provider_rows(
                    sym, rows, min_dte=min_dte, max_dte=max_dte, max_expirations=max_expirations,
                    target_dte=target_dte, spot=spot, source_label="cboe", build_rnd=build_rnd,
                    prefer_monthly=prefer_monthly, today=today,
                )
                if slices:
                    exps_all = {str(r.get("expiration") or "")[:10] for r in rows if r.get("expiration")}
                    diag["listed"] = len(exps_all)
                    diag["eligible"] = len(exps_all)
                    diag["checked"] = len(slices)
                    diag["two_sided"] = sum(sum(c.bid > 0 for c in s.calls + s.puts) for s in slices)
                    diag["provider"] = "cboe"
                    return slices

        elif p_key == "alpaca":
            try:
                from src.alpaca_data import fetch_option_chain_snapshots, has_credentials

                if not has_credentials():
                    continue
                rows = fetch_option_chain_snapshots(
                    sym, feed="indicative", min_dte=min_dte, max_dte=max_dte, spot=spot, today=today
                )
            except Exception:
                rows = []
            if rows:
                slices = _slices_from_provider_rows(
                    sym, rows, min_dte=min_dte, max_dte=max_dte, max_expirations=max_expirations,
                    target_dte=target_dte, spot=spot, source_label="alpaca_indicative", build_rnd=build_rnd,
                    prefer_monthly=prefer_monthly, today=today,
                )
                if slices:
                    exps_all = {str(r.get("expiration") or "")[:10] for r in rows if r.get("expiration")}
                    diag["listed"] = len(exps_all)
                    diag["eligible"] = len(exps_all)
                    diag["checked"] = len(slices)
                    diag["two_sided"] = sum(sum(c.bid > 0 for c in s.calls + s.puts) for s in slices)
                    diag["provider"] = "alpaca_indicative"
                    return slices

        elif p_key == "yfinance":
            try:
                import yfinance as yf
                from src.expiry_selection import select_expirations

                tkr = yf.Ticker(sym)
                exps = list(tkr.options or ())
                diag["listed"] = len(exps)
                if not exps:
                    continue
                if max_expirations == 1 and target_dte is not None:
                    sl_one = load_live_chain_slice(
                        sym, target_dte=target_dte, min_dte=min_dte, max_dte=max_dte,
                        spot=spot, build_rnd=build_rnd, max_expiry_checks=2, diagnostics=diag,
                    )
                    if sl_one is not None:
                        diag["provider"] = "yfinance"
                        return [sl_one]
                    continue

                chosen_exps = select_expirations(
                    exps, n=max(1, int(max_expirations)), today=today,
                    min_dte=min_dte, max_dte=max_dte, method="spread", prefer_monthly=prefer_monthly,
                )
                diag["eligible"] = len(chosen_exps)
                eff_spot = spot
                if eff_spot is None:
                    hist = tkr.history(period="5d")
                    if hist is not None and not hist.empty:
                        eff_spot = float(hist["Close"].dropna().iloc[-1])

                yf_rows: List[Dict[str, Any]] = []
                for exp in chosen_exps:
                    diag["checked"] = int(diag.get("checked", 0)) + 1
                    try:
                        ch = tkr.option_chain(exp)
                    except Exception:
                        continue
                    for ot, df in (("call", ch.calls), ("put", ch.puts)):
                        if df is None or df.empty:
                            continue
                        for rec in df.to_dict("records"):
                            yf_rows.append(
                                {
                                    "expiration": exp,
                                    "type": ot,
                                    "strike": rec.get("strike"),
                                    "bid": rec.get("bid"),
                                    "ask": rec.get("ask"),
                                    "iv": rec.get("impliedVolatility"),
                                    "open_interest": rec.get("openInterest"),
                                    "volume": rec.get("volume"),
                                    "last_trade_time": rec.get("lastTradeDate"),
                                    "spot": eff_spot,
                                    "source": "yfinance",
                                }
                            )
                slices = _slices_from_provider_rows(
                    sym, yf_rows, min_dte=min_dte, max_dte=max_dte, max_expirations=len(chosen_exps),
                    target_dte=None, spot=eff_spot, source_label="live", build_rnd=build_rnd,
                    prefer_monthly=prefer_monthly, today=today,
                )
                if slices:
                    diag["two_sided"] = sum(sum(c.bid > 0 for c in s.calls + s.puts) for s in slices)
                    diag["provider"] = "yfinance"
                    return slices
            except Exception:
                continue

    return []


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
