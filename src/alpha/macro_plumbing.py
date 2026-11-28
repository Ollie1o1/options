"""External Macro Plumbing & Alternative Data Aggregator (`src/alpha/macro_plumbing.py`).

100% free public endpoints cached locally to `data/macro_plumbing.json`:
  1. CBOE 4-Point VIX Curve     (`VIX9D`, `VIX`, `VIX3M`, `VIX6M`)
  2. FRED Fed Net Liquidity     (`WALCL - WTREGEN - RRPONTSYD` — 4w & 13w impulse in $B)
  3. CFTC COT Positioning       (free Socrata API — commercial/leveraged net positioning on ES/NQ/GC/CL/ZN)
  4. Polymarket Gamma API       (macro/Fed/tariff/regulatory odds filtered by strict `max_spread <= 0.05`)

Evidence label: CONTEXT (macro regime & risk-sizing overlay; never claims directional forecast).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.paths import repo_path

DEFAULT_PLUMBING_CACHE = "data/macro_plumbing.json"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
CBOE_IDX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
POLYMARKET_URL = "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=40&order=volume24hr&ascending=false"
CFTC_COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json?$order=report_date_as_yyyy_mm_dd%20DESC&$limit=120"

MACRO_KEYWORDS = (
    "fed", "fomc", "rate", "cpi", "inflation", "recession", "tariff", "gdp",
    "treasury", "powell", "cut", "hike", "oil", "gold", "s&p", "nasdaq",
)


@dataclass
class MacroPlumbingSnapshot:
    updated_at: str = ""
    vix_curve: Dict[str, Optional[float]] = field(default_factory=dict)  # vix9d, vix, vix3m, vix6m
    vix_term_label: str = "UNKNOWN"
    fed_net_liquidity_b: Optional[float] = None
    fed_liq_chg_4w_b: Optional[float] = None
    fed_liq_chg_13w_b: Optional[float] = None
    fed_liq_as_of: str = ""
    polymarket_macro: List[Dict[str, Any]] = field(default_factory=list)
    cot_signals: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MacroPlumbingSnapshot":
        return cls(
            updated_at=str(raw.get("updated_at") or ""),
            vix_curve=dict(raw.get("vix_curve") or {}),
            vix_term_label=str(raw.get("vix_term_label") or "UNKNOWN"),
            fed_net_liquidity_b=raw.get("fed_net_liquidity_b"),
            fed_liq_chg_4w_b=raw.get("fed_liq_chg_4w_b"),
            fed_liq_chg_13w_b=raw.get("fed_liq_chg_13w_b"),
            fed_liq_as_of=str(raw.get("fed_liq_as_of") or ""),
            polymarket_macro=list(raw.get("polymarket_macro") or []),
            cot_signals=list(raw.get("cot_signals") or []),
            errors=list(raw.get("errors") or []),
        )


def _fetch_text(url: str, timeout: int = 25) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "options-screener/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return ""


def _fetch_yf_vix_fallback(missing_keys: List[str]) -> Dict[str, float]:
    """Fetch VIX curve points via yfinance (`^VIX9D`, `^VIX`, `^VIX3M`, `^VIX6M`) when CBOE CSV fails."""
    if not missing_keys:
        return {}
    yf_map = {"vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vix6m": "^VIX6M"}
    out: Dict[str, float] = {}
    try:
        import math
        import yfinance as yf

        for key in missing_keys:
            sym = yf_map.get(key)
            if not sym:
                continue
            try:
                hist = yf.Ticker(sym).history(period="5d")
                if hist is not None and not hist.empty:
                    val = float(hist["Close"].dropna().iloc[-1])
                    if math.isfinite(val) and val > 0:
                        out[key] = round(val, 2)
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


def parse_fred_csv(text: str) -> List[Tuple[str, float]]:
    """Parse FRED `DATE,VALUE` CSV dropping `.` missing entries."""
    out: List[Tuple[str, float]] = []
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    for row in reader:
        if len(row) < 2:
            continue
        d, raw_v = row[0].strip(), row[1].strip()
        if not d or not raw_v or raw_v == ".":
            continue
        try:
            out.append((d[:10], float(raw_v)))
        except ValueError:
            continue
    return out


def _fred_api_key() -> Optional[str]:
    key = (os.environ.get("FRED_API_KEY") or "").strip()
    return key or None


def parse_fred_api_json(text: str) -> List[Tuple[str, float]]:
    """Parse official FRED / ALFRED JSON `observations` into `[(date, value)]`."""
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    out: List[Tuple[str, float]] = []
    for rec in (payload or {}).get("observations") or []:
        d = str(rec.get("date") or "")[:10]
        raw_v = str(rec.get("value") or "").strip()
        if not d or not raw_v or raw_v == ".":
            continue
        try:
            out.append((d, float(raw_v)))
        except ValueError:
            continue
    return out


def fetch_fred_observations(
    series_id: str,
    *,
    api_key: Optional[str] = None,
    realtime_start: Optional[str] = None,
    realtime_end: Optional[str] = None,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
    limit: Optional[int] = None,
    sort_order: str = "asc",
    fetcher: Optional[Callable[[str], str]] = None,
) -> List[Tuple[str, float]]:
    """Fetch FRED observations via official JSON API when `FRED_API_KEY` is set, else CSV fallback.

    Pass `realtime_start` / `realtime_end` (YYYY-MM-DD) to query **ALFRED point-in-time vintages**
    as originally published on that date (zero look-ahead revision bias for backtesting).
    """
    get = fetcher or _fetch_text
    key = api_key if api_key is not None else _fred_api_key()
    if key:
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "sort_order": sort_order,
        }
        if realtime_start:
            params["realtime_start"] = realtime_start[:10]
        if realtime_end:
            params["realtime_end"] = realtime_end[:10]
        if observation_start:
            params["observation_start"] = observation_start[:10]
        if observation_end:
            params["observation_end"] = observation_end[:10]
        if limit is not None and limit > 0:
            params["limit"] = int(limit)
        url = f"{FRED_API_URL}?{urllib.parse.urlencode(params)}"
        pts = parse_fred_api_json(get(url))
        if pts:
            return sorted(pts) if sort_order == "desc" else pts
        if realtime_start or realtime_end:
            return []
    if realtime_start or realtime_end:
        raise RuntimeError("ALFRED point-in-time vintages require FRED_API_KEY")
    return parse_fred_csv(get(FRED_CSV_URL.format(series_id=series_id)))


def fetch_alfred_vintage(
    series_id: str,
    *,
    as_of_date: str,
    observation_start: Optional[str] = None,
    api_key: Optional[str] = None,
    fetcher: Optional[Callable[[str], str]] = None,
) -> List[Tuple[str, float]]:
    """Point-in-time ALFRED vintage of `series_id` exactly as known on `as_of_date`."""
    d = as_of_date[:10]
    return fetch_fred_observations(
        series_id,
        api_key=api_key,
        realtime_start=d,
        realtime_end=d,
        observation_start=observation_start,
        observation_end=d,
        fetcher=fetcher,
    )


def compute_fed_net_liquidity(
    walcl: List[Tuple[str, float]],
    tga: List[Tuple[str, float]],
    rrp: List[Tuple[str, float]],
) -> Dict[str, Any]:
    """Compute Fed Net Liquidity ($B) = WALCL($M)/1000 - WTREGEN($M)/1000 - RRPONTSYD($B).

    Aligns weekly WALCL dates against the most recent prior TGA and RRP prints (zero lookahead).
    """
    if not walcl or not tga or not rrp:
        return {}
    tga_s = sorted(tga)
    rrp_s = sorted(rrp)

    def _latest_le(series: List[Tuple[str, float]], d: str) -> Optional[float]:
        val = None
        for sd, sv in series:
            if sd <= d:
                val = sv
            else:
                break
        return val

    pts: List[Tuple[str, float]] = []
    for d, w_m in sorted(walcl)[-30:]:
        t_m = _latest_le(tga_s, d)
        r_b = _latest_le(rrp_s, d)
        if t_m is None or r_b is None:
            continue
        # WALCL & WTREGEN are reported in millions; RRPONTSYD is in billions
        net_b = (w_m / 1000.0) - (t_m / 1000.0) - r_b
        pts.append((d, round(net_b, 1)))

    if not pts:
        return {}
    cur_d, cur_v = pts[-1]
    chg_4w = round(cur_v - pts[-5][1], 1) if len(pts) >= 5 else None
    chg_13w = round(cur_v - pts[-14][1], 1) if len(pts) >= 14 else None
    return {
        "fed_net_liquidity_b": cur_v,
        "fed_liq_chg_4w_b": chg_4w,
        "fed_liq_chg_13w_b": chg_13w,
        "fed_liq_as_of": cur_d,
    }


def parse_polymarket_macro(
    markets: List[Dict[str, Any]],
    *,
    max_spread: float = 0.05,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """Filter Polymarket active markets to liquid macro catalysts enforcing `spread <= max_spread`."""
    out: List[Dict[str, Any]] = []
    for m in markets or []:
        q = str(m.get("question") or "").strip()
        ql = q.lower()
        if not q or not any(kw in ql for kw in MACRO_KEYWORDS):
            continue
        try:
            bid = float(m["bestBid"]) if m.get("bestBid") is not None else None
            ask = float(m["bestAsk"]) if m.get("bestAsk") is not None else None
        except (TypeError, ValueError):
            bid, ask = None, None
        # Strict honesty rule matching src/predmarkets/kalshi.py: refuse wide/one-sided books!
        if bid is None or ask is None or bid <= 0.0 or ask < bid or (ask - bid) > max_spread:
            continue
        mid_pct = round(0.5 * (bid + ask) * 100.0, 1)
        vol24 = float(m.get("volume24hr") or 0.0)
        out.append(
            {
                "question": q[:85],
                "prob_pct": mid_pct,
                "spread_pts": round((ask - bid) * 100.0, 1),
                "volume_24h": round(vol24, 0),
                "source": "Polymarket",
            }
        )
    out.sort(key=lambda x: -x["volume_24h"])
    return out[:limit]


def parse_cftc_cot(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract latest weekly CFTC COT non-commercial (speculative) net positioning for macro futures."""
    targets = {
        "E-MINI S&P 500": "SPY/ES",
        "NASDAQ-100": "QQQ/NQ",
        "GOLD": "GLD/GC",
        "CRUDE OIL, LIGHT SWEET": "XLE/CL",
        "10-YEAR U.S. TREASURY": "TLT/ZN",
    }
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows or []:
        name = str(r.get("market_and_exchange_names") or "").upper()
        matched_lbl = None
        for key, lbl in targets.items():
            if key in name:
                matched_lbl = lbl
                break
        if not matched_lbl or matched_lbl in seen:
            continue
        try:
            n_long = float(r.get("noncomm_positions_long_all") or 0.0)
            n_short = float(r.get("noncomm_positions_short_all") or 0.0)
            oi = float(r.get("open_interest_all") or 0.0)
        except (TypeError, ValueError):
            continue
        if oi <= 0:
            continue
        seen.add(matched_lbl)
        net_pct_oi = round((n_long - n_short) / oi * 100.0, 1)
        bias = "CROWDED LONG" if net_pct_oi >= 20.0 else ("CROWDED SHORT" if net_pct_oi <= -20.0 else "BALANCED")
        out.append(
            {
                "asset": matched_lbl,
                "report_date": str(r.get("report_date_as_yyyy_mm_dd") or "")[:10],
                "spec_net_pct_oi": net_pct_oi,
                "bias": bias,
                "source": "CFTC COT",
            }
        )
    return out


def _read_local_vix_csv(name: str) -> Optional[float]:
    p = Path(repo_path(f"data/discovery/{name}_History.csv"))
    if not p.is_file():
        return None
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for ln in reversed(lines[1:]):
            parts = [x.strip() for x in ln.split(",")]
            if len(parts) >= 2:
                val = parts[-1]
                try:
                    f = float(val)
                    if f > 0:
                        return round(f, 2)
                except ValueError:
                    continue
    except OSError:
        return None
    return None


def load_macro_plumbing(cache_path: str = DEFAULT_PLUMBING_CACHE) -> MacroPlumbingSnapshot:
    """Instant offline read of cached macro plumbing + local CBOE VIX CSVs."""
    full = Path(repo_path(cache_path))
    snap = MacroPlumbingSnapshot()
    if full.is_file():
        try:
            snap = MacroPlumbingSnapshot.from_dict(json.loads(full.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            snap = MacroPlumbingSnapshot()

    # Always hydrate VIX / VIX3M from local discovery CSVs if absent in cache
    for k, csv_name in (("vix", "VIX"), ("vix3m", "VIX3M")):
        if snap.vix_curve.get(k) is None:
            v = _read_local_vix_csv(csv_name)
            if v is not None:
                snap.vix_curve[k] = v
    if snap.vix_curve.get("vix") is not None and snap.vix_curve.get("vix3m") is not None:
        snap.vix_term_label = (
            "BACKWARDATION"
            if float(snap.vix_curve["vix"] or 0) > float(snap.vix_curve["vix3m"] or 0)
            else "CONTANGO"
        )
    return snap


def refresh_macro_plumbing(
    *,
    cache_path: str = DEFAULT_PLUMBING_CACHE,
    fetcher: Optional[Callable[[str], str]] = None,
) -> MacroPlumbingSnapshot:
    """Refresh 4-point CBOE VIX curve, FRED Fed Net Liquidity, CFTC COT & Polymarket macro odds."""
    prior = load_macro_plumbing(cache_path=cache_path)
    get = fetcher or _fetch_text
    snap = MacroPlumbingSnapshot(updated_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    # 1. CBOE 4-point VIX curve (with yfinance ^VIX* fallback when CBOE CDN returns 403)
    for key, cname in (("vix9d", "VIX9D"), ("vix", "VIX"), ("vix3m", "VIX3M"), ("vix6m", "VIX6M")):
        try:
            txt = get(CBOE_IDX_URL.format(name=cname))
            lines = [ln for ln in txt.splitlines()[1:] if ln.strip()]
            if lines:
                val = float(lines[-1].split(",")[-1].strip())
                snap.vix_curve[key] = round(val, 2)
        except Exception:
            local_v = _read_local_vix_csv(cname)
            if local_v is not None:
                snap.vix_curve[key] = local_v
    if fetcher is None:
        missing_vix = [k for k in ("vix9d", "vix", "vix3m", "vix6m") if snap.vix_curve.get(k) is None]
        if missing_vix:
            for k, v in _fetch_yf_vix_fallback(missing_vix).items():
                snap.vix_curve[k] = v
    for k in ("vix9d", "vix", "vix3m", "vix6m"):
        if snap.vix_curve.get(k) is None and prior.vix_curve.get(k) is not None:
            snap.vix_curve[k] = prior.vix_curve[k]
    if snap.vix_curve.get("vix") is not None and snap.vix_curve.get("vix3m") is not None:
        snap.vix_term_label = (
            "BACKWARDATION"
            if float(snap.vix_curve["vix"] or 0) > float(snap.vix_curve["vix3m"] or 0)
            else "CONTANGO"
        )

    # 2. FRED Fed Net Liquidity (WALCL - TGA - RRP); preserve prior cached value on transient timeout
    try:
        if fetcher is None and _fred_api_key():
            walcl = fetch_fred_observations("WALCL", limit=60, sort_order="desc")
            tga = fetch_fred_observations("WTREGEN", limit=60, sort_order="desc")
            rrp = fetch_fred_observations("RRPONTSYD", limit=180, sort_order="desc")
        else:
            walcl = parse_fred_csv(get(FRED_CSV_URL.format(series_id="WALCL")))
            tga = parse_fred_csv(get(FRED_CSV_URL.format(series_id="WTREGEN")))
            rrp = parse_fred_csv(get(FRED_CSV_URL.format(series_id="RRPONTSYD")))
        liq = compute_fed_net_liquidity(walcl, tga, rrp)
        snap.fed_net_liquidity_b = liq.get("fed_net_liquidity_b")
        snap.fed_liq_chg_4w_b = liq.get("fed_liq_chg_4w_b")
        snap.fed_liq_chg_13w_b = liq.get("fed_liq_chg_13w_b")
        snap.fed_liq_as_of = str(liq.get("fed_liq_as_of") or "")
    except Exception as exc:
        snap.errors.append(f"fred_liq: {exc}")
        if prior.fed_net_liquidity_b is not None:
            snap.fed_net_liquidity_b = prior.fed_net_liquidity_b
            snap.fed_liq_chg_4w_b = prior.fed_liq_chg_4w_b
            snap.fed_liq_chg_13w_b = prior.fed_liq_chg_13w_b
            snap.fed_liq_as_of = prior.fed_liq_as_of

    # 3. Polymarket Macro Odds (honest spread guard)
    try:
        raw_pm = json.loads(get(POLYMARKET_URL))
        if isinstance(raw_pm, list):
            snap.polymarket_macro = parse_polymarket_macro(raw_pm)
    except Exception as exc:
        snap.errors.append(f"polymarket: {exc}")
        if prior.polymarket_macro:
            snap.polymarket_macro = list(prior.polymarket_macro)

    # 4. CFTC COT Positioning
    try:
        raw_cot = json.loads(get(CFTC_COT_URL))
        if isinstance(raw_cot, list):
            snap.cot_signals = parse_cftc_cot(raw_cot)
    except Exception as exc:
        snap.errors.append(f"cftc_cot: {exc}")
        if prior.cot_signals:
            snap.cot_signals = list(prior.cot_signals)

    full = Path(repo_path(cache_path))
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(full) + ".tmp")
    tmp.write_text(json.dumps(snap.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, full)
    return snap
