"""Macro layer: real VIX term structure, cross-asset rotation ratios, optional FRED.

All descriptive. Nothing here is a tested timing signal:
`docs/discovery/research_20261124.json` registered "VIX → next-week market
direction" and it was NOT_ESTABLISHED; the rotation ratios and FRED series
have not been tested in this repo at all. They are context for reading the
sector board, labelled as such wherever they are shown.

Sources
  * VIX / VIX3M / VVIX / SKEW: Cboe daily index histories archived at
    `data/discovery/<NAME>_History.csv` (refreshed by the discovery-coverage job).
  * Ratios: adjusted daily closes in `data/equity_ohlcv.db` (Yahoo).
  * FRED: https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>, cached in
    `data/fred_cache.json` by an explicit refresh only (never on a render path).
    FRED serves the *latest revised* history — fine for "where are we now",
    not a point-in-time series for backtests.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

CBOE_DIR = "data/discovery"
FRED_CACHE = "data/fred_cache.json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# (numerator, denominator, label, what a RISE means)
ROTATION_PAIRS: Tuple[Tuple[str, str, str, str], ...] = (
    ("RSP", "SPY", "Equal- vs cap-weight S&P 500", "broader participation"),
    ("IWM", "SPY", "Small caps vs S&P 500", "risk appetite / domestic cyclicals"),
    ("SMH", "SPY", "Semiconductors vs market", "cyclical-tech leadership"),
    ("XLY", "XLP", "Discretionary vs staples", "consumer risk appetite"),
    ("HYG", "IEF", "High-yield credit vs 7-10y Treasuries", "credit risk appetite"),
    ("HYG", "LQD", "High-yield vs investment-grade credit", "credit quality risk-on"),
    ("SPHB", "SPLV", "High beta vs low volatility", "risk appetite"),
    ("CPER", "GLD", "Copper vs gold", "growth over safety"),
    ("EEM", "SPY", "Emerging markets vs US", "global / weaker-dollar leadership"),
)

FRED_SERIES: Tuple[Tuple[str, str, str], ...] = (
    ("BAMLH0A0HYM2", "US high-yield OAS", "%"),
    ("T10Y2Y", "10y minus 2y Treasury", "%"),
    ("T10Y3M", "10y minus 3m Treasury", "%"),
    ("NFCI", "Chicago Fed financial conditions (weekly; >0 tighter than avg)", "idx"),
    ("VIXCLS", "Cboe VIX daily close", "pts"),
)


# ── VIX term structure ───────────────────────────────────────────────────────


def load_cboe_closes(name: str, base_dir: str = CBOE_DIR) -> List[Tuple[str, float]]:
    """[(YYYY-MM-DD, close)] from a Cboe `<NAME>_History.csv` (either OHLC or single-column)."""
    path = repo_path(os.path.join(base_dir, f"{name}_History.csv"))
    if not os.path.exists(path):
        return []
    out: List[Tuple[str, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return []
        cols = [h.strip().upper() for h in header]
        ci = cols.index("CLOSE") if "CLOSE" in cols else (len(cols) - 1)
        for row in reader:
            if len(row) <= ci:
                continue
            try:
                d = dt.datetime.strptime(row[0].strip(), "%m/%d/%Y").date().isoformat()
                v = float(row[ci])
            except (ValueError, IndexError):
                continue
            if math.isfinite(v):
                out.append((d, v))
    out.sort()
    return out


def _pctile(values: Sequence[float], x: float) -> Optional[float]:
    if not values:
        return None
    return round(100.0 * sum(1 for v in values if v <= x) / len(values), 0)


def vol_regime(*, base_dir: str = CBOE_DIR, today: Optional[dt.date] = None, stale_days: int = 5) -> Dict[str, Any]:
    """Real VIX / VIX3M / VVIX / SKEW: last close, 1y percentile, term ratio, as-of."""
    today = today or dt.date.today()
    out: Dict[str, Any] = {"source": f"Cboe daily histories ({base_dir}/*_History.csv)"}
    last_dates: List[str] = []
    for name in ("VIX", "VIX3M", "VVIX", "SKEW"):
        series = load_cboe_closes(name, base_dir)
        key = name.lower()
        if not series:
            out[key] = None
            continue
        d, v = series[-1]
        trail = [x for _, x in series[-252:]]
        out[key] = round(v, 2)
        out[f"{key}_pctile_1y"] = _pctile(trail, v)
        out[f"{key}_as_of"] = d
        last_dates.append(d)
    vix, v3 = out.get("vix"), out.get("vix3m")
    if vix is not None and v3:
        ratio = float(vix) / float(v3)
        out["term_ratio"] = round(ratio, 3)
        out["term"] = "BACKWARDATION" if ratio > 1.0 else "CONTANGO"
    else:
        out["term_ratio"] = None
        out["term"] = "UNKNOWN"
    out["as_of"] = min(last_dates) if last_dates else None
    out["stale"] = bool(
        out["as_of"] is None or (today - dt.date.fromisoformat(str(out["as_of"]))).days > stale_days
    )
    return out


# ── Rotation ratios ──────────────────────────────────────────────────────────


@dataclass
class RatioReading:
    num: str
    den: str
    label: str
    rise_means: str
    as_of: Optional[str] = None
    chg_21d_pct: Optional[float] = None
    chg_63d_pct: Optional[float] = None
    vs_200d_pct: Optional[float] = None
    pctile_1y: Optional[float] = None
    missing: Optional[str] = None

    @property
    def name(self) -> str:
        return f"{self.num}/{self.den}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name
        return d


def ratio_reading(ohlcv: OhlcvMap, num: str, den: str, label: str, rise_means: str) -> RatioReading:
    r = RatioReading(num=num, den=den, label=label, rise_means=rise_means)
    lacking = [s for s in (num, den) if s not in ohlcv]
    if lacking:
        r.missing = f"{'/'.join(lacking)} not in price archive (opt refresh --expand)"
        return r
    nd, nc, _ = ohlcv[num]
    dd, dc, _ = ohlcv[den]
    dmap = {d: float(c) for d, c in zip(dd, dc)}
    pairs = [(d, float(c) / dmap[d]) for d, c in zip(nd, nc) if d in dmap and dmap[d] > 0 and float(c) > 0]
    if len(pairs) < 64:
        r.missing = f"only {len(pairs)} common sessions"
        return r
    vals: np.ndarray = np.asarray([p[1] for p in pairs], dtype=float)
    r.as_of = pairs[-1][0]
    r.chg_21d_pct = round((vals[-1] / vals[-22] - 1.0) * 100.0, 2)
    r.chg_63d_pct = round((vals[-1] / vals[-64] - 1.0) * 100.0, 2)
    if len(vals) >= 200:
        r.vs_200d_pct = round((vals[-1] / float(np.mean(vals[-200:])) - 1.0) * 100.0, 2)
    r.pctile_1y = _pctile(vals[-252:].tolist(), float(vals[-1]))
    return r


# ── FRED (explicit refresh only) ─────────────────────────────────────────────


def fetch_fred_series(series_id: str, *, timeout: float = 12.0) -> List[Tuple[str, float]]:
    """Download one FRED series as [(date, value)] (missing '.' values dropped).

    Prefers the official FRED JSON API when `FRED_API_KEY` is configured (avoids
    `fredgraph.csv` edge timeouts), falling back seamlessly to CSV when unset.
    """
    api_key = (os.environ.get("FRED_API_KEY") or "").strip()
    if api_key:
        import urllib.parse

        q = urllib.parse.urlencode(
            {"series_id": series_id, "api_key": api_key, "file_type": "json", "sort_order": "desc", "limit": 900}
        )
        try:
            req = urllib.request.Request(f"{FRED_API_URL}?{q}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            api_out: List[Tuple[str, float]] = []
            for rec in (payload or {}).get("observations") or []:
                d = str(rec.get("date") or "")[:10]
                raw_v = str(rec.get("value") or "").strip()
                if not d or raw_v in ("", "."):
                    continue
                try:
                    api_out.append((d, float(raw_v)))
                except ValueError:
                    continue
            if api_out:
                api_out.sort()
                return api_out
        except Exception:
            pass

    url = FRED_URL.format(sid=series_id)
    # FRED's edge drops some custom User-Agents (connection timeout); the stock urllib header works.
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        text = resp.read().decode("utf-8", errors="replace")
    out: List[Tuple[str, float]] = []
    for line in text.splitlines()[1:]:
        parts = line.strip().split(",")
        if len(parts) < 2 or parts[1] in ("", "."):
            continue
        try:
            out.append((parts[0][:10], float(parts[1])))
        except ValueError:
            continue
    return out


def refresh_fred_cache(path: str = FRED_CACHE, *, keep: int = 800) -> Dict[str, Any]:
    """Fetch every FRED_SERIES and write the cache. Returns {series: ok|error}."""
    full = repo_path(path)
    cache = load_fred_cache(path)
    status: Dict[str, Any] = {}
    fetched_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for sid, _label, _unit in FRED_SERIES:
        try:
            obs = fetch_fred_series(sid)
            if not obs:
                raise ValueError("empty series")
            cache[sid] = {"fetched_at": fetched_at, "url": FRED_URL.format(sid=sid), "obs": obs[-keep:]}
            status[sid] = "ok"
        except Exception as exc:  # network/proxy/HTTP — keep the previous cache entry
            status[sid] = f"error: {str(exc)[:80]}"
    tmp = full + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh)
    os.replace(tmp, full)
    return status


def load_fred_cache(path: str = FRED_CACHE) -> Dict[str, Any]:
    full = repo_path(path)
    if not os.path.exists(full):
        return {}
    try:
        with open(full, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def fred_readings(cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sid, label, unit in FRED_SERIES:
        ent = cache.get(sid) or {}
        obs = [(str(d), float(v)) for d, v in (ent.get("obs") or [])]
        if not obs:
            rows.append({"series": sid, "label": label, "unit": unit, "value": None,
                         "missing": "not cached (opt discover --refresh-macro)"})
            continue
        d, v = obs[-1]
        cutoff = (dt.date.fromisoformat(d) - dt.timedelta(days=30)).isoformat()
        prior = [x for dd, x in obs if dd <= cutoff]
        yr = [x for dd, x in obs if dd >= (dt.date.fromisoformat(d) - dt.timedelta(days=365)).isoformat()]
        rows.append({
            "series": sid, "label": label, "unit": unit, "value": round(v, 3), "as_of": d,
            "chg_1m": round(v - prior[-1], 3) if prior else None,
            "pctile_1y": _pctile(yr, v),
            "fetched_at": ent.get("fetched_at"), "source": ent.get("url"),
        })
    return rows


# ── Snapshot ─────────────────────────────────────────────────────────────────


@dataclass
class MacroSnapshot:
    vol: Dict[str, Any]
    spy: Dict[str, Any]
    ratios: List[RatioReading] = field(default_factory=list)
    fred: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def ratio_tally(self) -> Tuple[int, int]:
        """(# ratios rising over 63d, # measurable). A descriptive count — NOT a tested signal."""
        live = [r for r in self.ratios if r.chg_63d_pct is not None]
        return sum(1 for r in live if (r.chg_63d_pct or 0.0) > 0), len(live)

    def to_dict(self) -> Dict[str, Any]:
        up, n = self.ratio_tally()
        return {
            "vol": self.vol, "spy": self.spy,
            "ratios": [r.to_dict() for r in self.ratios],
            "ratio_tally": {"rising_63d": up, "measurable": n},
            "fred": self.fred, "notes": list(self.notes),
        }


def spy_state(ohlcv: OhlcvMap) -> Dict[str, Any]:
    if "SPY" not in ohlcv or len(ohlcv["SPY"][1]) < 200:
        return {"as_of": None}
    ds, cs, _ = ohlcv["SPY"]
    c: np.ndarray = np.asarray(cs, dtype=float)
    sma200 = float(np.mean(c[-200:]))
    lr = np.diff(np.log(c[-21:]))
    rv20 = float(np.std(lr, ddof=1) * math.sqrt(252.0)) if len(lr) > 2 else None
    hi52 = float(np.max(c[-252:]))
    return {
        "as_of": str(ds[-1]),
        "last": round(float(c[-1]), 2),
        "vs_200d_pct": round((float(c[-1]) / sma200 - 1.0) * 100.0, 2),
        "above_200d": bool(c[-1] > sma200),
        "rv20_pct": None if rv20 is None else round(rv20 * 100.0, 1),
        "from_52w_high_pct": round((float(c[-1]) / hi52 - 1.0) * 100.0, 2),
        "ret_21d_pct": round((float(c[-1]) / float(c[-22]) - 1.0) * 100.0, 2),
        "source": "data/equity_ohlcv.db (Yahoo adjusted daily closes)",
    }


def build_macro_snapshot(ohlcv: OhlcvMap, *, fred_path: str = FRED_CACHE, cboe_dir: str = CBOE_DIR) -> MacroSnapshot:
    snap = MacroSnapshot(
        vol=vol_regime(base_dir=cboe_dir),
        spy=spy_state(ohlcv),
        ratios=[ratio_reading(ohlcv, n, d, lab, rise) for n, d, lab, rise in ROTATION_PAIRS],
        fred=fred_readings(load_fred_cache(fred_path)),
    )
    snap.notes.append(
        "Context only. VIX→direction was registered and NOT_ESTABLISHED (docs/discovery/research_20261124.json); "
        "the ratios and FRED series are untested here."
    )
    if snap.vol.get("stale"):
        snap.notes.append(f"Cboe VIX files are stale (as of {snap.vol.get('as_of')}).")
    return snap


__all__ = [
    "ROTATION_PAIRS", "FRED_SERIES", "RatioReading", "MacroSnapshot",
    "load_cboe_closes", "vol_regime", "ratio_reading", "spy_state",
    "fetch_fred_series", "refresh_fred_cache", "load_fred_cache", "fred_readings",
    "build_macro_snapshot",
]
