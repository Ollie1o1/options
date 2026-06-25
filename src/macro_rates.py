"""Hard macro / rates ingestion from FRED (no API key required).

FRED exposes every series as a plain CSV via the public graph endpoint:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10

No key, no auth, daily refresh. We pull the handful of daily series that
ground the screener's macro view in actual rates/curve/risk rather than
news sentiment:

    DGS10   — 10-year Treasury constant maturity yield (%)
    DGS2    — 2-year Treasury yield (%)
    T10Y2Y  — 10y minus 2y spread (percentage points; <0 = inverted curve)
    DFF     — effective federal funds rate (%)
    VIXCLS  — CBOE VIX close

Everything is best-effort: a missing series renders as N/A, never raises.
Results are disk-cached for a few hours (rates update at most daily).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

SERIES = ("DGS10", "DGS2", "T10Y2Y", "DFF", "VIXCLS")

_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                           "macro_rates_cache.json")
_CACHE_TTL_S = 6 * 3600


@dataclass
class RatesSnapshot:
    dgs10: Optional[float]
    dgs2: Optional[float]
    t10y2y: Optional[float]
    dff: Optional[float]
    vixcls: Optional[float]
    as_of: dict = field(default_factory=dict)
    # Populated only by the yfinance fallback (FRED has no key but its CSV
    # endpoint times out from some networks). Yahoo has no clean 2Y, so the
    # fallback reports the 3-month bill and the 10y-3m slope instead — which
    # is the curve inversion signal the Fed actually favours.
    dgs3mo: Optional[float] = None
    t10y3m: Optional[float] = None
    source: str = "FRED"


def parse_fred_csv(text: str) -> list:
    """Parse a FRED graph CSV into ``[(date, value_or_None), ...]``.

    Skips the header row (``observation_date,<series>`` or legacy
    ``DATE,VALUE``). FRED encodes missing observations as ``.``.
    """
    out: list = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "," not in line:
            continue
        date_str, _, val_str = line.partition(",")
        date_str = date_str.strip()
        val_str = val_str.strip()
        # Skip header: first column is not a date.
        if not (len(date_str) >= 8 and date_str[:4].isdigit() and "-" in date_str):
            continue
        if val_str in (".", "", "NaN"):
            out.append((date_str, None))
            continue
        try:
            out.append((date_str, float(val_str)))
        except ValueError:
            out.append((date_str, None))
    return out


def latest_valid(observations: list) -> tuple:
    """Return the most recent ``(date, value)`` whose value is not missing."""
    for date_str, value in reversed(observations):
        if value is not None:
            return (date_str, value)
    return (None, None)


_BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/124.0 Safari/537.36")

_SESSION = None


def _get_session():
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
        _SESSION.headers.update({"User-Agent": _BROWSER_UA})
    return _SESSION


def _http_fetch(series_id: str) -> str:
    # FRED throttles non-browser User-Agents (the custom UA hangs/times out)
    # and rate-limits rapid bursts from one IP. Present a browser UA over a
    # reused session. Single attempt with a tight timeout: this feeds a
    # startup panel, so a throttled FRED must never stall the dashboard
    # (fetch_rates_snapshot also fail-fasts after the first failure).
    sess = _get_session()
    resp = sess.get(_FRED_CSV.format(id=series_id), timeout=6)
    resp.raise_for_status()
    return resp.text


def _yahoo_fetch(symbol: str) -> Optional[float]:
    """Latest close for a Yahoo index symbol, or None. Best-effort, no raise."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period="5d")
        series = hist["Close"].dropna()
        if len(series):
            return float(series.iloc[-1])
    except Exception:
        pass
    return None


def yahoo_rates_fallback(fetcher: Callable[[str], Optional[float]] = _yahoo_fetch) -> RatesSnapshot:
    """Build a rates snapshot from yfinance Treasury indices when FRED is down.

    CBOE yield indices (^TNX/^IRX/^FVX/^TYX) quote 10x the percentage, so they
    are divided by 10. ^VIX is already the index level. Yahoo has no 2-year, so
    we report the 3-month bill and the 10y-3m slope.
    """
    def pct(sym):
        v = fetcher(sym)
        if v is None:
            return None
        # ^TNX & friends sometimes quote 10x the percent (CBOE convention,
        # e.g. 43.9 == 4.39%) and sometimes already-in-percent (4.39),
        # depending on the yfinance build. No Treasury yield is ~>20%, so a
        # value above 20 is the 10x form and gets scaled down.
        return round(v / 10.0 if v > 20 else v, 4)

    dgs10 = pct("^TNX")
    dgs3mo = pct("^IRX")
    vix = fetcher("^VIX")
    t10y3m = round(dgs10 - dgs3mo, 4) if (dgs10 is not None and dgs3mo is not None) else None
    stamp = time.strftime("%Y-%m-%d")
    as_of = {k: stamp for k, v in (("DGS10", dgs10), ("DGS3MO", dgs3mo),
                                   ("VIX", vix)) if v is not None}
    return RatesSnapshot(
        dgs10=dgs10, dgs2=None, t10y2y=None, dff=None,
        vixcls=(round(float(vix), 2) if vix is not None else None),
        dgs3mo=dgs3mo, t10y3m=t10y3m, source="yahoo", as_of=as_of,
    )


def _read_cache() -> Optional[dict]:
    try:
        with open(_CACHE_PATH) as fh:
            blob = json.load(fh)
        if time.time() - blob.get("_ts", 0) <= _CACHE_TTL_S:
            return blob
    except Exception:
        pass
    return None


def _write_cache(snap: RatesSnapshot) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w") as fh:
            json.dump({
                "_ts": time.time(),
                "dgs10": snap.dgs10, "dgs2": snap.dgs2, "t10y2y": snap.t10y2y,
                "dff": snap.dff, "vixcls": snap.vixcls, "as_of": snap.as_of,
                "dgs3mo": snap.dgs3mo, "t10y3m": snap.t10y3m, "source": snap.source,
            }, fh)
    except Exception:
        pass


def fetch_rates_snapshot(fetcher: Callable[[str], str] = _http_fetch,
                         use_cache: bool = True,
                         yahoo_fetcher: Callable[[str], Optional[float]] = _yahoo_fetch
                         ) -> RatesSnapshot:
    """Fetch the latest value of each macro series into a RatesSnapshot.

    ``fetcher`` is injected for testing; it maps a series id to raw CSV text.
    Any series that fails to fetch or has no valid observation is left None.
    When FRED yields nothing at all, fall back to yfinance Treasury yields via
    ``yahoo_fetcher`` so the panel still populates.
    """
    if use_cache:
        cached = _read_cache()
        if cached is not None:
            return RatesSnapshot(
                dgs10=cached.get("dgs10"), dgs2=cached.get("dgs2"),
                t10y2y=cached.get("t10y2y"), dff=cached.get("dff"),
                vixcls=cached.get("vixcls"), as_of=cached.get("as_of", {}),
                dgs3mo=cached.get("dgs3mo"), t10y3m=cached.get("t10y3m"),
                source=cached.get("source", "FRED"),
            )

    values: dict = {}
    as_of: dict = {}
    for i, sid in enumerate(SERIES):
        if i and fetcher is _http_fetch:
            time.sleep(0.4)  # stay under FRED's burst throttle
        try:
            date_str, value = latest_valid(parse_fred_csv(fetcher(sid)))
        except Exception:
            # Source unreachable. If even the first series fails, FRED is down
            # or throttling us — abort rather than wait on every timeout.
            if i == 0:
                break
            date_str, value = (None, None)
        values[sid] = value
        if date_str is not None:
            as_of[sid] = date_str

    snap = RatesSnapshot(
        dgs10=values.get("DGS10"), dgs2=values.get("DGS2"),
        t10y2y=values.get("T10Y2Y"), dff=values.get("DFF"),
        vixcls=values.get("VIXCLS"), as_of=as_of,
    )
    # FRED gave us nothing — fall back to yfinance Treasury yields so the panel
    # still populates (FRED's CSV endpoint times out from some networks).
    has_any = any(v is not None for v in
                  (snap.dgs10, snap.dgs2, snap.t10y2y, snap.dff, snap.vixcls))
    if not has_any:
        try:
            snap = yahoo_rates_fallback(yahoo_fetcher)
            has_any = any(v is not None for v in (snap.dgs10, snap.dgs3mo, snap.vixcls))
        except Exception:
            pass

    # Never cache a total failure — a transient outage must not poison the
    # panel for the whole TTL window.
    if use_cache and has_any:
        _write_cache(snap)
    return snap


def _fmt(value: Optional[float], suffix: str = "%", nd: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{nd}f}{suffix}"


def format_rates_panel(snap: RatesSnapshot, width: int = 100) -> list:
    """Render the macro snapshot as factual, display-ready lines."""
    try:
        from src import ui  # optional pretty rule
        header = ui.rule(width, title="MACRO / RATES (FRED)")
    except Exception:
        header = "  MACRO / RATES (FRED)"

    as_of_dates = [d for d in snap.as_of.values() if d]
    stamp = max(as_of_dates) if as_of_dates else "n/a"

    curve_note = ""
    if snap.t10y2y is not None:
        curve_note = "  (inverted)" if snap.t10y2y < 0 else "  (normal)"

    lines = [header]
    if all(v is None for v in (snap.dgs10, snap.dgs2, snap.t10y2y,
                               snap.dff, snap.vixcls, snap.dgs3mo)):
        lines.append("  rates: N/A (FRED + Yahoo unreachable)")
        return lines

    if snap.source == "yahoo":
        # Yahoo has no clean 2Y; report the 3M bill and the 10y-3m slope.
        slope_note = ""
        if snap.t10y3m is not None:
            slope_note = "  (inverted)" if snap.t10y3m < 0 else "  (normal)"
        lines.append(
            f"  10Y {_fmt(snap.dgs10)}   3M {_fmt(snap.dgs3mo)}   "
            f"10y-3m {_fmt(snap.t10y3m, suffix='pp')}{slope_note}"
        )
        lines.append(
            f"  VIX {_fmt(snap.vixcls, suffix='', nd=1)}   "
            f"as of {stamp}  (via Yahoo — FRED unreachable)"
        )
        return lines

    lines.append(
        f"  10Y {_fmt(snap.dgs10)}   2Y {_fmt(snap.dgs2)}   "
        f"2s10s {_fmt(snap.t10y2y, suffix='pp')}{curve_note}"
    )
    lines.append(
        f"  Fed funds {_fmt(snap.dff)}   VIX {_fmt(snap.vixcls, suffix='', nd=1)}   "
        f"as of {stamp}"
    )
    return lines


def print_macro_rates(width: int = 100) -> None:
    """Fetch and print the macro/rates panel; silent on total failure."""
    try:
        snap = fetch_rates_snapshot()
        for line in format_rates_panel(snap, width=width):
            print(line)
    except Exception:
        pass
