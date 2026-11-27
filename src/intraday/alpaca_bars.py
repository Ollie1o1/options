"""Download Alpaca 1-minute stock bars into monthly Parquet files.

Credentials come from APCA_API_KEY_ID / APCA_API_SECRET_KEY in the environment,
falling back to the repo's `.env`. A paper-account key is enough for market data.

    python -m src.intraday.alpaca_bars --symbols SPY QQQ IWM DIA --start 2016-01 --end 2026-11

Months already on disk are skipped unless they could still be incomplete (the
current month, or a month whose file was written before that month ended).
Only 04:00-16:00 ET bars are kept (premarket + regular session).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "intraday"
ET = "America/New_York"
KEEP_START_MIN = 4 * 60     # 04:00 ET
KEEP_END_MIN = 16 * 60      # bars starting before 16:00 ET
COLUMNS = ["t", "o", "h", "l", "c", "v", "n", "vw"]


def load_credentials(env_path: Optional[Path] = None) -> Tuple[str, str]:
    """Return (key_id, secret) from the environment, else from `.env`."""
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if key and secret:
        return key, secret
    path = env_path or (REPO_ROOT / ".env")
    values: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    key = key or values.get("APCA_API_KEY_ID")
    secret = secret or values.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set (environment or .env)")
    return key, secret


def month_range(start: str, end: str) -> List[Tuple[int, int]]:
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds_utc(year: int, month: int) -> Tuple[str, str]:
    start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
    nxt = dt.datetime(year + (month == 12), 1 if month == 12 else month + 1, 1, tzinfo=dt.timezone.utc)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), nxt.strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp_end(end_iso: str, now: Optional[dt.datetime] = None, delay_min: int = 16) -> str:
    """Free-plan SIP data refuses the last 15 minutes: cap `end` at now - delay."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cap = (now - dt.timedelta(minutes=delay_min)).replace(second=0, microsecond=0)
    end = dt.datetime.strptime(end_iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return min(end, cap).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_bars(get: Callable, headers: Dict[str, str], symbol: str, start: str, end: str,
               feed: str = "sip", max_retries: int = 6, pause: float = 0.31) -> pd.DataFrame:
    """All 1-minute bars for one symbol in [start, end), following page tokens.

    `get` is `requests.get`-compatible (injected so tests need no network).
    Retries 429/5xx with exponential backoff.
    """
    params = {"symbols": symbol, "timeframe": "1Min", "start": start, "end": end,
              "feed": feed, "adjustment": "raw", "limit": 10000, "sort": "asc"}
    rows: List[dict] = []
    token: Optional[str] = None
    while True:
        if token:
            params["page_token"] = token
        else:
            params.pop("page_token", None)
        for attempt in range(max_retries):
            resp = get(DATA_URL, headers=headers, params=dict(params), timeout=60)
            if resp.status_code == 200:
                break
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            raise RuntimeError(f"Alpaca {resp.status_code} for {symbol} {start}: {resp.text[:200]}")
        else:
            raise RuntimeError(f"Alpaca kept failing for {symbol} {start} after {max_retries} tries")
        body = resp.json()
        rows.extend((body.get("bars") or {}).get(symbol) or [])
        token = body.get("next_page_token")
        if pause:
            time.sleep(pause)
        if not token:
            break
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


def keep_session_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars starting in [04:00, 16:00) ET."""
    if df.empty:
        return df
    local = df["t"].dt.tz_convert(ET)
    minute = local.dt.hour * 60 + local.dt.minute
    return df[(minute >= KEEP_START_MIN) & (minute < KEEP_END_MIN)].reset_index(drop=True)


def month_path(out_dir: Path, symbol: str, year: int, month: int) -> Path:
    return out_dir / symbol / f"{year:04d}-{month:02d}.parquet"


def needs_download(path: Path, year: int, month: int, today: dt.date) -> bool:
    if not path.exists():
        return True
    next_month = dt.date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    written = dt.datetime.fromtimestamp(path.stat().st_mtime).date()
    # A file written before the month was over may be missing its last days.
    return written <= next_month and today > written


def download(symbols: List[str], start: str, end: str, out_dir: Path = DEFAULT_OUT,
             feed: str = "sip", get: Optional[Callable] = None,
             log: Callable[[str], None] = print) -> Dict[str, int]:
    if get is None:
        import requests
        get = requests.get
    key, secret = load_credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    today = dt.date.today()
    counts: Dict[str, int] = {}
    for symbol in symbols:
        total = 0
        for year, month in month_range(start, end):
            path = month_path(out_dir, symbol, year, month)
            if not needs_download(path, year, month, today):
                continue
            s, e = month_bounds_utc(year, month)
            e = clamp_end(e)
            if e <= s:
                continue
            df = keep_session_window(fetch_bars(get, headers, symbol, s, e, feed=feed))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, path)
            total += len(df)
            log(f"{symbol} {year}-{month:02d}: {len(df):,} bars")
        counts[symbol] = total
    return counts


CALENDAR_URL = "https://paper-api.alpaca.markets/v2/calendar"


def fetch_calendar(start: str = "2016-01-01", end: Optional[str] = None,
                   out_dir: Path = DEFAULT_OUT, get: Optional[Callable] = None,
                   refresh: bool = False) -> pd.DataFrame:
    """Exchange calendar (date, open, close as 'HH:MM'), cached as calendar.parquet."""
    path = out_dir / "calendar.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if get is None:
        import requests
        get = requests.get
    key, secret = load_credentials()
    end = end or (dt.date.today() + dt.timedelta(days=366)).isoformat()
    resp = get(CALENDAR_URL, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
               params={"start": start, "end": end}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Alpaca calendar {resp.status_code}: {resp.text[:200]}")
    cal = pd.DataFrame(resp.json())[["date", "open", "close"]]
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    path.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(path, index=False)
    return cal


def load_symbol(symbol: str, out_dir: Path = DEFAULT_OUT) -> pd.DataFrame:
    """All stored bars for a symbol, sorted, with `t` as UTC timestamps."""
    files = sorted((out_dir / symbol).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("t").drop_duplicates("t").reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y-%m"), help="YYYY-MM")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--feed", default="sip")
    a = ap.parse_args(argv)
    counts = download([s.upper() for s in a.symbols], a.start, a.end, Path(a.out), a.feed)
    for sym, n in counts.items():
        print(f"{sym}: {n:,} new bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
