"""Market snapshots for the holdings desk.

snapshot_from_closes is the pure, tested core. fetch_snapshots is the thin
network wrapper: ONE batched yf.download for the whole plan (a single HTTP
call — the banner path must never fan out per-ticker)."""
import logging
import math
from typing import Dict, List, Optional

from .zones import Snapshot

_MIN_CLOSES = 30
_SIGMA_WINDOW = 63

log = logging.getLogger(__name__)


def snapshot_from_closes(ticker: str, closes: List[float]) -> Optional[Snapshot]:
    clean = [float(c) for c in closes if c is not None and math.isfinite(float(c)) and c > 0]
    if len(clean) < _MIN_CLOSES:
        return None
    spot = clean[-1]
    tail = clean[-(_SIGMA_WINDOW + 1):]
    rets = [math.log(tail[i] / tail[i - 1]) for i in range(1, len(tail))]
    mean = sum(rets) / len(rets)
    daily_sigma = math.sqrt(sum((r - mean) ** 2 for r in rets) / len(rets))
    ma200 = (sum(clean[-200:]) / 200.0) if len(clean) >= 200 else None
    return Snapshot(ticker=ticker.upper(), spot=spot, high_52w=max(clean),
                    low_52w=min(clean), ma200=ma200, daily_sigma=daily_sigma,
                    closes=clean)


def fetch_snapshots(tickers: List[str]) -> Dict[str, Snapshot]:
    tickers = [t.upper() for t in tickers]
    if not tickers:
        return {}
    import yfinance as yf
    try:
        frame = yf.download(tickers, period="1y", auto_adjust=True,
                            progress=False, group_by="ticker", threads=True)
    except Exception as exc:
        log.debug("longterm fetch failed: %s", exc)
        return {}
    out: Dict[str, Snapshot] = {}
    for t in tickers:
        try:
            closes = (frame[t]["Close"] if len(tickers) > 1 else frame["Close"]).dropna()
            snap = snapshot_from_closes(t, list(closes))
            if snap:
                out[t] = snap
        except Exception as exc:
            log.debug("longterm snapshot skipped for %s: %s", t, exc)
    return out
