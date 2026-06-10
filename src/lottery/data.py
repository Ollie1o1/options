"""Option-data layer for the lottery sleeve — provider-agnostic by design.

Free now, paid later. The backtest's biggest uncertainty was the *guessed*
implied-vol model (a flat vol-risk-premium + skew). This module fetches REAL
current option chains (free, via yfinance — real IV, bid/ask, OI) and measures
the true VRP and skew, which are fed into the backtest in place of the guesses.

A `PolygonProvider` stub implements the same interface so a paid historical feed
(true per-contract premium history) can drop in without touching callers — just
subscribe, set POLYGON_API_KEY, and request provider="polygon".

Interface (all providers):
    get_chain(ticker, target_dte) -> {
        "spot": float, "dte": int, "t_years": float,
        "calls": [{"strike","iv","bid","ask","open_interest","last"}],
        "puts":  [ ... ],
    }
Only paid providers implement historical premium lookups; the free provider
returns the *current* surface, which is enough to calibrate the model.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional


# ── calibration math (pure) ────────────────────────────────────────────────────
def atm_iv_from_chain(chain: List[Dict[str, Any]], spot: float) -> Optional[float]:
    """IV of the strike nearest spot."""
    valid = [c for c in chain if c.get("iv") and c.get("strike")]
    if not valid:
        return None
    nearest = min(valid, key=lambda c: abs(c["strike"] - spot))
    return float(nearest["iv"])


def measure_vrp(atm_iv: float, realized_vol: float) -> Optional[float]:
    """Volatility risk premium = ATM implied / realized − 1."""
    if not realized_vol or realized_vol <= 0 or not atm_iv:
        return None
    return atm_iv / realized_vol - 1.0


def measure_skew_per_sigma(
    chain: List[Dict[str, Any]], spot: float, atm_iv: float, t_years: float
) -> Optional[float]:
    """Least-squares slope of (iv/atm_iv − 1) vs sigma-OTM over the OTM wing.

    sigma_otm = ln(strike/spot) / (atm_iv*sqrt(T)). Returns IV markup per sigma.
    """
    if not atm_iv or atm_iv <= 0 or not t_years or t_years <= 0 or not spot:
        return None
    unit = atm_iv * math.sqrt(t_years)
    if unit <= 0:
        return None
    xs, ys = [], []
    for c in chain:
        k, iv = c.get("strike"), c.get("iv")
        if not k or not iv or k <= spot:
            continue  # OTM-call wing only
        sigma_otm = math.log(k / spot) / unit
        if sigma_otm <= 0 or sigma_otm > 4:
            continue
        xs.append(sigma_otm)
        ys.append(iv / atm_iv - 1.0)
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def _median(vals: List[float]) -> Optional[float]:
    vals = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vals:
        return None
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2.0


def calibrate_from_chains(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate real VRP and skew across per-ticker chain samples (robust median).

    Each sample: {atm_iv, realized_vol, spot, t_years, calls:[...]}.
    Returns {vrp, skew_per_sigma, n_samples}. Falls back to sensible defaults
    when a measure can't be taken.
    """
    vrps, skews = [], []
    for s in samples:
        vrps.append(measure_vrp(s.get("atm_iv"), s.get("realized_vol")))
        skews.append(measure_skew_per_sigma(
            s.get("calls") or [], s.get("spot"), s.get("atm_iv"), s.get("t_years")))
    vrp = _median(vrps)
    skew = _median(skews)
    return {
        "vrp": vrp if vrp is not None else 0.12,
        "skew_per_sigma": skew if skew is not None else 0.08,
        "n_samples": sum(1 for v in vrps if v is not None),
    }


# ── providers ──────────────────────────────────────────────────────────────────
class OptionDataProvider:
    """Common interface. Concrete providers implement get_chain()."""

    name = "base"
    is_free = False

    def get_chain(self, ticker: str, target_dte: int = 14) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class YFinanceProvider(OptionDataProvider):
    """Free real current chains (real IV/bid/ask/OI). No history."""

    name = "yfinance"
    is_free = True

    def get_chain(self, ticker: str, target_dte: int = 14) -> Optional[Dict[str, Any]]:
        import warnings
        from datetime import datetime, date
        try:
            import yfinance as yf
        except Exception:
            return None
        try:
            tk = yf.Ticker(ticker)
            exps = tk.options
            if not exps:
                return None
            today = date.today()
            # pick the expiry closest to target_dte
            def _dte(e):
                return (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            exp = min(exps, key=lambda e: abs(_dte(e) - target_dte))
            dte = max(_dte(exp), 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chain = tk.option_chain(exp)
            try:
                spot = float(tk.fast_info.get("lastPrice"))
            except Exception:
                spot = float(chain.calls["strike"].median())

            def _rows(df):
                out = []
                for _, r in df.iterrows():
                    out.append({
                        "strike": float(r.get("strike")),
                        "iv": float(r.get("impliedVolatility")) if r.get("impliedVolatility") else None,
                        "bid": float(r.get("bid")) if r.get("bid") else None,
                        "ask": float(r.get("ask")) if r.get("ask") else None,
                        "open_interest": int(r.get("openInterest")) if r.get("openInterest") else 0,
                        "last": float(r.get("lastPrice")) if r.get("lastPrice") else None,
                    })
                return out

            return {
                "spot": spot, "dte": dte, "t_years": dte / 365.0,
                "calls": _rows(chain.calls), "puts": _rows(chain.puts),
            }
        except Exception:
            return None


class PolygonProvider(OptionDataProvider):
    """Paid historical feed (per-contract premium history). Stub until wired.

    Requires a Polygon options subscription + POLYGON_API_KEY. The scaffold in
    src/polygon_client.py already does auth + current snapshots; the historical
    aggregates endpoint (/v2/aggs/ticker/O:...) is the piece to add here.
    """

    name = "polygon"
    is_free = False

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "")

    def get_chain(self, ticker: str, target_dte: int = 14) -> Optional[Dict[str, Any]]:
        raise NotImplementedError(
            "PolygonProvider needs a Polygon options subscription and "
            "POLYGON_API_KEY. Not yet wired — use the free 'yfinance' provider, "
            "or subscribe and implement the /v2/aggs historical endpoint here."
        )

    def get_historical_premium(self, *args, **kwargs):
        raise NotImplementedError(
            "Historical per-contract premiums require a paid Polygon options plan."
        )


_PROVIDERS = {"yfinance": YFinanceProvider, "polygon": PolygonProvider}


def get_provider(name: Optional[str] = None) -> OptionDataProvider:
    """Return a data provider. Defaults to the free yfinance provider.

    Override via the argument or the LOTTERY_DATA_PROVIDER env var.
    """
    name = (name or os.environ.get("LOTTERY_DATA_PROVIDER") or "yfinance").lower()
    if name not in _PROVIDERS:
        raise ValueError(f"unknown option-data provider '{name}'; "
                         f"choose from {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]()


__all__ = [
    "atm_iv_from_chain", "measure_vrp", "measure_skew_per_sigma",
    "calibrate_from_chains", "OptionDataProvider", "YFinanceProvider",
    "PolygonProvider", "get_provider",
]
