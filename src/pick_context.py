"""Decision context stamped onto each pick — display only, never scoring.

Adds to the per-pick detail the things a careful trader looks up by hand:

- **History**: base rates from *your own* closed paper trades that match this
  setup (same strategy, similar DTE and |delta|) — an honest, self-calibrated
  complement to model PoP.
- **Events**: verified FOMC/CPI/NFP dates that fall inside the holding window.
- **Cohort**: whether this pick is gate-eligible (Long Call ≥ cohort DTE floor).
- **Book**: open positions you already hold on the same ticker.
- **Flow**: yesterday's OI delta for the ticker from the chain archive.
- **Insider**: 90d Form 4 read from a daily on-disk cache (filled lazily, one
  EDGAR fetch per ticker per day; failure-safe).

Every reader is failure-safe and cached per process so the per-pick cost is
one dict lookup. Nothing here touches quality_score or any weights.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

PAPER_DB = "paper_trades.db"
INSIDER_CACHE = os.path.join("data", "insider_cache.json")

_closed_trades_cache: Optional[List[Dict[str, Any]]] = None
_open_book_cache: Optional[Dict[str, List[str]]] = None


# ── History: setup analogs from your own closed trades ──────────────────────

def _load_closed_trades(db_path: str = PAPER_DB) -> List[Dict[str, Any]]:
    global _closed_trades_cache
    if _closed_trades_cache is not None:
        return _closed_trades_cache
    rows: List[Dict[str, Any]] = []
    try:
        with sqlite3.connect(db_path) as conn:
            for (strategy, date, expiration, entry_delta, pnl_pct) in conn.execute(
                    "SELECT strategy_name, date, expiration, entry_delta, pnl_pct "
                    "FROM trades WHERE status='CLOSED' AND pnl_pct IS NOT NULL"):
                try:
                    dte = (datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
                           - datetime.strptime(str(date)[:10], "%Y-%m-%d")).days
                except (TypeError, ValueError):
                    continue
                rows.append({"strategy": strategy, "dte": dte,
                             "delta": entry_delta, "pnl_pct": float(pnl_pct)})
    except sqlite3.Error:
        pass
    _closed_trades_cache = rows
    return rows


def analog_stats(strategy: str, dte: int, delta: Optional[float],
                 db_path: str = PAPER_DB,
                 dte_band: int = 10, delta_band: float = 0.10,
                 min_n: int = 5) -> Optional[Dict[str, Any]]:
    """Win rate / avg P&L of your closed trades matching this setup.
    Tries (strategy + DTE±band + |delta|±band); widens to (strategy + DTE±band)
    if too few; returns None when even that has < min_n trades."""
    trades = [t for t in _load_closed_trades(db_path)
              if t["strategy"] == strategy and abs(t["dte"] - dte) <= dte_band]
    matched = trades
    widened = True
    if delta is not None:
        tight = [t for t in trades if t["delta"] is not None
                 and abs(abs(float(t["delta"])) - abs(float(delta))) <= delta_band]
        if len(tight) >= min_n:
            matched, widened = tight, False
    if len(matched) < min_n:
        return None
    pnls = [t["pnl_pct"] for t in matched]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(matched),
        "win_rate": wins / len(matched),
        "avg_pnl_pct": sum(pnls) / len(pnls),
        "widened": widened,
    }


# ── Events inside the holding window ─────────────────────────────────────────

def events_in_window(expiration: str, today: Optional[str] = None) -> List[Dict[str, str]]:
    """Verified macro events (FOMC/CPI/NFP) dated inside [today, expiration]."""
    from src.macro_analyzer import _load_events_from_config
    today = today or datetime.now().strftime("%Y-%m-%d")
    exp = str(expiration)[:10]
    try:
        return sorted((e for e in _load_events_from_config({})
                       if today <= e["date"] <= exp), key=lambda e: e["date"])
    except Exception:
        return []


# ── Cohort eligibility ───────────────────────────────────────────────────────

def cohort_eligible(strategy: str, dte: int, min_dte: int = 30) -> bool:
    return strategy == "Long Call" and dte >= min_dte


# ── Open-book overlap ────────────────────────────────────────────────────────

def open_book(ticker: str, db_path: str = PAPER_DB) -> List[str]:
    """Strategy names of OPEN paper positions on this ticker."""
    global _open_book_cache
    if _open_book_cache is None:
        book: Dict[str, List[str]] = {}
        try:
            with sqlite3.connect(db_path) as conn:
                for t, s in conn.execute(
                        "SELECT ticker, strategy_name FROM trades WHERE status='OPEN'"):
                    book.setdefault(str(t).upper(), []).append(s)
        except sqlite3.Error:
            pass
        _open_book_cache = book
    return _open_book_cache.get((ticker or "").upper(), [])


# ── Flow: yesterday's OI delta from the chain archive ────────────────────────

def flow_summary(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        from src import uoa
        from src.chain_archive import DEFAULT_DB
        with sqlite3.connect(DEFAULT_DB) as conn:
            days = [r[0] for r in conn.execute(
                "SELECT DISTINCT snap_date FROM chain_snapshots WHERE symbol=? "
                "ORDER BY snap_date", ((symbol or "").upper(),))]
            if len(days) < 2:
                return None
            deltas = uoa.oi_deltas(conn, (symbol or "").upper(), days[-1])
        if not deltas:
            return None
        flow = uoa.symbol_flow(deltas)
        return {"date": days[-1], "call_oi_added": flow["call_oi_added"],
                "put_oi_added": flow["put_oi_added"],
                "net_call_share": flow["net_call_share"],
                "n_unusual": len(flow["unusual"])}
    except Exception:
        return None


# ── Insider: daily-cached Form 4 read ────────────────────────────────────────

def insider_summary(ticker: str, cache_path: str = INSIDER_CACHE,
                    fetch: bool = True) -> Optional[Dict[str, Any]]:
    """Cluster-score summary, cached per (ticker, day). One EDGAR round-trip
    per ticker per day at most; None on any failure."""
    ticker = (ticker or "").upper()
    today = datetime.now().strftime("%Y-%m-%d")
    cache: Dict[str, Any] = {}
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        pass
    hit = cache.get(ticker)
    if hit and hit.get("as_of") == today:
        return hit.get("summary")
    if not fetch:
        return None
    try:
        from src.insider import edgar
        from src.insider.parse import parse_form4
        from src.insider.signal import cluster_score
        cik = edgar.cik_for(ticker)
        summary = None
        if cik:
            txs: List[Dict[str, Any]] = []
            for filing in edgar.recent_form4(cik, max_filings=10, since_days=120):
                xml = edgar.fetch_form4_xml(cik, filing["accession"], filing["document"])
                if xml:
                    txs.extend(parse_form4(xml))
            summary = cluster_score(txs)
        cache[ticker] = {"as_of": today, "summary": summary}
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        return summary
    except Exception:
        return None


# ── Renderer ─────────────────────────────────────────────────────────────────

def context_lines(row: Dict[str, Any], today: Optional[str] = None,
                  db_path: str = PAPER_DB,
                  with_insider: bool = True) -> List[str]:
    """Up to three plain-text lines for the per-pick detail. Never raises."""
    out: List[str] = []
    try:
        opt_type = str(row.get("type") or "").lower()
        strategy = "Long Call" if opt_type == "call" else "Long Put"
        dte = int(float(row.get("T_years") or 0) * 365)
        delta = row.get("delta")
        delta = float(delta) if delta is not None else None
        symbol = str(row.get("symbol") or "")
        expiration = str(row.get("expiration") or "")[:10]

        # History
        a = analog_stats(strategy, dte, delta, db_path=db_path)
        if a:
            scope = f"{strategy}, DTE±10" + ("" if a["widened"] else ", |Δ|±0.10")
            out.append(f"History:   your book, similar setups ({scope}): "
                       f"n={a['n']} | win {a['win_rate']:.0%} | "
                       f"avg {a['avg_pnl_pct']:+.1%}")
        else:
            out.append(f"History:   no comparable closed trades yet "
                       f"({strategy}, ~{dte} DTE)")

        # Events + cohort + book + flow + insider on one signals line
        sig: List[str] = []
        evs = events_in_window(expiration, today=today) if expiration else []
        if evs:
            sig.append("⚠ " + ", ".join(f"{e['name']} {e['date'][5:]}" for e in evs[:3])
                       + " inside window")
        if cohort_eligible(strategy, dte):
            sig.append(f"COHORT ✓ ({dte} DTE)")
        held = open_book(symbol, db_path=db_path)
        if held:
            kinds = {}
            for s in held:
                kinds[s] = kinds.get(s, 0) + 1
            sig.append("Book: already holding "
                       + ", ".join(f"{v}× {k}" for k, v in kinds.items()))
        f = flow_summary(symbol)
        if f:
            sig.append(f"Flow Δ({f['date'][5:]}): calls +{f['call_oi_added']:,.0f} / "
                       f"puts +{f['put_oi_added']:,.0f} OI"
                       + (f", {f['n_unusual']} unusual" if f["n_unusual"] else ""))
        ins = insider_summary(symbol, fetch=with_insider) if symbol else None
        if ins and ins.get("label") not in (None, "NONE"):
            sig.append(f"Insider: {ins['label']} (${ins['buy_value']:,.0f}, 90d)")
        if sig:
            out.append("Signals:   " + "  |  ".join(sig))

        # Dolt verdict: per-segment LONG/SHORT/STAND-DOWN from the real-marks
        # recommender (display-only; only shown for symbols in a known segment).
        try:
            from src.dolt_verdict import verdict_line
            vl = verdict_line(symbol)
            if vl:
                out.append(vl)
        except Exception:
            pass

        # Quant read: net-of-cost EV + cheap/rich vs surface + VRP regime.
        ql = quant_read_line(row)
        if ql:
            out.append(ql)

        # Leverage read: risk-matched option-vs-leverage vehicle selector
        # (display-only; single-leg directional picks only).
        try:
            from src.leverage_selector import leverage_vehicle_line
            ll = leverage_vehicle_line(row)
            if ll:
                out.append(ll)
        except Exception:
            pass
    except Exception:
        pass
    return out


def quant_read_line(row: Dict[str, Any]) -> Optional[str]:
    """A one-line, decision-time 'quant read' for a pick (display-only): does the
    edge survive REAL round-trip costs, is it cheap/rich vs the fitted vol surface,
    and what does the vol-risk-premium regime favor. Synthesizes the session's
    findings (cost is the wall; relative value + VRP are the real levers) at the
    point of decision. Never raises; returns None if EV isn't available."""
    try:
        net = row.get("ev_per_contract")
        if net is None or (isinstance(net, float) and math.isnan(net)):
            return None
        gross = row.get("ev_gross_per_contract")
        cost = row.get("ev_cost_per_contract")
        bits: List[str] = []
        # Honest banding: net<=0 negative; 0<net<cost means costs ate most of the
        # edge (marginal); net>=cost means the surviving edge beats the toll paid.
        _cost = cost if (cost is not None and not (isinstance(cost, float) and math.isnan(cost))) else 0.0
        if net <= 0:
            verdict = "NEGATIVE EV after cost — pass or restructure"
        elif net < _cost:
            verdict = "MARGINAL EV — costs eat most of the edge"
        else:
            verdict = "POSITIVE EV after cost"
        head = f"net EV {net:+,.0f}/contract"
        if gross is not None and cost is not None and not (isinstance(gross, float) and math.isnan(gross)):
            head += f" (gross {gross:+,.0f} − cost {cost:,.0f})"
        bits.append(f"{verdict}: {head}")
        # cheap/rich vs the SVI surface (negative residual = cheap)
        resid = row.get("iv_surface_residual")
        try:
            resid = float(resid)
            if resid <= -0.01:
                bits.append("CHEAP vs surface")
            elif resid >= 0.01:
                bits.append("RICH vs surface")
        except (TypeError, ValueError):
            pass
        # VRP regime stance
        vrp = row.get("vrp_regime")
        if vrp and str(vrp).upper() not in ("UNKNOWN", "NONE", ""):
            bits.append(f"VRP: {vrp}")
        return "Quant read: " + "  |  ".join(bits)
    except Exception:
        return None


def reset_caches() -> None:
    """Test hook."""
    global _closed_trades_cache, _open_book_cache
    _closed_trades_cache = None
    _open_book_cache = None
