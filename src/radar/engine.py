"""Concurrent Radar aggregator, confluence merger, and atomic JSON cache."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.paths import repo_path
from .playbooks import TIER_RANK, PlaybookMeta, load_playbooks
from .scanners import (
    _load_latest_rev_breadth,
    _load_recent_post_earnings_crashes,
    _load_upcoming_earnings_map,
    compute_regime_header,
    load_local_ohlcv_safe,
    scan_cached_etf_bear_calls,
    scan_cached_insider_clusters,
    scan_equity_pullbacks,
    scan_holdings_buy_zones,
    scan_vol_managed_outlook,
)
from .schema import Opportunity, RadarSnapshot

DEFAULT_CACHE_PATH = "data/radar_cache.json"


def merge_confluence(opportunities: List[Opportunity]) -> List[Opportunity]:
    """Attach independent secondary playbook tags to overlapping tickers and sort deterministically."""
    if not opportunities:
        return []

    by_ticker: Dict[str, List[Opportunity]] = {}
    for opp in opportunities:
        by_ticker.setdefault(opp.ticker, []).append(opp)

    merged: List[Opportunity] = []
    for opp in opportunities:
        peers = by_ticker[opp.ticker]
        other_sources = sorted(
            {p.playbook_id for p in peers if p.playbook_id != opp.playbook_id}
        )
        other_badges = sorted(
            {
                b
                for p in peers
                if p.playbook_id != opp.playbook_id
                for b in p.badges
                if b not in opp.badges
            }
        )
        tags = tuple(dict.fromkeys(list(opp.confluence_tags) + other_sources + other_badges[:3]))
        c_count = 1 + len(other_sources)
        merged.append(replace(opp, confluence_count=c_count, confluence_tags=tags))

    merged.sort(
        key=lambda o: (
            TIER_RANK.get(o.evidence_tier, 99),
            -o.confluence_count,
            len(o.risk_flags),
            o.friction_value,
            o.ticker,
            o.playbook_id,
        )
    )
    return merged


def save_radar_cache(snapshot: RadarSnapshot, cache_path: str = DEFAULT_CACHE_PATH) -> str:
    full = repo_path(cache_path)
    parent = os.path.dirname(full)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{full}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(snapshot.to_dict(), fh, indent=2)
    os.replace(tmp, full)
    return full


def load_cached_radar(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[RadarSnapshot]:
    full = repo_path(cache_path)
    if not os.path.exists(full):
        return None
    try:
        with open(full, encoding="utf-8") as fh:
            raw = json.load(fh)
        return RadarSnapshot.from_dict(raw, from_cache=True)
    except Exception:
        return None


def run_radar(
    *,
    ohlcv_db: str = "data/equity_ohlcv.db",
    playbooks_path: str = "configs/playbooks.json",
    cache_path: Optional[str] = DEFAULT_CACHE_PATH,
    timeout_s: float = 10.0,
    extra_adapters: Optional[Dict[str, Callable[[], List[Opportunity]]]] = None,
) -> RadarSnapshot:
    """Execute all failure-isolated scanners concurrently, merge confluence, and persist cache."""
    registry: Dict[str, PlaybookMeta] = load_playbooks(playbooks_path)
    ohlcv = load_local_ohlcv_safe(ohlcv_db)
    header = compute_regime_header(ohlcv)
    degraded: List[str] = []

    core_alloc: List[Dict[str, Any]] = []
    posture_map: Dict[str, str] = {}
    all_opps: List[Opportunity] = []

    # Step 1: Core Vol-Managed Outlook (provides posture_map for downstream stock adapters)
    try:
        core_alloc, vm_opps, posture_map = scan_vol_managed_outlook(ohlcv, registry)
        all_opps.extend(vm_opps)
    except Exception as exc:
        degraded.append(f"vol_managed_outlook: {exc}")

    earnings_map = _load_upcoming_earnings_map()
    crash_flags = _load_recent_post_earnings_crashes(ohlcv)
    rev_breadth = _load_latest_rev_breadth()

    # Step 2: Run remaining adapters concurrently with strict per-adapter timeout isolation
    adapters: Dict[str, Callable[[], List[Opportunity]]] = {
        "equity_pullback": lambda: scan_equity_pullbacks(
            ohlcv,
            registry,
            posture_map=posture_map,
            earnings_map=earnings_map,
            crash_flags=crash_flags,
            rev_breadth=rev_breadth,
            regime_posture=header["regime_posture"],
        ),
        "holdings_buy_zone": lambda: scan_holdings_buy_zones(ohlcv, registry),
        "etf_bear_call": lambda: scan_cached_etf_bear_calls(registry),
        "insider_cluster": lambda: scan_cached_insider_clusters(registry, ohlcv),
    }
    if extra_adapters:
        adapters.update(extra_adapters)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, max(1, len(adapters)))) as pool:
        futs = {name: pool.submit(fn) for name, fn in adapters.items()}
        for name, fut in futs.items():
            try:
                rows = fut.result(timeout=timeout_s)
                if rows:
                    all_opps.extend(rows)
            except concurrent.futures.TimeoutError:
                degraded.append(f"{name}: timeout >{timeout_s:.0f}s")
            except Exception as exc:
                degraded.append(f"{name}: {exc}")

    # Enforce registry invariant: never leak CLOSED playbooks into default Radar
    filtered = [o for o in all_opps if o.evidence_tier != "CLOSED"]
    merged = merge_confluence(filtered)

    snap = RadarSnapshot(
        generated_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        regime_posture=header["regime_posture"],
        vix=header["vix"],
        vix_3m=header["vix_3m"],
        vix_term=header["vix_term"],
        spy_last=header["spy_last"],
        spy_chg_1d_pct=header["spy_chg_1d_pct"],
        qqq_chg_1d_pct=header["qqq_chg_1d_pct"],
        next_macro_event=header["next_macro_event"],
        core_allocation=core_alloc,
        opportunities=merged,
        degraded_scanners=degraded,
        from_cache=False,
    )
    if cache_path:
        try:
            save_radar_cache(snap, cache_path)
        except Exception as exc:
            snap.degraded_scanners.append(f"cache_write: {exc}")
    return snap
