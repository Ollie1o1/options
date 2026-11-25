"""Concurrent Radar aggregator, confluence merger, portfolio/IV enrichment, and atomic JSON cache."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.equity_playbooks.indicators import trailing_rv
from src.paths import repo_path
from src.structure.chain_data import load_latest_atm_iv_map
from .playbooks import TIER_RANK, PlaybookMeta, load_playbooks
from .scanners import (
    _load_latest_rev_breadth,
    _load_recent_post_earnings_crashes,
    _load_upcoming_earnings_map,
    compute_regime_header,
    get_sector_etf,
    get_sector_map,
    load_local_ohlcv_safe,
    load_open_exposure_map,
    scan_archived_etf_wide_spreads,
    scan_cached_etf_bear_calls,
    scan_cached_insider_clusters,
    scan_covered_call_holdings,
    scan_equity_pullbacks,
    scan_holdings_buy_zones,
    scan_vol_managed_outlook,
)
from .schema import Opportunity, RadarSnapshot

DEFAULT_CACHE_PATH = "data/radar_cache.json"


def merge_confluence(opportunities: List[Opportunity]) -> List[Opportunity]:
    """Attach independent same-direction secondary playbook tags to overlapping tickers and sort deterministically.

    Opposing-direction signals on the same ticker (e.g. Bull Put LONG + Bear Call SHORT)
    never inflate `confluence_count`; instead they attach `OPPOSING_SIGNAL` to `risk_flags`
    so the desk never mistakes an accidental Iron Condor for agreement.
    """
    if not opportunities:
        return []

    by_ticker: Dict[str, List[Opportunity]] = {}
    for opp in opportunities:
        by_ticker.setdefault(opp.ticker, []).append(opp)

    merged: List[Opportunity] = []
    for opp in opportunities:
        peers = [p for p in by_ticker[opp.ticker] if p.playbook_id != opp.playbook_id]
        same_dir_peers = [
            p for p in peers
            if p.direction == opp.direction
            or opp.direction not in ("LONG", "SHORT")
            or p.direction not in ("LONG", "SHORT")
        ]
        opp_dir_peers = [
            p for p in peers
            if opp.direction in ("LONG", "SHORT")
            and p.direction in ("LONG", "SHORT")
            and p.direction != opp.direction
        ]
        other_sources = sorted({p.playbook_id for p in same_dir_peers})
        other_badges = sorted(
            {
                b
                for p in same_dir_peers
                for b in p.badges
                if b not in opp.badges
            }
        )
        tags = tuple(dict.fromkeys(list(opp.confluence_tags) + other_sources + other_badges[:3]))
        c_count = 1 + len(other_sources)
        flags = list(opp.risk_flags)
        if opp_dir_peers and "OPPOSING_SIGNAL" not in flags:
            flags.append("OPPOSING_SIGNAL")
        merged.append(replace(opp, confluence_count=c_count, confluence_tags=tags, risk_flags=tuple(flags)))

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


_SHORT_VOL_PLAYBOOKS = frozenset({"etf_wide_spread", "etf_bear_call", "covered_call_holdings"})


def enrich_with_portfolio_and_iv(
    opportunities: List[Opportunity],
    ohlcv: Dict[str, Any],
    *,
    chain_db: str = "data/chain_archive.db",
    own_picks_db: str = "data/own_picks.db",
    paper_db: str = "paper_trades.db",
    longterm_db: str = "data/longterm.db",
    vix_term: Optional[str] = None,
) -> List[Opportunity]:
    """Attach descriptive portfolio-overlap warnings, VIX-term warnings, and IV-vs-RV badges."""
    if not opportunities:
        return []

    exposure = load_open_exposure_map(
        own_picks_db=own_picks_db,
        paper_db=paper_db,
        longterm_db=longterm_db,
    )
    held_tickers = exposure.get("tickers") or {}
    sector_counts = exposure.get("sector_counts") or {}

    unique_syms = sorted({o.ticker for o in opportunities})
    spots = {o.ticker: o.spot for o in opportunities if o.spot > 0}
    iv_map = load_latest_atm_iv_map(unique_syms, spots, chain_db=chain_db)

    out: List[Opportunity] = []
    for o in opportunities:
        badges = list(o.badges)
        flags = list(o.risk_flags)
        is_short_vol = o.playbook_id in _SHORT_VOL_PLAYBOOKS

        # Portfolio overlap warnings (skip covered_call_holdings which intentionally targets owned stock)
        if o.playbook_id != "covered_call_holdings":
            if o.ticker in held_tickers and "HOLDING_ALREADY" not in flags:
                flags.append("HOLDING_ALREADY")
            else:
                sec = get_sector_etf(o.ticker)
                if sec and sector_counts.get(sec, 0) >= 2:
                    tag = f"SECTOR_OVERLAP:{sec}"
                    if tag not in flags:
                        flags.append(tag)

        # VIX backwardation is an active stress warning for index/ETF short-vol spreads
        if (
            vix_term == "BACKWARDATION"
            and o.playbook_id in ("etf_wide_spread", "etf_bear_call")
            and "VIX_BACKWARDATION" not in flags
        ):
            flags.append("VIX_BACKWARDATION")

        # IV vs RV cheapness / richness badge from real archived chains
        iv_info = iv_map.get(o.ticker)
        if iv_info and o.ticker in ohlcv and len(ohlcv[o.ticker][1]) >= 25:
            rv20 = float(trailing_rv(ohlcv[o.ticker][1], 20)[-1])
            atm_iv = float(iv_info["atm_iv"])
            if np.isfinite(rv20) and rv20 > 0.02:
                ratio = atm_iv / rv20
                if ratio <= 0.90:
                    if is_short_vol:
                        if "VOL_CHEAP_FOR_SHORT" not in flags:
                            flags.append("VOL_CHEAP_FOR_SHORT")
                    elif "IV_CHEAP" not in badges:
                        badges.append("IV_CHEAP")
                elif ratio >= 1.25 and "IV_RICH" not in badges:
                    badges.append("IV_RICH")

        out.append(replace(o, badges=tuple(badges), risk_flags=tuple(flags)))
    return out


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
    chain_db: str = "data/chain_archive.db",
    candidates_db: str = "data/candidates.db",
    predmarkets_db: str = "data/predmarkets.db",
    own_picks_db: str = "data/own_picks.db",
    paper_db: str = "paper_trades.db",
    longterm_db: str = "data/longterm.db",
    plan_path: Optional[str] = None,
    timeout_s: float = 10.0,
    extra_adapters: Optional[Dict[str, Callable[[], List[Opportunity]]]] = None,
) -> RadarSnapshot:
    """Execute all failure-isolated scanners concurrently, enrich, merge confluence, and persist cache."""
    registry: Dict[str, PlaybookMeta] = load_playbooks(playbooks_path)
    ohlcv = load_local_ohlcv_safe(ohlcv_db)
    header = compute_regime_header(ohlcv, predmarkets_db=predmarkets_db)
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

    # The first get_sector_map() call imports src.data_fetching (~1s of NLP
    # imports). Pay it here, not inside an adapter's timeout budget.
    try:
        get_sector_map()
    except Exception:
        pass

    earnings_map = _load_upcoming_earnings_map()
    crash_flags = _load_recent_post_earnings_crashes(ohlcv)
    rev_breadth = _load_latest_rev_breadth()

    # Step 2: Run remaining local adapters concurrently with strict per-adapter timeout isolation
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
        "holdings_buy_zone": lambda: scan_holdings_buy_zones(
            ohlcv, registry, plan_path=plan_path, fills_db=longterm_db
        ),
        "covered_call_holdings": lambda: scan_covered_call_holdings(
            registry,
            ohlcv,
            longterm_db=longterm_db,
            own_picks_db=own_picks_db,
            chain_db=chain_db,
            earnings_map=earnings_map,
        ),
        "etf_wide_spread": lambda: scan_archived_etf_wide_spreads(
            registry, ohlcv, chain_db=chain_db
        ),
        "etf_bear_call": lambda: scan_cached_etf_bear_calls(
            registry, ohlcv=ohlcv, candidates_db=candidates_db
        ),
        "insider_cluster": lambda: scan_cached_insider_clusters(registry, ohlcv),
    }
    if extra_adapters:
        adapters.update(extra_adapters)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(adapters)))) as pool:
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
    enriched = enrich_with_portfolio_and_iv(
        filtered,
        ohlcv,
        chain_db=chain_db,
        own_picks_db=own_picks_db,
        paper_db=paper_db,
        longterm_db=longterm_db,
        vix_term=header.get("vix_term"),
    )
    merged = merge_confluence(enriched)

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
        kalshi_macro=header.get("kalshi_macro") or [],
        ohlcv_as_of=header.get("ohlcv_as_of"),
        ohlcv_symbol_count=int(header.get("ohlcv_symbol_count") or 0),
        spy_rv20=header.get("spy_rv20"),
        vix_as_of=header.get("vix_as_of"),
    )
    if cache_path:
        try:
            save_radar_cache(snap, cache_path)
        except Exception as exc:
            snap.degraded_scanners.append(f"cache_write: {exc}")
    return snap
