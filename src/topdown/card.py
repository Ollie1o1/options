"""The Evidence Card: one ticker, every serious question, every fact with its source and date.

Layered on the existing `TickerDossier` (`src/radar/dossier.py`) — it adds the
sector context from the board, the peer strip, which lenses list the ticker,
an explicit verdict, and a provenance table so a reader can see where each
number came from and how old it is.

Verdict vocabulary (deliberately not buy/sell):
  TESTED SETUP        a PROVEN/PROMISING playbook fired and nothing flags it
  SETUP + WARNINGS    a playbook fired, but a landmine/regime warning applies
  AVOID FOR NOW       no setup, and a measured negative or imminent event applies
  NO TESTED SETUP     context only — nothing with evidence behind it is active
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import repo_path
from src.radar.schema import RadarSnapshot, TickerDossier
from .lenses import LensResult, lens_hits
from .peers import PeerStrip, peer_strip
from .sectors import SectorBoard, SectorRow

OhlcvMap = Dict[str, Tuple[List[str], np.ndarray, np.ndarray]]

VERDICT_STYLE = {
    "TESTED SETUP": "green",
    "SETUP + WARNINGS": "amber",
    "AVOID FOR NOW": "red",
    "NO TESTED SETUP": "grey",
}


@dataclass
class Provenance:
    section: str
    source: str
    as_of: Optional[str]
    note: str = ""
    age_days: Optional[int] = None


@dataclass
class EvidenceCard:
    ticker: str
    built_at: str
    verdict: str
    verdict_style: str
    reasons: List[str]
    warnings: List[str]
    dossier: TickerDossier
    sector: Optional[SectorRow]
    peers: PeerStrip
    lens_hits: List[Tuple[str, str]] = field(default_factory=list)
    provenance: List[Provenance] = field(default_factory=list)
    prob_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker, "built_at": self.built_at,
            "verdict": self.verdict, "verdict_style": self.verdict_style,
            "reasons": list(self.reasons), "warnings": list(self.warnings),
            "dossier": self.dossier.to_dict(),
            "sector": self.sector.to_dict() if self.sector else None,
            "peers": self.peers.to_dict(),
            "lens_hits": [list(x) for x in self.lens_hits],
            "provenance": [asdict(p) for p in self.provenance],
            "prob_label": self.prob_label,
        }


def _age(as_of: Optional[str], today: dt.date) -> Optional[int]:
    if not as_of:
        return None
    try:
        return (today - dt.date.fromisoformat(str(as_of)[:10])).days
    except ValueError:
        return None


def _mtime_date(path: str) -> Optional[str]:
    full = repo_path(path)
    if not os.path.exists(full):
        return None
    return dt.datetime.fromtimestamp(os.path.getmtime(full)).date().isoformat()


def probability_label(dos: TickerDossier) -> str:
    """Say plainly whether the ±5% odds are market-implied or a model."""
    if dos.rnd_prob_up_5pct is None:
        return "unavailable"
    if dos.rnd_source not in ("lognormal", "", None) and dos.chain_source != "rv_proxy":
        return f"market-implied ({dos.rnd_source} density from {dos.chain_source} chain, {dos.chain_as_of or '?'})"
    basis = "ATM IV" if dos.atm_iv is not None else "20d realized vol"
    return f"MODEL, not market-implied (lognormal on {basis})"


def build_evidence_card(
    ticker: str,
    *,
    ohlcv: OhlcvMap,
    board: SectorBoard,
    lens_results: Sequence[LensResult] = (),
    snapshot: Optional[RadarSnapshot] = None,
    dossier: Optional[TickerDossier] = None,
    ohlcv_db: str = "data/equity_ohlcv.db",
    cache_path: str = "data/radar_cache.json",
    chain_db: str = "data/chain_archive.db",
    estimates_as_of: Optional[str] = None,
    with_peer_iv: bool = True,
    today: Optional[dt.date] = None,
) -> EvidenceCard:
    sym = ticker.upper().strip()
    today = today or dt.date.today()
    if dossier is None:
        from src.radar.dossier import build_dossier

        dossier = build_dossier(sym, ohlcv_db=ohlcv_db, cache_path=cache_path, chain_db=chain_db, fetch_chain=False)
    sector_row = board.row(dossier.sector_etf) if dossier.sector_etf else board.row(sym)
    if sector_row is not None and dossier.sector_etf:
        dossier.sector_posture = sector_row.posture
    strip = peer_strip(sym, ohlcv, chain_db=chain_db, with_iv=with_peer_iv)
    hits = lens_hits(sym, lens_results)

    # ── verdict ──
    setups = [s for s in dossier.active_signals if s.evidence_tier in ("VALIDATED", "LEAD")]
    warnings: List[str] = []
    avoid: List[str] = []
    if dossier.days_to_earnings is not None and dossier.days_to_earnings <= 7:
        avoid.append(f"Earnings in {dossier.days_to_earnings}d ({dossier.earnings_date}) — gap risk")
    if dossier.post_earnings_crash_flag:
        avoid.append("Post-earnings breakdown in the last 20 sessions (measured -1.16% vs SPY after)")
    if sector_row is not None and sector_row.posture == "AVOID" and sector_row.etf != sym:
        avoid.append(f"Sector {sector_row.etf} has a regime-gated BEARISH Outlook call")
    if dossier.dist_sma200_pct is not None and dossier.dist_sma200_pct < 0:
        warnings.append("Below its 200-day average")
    spy_tuple = ohlcv.get("SPY")
    if spy_tuple is not None and len(spy_tuple[1]) >= 200:
        sc = np.asarray(spy_tuple[1], dtype=float)
        if sc[-1] < float(np.mean(sc[-200:])) and any(s.playbook_id == "equity_pullback" for s in setups):
            warnings.append("SPY below its 200d: uptrend pullbacks lost money in 2018 and 2022")
    reasons = [f"{_tier_word(s.evidence_tier)} · {s.playbook_label}: {s.setup_summary}" for s in setups]
    if setups and not avoid:
        verdict = "TESTED SETUP"
    elif setups:
        verdict = "SETUP + WARNINGS"
    elif avoid:
        verdict = "AVOID FOR NOW"
    else:
        verdict = "NO TESTED SETUP"
        reasons = ["No PROVEN or PROMISING playbook is active for this ticker; everything below is context."]
        if sector_row is not None and sector_row.weight > 0 and sector_row.etf != sym:
            reasons.append(
                f"Its sector ETF {sector_row.etf} IS held by the PROVEN rotation ({sector_row.weight*100:.0f}% this "
                f"month) — owning {sector_row.etf} is the tested way to hold this exposure."
            )
    warnings = avoid + warnings

    # ── provenance ──
    prov: List[Provenance] = []
    last_bar = str(ohlcv[sym][0][-1]) if sym in ohlcv and ohlcv[sym][0] else None
    spy_bar = str(spy_tuple[0][-1]) if spy_tuple is not None and spy_tuple[0] else None
    prov.append(Provenance("Price, trend, levels, RV", f"{os.path.basename(ohlcv_db)} (Yahoo, adjusted)", last_bar,
                           "behind SPY's last bar" if last_bar and spy_bar and last_bar < spy_bar else ""))
    prov.append(Provenance("Sector posture & rank", "outlook/vm_rotation.py", board.as_of,
                           "VALIDATED ordering"))
    prov.append(Provenance("Peers", "126d return correlation", last_bar, strip.method))
    prov.append(Provenance(
        "Options IV / skew", os.path.basename(chain_db) if dossier.chain_source != "rv_proxy" else "none — not in chain archive",
        dossier.chain_as_of, "" if dossier.chain_source == "rv_proxy" else dossier.chain_source))
    prov.append(Provenance("±5% odds (30d)", "computed", dossier.chain_as_of or last_bar, probability_label(dossier)))
    prov.append(Provenance("Earnings date", "dolt_options.db earnings_cal", _mtime_date("data/dolt_options.db"),
                           f"next {dossier.earnings_date}" if dossier.earnings_date else "none upcoming on file"))
    prov.append(Provenance("Estimate revisions", "estimates_archive.db eps_revisions", estimates_as_of,
                           "TESTING — readout >= 2027-01-15"))
    if dossier.analyst_actions:
        prov.append(Provenance("Analyst actions", "estimates_archive.db analyst_actions",
                               dossier.analyst_actions[0].get("date"), "tested NULL as a signal (2026-10-14)"))
    if dossier.headlines:
        prov.append(Provenance("Headlines", "news_archive.db", dossier.headlines[0].get("date"),
                               "context; news types tested NULL"))
    if dossier.kalshi_macro:
        prov.append(Provenance("Macro odds", "predmarkets.db (Kalshi)", _mtime_date("data/predmarkets.db")))
    gen = snapshot.generated_at[:10] if snapshot is not None else None
    prov.append(Provenance("Active playbook signals", os.path.basename(cache_path), gen or _mtime_date(cache_path),
                           "Radar snapshot"))
    for p in prov:
        p.age_days = _age(p.as_of, today)

    return EvidenceCard(
        ticker=sym,
        built_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        verdict=verdict, verdict_style=VERDICT_STYLE[verdict],
        reasons=reasons, warnings=warnings, dossier=dossier, sector=sector_row, peers=strip,
        lens_hits=hits, provenance=prov, prob_label=probability_label(dossier),
    )


def _tier_word(tier: str) -> str:
    return {"VALIDATED": "PROVEN", "LEAD": "PROMISING"}.get(tier, tier)


__all__ = ["EvidenceCard", "Provenance", "build_evidence_card", "probability_label", "VERDICT_STYLE"]
