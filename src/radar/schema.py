"""Data contracts and JSON serialization for the Unified Radar & Dossier."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Opportunity:
    ticker: str
    asset_class: str  # "stock" | "etf" | "option" | "crypto"
    playbook_id: str
    playbook_label: str
    evidence_tier: str  # "VALIDATED" | "LEAD" | "CONTEXT" | "RESEARCH" | "CLOSED"
    direction: str  # "LONG" | "SHORT" | "NEUTRAL"
    spot: float
    chg_1d_pct: Optional[float]
    chg_5d_pct: Optional[float]
    horizon_label: str
    setup_summary: str
    badges: Tuple[str, ...] = ()
    risk_flags: Tuple[str, ...] = ()
    friction_label: str = ""
    friction_value: float = 0.001
    confluence_count: int = 1
    confluence_tags: Tuple[str, ...] = ()
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    as_of: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["badges"] = list(self.badges)
        d["risk_flags"] = list(self.risk_flags)
        d["confluence_tags"] = list(self.confluence_tags)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Opportunity":
        return cls(
            ticker=str(d["ticker"]).upper(),
            asset_class=str(d["asset_class"]),
            playbook_id=str(d["playbook_id"]),
            playbook_label=str(d["playbook_label"]),
            evidence_tier=str(d["evidence_tier"]),
            direction=str(d["direction"]),
            spot=float(d["spot"]),
            chg_1d_pct=None if d.get("chg_1d_pct") is None else float(d["chg_1d_pct"]),
            chg_5d_pct=None if d.get("chg_5d_pct") is None else float(d["chg_5d_pct"]),
            horizon_label=str(d.get("horizon_label", "")),
            setup_summary=str(d.get("setup_summary", "")),
            badges=tuple(d.get("badges") or ()),
            risk_flags=tuple(d.get("risk_flags") or ()),
            friction_label=str(d.get("friction_label", "")),
            friction_value=float(d.get("friction_value", 0.001)),
            confluence_count=int(d.get("confluence_count", 1)),
            confluence_tags=tuple(d.get("confluence_tags") or ()),
            suggested_stop=None if d.get("suggested_stop") is None else float(d["suggested_stop"]),
            suggested_target=None if d.get("suggested_target") is None else float(d["suggested_target"]),
            as_of=str(d.get("as_of", "")),
            extra=dict(d.get("extra") or {}),
        )


@dataclass
class RadarSnapshot:
    generated_at: str
    regime_posture: str  # "RISK_ON" | "NEUTRAL" | "RISK_OFF"
    vix: Optional[float]
    vix_3m: Optional[float]
    vix_term: str  # "CONTANGO" | "BACKWARDATION" | "UNKNOWN"
    spy_last: Optional[float]
    spy_chg_1d_pct: Optional[float]
    qqq_chg_1d_pct: Optional[float]
    next_macro_event: str
    core_allocation: List[Dict[str, Any]] = field(default_factory=list)
    opportunities: List[Opportunity] = field(default_factory=list)
    degraded_scanners: List[str] = field(default_factory=list)
    from_cache: bool = False
    kalshi_macro: List[Dict[str, Any]] = field(default_factory=list)
    ohlcv_as_of: Optional[str] = None
    ohlcv_symbol_count: int = 0
    # SPY 20d realized vol (annualised %). Until 2026-11-25 this number was
    # stored in `vix` and shown as "RV/VIX"; `vix` is now the real Cboe VIX close.
    spy_rv20: Optional[float] = None
    vix_as_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "regime_posture": self.regime_posture,
            "vix": self.vix,
            "vix_3m": self.vix_3m,
            "vix_term": self.vix_term,
            "spy_last": self.spy_last,
            "spy_chg_1d_pct": self.spy_chg_1d_pct,
            "qqq_chg_1d_pct": self.qqq_chg_1d_pct,
            "next_macro_event": self.next_macro_event,
            "core_allocation": list(self.core_allocation),
            "opportunities": [o.to_dict() for o in self.opportunities],
            "degraded_scanners": list(self.degraded_scanners),
            "from_cache": bool(self.from_cache),
            "kalshi_macro": list(self.kalshi_macro),
            "ohlcv_as_of": self.ohlcv_as_of,
            "ohlcv_symbol_count": int(self.ohlcv_symbol_count),
            "spy_rv20": self.spy_rv20,
            "vix_as_of": self.vix_as_of,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, from_cache: bool = False) -> "RadarSnapshot":
        legacy = "spy_rv20" not in d  # pre-2026-11-25 caches stored SPY RV20 under "vix"
        vix_raw = None if legacy else d.get("vix")
        rv_raw = d.get("vix") if legacy else d.get("spy_rv20")
        return cls(
            generated_at=str(d.get("generated_at", "")),
            regime_posture=str(d.get("regime_posture", "NEUTRAL")),
            vix=None if vix_raw is None else float(vix_raw),
            vix_3m=None if d.get("vix_3m") is None else float(d["vix_3m"]),
            vix_term=str(d.get("vix_term", "UNKNOWN")),
            spy_last=None if d.get("spy_last") is None else float(d["spy_last"]),
            spy_chg_1d_pct=None if d.get("spy_chg_1d_pct") is None else float(d["spy_chg_1d_pct"]),
            qqq_chg_1d_pct=None if d.get("qqq_chg_1d_pct") is None else float(d["qqq_chg_1d_pct"]),
            next_macro_event=str(d.get("next_macro_event", "none")),
            core_allocation=list(d.get("core_allocation") or []),
            opportunities=[Opportunity.from_dict(x) for x in (d.get("opportunities") or [])],
            degraded_scanners=list(d.get("degraded_scanners") or []),
            from_cache=from_cache,
            kalshi_macro=list(d.get("kalshi_macro") or []),
            ohlcv_as_of=d.get("ohlcv_as_of"),
            ohlcv_symbol_count=int(d.get("ohlcv_symbol_count") or 0),
            spy_rv20=None if rv_raw is None else float(rv_raw),
            vix_as_of=d.get("vix_as_of"),
        )


@dataclass
class TickerDossier:
    ticker: str
    as_of: str
    spot: Optional[float]
    chg_1d_pct: Optional[float]
    chg_5d_pct: Optional[float]
    chg_21d_pct: Optional[float]
    mom_12_1_pct: Optional[float]
    dist_sma20_pct: Optional[float]
    dist_sma50_pct: Optional[float]
    dist_sma200_pct: Optional[float]
    rsi_2: Optional[float]
    rsi_14: Optional[float]
    rv_20_ann: Optional[float]
    supports: List[Dict[str, Any]] = field(default_factory=list)
    resistances: List[Dict[str, Any]] = field(default_factory=list)
    cdr_ticker: Optional[str] = None
    sector_etf: Optional[str] = None
    sector_posture: Optional[str] = None
    days_to_earnings: Optional[int] = None
    earnings_date: Optional[str] = None
    earnings_source: str = "unknown"
    next_macro_event: str = "none"
    rev_breadth_30d: Optional[float] = None
    post_earnings_crash_flag: bool = False
    headlines: List[Dict[str, Any]] = field(default_factory=list)
    atm_iv: Optional[float] = None
    iv_rank: Optional[float] = None
    implied_move_1s_pct: Optional[float] = None
    rnd_prob_up_5pct: Optional[float] = None
    rnd_prob_dn_5pct: Optional[float] = None
    active_signals: List[Opportunity] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    closes_60d: List[float] = field(default_factory=list)
    chain_source: str = "rv_proxy"
    chain_as_of: Optional[str] = None
    rnd_source: str = "lognormal"
    iv_vs_rv_ratio: Optional[float] = None
    skew_25d_vp: Optional[float] = None
    kalshi_macro: List[Dict[str, Any]] = field(default_factory=list)
    analyst_actions: List[Dict[str, Any]] = field(default_factory=list)
    open_position_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["active_signals"] = [s.to_dict() for s in self.active_signals]
        return d
