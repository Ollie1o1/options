"""Canadian Wealthsimple TFSA Tax-Drag Calculator, ETF Equivalents & CRA Safety Meter (`src/tfsa_tax.py`).

Two Canadian-specific TFSA realities that directly improve net compounding:
  1. **15% IRS US Dividend Withholding Tax:** Unlike an RRSP (treaty-exempt), US-listed
     equity/sector ETF dividends paid into a Canadian TFSA suffer a non-recoverable 15%
     IRS withholding tax. Capital gains and covered-call/long-option gains are 0% taxed.
     Comparing US ETFs against Canadian swap/total-return or T-bill equivalents (`UBIL.U`,
     `CASH.TO`, `CBIL.TO`, `HXS`, `HXQ`, CDRs in `cdr_map.json`) quantifies the exact drag.
  2. **CRA "Carrying on a Business" Safety Guard:** Frequent short-term trading inside a TFSA
     risks CRA re-characterizing tax-free gains as 100% taxable business income. Monitoring
     annual trade count, average holding period, short-term (<5d) turnover, and annual TFSA
     contribution room keeps the account safely in investor territory.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.paths import repo_path

US_DIVIDEND_WITHHOLDING_RATE = 0.15

# Approximate trailing annual dividend yields for common TFSA Core ETFs & T-bill ETFs
# (descriptive estimates used to quantify 15% IRS withholding drag in a TFSA).
ETF_YIELD_ESTIMATES: Dict[str, Dict[str, Any]] = {
    "SGOV": {"yield_pct": 4.3, "us_withholding": False, "note": "US T-bill interest (portfolio-interest exempt / 0% WHT on qualified interest distributions; verify broker slip)", "cad_alt": "UBIL.U / CBIL.TO / CASH.TO"},
    "BIL":  {"yield_pct": 4.3, "us_withholding": False, "note": "US T-bill interest (typically 0% WHT; verify broker slip)", "cad_alt": "UBIL.U / CBIL.TO"},
    "CBIL.TO": {"yield_pct": 3.4, "us_withholding": False, "note": "Canadian T-bill ETF (0% US WHT, CAD)", "cad_alt": "CASH.TO"},
    "CASH.TO": {"yield_pct": 3.5, "us_withholding": False, "note": "Canadian HISA ETF (0% US WHT, CAD)", "cad_alt": "CBIL.TO"},
    "UBIL.U": {"yield_pct": 4.2, "us_withholding": False, "note": "Canadian-domiciled USD T-bill ETF (0% US WHT, USD)", "cad_alt": "SGOV"},
    "SPY":  {"yield_pct": 1.25, "us_withholding": True, "note": "15% WHT on ~1.25% div = ~19 bps/yr drag", "cad_alt": "HXS.TO / VFV.TO"},
    "QQQ":  {"yield_pct": 0.60, "us_withholding": True, "note": "15% WHT on ~0.60% div = ~9 bps/yr drag (minimal)", "cad_alt": "HXQ.TO / QQC.TO"},
    "IWM":  {"yield_pct": 1.35, "us_withholding": True, "note": "15% WHT on ~1.35% div = ~20 bps/yr drag", "cad_alt": "—"},
    "XLK":  {"yield_pct": 0.65, "us_withholding": True, "note": "Low dividend (~10 bps/yr drag)", "cad_alt": "—"},
    "XLF":  {"yield_pct": 1.55, "us_withholding": True, "note": "Moderate dividend (~23 bps/yr drag)", "cad_alt": "ZEB.TO (CAD banks)"},
    "XLE":  {"yield_pct": 3.25, "us_withholding": True, "note": "High dividend (~49 bps/yr TFSA drag)", "cad_alt": "XEG.TO (CAD energy, 0% US WHT)"},
    "XLV":  {"yield_pct": 1.55, "us_withholding": True, "note": "Moderate dividend (~23 bps/yr drag)", "cad_alt": "—"},
    "XLI":  {"yield_pct": 1.40, "us_withholding": True, "note": "Moderate dividend (~21 bps/yr drag)", "cad_alt": "—"},
    "XLP":  {"yield_pct": 2.65, "us_withholding": True, "note": "Higher dividend (~40 bps/yr TFSA drag)", "cad_alt": "—"},
    "XLY":  {"yield_pct": 0.75, "us_withholding": True, "note": "Low dividend (~11 bps/yr drag)", "cad_alt": "—"},
    "XLU":  {"yield_pct": 3.10, "us_withholding": True, "note": "High dividend (~47 bps/yr TFSA drag)", "cad_alt": "ZUT.TO (CAD util, 0% US WHT)"},
    "XLB":  {"yield_pct": 1.90, "us_withholding": True, "note": "Moderate dividend (~29 bps/yr drag)", "cad_alt": "—"},
    "XLRE": {"yield_pct": 3.40, "us_withholding": True, "note": "High REIT dividend (~51 bps/yr TFSA drag)", "cad_alt": "XRE.TO (CAD REIT)"},
    "XLC":  {"yield_pct": 0.90, "us_withholding": True, "note": "Low dividend (~14 bps/yr drag)", "cad_alt": "—"},
    "SMH":  {"yield_pct": 0.50, "us_withholding": True, "note": "Very low dividend (~8 bps/yr drag)", "cad_alt": "—"},
    "GLD":  {"yield_pct": 0.00, "us_withholding": False, "note": "No dividend (0% WHT drag)", "cad_alt": "PHYS.TO"},
    "SLV":  {"yield_pct": 0.00, "us_withholding": False, "note": "No dividend (0% WHT drag)", "cad_alt": "PSLV.TO"},
    "TLT":  {"yield_pct": 4.20, "us_withholding": False, "note": "US Treasury interest (typically portfolio-interest exempt)", "cad_alt": "ZTL.TO"},
}


def load_cdr_map(path: str = "cdr_map.json") -> Dict[str, str]:
    full = Path(repo_path(path))
    if not full.is_file():
        return {}
    try:
        raw = json.loads(full.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k).upper(): str(v).upper() for k, v in raw.items() if not str(k).startswith("_")}
    except (OSError, ValueError):
        pass
    return {}


@dataclass(frozen=True)
class TaxDragRow:
    ticker: str
    position_usd: float
    est_yield_pct: float
    withholding_bps: float
    annual_drag_usd: float
    cad_alternative: str
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TfsaSafetyReport:
    account_type: str
    total_tax_drag_usd_yr: float
    weighted_tax_drag_bps: float
    tax_drag_rows: List[TaxDragRow] = field(default_factory=list)
    trades_ytd: int = 0
    closed_trades_ytd: int = 0
    avg_hold_days: Optional[float] = None
    short_hold_trades_lt5d: int = 0
    cra_status: str = "SAFE_INVESTOR"   # "SAFE_INVESTOR" | "WATCH_CADENCE" | "HIGH_TURNOVER_RISK"
    cra_notes: List[str] = field(default_factory=list)
    room_cad: Optional[float] = None
    contributed_ytd_cad: float = 0.0
    room_remaining_cad: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "tax_drag_rows": [r.to_dict() for r in self.tax_drag_rows],
        }


def estimate_tfsa_tax_drag(
    allocations_usd: Dict[str, float],
    *,
    cdr_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[TaxDragRow], float, float]:
    """Compute per-ticker and total annual 15% US dividend withholding drag inside a Canadian TFSA."""
    cdrs = cdr_map if cdr_map is not None else load_cdr_map()
    rows: List[TaxDragRow] = []
    total_usd = 0.0
    total_drag = 0.0

    for tk, val in sorted(allocations_usd.items(), key=lambda x: -x[1]):
        sym = str(tk).upper().strip()
        pos_usd = max(0.0, float(val or 0.0))
        if not sym or pos_usd <= 0:
            continue
        total_usd += pos_usd
        info = ETF_YIELD_ESTIMATES.get(sym)
        if info is not None:
            y_pct = float(info["yield_pct"])
            applies = bool(info["us_withholding"])
            alt = str(info.get("cad_alt") or cdrs.get(sym) or "—")
            note = str(info.get("note") or "")
        else:
            is_cad = sym.endswith(".TO") or sym.endswith(".NE") or sym.endswith(".V")
            y_pct = 1.2 if not is_cad else 2.0
            applies = not is_cad
            alt = cdrs.get(sym, "—")
            note = (
                "Canadian-listed (0% US WHT)"
                if is_cad
                else (f"Est. US equity (~1.2% div · CDR alt: {alt})" if alt != "—" else "Est. US equity (~1.2% div · 15% WHT)")
            )

        drag_bps = round(y_pct * US_DIVIDEND_WITHHOLDING_RATE * 100.0, 1) if applies else 0.0
        drag_usd = round(pos_usd * (drag_bps / 10000.0), 2)
        total_drag += drag_usd
        rows.append(
            TaxDragRow(
                ticker=sym,
                position_usd=round(pos_usd, 2),
                est_yield_pct=y_pct,
                withholding_bps=drag_bps,
                annual_drag_usd=drag_usd,
                cad_alternative=alt,
                note=note,
            )
        )

    weighted_bps = round((total_drag / total_usd) * 10000.0, 1) if total_usd > 0 else 0.0
    return rows, round(total_drag, 2), weighted_bps


def compute_tfsa_safety_report(
    *,
    profile: Any,
    core_allocation: Sequence[Dict[str, Any]] = (),
    equity_picks: Sequence[Dict[str, Any]] = (),
    today: Optional[dt.date] = None,
) -> TfsaSafetyReport:
    """Build complete Canadian TFSA tax-drag + CRA turnover + contribution-room safety report."""
    now_d = today or dt.date.today()
    year_prefix = f"{now_d.year}-"

    # 1. Estimate USD allocations for tax-drag calculation
    alloc_usd: Dict[str, float] = {}
    acct_usd = float(getattr(profile, "tfsa_usd", 5000.0) or 5000.0)
    cash_etf = str(getattr(profile, "cash_etf", "SGOV") or "SGOV").upper()
    for c in core_allocation or []:
        tk = str(c.get("ticker") or "").upper()
        w = float(c.get("target_weight") or 0.0)
        if not tk or w <= 0:
            continue
        eff_tk = cash_etf if tk == "CASH" else tk
        alloc_usd[eff_tk] = alloc_usd.get(eff_tk, 0.0) + acct_usd * w
    for p in equity_picks or []:
        if str(p.get("status") or "").upper() == "OPEN":
            tk = str(p.get("ticker") or "").upper()
            px = float(p.get("current_price") or p.get("entry_price") or 0.0)
            sh = float(p.get("shares") or 0.0)
            if tk and px > 0 and sh > 0:
                alloc_usd[tk] = alloc_usd.get(tk, 0.0) + px * sh

    drag_rows, total_drag, weighted_bps = estimate_tfsa_tax_drag(alloc_usd)

    # 2. CRA activity / holding-period turnover telemetry (YTD)
    trades_ytd = 0
    closed_ytd = 0
    hold_days: List[int] = []
    short_lt5d = 0
    for p in equity_picks or []:
        d_ent = str(p.get("date") or "")[:10]
        if d_ent.startswith(year_prefix):
            trades_ytd += 1
        st = str(p.get("status") or "").upper()
        d_ex = str(p.get("exit_date") or "")[:10]
        if st == "CLOSED" and (d_ent.startswith(year_prefix) or d_ex.startswith(year_prefix)):
            closed_ytd += 1
            try:
                days = max(0, (dt.date.fromisoformat(d_ex) - dt.date.fromisoformat(d_ent)).days)
                hold_days.append(days)
                if days < 5:
                    short_lt5d += 1
            except ValueError:
                pass

    avg_hold = round(sum(hold_days) / len(hold_days), 1) if hold_days else None
    notes: List[str] = []
    cra_status = "SAFE_INVESTOR"

    if trades_ytd >= 60 or (closed_ytd >= 10 and short_lt5d / max(closed_ytd, 1) >= 0.50):
        cra_status = "HIGH_TURNOVER_RISK"
        notes.append(
            f"High short-term cadence ({trades_ytd} YTD trades, {short_lt5d} closed <5d) — CRA can audit frequent TFSA trading as business income."
        )
    elif trades_ytd >= 25 or (avg_hold is not None and avg_hold < 14.0 and closed_ytd >= 5):
        cra_status = "WATCH_CADENCE"
        notes.append(
            f"Moderate TFSA turnover ({trades_ytd} YTD trades, avg hold {avg_hold}d) — prefer monthly VM_OUTLOOK rotation & 20–90d thesis holds."
        )
    else:
        notes.append(
            f"Investor-grade TFSA cadence ({trades_ytd} YTD trades) — low turnover protects tax-free status."
        )

    if weighted_bps >= 25.0:
        notes.append(
            f"Est. 15% IRS US dividend withholding drag: ~${total_drag:,.2f}/yr ({weighted_bps:.0f} bps). High-yield US ETFs (XLE/XLP/XLU/XLRE) lose ~40–50 bps/yr in a TFSA."
        )
    else:
        notes.append(
            f"Low US dividend tax drag (~${total_drag:,.2f}/yr · {weighted_bps:.0f} bps). Capital gains & covered calls stay 0% taxed."
        )

    room_cad = getattr(profile, "tfsa_room_cad", None)
    contrib_cad = float(getattr(profile, "tfsa_contributed_ytd_cad", 0.0) or 0.0)
    rem_cad = round(max(0.0, float(room_cad) - contrib_cad), 2) if room_cad is not None else None
    if rem_cad is not None and rem_cad < 500.0:
        notes.append(f"TFSA contribution room nearly full (${rem_cad:,.0f} CAD remaining) — avoid 1%/mo CRA over-contribution penalty!")

    return TfsaSafetyReport(
        account_type=str(getattr(profile, "account_type", "tfsa") or "tfsa"),
        total_tax_drag_usd_yr=total_drag,
        weighted_tax_drag_bps=weighted_bps,
        tax_drag_rows=drag_rows,
        trades_ytd=trades_ytd,
        closed_trades_ytd=closed_ytd,
        avg_hold_days=avg_hold,
        short_hold_trades_lt5d=short_lt5d,
        cra_status=cra_status,
        cra_notes=notes,
        room_cad=float(room_cad) if room_cad is not None else None,
        contributed_ytd_cad=round(contrib_cad, 2),
        room_remaining_cad=rem_cad,
    )
