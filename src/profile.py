"""Unified Wealthsimple TFSA & Account Profile, Persistent Core Holdings, and CSV Importer.

Replaces scattered hardcoded `$500 / $750 / $5,000 / $50,000` defaults with ONE persistent
user profile (`data/tfsa_profile.json`) tailored for Canadian Wealthsimple USD TFSA trading:
  - Tracks TFSA balance, risk budget, defensive T-bill ETF (`SGOV` / `BIL` / `CBIL.TO`),
    persisted Core ETF share counts, and last monthly rebalance date.
  - Estimates live risk-free yield on the uninvested volatility buffer from `macro_rates_cache.json`.
  - Imports Wealthsimple Activity/Monthly Statement CSV exports into Core holdings & `own_picks.db`.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.own_picks import log_equity_pick
from src.paths import repo_path

DEFAULT_PROFILE_PATH = "data/tfsa_profile.json"

CORE_OUTLOOK_ETFS: frozenset[str] = frozenset(
    {
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
        "GLD", "SLV", "TLT", "SMH", "IWM", "QQQ", "SPY", "SGOV", "BIL", "CBIL.TO",
    }
)


@dataclass
class UserProfile:
    broker: str = "wealthsimple_usd"     # "wealthsimple_usd" | "ibkr_canada"
    account_type: str = "tfsa"           # "tfsa" | "taxable"
    tfsa_usd: float = 5000.0
    taxable_usd: float = 0.0
    risk_budget_usd: float = 500.0
    max_portfolio_heat_pct: float = 10.0
    cash_etf: str = "SGOV"               # T-bill ETF for VM_OUTLOOK cash buffer
    fractional_shares: bool = True
    drift_threshold_pct: float = 5.0
    rebalance_interval_days: int = 28
    last_rebalance_date: Optional[str] = None
    tfsa_room_cad: Optional[float] = None
    tfsa_contributed_ytd_cad: float = 0.0
    core_holdings: Dict[str, float] = field(default_factory=dict)
    imported_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_profile(path: str = DEFAULT_PROFILE_PATH) -> UserProfile:
    """Load persisted user profile (never raises; falls back gracefully to defaults + settings.json)."""
    full = Path(repo_path(path))
    data: Dict[str, Any] = {}
    if full.is_file():
        try:
            raw = json.loads(full.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
        except (OSError, ValueError):
            data = {}

    # Inherit account_size / default_budget from settings.json if profile not yet customized
    if "tfsa_usd" not in data or "risk_budget_usd" not in data:
        try:
            from src.settings import load_settings

            st = load_settings()
            if "tfsa_usd" not in data and st.get("account_size"):
                data["tfsa_usd"] = float(st["account_size"])
            if "risk_budget_usd" not in data and st.get("default_budget"):
                data["risk_budget_usd"] = float(st["default_budget"])
        except Exception:
            pass

    base = UserProfile()
    holdings = {
        str(k).upper().strip(): round(float(v), 6)
        for k, v in (data.get("core_holdings") or {}).items()
        if str(k).strip() and float(v) > 0
    }
    return UserProfile(
        broker=str(data.get("broker") or base.broker),
        account_type=str(data.get("account_type") or base.account_type).lower(),
        tfsa_usd=max(100.0, float(data.get("tfsa_usd", base.tfsa_usd))),
        taxable_usd=max(0.0, float(data.get("taxable_usd", base.taxable_usd))),
        risk_budget_usd=max(25.0, float(data.get("risk_budget_usd", base.risk_budget_usd))),
        max_portfolio_heat_pct=max(1.0, float(data.get("max_portfolio_heat_pct", base.max_portfolio_heat_pct))),
        cash_etf=str(data.get("cash_etf") or base.cash_etf).upper().strip(),
        fractional_shares=bool(data.get("fractional_shares", base.fractional_shares)),
        drift_threshold_pct=max(1.0, float(data.get("drift_threshold_pct", base.drift_threshold_pct))),
        rebalance_interval_days=max(7, int(data.get("rebalance_interval_days", base.rebalance_interval_days))),
        last_rebalance_date=data.get("last_rebalance_date"),
        tfsa_room_cad=(float(data["tfsa_room_cad"]) if data.get("tfsa_room_cad") is not None else None),
        tfsa_contributed_ytd_cad=max(0.0, float(data.get("tfsa_contributed_ytd_cad", 0.0) or 0.0)),
        core_holdings=holdings,
        imported_hashes=[str(x) for x in (data.get("imported_hashes") or [])[-500:]],
    )


def save_profile(profile: UserProfile, path: str = DEFAULT_PROFILE_PATH) -> str:
    full = Path(repo_path(path))
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(full) + ".tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, full)
    return str(full)


def estimate_cash_yield_annual(
    rates_cache: str = "data/macro_rates_cache.json",
    default: float = 0.043,
) -> Tuple[float, str]:
    """Return `(annual_yield_decimal, label)` from 3-month T-bill cache for SGOV/BIL buffer."""
    full = repo_path(rates_cache)
    try:
        raw = json.loads(Path(full).read_text(encoding="utf-8"))
        val = raw.get("dgs3mo")
        if val is not None and math.isfinite(float(val)) and 0.0 <= float(val) < 20.0:
            y = float(val) / 100.0
            as_of = (raw.get("as_of") or {}).get("DGS3MO") or "cached"
            return y, f"{y * 100:.2f}% (3m T-bill {as_of})"
    except (OSError, ValueError, TypeError):
        pass
    return default, f"{default * 100:.2f}% (est. T-bill)"


def rebalance_due_status(
    profile: UserProfile,
    *,
    max_drift_pct: float,
    rebalance_recommended: bool,
    today: Optional[dt.date] = None,
) -> Tuple[bool, str]:
    """Check whether monthly TFSA Core rebalance is due (drift threshold OR calendar cadence)."""
    now_d = today or dt.date.today()
    days_since: Optional[int] = None
    if profile.last_rebalance_date:
        try:
            days_since = (now_d - dt.date.fromisoformat(profile.last_rebalance_date[:10])).days
        except ValueError:
            days_since = None

    if not profile.core_holdings:
        return True, f"Initial TFSA Core deployment ready (${profile.tfsa_usd:,.0f} USD)"
    if days_since is not None and days_since >= profile.rebalance_interval_days and rebalance_recommended:
        return True, f"Monthly rebalance due ({days_since}d since {profile.last_rebalance_date}, drift {max_drift_pct:.1f}%)"
    if rebalance_recommended and max_drift_pct >= profile.drift_threshold_pct:
        return True, f"Drift {max_drift_pct:.1f}% >= {profile.drift_threshold_pct:.1f}% threshold"
    if days_since is not None:
        left = max(0, profile.rebalance_interval_days - days_since)
        return False, f"On target (drift {max_drift_pct:.1f}% · next monthly check in {left}d)"
    return False, f"On target (drift {max_drift_pct:.1f}%)"


def apply_rebalance_to_profile(
    profile: UserProfile,
    target_shares_by_ticker: Dict[str, float],
    *,
    rebalance_date: Optional[str] = None,
    path: str = DEFAULT_PROFILE_PATH,
) -> UserProfile:
    """Persist new target Core ETF shares and stamp `last_rebalance_date`."""
    cleaned = {
        str(k).upper(): round(float(v), 4)
        for k, v in target_shares_by_ticker.items()
        if str(k).upper() not in ("CASH", "CASH/SGOV") and float(v) > 0
    }
    profile.core_holdings = cleaned
    profile.last_rebalance_date = rebalance_date or dt.date.today().isoformat()
    save_profile(profile, path=path)
    return profile


def _col(row: Dict[str, str], *candidates: str) -> str:
    lowered = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k is not None}
    for c in candidates:
        if c in lowered and lowered[c] != "":
            return lowered[c]
    return ""


def _parse_float(raw: str) -> Optional[float]:
    cleaned = raw.replace("$", "").replace(",", "").replace("USD", "").replace("CAD", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def import_wealthsimple_csv(
    csv_path: str,
    *,
    profile_path: str = DEFAULT_PROFILE_PATH,
    own_picks_db: str = "data/own_picks.db",
    core_etfs: Set[str] = set(CORE_OUTLOOK_ETFS),
) -> Dict[str, Any]:
    """Idempotently import Wealthsimple Activity/Statement CSV rows into TFSA profile & own_picks.db.

    Handles standard Wealthsimple CSV columns (`transaction`/`activity`, `date`, `symbol`/`ticker`,
    `quantity`/`shares`, `price`/`unit_price`, `amount`, `description`).
      - Core Outlook ETF buys/sells update `profile.core_holdings`.
      - Non-core stock/ETF BUY trades are logged into `own_picks.db` as `wealthsimple_import`.
      - Row SHA-256 fingerprints are recorded in `profile.imported_hashes` so re-importing the
        same monthly CSV never double-counts.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    profile = load_profile(profile_path)
    seen = set(profile.imported_hashes)
    core_updates = 0
    picks_logged = 0
    skipped_dupes = 0
    skipped_other = 0

    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tx = _col(row, "transaction", "activity", "type", "action").upper()
            if not any(tok in tx for tok in ("BUY", "SELL")):
                skipped_other += 1
                continue
            is_buy = "BUY" in tx
            sym = _col(row, "symbol", "ticker", "security").upper().replace("$", "")
            date_s = _col(row, "date", "settlement_date", "trade_date", "transaction_date")[:10]
            qty = _parse_float(_col(row, "quantity", "shares", "units"))
            px = _parse_float(_col(row, "price", "unit_price", "unit price", "fill_price"))
            amt = _parse_float(_col(row, "amount", "net_amount", "total"))
            if not sym or qty is None or abs(qty) <= 0:
                skipped_other += 1
                continue
            qty_abs = abs(qty)
            if (px is None or px <= 0) and amt is not None and qty_abs > 0:
                px = abs(amt) / qty_abs
            if px is None or px <= 0:
                skipped_other += 1
                continue

            fp = hashlib.sha256(f"{date_s}|{tx}|{sym}|{qty_abs:.6f}|{px:.4f}".encode("utf-8")).hexdigest()[:16]
            if fp in seen:
                skipped_dupes += 1
                continue
            seen.add(fp)
            profile.imported_hashes.append(fp)

            if sym in core_etfs:
                cur = float(profile.core_holdings.get(sym, 0.0))
                nxt = round(cur + qty_abs if is_buy else max(0.0, cur - qty_abs), 6)
                if nxt > 1e-6:
                    profile.core_holdings[sym] = nxt
                else:
                    profile.core_holdings.pop(sym, None)
                core_updates += 1
            elif is_buy:
                log_equity_pick(
                    sym,
                    float(px),
                    shares=qty_abs,
                    side="LONG",
                    asset_class="stock",
                    playbook_id="wealthsimple_import",
                    notes=f"Imported from {p.name}",
                    entry_date=date_s or None,
                    db_path=own_picks_db,
                )
                picks_logged += 1
            else:
                skipped_other += 1

    save_profile(profile, path=profile_path)
    return {
        "file": str(p),
        "core_updates": core_updates,
        "picks_logged": picks_logged,
        "skipped_duplicates": skipped_dupes,
        "skipped_non_trade": skipped_other,
        "core_holdings": dict(profile.core_holdings),
    }


def import_wealthsimple_csv_text(
    csv_text: str,
    *,
    filename: str = "wealthsimple_upload.csv",
    profile_path: str = DEFAULT_PROFILE_PATH,
    own_picks_db: str = "data/own_picks.db",
    core_etfs: Set[str] = set(CORE_OUTLOOK_ETFS),
) -> Dict[str, Any]:
    """Import Wealthsimple CSV from raw string text (used by Web Cockpit drag-and-drop)."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", delete=False) as tf:
        tf.write(csv_text)
        tmp_path = tf.name
    try:
        res = import_wealthsimple_csv(
            tmp_path,
            profile_path=profile_path,
            own_picks_db=own_picks_db,
            core_etfs=core_etfs,
        )
        res["file"] = filename
        return res
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def find_latest_wealthsimple_csv(downloads_dir: Optional[str] = None) -> Optional[str]:
    """Locate the newest Wealthsimple activity/statement CSV in ~/Downloads (zero-typing sync)."""
    base = Path(downloads_dir) if downloads_dir else (Path.home() / "Downloads")
    if not base.is_dir():
        return None
    candidates: List[Tuple[float, Path]] = []
    for p in base.glob("*.csv"):
        nm = p.name.lower()
        if any(tok in nm for tok in ("wealthsimple", "activity", "statement", "tfsa", "monthly-statement")):
            try:
                candidates.append((p.stat().st_mtime, p))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return str(candidates[0][1])
