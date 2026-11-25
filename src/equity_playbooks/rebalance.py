"""TFSA Core Portfolio Rebalance Calculator for VALIDATED Playbook `VM_OUTLOOK_TOP3`.

Turns the target weights from `scan_vol_managed_outlook` (top-3 Outlook ETFs scaled by
`clip(0.14 / sigma_20, 0.25, 1.00)` + remainder in CASH/T-bills) into exact share
counts, BUY/SELL/HOLD delta orders, and drift checks so the user can execute their
monthly TFSA rebalance in 60 seconds with zero spreadsheet work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class RebalanceOrder:
    ticker: str
    action: str  # "BUY" | "SELL" | "HOLD" | "RESERVE"
    target_weight_pct: float
    current_weight_pct: float
    drift_pct: float
    spot: float
    current_shares: float
    target_shares: float
    delta_shares: float
    delta_usd: float
    target_usd: float


@dataclass(frozen=True)
class RebalancePlan:
    account_usd: float
    vol_scale: float
    spy_rv20: Optional[float]
    max_drift_pct: float
    rebalance_recommended: bool
    drift_threshold_pct: float
    fractional_shares: bool
    estimated_cash_after_usd: float
    orders: List[RebalanceOrder] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_usd": self.account_usd,
            "vol_scale": self.vol_scale,
            "spy_rv20": self.spy_rv20,
            "max_drift_pct": self.max_drift_pct,
            "rebalance_recommended": self.rebalance_recommended,
            "drift_threshold_pct": self.drift_threshold_pct,
            "fractional_shares": self.fractional_shares,
            "estimated_cash_after_usd": self.estimated_cash_after_usd,
            "orders": [asdict(o) for o in self.orders],
        }


def parse_holdings_spec(raw: str) -> Dict[str, float]:
    """Parse `'XLK:10,GLD:5.5,SMH:4'` into `{'XLK': 10.0, 'GLD': 5.5, 'SMH': 4.0}`."""
    out: Dict[str, float] = {}
    if not raw or not raw.strip():
        return out
    for part in raw.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        tk, qty_s = item.split(":", 1)
        tk_u = tk.strip().upper()
        try:
            qty = float(qty_s.strip())
        except ValueError:
            continue
        if tk_u and qty >= 0:
            out[tk_u] = out.get(tk_u, 0.0) + qty
    return out


def compute_tfsa_rebalance(
    core_allocation: Sequence[Dict[str, Any]],
    *,
    account_usd: float,
    current_shares: Optional[Dict[str, float]] = None,
    spot_lookup: Optional[Dict[str, float]] = None,
    fractional: bool = False,
    drift_threshold_pct: float = 5.0,
    min_order_usd: float = 25.0,
) -> RebalancePlan:
    """Compute exact BUY/SELL/HOLD orders to align a TFSA account with `VM_OUTLOOK_TOP3`."""
    if account_usd <= 0:
        raise ValueError("account_usd must be positive")
    cur_map = {str(k).upper(): float(v) for k, v in (current_shares or {}).items() if float(v) > 0}
    extra_spots = {str(k).upper(): float(v) for k, v in (spot_lookup or {}).items() if float(v) > 0}

    target_by_ticker: Dict[str, Dict[str, Any]] = {}
    vol_scale = 1.0
    spy_rv20: Optional[float] = None
    for row in core_allocation:
        tk = str(row.get("ticker", "")).upper()
        if not tk:
            continue
        target_by_ticker[tk] = dict(row)
        if row.get("vol_scale") is not None:
            vol_scale = float(row["vol_scale"])
        if row.get("spy_rv20") is not None:
            spy_rv20 = float(row["spy_rv20"])

    # Include any currently held ETFs that dropped out of the top-3 (must be SOLD)
    all_etfs: List[str] = [tk for tk in target_by_ticker if tk != "CASH"]
    for tk in sorted(cur_map.keys()):
        if tk != "CASH" and tk not in all_etfs:
            all_etfs.append(tk)

    orders: List[RebalanceOrder] = []
    max_drift = 0.0
    deployed_after = 0.0

    for tk in all_etfs:
        row = target_by_ticker.get(tk)
        t_weight = float(row.get("target_weight", 0.0)) if row else 0.0
        spot = (
            float(row["spot"])
            if (row and row.get("spot") and float(row["spot"]) > 0)
            else extra_spots.get(tk, 0.0)
        )
        cur_sh = cur_map.get(tk, 0.0)
        cur_usd = cur_sh * spot
        cur_w = cur_usd / account_usd if account_usd > 0 else 0.0
        target_usd = account_usd * t_weight

        if spot > 0:
            raw_target_sh = target_usd / spot
            target_sh = round(raw_target_sh, 4) if fractional else float(int(raw_target_sh))
        else:
            target_sh = 0.0

        delta_sh = round(target_sh - cur_sh, 4) if fractional else float(round(target_sh - cur_sh))
        delta_usd = delta_sh * spot
        drift_pct = (t_weight - cur_w) * 100.0
        max_drift = max(max_drift, abs(drift_pct))

        if abs(delta_usd) < min_order_usd or abs(delta_sh) < 1e-6:
            action = "HOLD"
            delta_sh = 0.0
            delta_usd = 0.0
            deployed_after += cur_sh * spot
        elif delta_sh > 0:
            action = "BUY"
            deployed_after += target_sh * spot
        else:
            action = "SELL"
            deployed_after += target_sh * spot

        orders.append(
            RebalanceOrder(
                ticker=tk,
                action=action,
                target_weight_pct=round(t_weight * 100.0, 1),
                current_weight_pct=round(cur_w * 100.0, 1),
                drift_pct=round(drift_pct, 1),
                spot=round(spot, 2),
                current_shares=round(cur_sh, 4),
                target_shares=round(target_sh, 4),
                delta_shares=round(delta_sh, 4),
                delta_usd=round(delta_usd, 2),
                target_usd=round(target_usd, 2),
            )
        )

    # Cash / T-bill reserve row
    cash_row = target_by_ticker.get("CASH")
    cash_target_w = float(cash_row.get("target_weight", 0.0)) if cash_row else max(0.0, 1.0 - sum(
        float(target_by_ticker[t].get("target_weight", 0.0)) for t in target_by_ticker if t != "CASH"
    ))
    est_cash_after = max(0.0, account_usd - deployed_after)
    cur_etf_total = sum(cur_map.get(tk, 0.0) * (
        float(target_by_ticker[tk]["spot"]) if (tk in target_by_ticker and target_by_ticker[tk].get("spot"))
        else extra_spots.get(tk, 0.0)
    ) for tk in all_etfs)
    cur_cash_w = max(0.0, (account_usd - cur_etf_total) / account_usd)
    cash_drift = (cash_target_w - cur_cash_w) * 100.0
    max_drift = max(max_drift, abs(cash_drift))

    orders.append(
        RebalanceOrder(
            ticker="CASH/SGOV",
            action="RESERVE",
            target_weight_pct=round(cash_target_w * 100.0, 1),
            current_weight_pct=round(cur_cash_w * 100.0, 1),
            drift_pct=round(cash_drift, 1),
            spot=1.0,
            current_shares=round(max(0.0, account_usd - cur_etf_total), 2),
            target_shares=round(est_cash_after, 2),
            delta_shares=0.0,
            delta_usd=round(est_cash_after - max(0.0, account_usd - cur_etf_total), 2),
            target_usd=round(account_usd * cash_target_w, 2),
        )
    )

    # Order SELLs first (free up cash), then BUYs, then HOLDs, then RESERVE
    action_rank = {"SELL": 0, "BUY": 1, "HOLD": 2, "RESERVE": 3}
    orders.sort(key=lambda o: (action_rank.get(o.action, 9), -abs(o.delta_usd), o.ticker))

    has_actionable = any(o.action in ("BUY", "SELL") for o in orders)
    rebalance_rec = bool(has_actionable and (max_drift >= drift_threshold_pct or not cur_map))

    return RebalancePlan(
        account_usd=round(account_usd, 2),
        vol_scale=round(vol_scale, 3),
        spy_rv20=round(spy_rv20, 4) if spy_rv20 is not None else None,
        max_drift_pct=round(max_drift, 1),
        rebalance_recommended=rebalance_rec,
        drift_threshold_pct=drift_threshold_pct,
        fractional_shares=fractional,
        estimated_cash_after_usd=round(est_cash_after, 2),
        orders=orders,
    )
