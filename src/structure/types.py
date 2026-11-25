"""Shared types and measured constants for the structure expression engine.

Base rates are measured, not assumed - see docs/OUTLOOK_FINDINGS.md.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

# docs/OUTLOOK_FINDINGS.md: bullish calls hit 66-72%, bearish calls ~30%.
# The asymmetry is the whole point - bearish views are anti-predictive here.
BULLISH_BASE_RATE = 0.68
BEARISH_BASE_RATE = 0.30

LEG_COUNT = {"Long Call": 1, "Long Put": 1, "Short Put": 1,
             "Bull Put": 2, "Bear Call": 2, "Iron Condor": 4}

DEBIT_STRUCTURES = frozenset({"Long Call", "Long Put"})
CREDIT_STRUCTURES = frozenset({"Bull Put", "Bear Call", "Iron Condor",
                               "Short Put"})

BULLISH_STRUCTURES = frozenset({"Bull Put", "Long Call"})
BEARISH_STRUCTURES = frozenset({"Long Put", "Bear Call"})
NEUTRAL_STRUCTURES = frozenset({"Bull Put", "Bear Call", "Iron Condor"})


@dataclass
class StructureMargin:
    strategy: str
    n: int
    wins: int
    losses: int
    avg_win: float
    avg_loss: float
    breakeven_hit: float
    realized_hit: float
    margin: float
    state: str            # ACTIVE | BENCHED | UNPROVEN
    ci_lo: float
    ci_hi: float

    @property
    def ci_includes_zero(self) -> bool:
        return self.ci_lo <= 0.0 <= self.ci_hi


@dataclass
class View:
    symbol: str
    direction: str        # BULLISH | BEARISH | NEUTRAL
    confidence: float     # [0, 1]
    drivers: List[str] = field(default_factory=list)


@dataclass
class Expression:
    strategy: str
    margin: float
    breakeven_hit: float
    realized_hit: float
    capital_required: float
    cost_drag_pct: float
    legs: int
    # Set when a debit structure passes the forward-looking confidence gate
    # despite a negative backward-looking margin. The view is overriding the
    # structure's own track record, and the user must see that explicitly -
    # "my view is better than history" is exactly the unfalsifiable optimism
    # this engine exists to prevent.
    warning: str = ""


@dataclass
class Rejection:
    strategy: str
    reason: str


@dataclass(frozen=True)
class DetailedView:
    """Instrument-agnostic trade view for Shares-vs-Options Workbench (`compare_vehicles`)."""
    symbol: str
    direction: str             # "BULLISH" | "BEARISH" | "NEUTRAL"
    target_move_pct: float     # e.g. +0.08 for +8%, -0.06 for -6%
    horizon_days: int          # e.g. 30
    vol_mult: float = 1.0      # 1.0 = agree with market vol; <1 = expect crush
    risk_budget_usd: float = 500.0
    account_type: str = "tfsa" # "tfsa" | "taxable"
    owns_100_shares: bool = False
    stop_loss_pct: float = 0.06


@dataclass(frozen=True)
class VehicleComparison:
    vehicle: str               # "Shares" | "Deep-ITM LEAPS" | "Debit Vertical" | "Wide Credit Spread" | "Covered Call" | "Single Option"
    description: str           # e.g. "Buy 25 sh @ $200.00 (stop $188.00)" or "Bull Put 190/180 90d"
    legs: int
    tfsa_legal: bool
    capital_at_risk_usd: float
    cash_outlay_usd: float
    max_profit_usd: float
    friction_usd: float
    friction_pct_of_reward: float
    ev_mkt_net_usd: float
    ev_view_net_usd: float
    pop_view: float
    rr_ratio: float
    theta_per_day_usd: float
    delta_dollars: float
    walk_away_label: str
    walk_away_steps: Tuple[str, ...] = ()
    recommended: bool = False
    caution: str = ""
