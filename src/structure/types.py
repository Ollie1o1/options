"""Shared types and measured constants for the structure expression engine.

Base rates are measured, not assumed - see docs/OUTLOOK_FINDINGS.md.
"""
from dataclasses import dataclass, field
from typing import List

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


@dataclass
class Rejection:
    strategy: str
    reason: str
