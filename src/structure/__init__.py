"""Structure expression engine - view in, tradeable structure out.

Display-only. Never writes to paper_trades.db and never touches the Phase-1 gate.
"""
from .types import (BEARISH_BASE_RATE, BEARISH_STRUCTURES, BULLISH_BASE_RATE,
                    BULLISH_STRUCTURES, CREDIT_STRUCTURES, DEBIT_STRUCTURES,
                    LEG_COUNT, NEUTRAL_STRUCTURES, Expression, Rejection,
                    StructureMargin, View)

__all__ = ["StructureMargin", "View", "Expression", "Rejection", "LEG_COUNT",
           "DEBIT_STRUCTURES", "CREDIT_STRUCTURES", "BULLISH_STRUCTURES",
           "BEARISH_STRUCTURES", "NEUTRAL_STRUCTURES", "BULLISH_BASE_RATE",
           "BEARISH_BASE_RATE"]
