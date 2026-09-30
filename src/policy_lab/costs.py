"""Exit pricing under three cost assumptions.

`mid` exists only as a diagnostic: a lab that replays exits at mid will report
an edge that no one could have traded. `cross` is the default everywhere and is
the only setting a promotion decision may use, matching the `"cross"` convention
already validated against real quotes in src/candidate_verdict.py.

MANDATORY AMENDMENT (Task 2, 2026-09-22): the primary research corpus stores
spread marks with bid and ask 100% NULL for a large share of rows — only `mid`
exists (94,479 of 140,939 marks; 827,330 of 827,330 marks attached to closed
Bull Put positions). A loader synthesizing bid=ask=mid for those rows would
make `cross()` silently return mid — a free-fill replay that manufactures
edge nobody could have traded. `PathPoint.spread_imputed` marks those rows,
and `close_price` widens an imputed mid by `imputed_half_spread` instead of
trusting the (meaningless) synthesized bid/ask.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """How much it costs to close a position on a given day.

    A credit structure is closed by buying it back, so crossing means paying
    the ASK. A debit structure is closed by selling, so crossing means taking
    the BID. Both directions cost the trader money; a result where `cross`
    beats `mid` is a sign error, not a discovery.

    `imputed_half_spread` is the fallback cost applied to a `PathPoint` whose
    bid/ask were synthesized from mid (`spread_imputed=True`), under `cross`
    and `surface`. The default, 0.095, is the measured mean
    `(ask - bid) / mid` on the 46,460 real single-leg quotes in the corpus.
    """
    setting: str
    imputed_half_spread: float = 0.095

    SETTINGS = ("mid", "cross", "surface")

    def __post_init__(self) -> None:
        if self.setting not in self.SETTINGS:
            raise ValueError(
                f"unknown cost setting {self.setting!r}; "
                f"expected one of {self.SETTINGS}")

    @classmethod
    def mid(cls) -> "CostModel":
        return cls("mid")

    @classmethod
    def cross(cls) -> "CostModel":
        return cls("cross")

    @classmethod
    def surface(cls) -> "CostModel":
        return cls("surface")

    def close_price(self, point, is_credit: bool) -> float:
        """What closing costs (credit) or yields (debit) on this day.

        For a point with `spread_imputed=True`, `cross` and `surface` ignore
        the synthesized bid/ask entirely and widen `mid` by
        `imputed_half_spread` instead — synthesized bid/ask are not a real
        quote and must never be crossed as one. `mid` ignores
        `spread_imputed` in all cases.
        """
        if self.setting == "mid":
            return float(point.mid)
        # "cross" and "surface" (surface is deliberately identical to cross
        # until the Corpus B fit lands in a later task, which is the
        # conservative direction).
        if point.spread_imputed:
            mid = float(point.mid)
            half = self.imputed_half_spread
            if is_credit:
                return mid * (1 + half / 2)
            return mid * (1 - half / 2)
        return float(point.ask) if is_credit else float(point.bid)
