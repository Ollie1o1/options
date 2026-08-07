"""The starting setup library.

Four tiers, and the last two matter most:

  SIGNAL   The open question. Does conditioning on IV rank, trend or a recent
           drop beat selling on no condition at all? Never measured here.
  ACCESS   The same premium reached with less capital (spreads) or more
           (naked/secured), so the capital constraint is measured, not assumed.
  PROBE    Index versus single name. Prior research found index VRP positive and
           single-name VRP absent; these two make that testable rather than folklore.
  CONTROL  benchmark / null_control / known_negative. Selectivity made this book
           WORSE monotonically, so any signalled setup must beat an unselected
           benchmark before it is believed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .record import StrategyRecord
from .spec import StrategySpec

CREATED = "2026-08-06"
CAP = {"max_capital_at_risk": 4000, "max_concurrent": 2}

_CSP_CAPITAL = ("Cash-secured: needs strike x 100 in cash. At $4,000 that is "
                "roughly sub-$40 underlyings. Ledger median for a CSP was "
                "$21,855 and only 15% were affordable.")
_SPREAD_CAPITAL = ("Defined risk: capital is width x 100 minus credit, about "
                   "$500 for a $5-wide. The capital-efficient route to the same "
                   "premium.")
_COVERED_CAPITAL = ("Requires 100 shares already held. No new capital if the "
                    "HOLDINGS book owns them — currently that book is empty.")
_NAKED_CAPITAL = ("Margin-secured. Taxable account only; registered accounts "
                  "cannot sell naked.")


def _s(sid: str, structure: str, entry: Dict[str, Any], exit_: Dict[str, Any],
       universe: Optional[Dict[str, Any]] = None) -> StrategySpec:
    return StrategySpec(id=sid, version=1, structure=structure,
                        universe=universe or {"strata": ["liquid"],
                                              "max_price": 40},
                        entry=entry, exit=exit_, sizing=CAP,
                        created=CREATED, trial_count=0)


def _r(spec: StrategySpec, name: str, hypothesis: str, signal: Dict[str, Any],
       accounts: List[str], capital: str, role: str,
       status: str = "specified",
       links: Optional[List[str]] = None) -> StrategyRecord:
    return StrategyRecord(
        spec=spec, name=name, hypothesis=hypothesis, signal=signal,
        accounts=accounts, capital_note=capital, status=status,
        evidence={}, cost_profile={}, verdict=None,
        provenance={"created": CREATED, "role": role},
        links=links or ["docs/PROFITABILITY_FINDINGS.md"], amendments=[])


_MANAGED = {"profit_target": 0.5, "stop": 2.0, "hold_to_expiry": False}
_HELD = {"hold_to_expiry": True}

LIBRARY = [
    # ── SIGNAL: the open question — does timing help? ──
    _r(_s("wheel_csp_ivr50", "short_put",
          {"dte": [30, 45], "short_delta": 0.25, "iv_rank_min": 50}, _MANAGED),
       "Wheel entry: cash-secured put, IV rank > 50",
       "Implied vol rich against its own history means the variance premium is "
       "unusually fat. Sell puts on a name you would be content to own; if "
       "assigned, the wheel continues into covered calls.",
       {"iv_rank_min": 50, "no_earnings_before_expiry": True},
       ["tfsa", "taxable"], _CSP_CAPITAL, "candidate", status="idea"),
    _r(_s("csp_post_drop", "short_put",
          {"dte": [30, 45], "short_delta": 0.25, "drop_pct_min": 7,
           "rsi_max": 35}, _MANAGED),
       "Cash-secured put after a sharp drop",
       "Implied vol spikes when price falls, so a post-drop put sells fear at "
       "its most expensive. The risk is that the drop continues — this setup "
       "exists to find out whether the premium compensates.",
       {"drop_pct_min": 7, "rsi_max": 35, "iv_rank_min": 40},
       ["tfsa", "taxable"], _CSP_CAPITAL, "candidate", status="idea"),
    _r(_s("csp_earnings_crush", "short_put",
          {"dte": [7, 21], "short_delta": 0.20, "earnings_within": 5},
          {"close_after_earnings": True, "hold_to_expiry": False}),
       "Cash-secured put into earnings, closed after the crush",
       "Implied vol is systematically bid before earnings and collapses after. "
       "Sell the anticipation, close once it deflates. Highest-variance setup "
       "here: a gap through the strike is exactly the tail this book has "
       "never observed.",
       {"earnings_within": 5, "iv_rank_min": 60},
       ["tfsa", "taxable"], _CSP_CAPITAL, "candidate", status="idea"),
    _r(_s("covered_call_holdings", "covered_call",
          {"dte": [30, 45], "short_delta": 0.25, "above_cost_basis": True},
          _MANAGED, universe={"strata": ["liquid"]}),
       "Covered call against long-term holdings",
       "The most capital-efficient premium available: shares already owned, so "
       "no new capital and no assignment risk beyond losing upside. Ties the "
       "HOLDINGS/TFSA book to income.",
       {"iv_rank_min": 40, "above_cost_basis": True},
       ["tfsa", "taxable"], _COVERED_CAPITAL, "candidate", status="idea"),
    _r(_s("naked_call_extended", "naked_call",
          {"dte": [30, 45], "short_delta": 0.20, "rsi_min": 70}, _MANAGED),
       "Naked call on an extended name  [TAXABLE ONLY]",
       "Selling calls into an overbought, over-extended move. Uncapped loss if "
       "the move continues, which is why it is taxable-only and capped hard.",
       {"rsi_min": 70, "iv_rank_min": 50},
       ["taxable"], _NAKED_CAPITAL, "candidate", status="idea"),

    # ── DIRECTIONAL: a view on where the stock goes, expressed as short premium ──
    _r(_s("bullish_trend_put_spread", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "above_sma50": True, "rsi_min": 50}, _MANAGED,
          universe={"strata": ["liquid"]}),
       "Bullish trend: sell put spread below an uptrend",
       "A bullish view expressed so it wins on up, sideways OR slightly down, "
       "rather than needing the move to clear strike plus premium by a date. "
       "The same view bought as a call won 38% of the time in this book; sold "
       "as a put spread it won 66%.",
       {"above_sma50": True, "rsi_min": 50},
       ["tfsa", "taxable"], _SPREAD_CAPITAL, "directional"),
    _r(_s("bearish_trend_call_spread", "bear_call",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "below_sma50": True, "rsi_max": 50}, _MANAGED,
          universe={"strata": ["liquid"]}),
       "Bearish trend: sell call spread above a downtrend",
       "The bearish mirror. Also the cheapest structure to cross at "
       "$0.050/share, so the directional view is not eaten by friction.",
       {"below_sma50": True, "rsi_max": 50},
       ["tfsa", "taxable"], _SPREAD_CAPITAL, "directional"),
    _r(_s("directional_long_call_control", "long_call",
          {"dte": [30, 45], "target_delta": 0.40, "above_sma50": True,
           "rsi_min": 50}, _MANAGED, universe={"strata": ["liquid"]}),
       "Same bullish view, bought as a call  [EXPRESSION CONTROL]",
       "Identical entry signal to bullish_trend_put_spread, expressed as a long "
       "call instead. Isolates EXPRESSION from SIGNAL: if the spread wins and "
       "this loses on the same days, the lesson is about structure, not "
       "prediction.",
       {"above_sma50": True, "rsi_min": 50},
       ["tfsa", "taxable"],
       "Debit paid up front, ~$700 median. Affordable but historically negative-EV.",
       "expression_control"),

    # ── ACCESS: same premium, different capital ──
    _r(_s("put_spread_ivr50", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "iv_rank_min": 50}, _MANAGED,
          universe={"strata": ["liquid"]}),
       "Put credit spread, IV rank > 50",
       "The capital-efficient route to the same premium as wheel_csp_ivr50: "
       "about $500 at risk instead of $21,855. Directly measures what the "
       "affordability cap costs in edge.",
       {"iv_rank_min": 50}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "candidate"),
    _r(_s("put_spread_ivr50_hold", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "iv_rank_min": 50}, _HELD, universe={"strata": ["liquid"]}),
       "Put credit spread, IV rank > 50, held to expiry",
       "Holding pays only the opening legs and roughly halves the toll — 53% of "
       "credit falls to 27% on measured friction. Paired with the managed "
       "version, this IS the cost experiment.",
       {"iv_rank_min": 50}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "candidate"),
    _r(_s("call_spread_extended", "bear_call",
          {"dte": [30, 45], "short_delta": 0.20, "width": 5.0, "rsi_min": 70},
          _MANAGED, universe={"strata": ["liquid"]}),
       "Call credit spread on an extended name",
       "The defined-risk form of naked_call_extended, and the cheapest measured "
       "structure to cross at $0.050/share. Legal in a registered account.",
       {"rsi_min": 70, "iv_rank_min": 50},
       ["tfsa", "taxable"], _SPREAD_CAPITAL, "candidate"),

    # ── PROBE: index versus single name ──
    _r(_s("csp_index_only", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "iv_rank_min": 50}, _MANAGED,
          universe={"symbols": ["SPY"], "strata": ["legacy"]}),
       "Put spread on the index only",
       "Prior VRP research found the index premium positive. This is that claim "
       "as a testable setup rather than folklore.",
       {"iv_rank_min": 50}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "index_probe",
       links=["docs/DOLT_NEXT_STEPS.md"]),
    _r(_s("csp_single_names", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "iv_rank_min": 50}, _MANAGED,
          universe={"strata": ["broad"], "exclude_etfs": True}),
       "Put spread on single names only",
       "Prior research found single-name equity VRP absent. If this loses while "
       "csp_index_only wins, that result replicates and the desk should stop "
       "selling single-name premium.",
       {"iv_rank_min": 50}, ["tfsa", "taxable"], _SPREAD_CAPITAL,
       "single_name_probe", links=["docs/DOLT_NEXT_STEPS.md"]),

    # ── CONTROL: what makes any of the above believable ──
    _r(_s("benchmark_unselected", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0}, _MANAGED,
          universe={"strata": ["liquid"]}),
       "Unselected put spread, every eligible day  [BENCHMARK]",
       "No signal at all. Unselected short premium was the single positive cell "
       "in the profitability analysis, and selectivity degraded results "
       "monotonically. Every signalled setup above must beat THIS or its signal "
       "is decoration.",
       {}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "benchmark"),
    _r(_s("null_random_days", "bull_put",
          {"dte": [30, 45], "short_delta": 0.25, "width": 5.0,
           "entry_days": "random"}, _MANAGED, universe={"strata": ["liquid"]}),
       "Put spread on random entry days  [NULL CONTROL]",
       "Entry days chosen at random. Separates 'the premium exists' from 'our "
       "timing finds it'. Given no stored metric has ever predicted within a "
       "family, expect this to be uncomfortably competitive.",
       {}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "null_control"),
    _r(_s("null_random_strikes", "bull_put",
          {"dte": [30, 45], "strike_selection": "random", "width": 5.0},
          _MANAGED, universe={"strata": ["liquid"]}),
       "Put spread with random strikes  [NULL CONTROL]",
       "Strike chosen at random within the window. If a delta-targeted setup "
       "cannot beat this, strike selection carries no information.",
       {}, ["tfsa", "taxable"], _SPREAD_CAPITAL, "null_control"),
    _r(_s("known_negative_long_call", "long_call",
          {"dte": [30, 45], "target_delta": 0.40}, _MANAGED,
          universe={"strata": ["liquid"]}),
       "Long call 40-delta  [KNOWN-NEGATIVE TRIPWIRE]",
       "98 live paper trades produced -$17,620 at a scorer IC of -0.020. If the "
       "engine ever reports this profitable, the ENGINE is wrong. Kept dead "
       "deliberately as a tripwire.",
       {}, ["tfsa", "taxable"],
       "Debit paid up front; ~$700 median. Affordable but negative-EV.",
       "known_negative", status="dead"),
]


def seed_library(root: str) -> List[StrategyRecord]:
    """Write every setup into a registry. Idempotent."""
    from .registry import Registry
    reg = Registry(root)
    out = []
    for rec in LIBRARY:
        reg.save(rec.spec)
        out.append(rec)
    return out
