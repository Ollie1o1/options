"""Deterministic "what to do" playbook — no AI, just rules.

A fixed library of pre-written scenario lines. Each scenario has a `when`
predicate over the live briefing `PlaybookState` and a `priority`. `select`
returns the highest-priority matching line (plus a runner-up), with concrete
numbers filled in from the state. A catch-all at priority 0 guarantees a line
always shows. Because it is pure rules, tests assert exactly which line fires
for each archetypal situation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class PlaybookState:
    # Verdict + core signals
    verdict: str = "NEUTRAL"
    trend: float = 0.0          # [-1,1]
    momentum: float = 0.0       # [-1,1]
    rsi: Optional[float] = None
    bounce_rate: Optional[float] = None
    bounce_n: int = 0
    support_dist: Optional[float] = None   # abs fraction to nearest support
    # Structure / regime flags
    below_200d: bool = False
    death_cross: bool = False
    reclaimed_50d: bool = False
    new_60d_low: bool = False
    parabolic: bool = False
    near_resistance: bool = False
    near_200d: bool = False
    # Options / vol
    iv_rank: Optional[float] = None
    skew: Optional[float] = None
    term_backwardated: bool = False
    # Events / flow
    days_to_earnings: Optional[int] = None
    analyst_raises: int = 0
    analyst_cuts: int = 0
    news_sentiment: Optional[float] = None
    price_down_5d: bool = False
    regime: str = ""
    top_driver: str = ""
    # Pre-formatted strings for text templating (e.g. "$189")
    fmt: Dict[str, str] = field(default_factory=dict)


@dataclass
class Scenario:
    id: str
    priority: int
    when: Callable[[PlaybookState], bool]
    text: str


def _has(s: PlaybookState, *keys: str) -> bool:
    return all(k in s.fmt for k in keys)


# Ordered library. Higher priority wins when multiple match.
SCENARIOS: List[Scenario] = [
    # ── Earnings / catalyst (high priority: event risk dominates) ──
    Scenario("earnings_imminent", 95,
             lambda s: s.days_to_earnings is not None and 0 <= s.days_to_earnings <= 3,
             "Earnings in {dte_earn} days — a binary event. Either wait for the "
             "print or size small enough to survive a full gap against you."),
    Scenario("earnings_soon_iv_ramp", 80,
             lambda s: s.days_to_earnings is not None and 4 <= s.days_to_earnings <= 10,
             "Earnings in {dte_earn} days. IV will ramp into it, so long options "
             "bleed unless you specifically want the event — otherwise wait."),
    Scenario("earnings_just_passed", 60,
             lambda s: s.days_to_earnings is not None and -2 <= s.days_to_earnings < 0,
             "Earnings just passed — IV crush is done, so this is a cleaner read. "
             "Trade the reaction, not the anticipation."),

    # ── Downtrend / avoid ──
    Scenario("falling_knife", 90,
             lambda s: s.new_60d_low and s.momentum < -0.2,
             "New multi-week low with momentum still down — a falling knife. No "
             "long until it stops making lower lows; watch {nearest_support}."),
    Scenario("primary_downtrend", 85,
             lambda s: s.below_200d and s.trend < -0.2,
             "Primary trend is down (below the 200-day). This is a downtrend, not "
             "a dip — avoid new longs until it reclaims {ma200}."),
    Scenario("counter_trend_bounce", 78,
             lambda s: s.below_200d and s.near_resistance,
             "A bounce into resistance ({resist}) inside a downtrend. That's a "
             "place to trim or avoid, not to buy."),
    Scenario("fresh_death_cross", 72,
             lambda s: s.death_cross and s.trend < 0,
             "The 50-day just crossed below the 200-day — trend is deteriorating. "
             "Stand aside until it stabilizes."),

    # ── Oversold / capitulation ──
    Scenario("oversold_at_support_bounce", 75,
             lambda s: s.rsi is not None and s.rsi < 30 and (s.support_dist or 1) < 0.03
                       and (s.bounce_rate or 0) >= 0.6 and s.bounce_n >= 10,
             "Oversold right at support with a strong historical bounce "
             "({bounce_rate} in ~2wk). A tactical long with tight risk below "
             "{nearest_support} is reasonable."),
    Scenario("oversold_in_downtrend", 70,
             lambda s: s.rsi is not None and s.rsi < 30 and s.trend < 0,
             "Oversold, but in a downtrend — oversold can stay oversold. Wait for "
             "a reclaim of {ma50} before trusting a bounce."),
    Scenario("big_drop_strong_bounce", 68,
             lambda s: s.price_down_5d and (s.bounce_rate or 0) >= 0.65 and s.bounce_n >= 10,
             "After drops like this, it was higher within ~2 weeks {bounce_rate} "
             "of the time. Odds favor a bounce — a small starter that you scale "
             "into is the play, not the whole position."),
    Scenario("big_drop_thin_sample", 66,
             lambda s: s.price_down_5d and s.bounce_rate is not None and s.bounce_n < 10,
             "Big drop, but too few similar cases to trust the bounce stat "
             "(n={bounce_n}). Treat it as wait until momentum steadies."),

    # ── Pullback / entry timing ──
    Scenario("pullback_to_200d_support", 64,
             lambda s: s.trend > 0 and s.momentum < 0 and s.near_200d
                       and _has(s, "support_200d"),
             "Pullback to a high-value support ({support_200d}, the 200-day) inside "
             "an uptrend. Scale in here with a stop just below it."),
    Scenario("pullback_to_support_falling", 62,
             lambda s: s.verdict == "WAIT" and s.trend > 0 and s.momentum < 0
                       and (s.support_dist or 1) < 0.03,
             "Pullback to support in an uptrend, but momentum is still falling. "
             "Wait for it to stabilize near {nearest_support} before adding; a "
             "starter only if it reclaims {ma50}."),
    Scenario("shallow_dip_wait_for_support", 55,
             lambda s: s.trend > 0 and s.momentum < 0 and (s.support_dist or 0) >= 0.03,
             "A shallow dip that hasn't reached support yet. Be patient — nearest "
             "support is {nearest_support}; let it come to you."),
    Scenario("trend_repairing_reclaimed_50d", 52,
             lambda s: s.trend > 0 and s.reclaimed_50d,
             "Trend is repairing — it just reclaimed the 50-day ({ma50}). A "
             "starter with a stop back below {ma50} is reasonable."),

    # ── Overbought / extended ──
    Scenario("parabolic_let_cool", 58,
             lambda s: s.parabolic,
             "Move is parabolic — extended well beyond its normal pace. Let it "
             "cool off rather than chasing the top tick."),
    Scenario("overbought_extended", 50,
             lambda s: s.rsi is not None and s.rsi > 70 and s.trend > 0,
             "Overbought and extended above the 50-day. Don't chase — wait for a "
             "pullback toward {ma50} for better risk/reward."),
    Scenario("at_resistance_buy_breakout", 48,
             lambda s: s.near_resistance and s.trend >= 0,
             "Sitting just under resistance ({resist}). Expect a stall — buy the "
             "breakout above it, not into it."),

    # ── Volatility / options structure ──
    Scenario("iv_rich_prefer_spreads", 44,
             lambda s: s.iv_rank is not None and s.iv_rank > 0.7,
             "Options are expensive (IV rank {iv_rank}). Favor spreads that sell "
             "premium over outright long calls — you're overpaying for vol."),
    Scenario("iv_cheap_long_premium", 42,
             lambda s: s.iv_rank is not None and s.iv_rank < 0.3,
             "Options are cheap (IV rank {iv_rank}). Long premium (calls/puts) is "
             "favored over spreads here."),
    Scenario("skew_rich_puts", 40,
             lambda s: s.skew is not None and s.skew > 0.15,
             "Downside is well bid (rich put skew). If hedging, put spreads are "
             "more cost-effective than outright puts."),
    Scenario("term_backwardated", 38,
             lambda s: s.term_backwardated,
             "The term structure is backwarded — near-term stress is priced in. "
             "Expect implied vol to mean-revert lower once it settles."),

    # ── News / analyst ──
    Scenario("analyst_cuts_headwind", 36,
             lambda s: s.analyst_cuts >= 2 and s.analyst_raises == 0,
             "Multiple analyst downgrades and no upgrades lately — an institutional "
             "headwind for longs. Demand a better setup before buying."),
    Scenario("analyst_raises_tailwind", 34,
             lambda s: s.analyst_raises >= 2 and s.analyst_cuts == 0,
             "Multiple analyst upgrades and no cuts — a tailwind. Check it isn't "
             "already fully priced before chasing."),
    Scenario("bad_news_already_down", 32,
             lambda s: (s.news_sentiment or 0) < -0.2 and s.price_down_5d,
             "Negative news, but price is already well off its highs — a lot may "
             "be priced in. Watch for stabilization rather than fresh selling."),

    # ── Trend / regime (healthy) ──
    Scenario("healthy_uptrend_entry", 46,
             lambda s: s.trend > 0.3 and s.momentum > 0.15
                       and not (s.rsi is not None and s.rsi > 70),
             "Healthy uptrend that isn't overextended — a normal entry zone. Buy a "
             "starter and add on dips toward {ma50}."),
    Scenario("uptrend_defensive_regime", 30,
             lambda s: s.trend > 0 and "DEFENSIVE" in (s.regime or ""),
             "Trend is up but the broad tape is defensive ({regime}). The setup is "
             "fine — just size smaller than usual until the market steadies."),
    Scenario("sideways_no_edge", 20,
             lambda s: abs(s.trend) <= 0.2 and abs(s.momentum) <= 0.15,
             "Range-bound with no clear edge either way. Wait for a break of the "
             "range — resistance {resist}, support {nearest_support}."),

    # ── Conflicting / fallback ──
    Scenario("signals_conflict", 10,
             lambda s: s.verdict == "NEUTRAL" and s.trend * s.momentum < 0,
             "Signals conflict (trend and momentum disagree). No clean setup — the "
             "firmest single fact is {top_driver}. Wait for them to align."),
    Scenario("catch_all", 0,
             lambda s: True,
             "Mixed picture, no decisive setup. Watch {nearest_support} as support "
             "and {resist} as the level to reclaim before committing."),
]


def select(state: PlaybookState) -> Tuple[Optional[str], Optional[str]]:
    """Return (primary_text, secondary_text), numbers filled in.

    Evaluates every scenario; among those whose `when` is true *and* whose text
    template can be filled from the state, returns the two highest priorities.
    The catch-all guarantees at least one result.
    """
    matches: List[Tuple[int, str, str]] = []
    for sc in SCENARIOS:
        try:
            if sc.when(state):
                text = sc.text.format(**_fill(state))
                matches.append((sc.priority, sc.id, text))
        except (KeyError, IndexError, ValueError):
            # Template needs a field the state didn't provide — skip gracefully.
            continue
    matches.sort(key=lambda m: m[0], reverse=True)
    primary = matches[0][2] if matches else None
    secondary = matches[1][2] if len(matches) > 1 else None
    return primary, secondary


def select_id(state: PlaybookState) -> Optional[str]:
    """The id of the primary scenario (used by tests)."""
    matches: List[Tuple[int, str]] = []
    for sc in SCENARIOS:
        try:
            if sc.when(state):
                sc.text.format(**_fill(state))  # ensure fillable
                matches.append((sc.priority, sc.id))
        except (KeyError, IndexError, ValueError):
            continue
    matches.sort(key=lambda m: m[0], reverse=True)
    return matches[0][1] if matches else None


def _fill(state: PlaybookState) -> Dict[str, str]:
    """Templating dict: pre-formatted strings + a few derived ones."""
    d = dict(state.fmt)
    if state.days_to_earnings is not None:
        d.setdefault("dte_earn", str(abs(state.days_to_earnings)))
    if state.bounce_rate is not None:
        d.setdefault("bounce_rate", f"{state.bounce_rate:.0%}")
    d.setdefault("bounce_n", str(state.bounce_n))
    if state.iv_rank is not None:
        d.setdefault("iv_rank", f"{state.iv_rank:.0%}")
    if state.regime:
        d.setdefault("regime", state.regime)
    if state.top_driver:
        d.setdefault("top_driver", state.top_driver)
    return d
