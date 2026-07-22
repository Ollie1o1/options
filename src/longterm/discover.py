"""Long-term discovery scan: find candidates worth watching in a sector.

Two-tier design (see docs/superpowers/specs/2026-07-20-longterm-discover-design.md):

  FAST tier (every candidate, zero extra network calls beyond the one batched
  price fetch already used by the buy-zone watcher): drawdown, distance below
  the 200-day moving average, 12-1 month momentum, real support/resistance
  levels, and this name's own empirical bounce-odds after drops this size.

  DEEP tier (bounded to the top ~6 ranked candidates, each a separate network
  round-trip): insider cluster-buy activity (EDGAR Form 4), days to next
  earnings, and lightweight fundamentals (P/E, margins, growth).

No claim of predictive edge anywhere in this module. Drawdown, support
levels, and bounce rates are historical facts about a stock's own past
behavior, not forecasts. Insider buying and fundamentals are descriptive
context, not a "buy signal" — a name can show every positive sign here and
still be a value trap.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .cdr import cdr_for
from .plan import Tranche
from .zones import Snapshot

logger = logging.getLogger(__name__)

# Mid-cap-or-larger, average volume > 200k shares/day, US-listed — screens out
# illiquid and penny-stock noise before anything else runs. Every sector
# filter below is built on this same gate.
_QUALITY_GATE = "cap_midover,sh_avgvol_o200,geo_usa"

# Finviz screener filter-code fragments (https://finviz.com/screener.ashx).
# ind_* = industry-level filter (narrower); sec_* = sector-level (broader).
# CONSUMER intentionally maps to Consumer Defensive (staples), not Cyclical —
# staples is the more common "value/long-term hold" hunting ground.
SECTOR_FILTERS: Dict[str, str] = {
    "SEMICONDUCTORS": f"ind_semiconductors,{_QUALITY_GATE}",
    "TECH": f"sec_technology,{_QUALITY_GATE}",
    "BANKS": f"ind_banks,{_QUALITY_GATE}",
    "HEALTHCARE": f"sec_healthcare,{_QUALITY_GATE}",
    "ENERGY": f"sec_energy,{_QUALITY_GATE}",
    "CONSUMER": f"sec_consumerdefensive,{_QUALITY_GATE}",
    "INDUSTRIALS": f"sec_industrials,{_QUALITY_GATE}",
    "UTILITIES": f"sec_utilities,{_QUALITY_GATE}",
    "REALESTATE": f"sec_realestate,{_QUALITY_GATE}",
    "MATERIALS": f"sec_basicmaterials,{_QUALITY_GATE}",
    "COMMUNICATIONS": f"sec_communicationservices,{_QUALITY_GATE}",
}


def universe(sector_keyword: str, limit: int = 30) -> List[str]:
    """Ticker list for a sector keyword, quality-filtered, most-liquid first.

    Raises ValueError (with the valid keyword list) for an unrecognized
    keyword — a typo should never silently scan the wrong thing or an empty
    universe. A recognized keyword whose live Finviz fetch fails returns an
    empty list rather than raising: unlike get_squeeze_universe() in
    src/squeeze/universe.py, finviz_tickers() itself has NO try/except of
    its own around its network calls, so this function wraps the call
    directly and degrades to [] on any Exception, so the caller can
    distinguish "bad input" (ValueError) from "the network is down right
    now" ([] — a legitimate scan result caller code already treats as "no
    candidates").
    """
    from src.squeeze.universe import finviz_tickers

    keyword = sector_keyword.upper()
    f_params = SECTOR_FILTERS.get(keyword)
    if f_params is None:
        valid = ", ".join(sorted(SECTOR_FILTERS))
        raise ValueError(f"unknown sector {sector_keyword!r} — valid: {valid}")
    try:
        return finviz_tickers(f_params, order="-averagevolume", limit=limit)
    except Exception as exc:
        logger.warning("Finviz fetch failed for sector %s (%s); returning empty universe",
                       keyword, exc)
        return []


# Fallback ladder steps when a name has too little history for real support
# levels (support_resistance_levels needs >= 50 closes for even its weakest
# candidate, the 50d MA). -10% / -20% mirrors the buy-zone watcher's own
# default philosophy: transparent, round, never dressed up as a prediction.
_FALLBACK_STEPS = (0.0, -0.10, -0.20)

# Deep-tier and board sizing (also used by scan() in Task 4/5) — named here
# since suggest_ladder's "at most 2 supports" rule is part of the same
# "3 tranches total" contract as the rest of this module.
_MAX_LADDER_SUPPORTS = 2


@dataclass
class CandidateRead:
    """One sector-scan candidate's free, per-candidate context.

    Every field here comes from the single batched price-history fetch
    already paid for by the caller (see scan() in a later task) — nothing
    in this dataclass costs an extra network round-trip to produce.

    Fields:
      ticker              -- the underlying symbol.
      spot                -- current price used for every distance/pct calc below.
      drawdown_pct        -- (spot/high_52w - 1) * 100; percent below the 52w high,
                              always <= 0 (0 only if spot == high_52w).
      ma200_distance_pct  -- (spot/ma200 - 1) * 100; percent above/below the 200d
                              moving average, signed (negative = below). None if
                              the snapshot doesn't carry a 200d MA (< 200 closes).
      momentum_12_1       -- 12-month return skipping the most recent month, as a
                              fraction (e.g. -0.18 means -18%). None if the
                              snapshot has fewer than 252 closes of history.
      supports            -- support levels below spot, nearest first; each entry
                              is {"label", "level", "pct"} from
                              levels.support_resistance_levels()["supports"].
                              Historical technical levels, not a prediction.
      bounce              -- this name's own empirical bounce-odds after selloffs
                              of its current magnitude, from levels.bounce_stats().
                              A backward-looking base rate, not a forecast.
      suggested_ladder    -- starting-point 3-tranche buy ladder from
                              suggest_ladder(); every level is editable before ADD.
      rsi                 -- Wilder RSI(14) on `snapshot.closes` (src/levels.py:rsi).
                              None if fewer than 15 closes. Textbook zones only
                              (<30 oversold, >70 overbought) — descriptive, not
                              a signal.
      ann_vol_pct         -- annualized realized volatility, i.e.
                              snapshot.daily_sigma * sqrt(252) * 100, from the
                              same 63-day window data.snapshot_from_closes()
                              already computes. Zero extra network cost.
      cdr_ticker          -- the Canadian Depositary Receipt ticker for this
                              name, if one exists in cdr_map.json (see
                              src/longterm/cdr.py), else None. Annotation
                              only — not a recommendation.
    """
    ticker: str
    spot: float
    drawdown_pct: float
    ma200_distance_pct: Optional[float]
    momentum_12_1: Optional[float]
    supports: List[Any] = field(default_factory=list)
    bounce: Dict[str, Any] = field(default_factory=dict)
    suggested_ladder: List[Tranche] = field(default_factory=list)
    rsi: Optional[float] = None
    ann_vol_pct: Optional[float] = None
    cdr_ticker: Optional[str] = None


def suggest_ladder(spot: float, supports: List[Any]) -> List[Tranche]:
    """A 3-tranche equal-weight ladder: spot, then up to 2 real support
    levels strictly below spot (nearest first), preferring real levels over
    synthetic percentage steps wherever at least one real level exists.

    Exact behavior by how many valid supports (entries with "level" < spot)
    are available:
      0 supports -> full synthetic fallback: spot, spot*0.90, spot*0.80
        (i.e. spot / -10% / -20%).
      1 support  -> spot, that one real support, then a single synthetic
        step below it: spot*0.80 (spot / -20%) — the real level is never
        discarded; only the third tranche is synthetic, since there's no
        second real level to use.
      2+ supports -> spot plus the 2 nearest real supports below spot,
        nearest first (the 3rd-nearest and beyond are ignored — this
        function only ever returns 3 tranches).

    This is a starting point, never a prediction — every level here is
    editable before ADD.

    Args:
      spot: current price; always becomes the ladder's first tranche level.
      supports: candidate support entries, each a dict with at least
        "level" (float). Entries at or above spot are ignored.

    Returns:
      A list of Tranche(level, weight) with equal weights that sum to 1.0,
      ordered spot-first then descending by level.
    """
    below = sorted((s for s in supports if s["level"] < spot),
                   key=lambda s: -s["level"])[:_MAX_LADDER_SUPPORTS]
    if len(below) == 0:
        levels = [spot] + [spot * (1.0 + step) for step in _FALLBACK_STEPS[1:]]
    elif len(below) == 1:
        real_level = below[0]["level"]
        levels = [spot, real_level, spot * (1.0 + _FALLBACK_STEPS[2])]
    else:
        levels = [spot] + [s["level"] for s in below]
    weight = 1.0 / len(levels)
    return [Tranche(level=lvl, weight=weight) for lvl in levels]


def fast_context(snapshot: Snapshot) -> CandidateRead:
    """Free per-candidate context: drawdown, MA-distance, momentum, real
    support/resistance levels, and this name's own empirical bounce-odds —
    all derived from `snapshot.closes`, the price history already fetched
    for the whole scan universe. Zero additional network calls.

    Pure function: no I/O, no network, deterministic given the snapshot.
    See CandidateRead's field docs for exact units/sign conventions.
    """
    from src.levels import bounce_stats, rsi, support_resistance_levels
    from src.outlook.factors import mom_12_1
    import math

    drawdown_pct = (snapshot.spot / snapshot.high_52w - 1.0) * 100.0
    ma200_distance_pct = (
        (snapshot.spot / snapshot.ma200 - 1.0) * 100.0
        if snapshot.ma200 else None
    )
    momentum = mom_12_1(snapshot.closes)
    levels = support_resistance_levels(snapshot.closes, snapshot.spot)
    bounce = bounce_stats(snapshot.closes)
    ladder = suggest_ladder(snapshot.spot, levels["supports"])
    ann_vol_pct = snapshot.daily_sigma * math.sqrt(252) * 100.0 if snapshot.daily_sigma else None
    return CandidateRead(
        ticker=snapshot.ticker,
        spot=snapshot.spot,
        drawdown_pct=drawdown_pct,
        ma200_distance_pct=ma200_distance_pct,
        momentum_12_1=momentum,
        supports=levels["supports"],
        bounce=bounce,
        suggested_ladder=ladder,
        rsi=rsi(snapshot.closes),
        ann_vol_pct=ann_vol_pct,
        cdr_ticker=cdr_for(snapshot.ticker),
    )


# Bounded to the deep tier only (see scan() in a later task) — this is the
# one function in this module that touches yfinance's slow, unbatched
# per-ticker `.info` endpoint. Never called for the full scan universe.
_FUNDAMENTALS_FIELDS = (
    "trailingPE", "forwardPE", "profitMargins",
    "revenueGrowth", "earningsGrowth", "returnOnEquity",
)


@dataclass
class DeepRead:
    """Bounded-cost, per-ticker context — one network round-trip per field,
    fetched only for the top-ranked handful of candidates (see scan()).
    Each field is independently None on failure; a bad insider fetch never
    blocks earnings or fundamentals from showing.
    """
    ticker: str
    insider: Optional[Dict[str, Any]] = None
    earnings_days: Optional[int] = None
    fundamentals: Optional[Dict[str, Any]] = None


def _insider_read(ticker: str) -> Optional[Dict[str, Any]]:
    """EDGAR Form 4 cluster-buy score for the trailing 90 days, or None.

    Chains cik_for -> recent_form4 -> fetch_form4_xml/parse_form4 per filing
    -> cluster_score, the same sequence src/insider/__main__.py's CLI uses.
    Every step degrades to None/[] internally per src/insider/edgar.py's own
    contract; this wrapper adds one outer try/except so a change in that
    contract can never propagate into a crashed scan.
    """
    from src.insider import edgar
    from src.insider.parse import parse_form4
    from src.insider.signal import cluster_score

    try:
        cik = edgar.cik_for(ticker)
        if not cik:
            return None
        transactions: List[Dict[str, Any]] = []
        for filing in edgar.recent_form4(cik, since_days=120):
            xml = edgar.fetch_form4_xml(cik, filing["accession"], filing["document"])
            if xml:
                transactions.extend(parse_form4(xml))
        return cluster_score(transactions, window_days=90)
    except Exception:
        return None


def _earnings_read(ticker: str) -> Optional[int]:
    """Days until next earnings (0 == today, negative never happens since
    next_earnings_date only returns future dates), or None (no key
    configured, no data, or any error) — same config.json/env-var key
    resolution already used by board.py's buy-zone banner
    (`_earnings_flags`), duplicated here rather than imported since it
    returns a different shape (days, not a %m-%d label) for this module's
    narrative use.
    """
    import datetime as dt
    import json

    from src.earnings_provider import next_earnings_date, resolve_api_key

    try:
        try:
            with open("config.json") as f:
                cfg = json.load(f)
        except Exception:
            cfg = None
        when = next_earnings_date(ticker, api_key=resolve_api_key(cfg))
        if when is None:
            return None
        return (when.date() - dt.date.today()).days
    except Exception:
        return None


def _fundamentals_read(ticker: str) -> Optional[Dict[str, Any]]:
    """Six raw fundamentals fields from yfinance's `.info`, or None.

    Fields: trailingPE, forwardPE, profitMargins, revenueGrowth,
    earningsGrowth, returnOnEquity — passed through exactly as yfinance
    reports them (fractions for margins/growth/ROE, e.g. 0.22 == 22%; raw
    ratios for the two P/E fields). A field yfinance doesn't have for this
    name comes back None (dict.get), the whole result is None only if the
    `.info` call itself fails.

    Never compared against a "sector average" — a 6-ticker deep tier is too
    small a sample to make that claim honestly (see module docstring).
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        return {name: info.get(name) for name in _FUNDAMENTALS_FIELDS}
    except Exception:
        return None


def deep_context(ticker: str) -> DeepRead:
    """Bounded-cost context for one candidate: insider activity, earnings
    timing, fundamentals — each independently fetched and independently
    allowed to fail. A crash inside any single source (not just the
    None-degrading paths each source already handles internally) is caught
    here too, so one broken source can never take the other two down with
    it. Call only for the top-ranked handful of candidates (see scan()),
    never for a whole scan universe.
    """
    insider: Optional[Dict[str, Any]] = None
    earnings_days: Optional[int] = None
    fundamentals: Optional[Dict[str, Any]] = None
    try:
        insider = _insider_read(ticker)
    except Exception:
        pass
    try:
        earnings_days = _earnings_read(ticker)
    except Exception:
        pass
    try:
        fundamentals = _fundamentals_read(ticker)
    except Exception:
        pass
    return DeepRead(
        ticker=ticker,
        insider=insider,
        earnings_days=earnings_days,
        fundamentals=fundamentals,
    )


def _fmt_pct(value: Optional[float], from_fraction: bool = False) -> str:
    """Consistent %-sign formatting for the narrative; 'n/a' for None.

    Args:
      value: a percent already (e.g. -8.5) unless from_fraction is True, in
        which case it's a fraction (e.g. 0.22 == 22%) as yfinance reports
        margins/growth/ROE.
    """
    if value is None:
        return "n/a"
    pct = value * 100.0 if from_fraction else value
    return f"{pct:+.0f}%"


# Default cluster_score() window (see src/insider/signal.py and
# _insider_read's own call above); used only as a fallback label if an
# insider dict predates the "window_days" key, so the sentence never claims
# a window it didn't actually measure.
_DEFAULT_INSIDER_WINDOW_DAYS = 90


def insight_line(candidate: CandidateRead, deep: Optional[DeepRead]) -> str:
    """One synthesized sentence per candidate. Every clause traces to a
    concrete field on `candidate`/`deep` — no clause is invented, and a
    missing deep-tier field simply drops its clause rather than printing a
    placeholder ('None' must never appear in the returned string).

    Never claims predictive edge: bounce odds are labeled as a historical
    base rate with its sample size (n=), never a forecast; insider activity
    and fundamentals are descriptive context, never framed as a
    recommendation or a "buy signal".

    Args:
      candidate: this ticker's free-tier context (always present).
      deep: this ticker's bounded-cost context, or None if it wasn't in the
        deep-tier count for this scan (see scan()).

    Returns:
      A single "; "-joined sentence ending in ".". Clause count varies from
      1 (drawdown only, no supports/bounce data, no deep tier) up to 5 (all
      fast-tier clauses present plus all three deep-tier clauses).
    """
    clauses = [f"{candidate.ticker}: {candidate.drawdown_pct:+.0f}% off ATH"]

    if candidate.supports:
        nearest = candidate.supports[0]  # nearest-first, see CandidateRead docstring
        clauses.append(f"${candidate.spot:,.0f} ≈ {nearest['label']} support")

    horizons = (candidate.bounce or {}).get("by_horizon") or {}
    twenty_day = horizons.get(20)
    if twenty_day and twenty_day.get("n"):
        rate = twenty_day["bounce_rate"]
        n = twenty_day["n"]
        clauses.append(
            f"historically finished higher {rate * 100:.0f}% of the time "
            f"20 trading days after a drop this size (n={n})"
        )

    if deep and deep.insider and deep.insider.get("label") not in (None, "NONE"):
        buyers = deep.insider.get("n_buyers", 0)
        value = deep.insider.get("buy_value", 0.0)
        window = deep.insider.get("window_days", _DEFAULT_INSIDER_WINDOW_DAYS)
        clauses.append(
            f"insiders bought ${value:,.0f} across {buyers} buyer"
            f"{'s' if buyers != 1 else ''} in the last {window}d"
        )

    if deep and deep.earnings_days is not None:
        clauses.append(f"earnings in {deep.earnings_days} days")

    if deep and deep.fundamentals:
        pe = deep.fundamentals.get("trailingPE")
        margin = deep.fundamentals.get("profitMargins")
        if pe is not None:
            margin_txt = (f", margins {_fmt_pct(margin, from_fraction=True)}"
                          if margin is not None else "")
            clauses.append(f"P/E {pe:.0f}{margin_txt}")

    return "; ".join(clauses) + "."


def scan(sector_keyword: str, universe_limit: int = 30, board_limit: int = 15,
         deep_limit: int = 6) -> List[Tuple[CandidateRead, Optional[DeepRead]]]:
    """Full discovery scan: universe -> one batched price fetch -> free
    context for every candidate -> rank by drawdown -> bounded-cost deep
    context for the top `deep_limit` ranked candidates.

    Args:
      sector_keyword: key into SECTOR_FILTERS (case-insensitive); see
        universe()'s own contract for the ValueError on an unknown keyword.
      universe_limit: how many tickers to pull from the sector screen before
        any price data is fetched.
      board_limit: how many ranked candidates to return in total (after the
        drawdown sort); the "board" the caller renders.
      deep_limit: how many of those ranked candidates (from the top,
        i.e. most beaten-down first) get the extra, per-ticker deep_context()
        network round-trips. deep_limit <= board_limit in normal use; a
        larger deep_limit has no effect since only board_limit candidates
        ever exist to deepen.

    Returns:
      A list of (CandidateRead, Optional[DeepRead]) pairs, length
      <= board_limit, ordered most-beaten-down-first (ascending
      drawdown_pct, i.e. most negative — the biggest drop — sorts first).
      DeepRead is non-None only for the first `deep_limit` entries in that
      order; every entry after that carries None rather than a stale or
      empty DeepRead. A ticker returned by universe() with no snapshot (a
      single failed fetch inside the one batched call) is silently skipped,
      not raised — the scan degrades to "fewer candidates," never a crash,
      matching the rest of this module's failure philosophy.
    """
    from .data import fetch_snapshots

    tickers = universe(sector_keyword, limit=universe_limit)
    snapshots = fetch_snapshots(tickers)
    candidates = [fast_context(snapshots[t]) for t in tickers if t in snapshots]
    candidates.sort(key=lambda c: c.drawdown_pct)
    ranked = candidates[:board_limit]

    results: List[Tuple[CandidateRead, Optional[DeepRead]]] = []
    for i, candidate in enumerate(ranked):
        deep = deep_context(candidate.ticker) if i < deep_limit else None
        results.append((candidate, deep))
    return results
