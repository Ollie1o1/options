"""🗺️ Discover tab for the Streamlit web cockpit — same four numbered sections as the CLI/TUI.

The DataFrame builders are pure (no Streamlit) so they can be unit-tested; only
`render_discover` touches `st`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .card import EvidenceCard
from src.paths import repo_path

from .expiry_replay import (
    ARCHIVE_DB, EQUITY_DB, ArchivedEarnings, ReplayQuote, archive_days, archive_version, expiry_outcome,
    load_archived_earnings, load_replay_quotes, load_replay_timeline, mark_outcome, read_replay_record, replay_record,
    replay_sessions,
)
from .hub import DiscoveryHub
from .lenses import LENSES, LENS_TAG_MEANING, NULL_LEDGER, LensResult, tags_by_ticker
from .render import LABEL_MEANING

CHIP_BG = {
    "PROVEN": "#5ec98d", "PROMISING": "#82aad2", "TESTING": "#c678dd", "WATCHLIST": "#d6a452",
    "CONTEXT": "#8b949e", "AVOID": "#e06c75", "RETIRED": "#8b949e",
}
VERDICT_BG = {"green": "#5ec98d", "amber": "#d6a452", "red": "#e06c75", "grey": "#8b949e"}


def chip_html(label: str) -> str:
    bg = CHIP_BG.get(label, "#8b949e")
    return (f'<span style="background:{bg};color:#0e1117;padding:1px 7px;border-radius:4px;'
            f'font-weight:700;font-size:0.78rem;letter-spacing:0.03em">{label}</span>')


def board_frame(hub: DiscoveryHub) -> pd.DataFrame:
    return pd.DataFrame([{
        "Rank": r.rank, "ETF": r.etf, "Sector / asset": r.name, "Call": r.posture,
        "Sleeve %": round(r.weight * 100.0, 1), "RS vs SPY 1m %": r.rs_21d_pct, "RS vs SPY 3m %": r.rs_63d_pct,
        "vs 200d %": r.vs_200d_pct, "Stocks >50d %": r.pct_above_50d, "Stocks >200d %": r.pct_above_200d,
        "Stocks (n)": r.n_members or None,
    } for r in hub.board.rows])


def ratios_frame(hub: DiscoveryHub) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ratio": r.name, "What it compares": r.label, "A rise means": r.rise_means,
        "1m %": r.chg_21d_pct, "3m %": r.chg_63d_pct, "vs 200d %": r.vs_200d_pct, "1y percentile": r.pctile_1y,
        "As of": r.as_of or (r.missing or ""),
    } for r in hub.macro.ratios])


def lens_frame(res: LensResult, all_results: List[LensResult]) -> pd.DataFrame:
    tags = tags_by_ticker(all_results)
    return pd.DataFrame([{
        "Ticker": r.ticker, "Sector": r.sector_etf or "", "Listed in": " ".join(t for t, _ in tags.get(r.ticker, [])),
        "Evidence": r.evidence_label or res.lens.label,
        "Warnings": ", ".join(r.flags), "Why it is listed": r.why, "As of": r.as_of or "",
    } for r in res.rows])


def members_frame(hub: DiscoveryHub, etf: str) -> pd.DataFrame:
    from .peers import _metrics
    from .sectors import sector_members

    tags = tags_by_ticker(hub.lenses)
    rows = []
    for sym in sector_members(etf, hub.ohlcv):
        m = _metrics(hub.ohlcv, sym)
        rows.append({
            "Ticker": sym, "12-1m %": m["ret_12_1_pct"], "1m %": m["ret_21d_pct"], "3m %": m["ret_63d_pct"],
            "vs 200d %": m["vs_200d_pct"], "Vol 20d %": m["rv20_pct"], "From 52w high %": m["from_52w_high_pct"],
            "Listed in": " ".join(t for t, _ in tags.get(sym, [])),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("12-1m %", ascending=False, na_position="last") if not df.empty else df


def peers_frame(card: EvidenceCard) -> pd.DataFrame:
    return pd.DataFrame([{
        "Ticker": ("▶ " if r.is_subject else "") + r.ticker + ("" if r.in_sector else " ·"),
        f"Corr w/ {card.ticker}": r.corr_126d, "Beta vs SPY": r.beta_126d, "1m %": r.ret_21d_pct,
        "3m %": r.ret_63d_pct, "12-1m %": r.ret_12_1_pct, "Vol 20d %": r.rv20_pct, "vs 200d %": r.vs_200d_pct,
        "IV/RV": r.iv_rv, "IV as of": r.iv_as_of or "",
    } for r in card.peers.rows])


def provenance_frame(card: EvidenceCard) -> pd.DataFrame:
    return pd.DataFrame([{
        "Section": p.section, "Source": p.source, "As of": p.as_of or "—",
        "Age (days)": p.age_days, "Note": p.note,
    } for p in card.provenance])


def null_ledger_frame() -> pd.DataFrame:
    return pd.DataFrame([{"Idea (tested here)": a, "Result": b, "Date": c, "Source": d} for a, b, c, d in NULL_LEDGER])


def dte_strip_frame(bundle: Any) -> pd.DataFrame:
    rows = []
    for r in bundle.dte_rows:
        p_s = (
            f"{r.p_win*100:.0f}% [{r.p_win_ci95[0]*100:.0f}–{r.p_win_ci95[1]*100:.0f}%] (n={r.n_backtest})"
            if (r.p_win is not None and r.n_backtest > 0 and r.p_win_ci95)
            else "—"
        )
        rows.append(
            {
                "DTE Bucket": r.bucket_label,
                "Study verdict": r.evidence_label,
                "Historical win rate [95% CI, n]": p_s,
                "Average net return %": round(r.mean_roc * 100.0, 2) if r.mean_roc is not None else None,
                "Average loss %": round(r.avg_loss * 100.0, 1) if r.avg_loss is not None else None,
                "Example contract": f"{r.ticker} {r.strikes} ({r.dte}d at snapshot)" if r.ticker else "—",
                "Quote context": r.quote_context,
                "Mid $": r.credit_mid,
                "Modeled credit $ (k≤0.35)": r.limit_k035,
                "Max loss $ before fees": r.max_risk_usd,
                "Friction %": round(r.friction_pct * 100.0, 1) if r.friction_pct is not None else None,
                "Profit factor": r.pf,
                "Δ-Adj Excess %": round(r.excess_limit * 100.0, 2) if r.excess_limit is not None else None,
                "Why": r.why,
            }
        )
    return pd.DataFrame(rows)


def _box(st: Any, label: str, title: str, body: str) -> None:
    border = CHIP_BG.get(label, "#8b949e")
    st.markdown(
        f'<div style="border-left:4px solid {border};background:#161b22;padding:0.55rem 0.9rem;'
        f'border-radius:4px;margin:0.3rem 0 0.6rem 0">{chip_html(label)} <b>{title}</b><br>'
        f'<span style="color:#c9d1d9">{body}</span></div>',
        unsafe_allow_html=True,
    )


def _replay_cache(st: Any, slot: str, key: tuple, loader: Callable[[], Any]) -> Any:
    state_key = f"_expiry-replay-{slot}"
    entry = st.session_state.get(state_key)
    if entry is None or entry[0] != key:
        entry = (key, loader())
        st.session_state[state_key] = entry
    return entry[1]


def render_expiry_replay(
    st: Any, *, default_ticker: Optional[str] = None, chain_db: str = ARCHIVE_DB, equity_db: str = EQUITY_DB,
) -> None:
    st.subheader("Compare expirations")
    st.caption("Compare the same strike across expiration dates using archived quotes.")
    mode = st.radio("Comparison source", ("Option archive", "Open saved replay"), horizontal=True, key="expiry-replay-mode")
    if mode == "Open saved replay":
        upload = st.file_uploader("Replay JSON", type=["json"], key="expiry-replay-upload")
        if upload is None:
            return
        try:
            saved = read_replay_record(upload.getvalue())
        except ValueError as exc:
            st.error(str(exc))
            return
        first = saved.quotes[0]
        st.text(f"{first.symbol} · {first.snapshot_date} · {first.option_type.title()} · ${first.strike:g} strike")
        key = f"saved-{hashlib.sha256(upload.getvalue()).hexdigest()}"
        _render_replay_results(st, saved.quotes, saved.expiration_price, saved.fee, key,
                               chain_db, equity_db, earnings=saved.earnings)
        return

    c1, c2, c3 = st.columns(3)
    symbol = c1.text_input("Stock or ETF ticker", value=default_ticker or "SPY", key="expiry-replay-symbol").strip().upper()
    if not symbol:
        return
    archive_key = (chain_db, archive_version(chain_db), symbol)
    try:
        sessions = _replay_cache(st, "sessions", archive_key, lambda: replay_sessions(symbol, chain_db=chain_db))
    except sqlite3.Error:
        st.info("The option archive is unavailable. Use Market and studies to see historical results.")
        return
    if not sessions:
        st.info(f"No archive sessions were found for {symbol}.")
        with st.expander("Archive coverage by date"):
            _render_archive_coverage(st, symbol, dt.date.today().isoformat(), chain_db, equity_db, archive_key)
        return
    session = c2.selectbox("Archived session", sessions, key=f"expiry-replay-session-{symbol}")
    option_type = c3.radio("Contract type", ("Call", "Put"), horizontal=True, key="expiry-replay-type").lower()
    try:
        quotes = _replay_cache(st, "quotes", (*archive_key, session, option_type),
                               lambda: load_replay_quotes(symbol, session, option_type, chain_db=chain_db))
        earnings = _replay_cache(st, "earnings", (*archive_key, session),
                                 lambda: load_archived_earnings(symbol, session, chain_db=chain_db))
    except sqlite3.Error:
        st.info("The selected session could not be read from the option archive.")
        return
    with st.expander("Archive coverage by date"):
        _render_archive_coverage(st, symbol, session, chain_db, equity_db, archive_key)
    if not quotes:
        st.info(f"No usable two-sided {option_type} quotes were archived for {symbol} on {session}.")
        return

    prefix = f"expiry-replay-{symbol}-{session}-{option_type}"
    sources = sorted({q.source or "Unknown" for q in quotes})
    source = st.selectbox("Archived provider", sources, key=f"{prefix}-source") if len(sources) > 1 else sources[0]
    quotes = [q for q in quotes if (q.source or "Unknown") == source]
    expiry_count = len({q.expiration for q in quotes})
    st.caption(f"Archive coverage · {len(quotes):,} usable {option_type} quotes across {expiry_count} expiration dates")
    first_dte = min(q.days_to_expiration for q in quotes)
    last_dte = max(q.days_to_expiration for q in quotes)
    if first_dte == last_dte:
        st.info("This session has one usable expiration from the selected provider. Choose another session to compare dates.")
        return
    dte_from, dte_to = st.slider(
        "Days until expiration at the archived session", first_dte, last_dte, (first_dte, last_dte),
        key=f"{prefix}-{source}-dte",
    )
    by_strike: dict[float, dict[str, ReplayQuote]] = {}
    for quote in quotes:
        if dte_from <= quote.days_to_expiration <= dte_to:
            by_strike.setdefault(quote.strike, {}).setdefault(quote.expiration, quote)
    strikes = sorted(strike for strike, dates in by_strike.items() if len(dates) > 1)
    if not strikes:
        st.info("No strike appears on two usable expiration dates in this range. Widen the range or choose another session.")
        return

    c1, c2 = st.columns([1, 2])
    selection_key = f"{prefix}-{source}-{dte_from}-{dte_to}"
    archive_spots = {q.recorded_stock_price for q in quotes if q.recorded_stock_price is not None}
    first_strike = min(strikes, key=lambda k: abs(k - next(iter(archive_spots)))) if len(archive_spots) == 1 else strikes[0]
    strike = c1.selectbox(
        "Strike price", strikes, index=strikes.index(first_strike),
        format_func=lambda x: f"${x:,.2f}", key=f"{selection_key}-strike",
    )
    matching = by_strike[strike]
    dates = sorted(matching)
    selected = c2.multiselect(
        "Expiration dates", dates, default=dates[:2], max_selections=3,
        format_func=lambda day: f"{day} · {matching[day].days_to_expiration} days",
        key=f"{selection_key}-{strike}-dates",
    )
    if len(selected) < 2:
        st.info("Select two or three expiration dates to compare.")
        return

    selected_quotes = tuple(matching[day] for day in sorted(selected))
    _render_replay_results(st, selected_quotes, None, 0.0, f"{selection_key}-{strike}", chain_db, equity_db, earnings=earnings)


def _render_replay_results(
    st: Any, selected_quotes: tuple[ReplayQuote, ...], saved_price: Optional[float], initial_fee: float,
    selection_key: str, chain_db: str, equity_db: str, *, earnings: Optional[ArchivedEarnings],
) -> None:
    first = selected_quotes[0]
    symbol, session, strike = first.symbol, first.snapshot_date, first.strike
    source = first.source or "Unknown"
    recorded_prices = {q.recorded_stock_price for q in selected_quotes}
    recorded_spot = next(iter(recorded_prices)) if len(recorded_prices) == 1 else None
    if recorded_spot is not None:
        st.text(f"Recorded stock price ${recorded_spot:,.2f} · Archived provider {source}")
    else:
        st.text(f"Archived provider {source} · No single recorded stock price for this comparison")
    c1, c2 = st.columns(2)
    default_price = saved_price if saved_price is not None else round(recorded_spot or strike, 2)
    expiration_price = float(c1.number_input(
        "Stock price at each expiration", min_value=0.0, max_value=1_000_000.0, value=default_price, step=0.5,
        key=f"{selection_key}-price",
    ))
    fee = float(c2.number_input(
        "Simulated fee per contract per side", min_value=0.0, max_value=100_000.0, value=initial_fee, step=0.5,
        key="expiry-replay-fee" if saved_price is None else f"{selection_key}-fee",
    ))
    outcomes = [(quote, expiry_outcome(quote, expiration_price, fee)) for quote in selected_quotes]
    display = []
    for card, (quote, outcome) in zip(st.columns(len(outcomes)), outcomes):
        break_even_label = "No attainable price"
        if outcome.break_even is not None:
            break_even_label = f"${outcome.break_even:,.4f}".rstrip("0").rstrip(".")
            if recorded_spot:
                change = outcome.break_even / recorded_spot - 1.0
                break_even_label += f" ({change:+.2%} from recorded price)"
        with card.container(border=True):
            st.markdown(f"**{quote.expiration}** · {quote.days_to_expiration} days at capture")
            st.write(f"Most you could lose **${outcome.max_loss:,.2f}**")
            st.write(f"Break-even at expiry **{break_even_label}**")
            if earnings is not None and earnings.day <= quote.expiration:
                st.write(f"**Earnings on the archived calendar · {earnings.day}**")
            else:
                st.write(f"Bid–ask gap **${outcome.bid_ask_gap:,.2f} per contract**")
        display.append({
            "Expiration": quote.expiration,
            "Simulated gain or loss": f"{outcome.pnl_at_ask:+,.2f} USD",
        })
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Expiration results use the archived ask and one fee for one assumed 100-share contract. "
        + ("Earnings history is unavailable for this capture." if earnings is None else
           "The earnings date comes from the calendar visible during capture and can change.")
    )
    with st.expander("What happened before expiration?", expanded=True):
        _render_replay_timeline(st, selected_quotes, fee, selection_key, chain_db, equity_db)
    with st.expander("See the payoff across stock prices"):
        thresholds = [outcome.break_even for _, outcome in outcomes if outcome.break_even is not None]
        anchors = [strike, expiration_price, *thresholds]
        low, high = max(0.0, min(anchors) - strike * 0.05), max(anchors) + strike * 0.05
        prices = sorted(set(anchors + [low + (high - low) * step / 40 for step in range(41)]))
        curve = pd.DataFrame({
            "Stock price at expiry (USD)": prices,
            **{
                quote.expiration: [expiry_outcome(quote, price, fee).pnl_at_ask for price in prices]
                for quote in selected_quotes
            },
        })
        st.line_chart(curve, x="Stock price at expiry (USD)", y_label="Gain or loss per contract (USD)")
    with st.expander("Quote details and price assumptions"):
        st.caption("The provider timestamp is the original archive field and can be a fetch or trade time. Quote time is shown when supplied separately.")
        st.dataframe([
            {
                "Expiration": quote.expiration, "Bid per share": quote.bid, "Ask per share": quote.ask,
                "Quote time": quote.quote_at or "Unknown", "Fetch completed": quote.fetched_at or "Unknown",
                "Provider timestamp": quote.stored_timestamp or "Unknown", "Last trade time": quote.last_trade_at or "Unknown",
                "Result at midpoint (USD)": outcome.pnl_at_midpoint,
            }
            for quote, outcome in outcomes
        ], width="stretch", hide_index=True)
        if earnings is not None:
            st.text(f"Earnings calendar · {earnings.source} · observed {earnings.observed_at}")
        st.caption("The midpoint is an alternate price assumption. The archive contains no measured fills for this replay.")
        st.download_button(
            "Download replay inputs and results",
            data=json.dumps(replay_record(selected_quotes, expiration_price, fee, earnings=earnings), indent=2),
            file_name=f"{symbol}-{session}-expiry-replay.json", mime="application/json",
        )


def _render_archive_coverage(st: Any, symbol: str, session: str, chain_db: str, equity_db: str, key: tuple) -> None:
    start = (dt.date.fromisoformat(session) - dt.timedelta(days=29)).isoformat()
    try:
        days = _replay_cache(st, "coverage", (*key, session, equity_db, archive_version(equity_db)),
                             lambda: archive_days(symbol, start, session, chain_db=chain_db, equity_db=equity_db))
    except sqlite3.Error:
        st.info("Archive coverage for this period is unavailable.")
        return
    if not days:
        st.caption("There are no capture attempts or reference market dates for this period.")
        return
    missing = sum(day.market_session and not day.rows for day in days)
    st.caption(f"30 days ending {session} · {sum(bool(day.rows) for day in days)} captures · {missing} gaps on known market sessions")
    st.caption("Market sessions come from the local SPY price calendar. Capture counts include every stored quote.")
    st.dataframe([
        {"Session": day.day, "Capture status": day.status, "Stored quotes": day.rows, "Expirations": day.expiries}
        for day in reversed(days)
    ], width="stretch", hide_index=True)


def _render_replay_timeline(
    st: Any, quotes: tuple[ReplayQuote, ...], fee: float, selection_key: str, chain_db: str, equity_db: str,
) -> None:
    import altair as alt

    key = (chain_db, archive_version(chain_db), equity_db, archive_version(equity_db), quotes)
    try:
        marks = _replay_cache(st, "timeline", key,
                              lambda: load_replay_timeline(quotes, chain_db=chain_db, equity_db=equity_db))
    except sqlite3.Error:
        st.info("The local archive is unavailable for later quotes. The saved expiration scenarios are available above.")
        return
    entry = quotes[0].snapshot_date
    by_expiration = {q.expiration: q for q in quotes}
    outcomes = [(mark, mark_outcome(by_expiration[mark.expiration], mark, fee)) for mark in marks]
    dates = sorted({mark.day for mark, _ in outcomes if mark.day > entry})
    if not dates:
        st.info("The archive has the entry snapshot. Later sessions will appear here as the archive grows.")
        return
    st.caption("Compare on one observed date through the earliest expiry. A simulated early exit uses the archived bid and a fee on each side.")
    records = []
    stocks_by_day: dict[str, set[float]] = {}
    segment = {expiry: 0 for expiry in by_expiration}
    for mark, outcome in outcomes:
        if mark.stock_price is not None:
            stocks_by_day.setdefault(mark.day, set()).add(mark.stock_price)
        if outcome is None:
            segment[mark.expiration] += 1
        else:
            records.append({"Session": mark.day, "Expiration": mark.expiration, "Net (USD)": outcome.net,
                            "Segment": segment[mark.expiration]})
    if any(record["Session"] > entry for record in records):
        options_chart = alt.Chart(pd.DataFrame(records)).mark_line(point=True).encode(
            x="Session:T", y=alt.Y("Net (USD):Q", title="Simulated exit gain or loss (USD)"), color="Expiration:N",
            detail="Segment:N", tooltip=["Session:T", "Expiration:N", "Net (USD):Q"],
        ).properties(height=210)
        spot_records = []
        spot_segment = 0
        for session in (entry, *dates):
            stock_prices = stocks_by_day.get(session, set())
            if len(stock_prices) == 1:
                spot_records.append({"Session": session, "Stock (USD)": next(iter(stock_prices)), "Segment": spot_segment})
            else:
                spot_segment += 1
        if len(spot_records) > 1:
            stock_chart = alt.Chart(pd.DataFrame(spot_records)).mark_line(point=True).encode(
                x="Session:T", y=alt.Y("Stock (USD):Q", title="Recorded stock price (USD)", scale=alt.Scale(zero=False)),
                detail="Segment:N", tooltip=["Session:T", "Stock (USD):Q"],
            ).properties(height=105)
            st.altair_chart(alt.vconcat(options_chart, stock_chart).resolve_scale(x="shared"), width="stretch")
        else:
            st.altair_chart(options_chart, width="stretch")
    else:
        st.info("No later session has a matching two-sided quote from the entry provider.")

    day = st.selectbox("Session to compare", dates, index=len(dates) - 1, key=f"{selection_key}-timeline-day")
    selected = [(mark, outcome) for mark, outcome in outcomes if mark.day == day]
    stocks = stocks_by_day.get(day, set())
    entries = {q.recorded_stock_price for q in quotes}
    if len(stocks) == 1:
        price = next(iter(stocks))
        start = next(iter(entries)) if len(entries) == 1 else None
        move = f" · {price / start - 1:+.2%} since the entry snapshot" if start else ""
        st.write(f"Recorded stock price **${price:,.2f}**{move}")
    else:
        st.caption("There is no single stock price recorded for the selected date.")
    st.dataframe([
        {"Expiration": mark.expiration, "Quote status": mark.status,
         "Option midpoint change (USD)": outcome.midpoint_change if outcome else None,
         "Bid–ask cost (USD)": outcome.spread_cost if outcome else None,
         "Fees (USD)": outcome.fees if outcome else None,
         "Simulated exit (USD)": outcome.net if outcome else None}
        for mark, outcome in selected
    ], width="stretch", hide_index=True)


def render_breakout_research(st: Any, *, report_path: str = "configs/ema_breakout_study.json") -> None:
    st.subheader("Do moving averages improve breakout forecasts?")
    try:
        report = json.loads(Path(repo_path(report_path)).read_text(encoding="utf-8"))
        horizons = report["horizons"]
        last_day = report["last_cached_session"]
        rows = []
        for horizon in horizons:
            for arm in horizon["comparisons"]:
                before = horizon["baseline_brier"]
                low, high = (v / before for v in arm["brier_gain_interval"])
                rows.append({
                    "Forecast window": f"{horizon['horizon_sessions']} sessions", "EMA window": f"{arm['ema_sessions']} sessions",
                    "Less forecast error": f"{arm['brier_skill']:+.2%}", "Uncertainty range": f"{low:+.2%} to {high:+.2%}",
                    "Test observations": horizon["holdout_observations"],
                    "Review": "Research follow-up" if arm["follow_up"] else "Below threshold",
                })
    except (OSError, TypeError, KeyError, ValueError, ZeroDivisionError):
        st.info("The historical study report is unavailable.")
        return
    if any(row["Review"] == "Research follow-up" for row in rows):
        st.info("At least one comparison met the threshold for another historical study.")
    else:
        st.info("None of the tested EMA windows met the threshold for another historical study.")
    st.caption(f"Held out 2024 through {last_day} · {report['symbols_with_history']} cached symbols · group forecasts")
    st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("How the test works"):
        st.write(
            "An exponential moving average (EMA) gives recent prices more weight. The baseline uses distance "
            "from the recent high and recent volatility. Each comparison adds only whether "
            "the close was above the moving average. Positive values mean lower forecast error. The uncertainty range "
            "adjusts for testing six comparisons. The review threshold was a reduction in error of at least 1%, "
            "an uncertainty range above zero, and improvement in both test periods."
        )
        st.dataframe([
            {"Forecast window": h["horizon_sessions"], "Historical breakout rate": f"{h['holdout_breakout_rate']:.1%}",
             "Unresolved observations": h["unresolved_observations"]} for h in horizons
        ], width="stretch", hide_index=True)
        st.caption("This study measures forecast accuracy in the cached universe. A profit comparison requires a separate test with costs.")


def render_discover(st: Any, hub: Optional[DiscoveryHub], *, get_card: Callable[[str], EvidenceCard],
                    default_ticker: Optional[str] = None, load_hub: Optional[Callable[[], DiscoveryHub]] = None) -> None:
    view = st.radio("Discover view", ("Compare expirations", "Market and studies", "Breakout research"),
                    horizontal=True, key="discover-view")
    if view == "Compare expirations":
        render_expiry_replay(st, default_ticker=default_ticker)
        return
    if view == "Breakout research":
        render_breakout_research(st)
        return
    if hub is None and load_hub is not None:
        hub = load_hub()
    if hub is None:
        st.info("The market view is unavailable.")
        return
    f = hub.freshness()
    st.caption(
        f"① Market › ② Sectors › ③ Ideas › ④ Evidence card — top-down, each step labelled by its evidence. "
        f"Data: {f['symbols']} symbols, SPY through {f['spy_last']}"
        + (f" · {f['behind']} symbols behind — run `opt refresh`" if f["behind"] else " · all current")
    )
    st.markdown(" ".join(f"{chip_html(k)} <span style='color:#8b949e;font-size:0.8rem'>{v}</span>"
                         for k, v in LABEL_MEANING.items()), unsafe_allow_html=True)

    # 0 · BEST ACTION TODAY + DTE-SORTED OPTIONS STRIP
    from .best_action import build_best_action

    st.markdown("### 0 · Best Action Today — face-value picks & backtested odds")
    d_c1, d_c2 = st.columns([2, 1])
    sort_choice = d_c1.radio(
        "Options DTE sort",
        ["dte", "evidence"],
        format_func=lambda v: "Shortest DTE first (0 to 240 days)" if v == "dte" else "Best evidence first",
        horizontal=True,
        label_visibility="collapsed",
        key="disc-dte-sort",
    )
    live_dte_flag = d_c2.checkbox("Request provider snapshots", value=False, key="disc-dte-live")
    ba = build_best_action(hub, live_options=bool(live_dte_flag), sort_dte_by=str(sort_choice))
    if ba.profile_summary:
        st.caption(f"Profile: {ba.profile_summary} · {ba.o1_fill_meter}")
    for act in ba.actions:
        sz_html = f"<span style='color:#5ec98d;font-weight:600'>{act.sizing_text}</span><br>" if act.sizing_text else ""
        _box(
            st,
            "AVOID" if act.evidence_label == "CLOSED" else act.evidence_label,
            f"[{act.account}] {act.headline}",
            f"{sz_html}<b>{act.prob_text}</b><br><span style='color:#8b949e'>{act.why}</span>",
        )
    with st.expander(f"Options by Expiry (0DTE → 240D) [TAXABLE_ONLY] · Clock: {ba.execution_window}", expanded=True):
        st.caption("Historical results describe the tested spread groups. Example contracts have no individual win estimate.")
        if ba.o1_fill_meter:
            st.caption(f"O1 Fill Quality Meter: **{ba.o1_fill_meter}**")
        st.dataframe(dte_strip_frame(ba), width="stretch", hide_index=True)
        st.caption(
            f"{ba.execution_note}"
            + (f" · {ba.implied_calibration_note}" if ba.implied_calibration_note else "")
        )

    # ① MARKET
    from .render import regime_read

    st.markdown("### ① Market — what kind of market is this?")
    read, _style = regime_read(hub)
    st.markdown(f"**Read:** {read} <span style='color:#8b949e'>(a description of today, not a forecast)</span>",
                unsafe_allow_html=True)
    s, v = hub.macro.spy, hub.macro.vol
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SPY vs 200-day", f"{s.get('vs_200d_pct', 0):+.1f}%" if s.get("as_of") else "—")
    c2.metric("SPY last month", f"{s.get('ret_21d_pct', 0):+.1f}%" if s.get("as_of") else "—")
    c3.metric("VIX", f"{v['vix']:.1f}" if v.get("vix") is not None else "—",
              help=f"{v.get('vix_pctile_1y', '—')}th percentile of the last year")
    c4.metric("VIX curve (VIX/VIX3M)", str(v.get("term", "—")).title(),
              f"{v.get('term_ratio'):.2f}" if v.get("term_ratio") else None, delta_color="off")
    c5.metric("SPY realized vol (20d)", f"{s.get('rv20_pct') or 0:.0f}%" if s.get("as_of") else "—")
    up, n = hub.macro.ratio_tally()
    with st.expander(f"Rotation ratios — {up}/{n} risk-appetite ratios rising over 3m (descriptive)", expanded=False):
        st.dataframe(ratios_frame(hub), width="stretch", hide_index=True)
        fred = [x for x in hub.macro.fred if x.get("value") is not None]
        if fred:
            st.dataframe(pd.DataFrame(fred)[["label", "value", "unit", "chg_1m", "pctile_1y", "as_of"]],
                         width="stretch", hide_index=True)
        else:
            st.caption("FRED credit-spread / yield-curve series not cached — run `opt discover --refresh-macro`.")
    st.caption(f"Source · SPY {s.get('as_of')} (data/equity_ohlcv.db) · Cboe VIX {v.get('as_of')} · "
               "context only: VIX→direction tested NOT_ESTABLISHED; ratios untested.")

    # ② SECTORS
    st.markdown("### ② Sectors — where is the strength?")
    b = hub.board
    held = ", ".join(f"{r.etf} {r.weight*100:.0f}%" for r in b.rows if r.weight > 0) or "nothing"
    _box(st, "PROVEN", "Ranked by the validated rotation model",
         f"This month's TFSA core sleeve: <b>{held}</b> · cash/T-bills {(b.sleeve_cash or 0)*100:.0f}% "
         f"(scaled {b.sleeve_scale or 0:.2f}× — basket 20d vol {(b.basket_rv20 or 0)*100:.1f}% vs 14% target). "
         "FAVOR = held · AVOID = bearish call (only fires when SPY is below trend). "
         "Relative-strength and breadth columns are description only.")
    st.dataframe(
        board_frame(hub), width="stretch", hide_index=True, height=35 * (len(b.rows) + 1) + 3,
        column_config={
            "Sleeve %": st.column_config.NumberColumn(format="%.0f%%"),
            "RS vs SPY 1m %": st.column_config.NumberColumn(format="%+.1f%%"),
            "RS vs SPY 3m %": st.column_config.NumberColumn(format="%+.1f%%"),
            "vs 200d %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Stocks >50d %": st.column_config.NumberColumn(format="%.0f%%"),
            "Stocks >200d %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    sectors = [r.etf for r in b.rows if r.n_members]
    pick = st.selectbox("Drill into a sector's stocks (② › constituents)", sectors, index=0 if sectors else None,
                        format_func=lambda e: f"{e} — {next((r.name for r in b.rows if r.etf == e), e)}")
    if pick:
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(members_frame(hub, pick), width="stretch", hide_index=True, column_config={
            "12-1m %": pct, "1m %": pct, "3m %": pct, "vs 200d %": pct, "From 52w high %": pct,
            "Vol 20d %": st.column_config.NumberColumn(format="%.0f%%"),
        })
        st.caption("Sorted by 12-1 month return — description, not a forecast. 'Listed in' tags: "
                   + " · ".join(f"{k} = {v}" for k, v in LENS_TAG_MEANING.items()))
    st.caption(f"Source · prices as of {b.as_of} · docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md")

    # ③ IDEAS
    st.markdown("### ③ Ideas — why look here?")
    lens_ids = [lz.id for lz in LENSES]
    titles = {lz.id: f"[{lz.label}] {lz.title}" for lz in LENSES}
    lid = st.radio("Lens", lens_ids, format_func=lambda i: titles[i], horizontal=True, label_visibility="collapsed")
    res = hub.lens(lid)
    if res is not None:
        _box(st, res.lens.label, f"{res.lens.title} — {res.lens.question}",
             f"<b>Evidence:</b> {res.lens.evidence}<br><span style='color:#8b949e'>Sorted by: {res.lens.sort_note} · "
             f"{len(res.rows)} row(s) · source {res.lens.source}</span>"
             + (f"<br><span style='color:#d6a452'>{res.note}</span>" if res.note else ""))
        if res.rows:
            st.dataframe(lens_frame(res, hub.lenses), width="stretch", hide_index=True)
        else:
            st.caption("Nothing listed right now.")
    with st.expander("Already tested here — does NOT work (don't re-chase)"):
        st.dataframe(null_ledger_frame(), width="stretch", hide_index=True)

    # ④ EVIDENCE CARD
    st.markdown("### ④ Evidence card — one ticker, every number with its source")
    choices: List[str] = list(dict.fromkeys(
        ([default_ticker] if default_ticker else []) + [x.ticker for r in hub.lenses for x in r.rows]))
    cc1, cc2 = st.columns([1, 1])
    pick_t = cc1.selectbox("Ticker from the lenses above", choices, index=0 if choices else None)
    typed = cc2.text_input("…or type any ticker", value="").strip().upper()
    ticker = typed or pick_t
    if not ticker:
        return
    card = get_card(ticker)
    _render_card(st, card)


def _render_card(st: Any, card: EvidenceCard) -> None:
    d = card.dossier
    bg = VERDICT_BG.get(card.verdict_style, "#8b949e")
    spot = f"${d.spot:.2f}" if d.spot is not None else "price n/a"
    reasons = "".join(f"<li>{r}</li>" for r in card.reasons)
    warns = "".join(f"<li style='color:#d6a452'>⚠ {w}</li>" for w in card.warnings)
    hits = " ".join(f"{chip_html(lab)} {title}" for title, lab in card.lens_hits)
    st.markdown(
        f'<div style="border:2px solid {bg};border-radius:6px;padding:0.6rem 1rem;background:#161b22">'
        f'<span style="background:{bg};color:#0e1117;padding:2px 9px;border-radius:4px;font-weight:800">'
        f'{card.verdict}</span> <b style="font-size:1.15rem">{card.ticker}</b> {spot}'
        f'<ul style="margin:0.4rem 0 0.2rem 1rem">{reasons}{warns}</ul>'
        + (f'<div style="color:#8b949e">Listed in: {hits}</div>' if hits else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**2 · Price structure**")
        try:
            from src.dashboard import build_dossier_price_figure

            st.plotly_chart(build_dossier_price_figure(d), width="stretch", key=f"disc-price-{card.ticker}")
        except Exception:
            pass
        st.markdown(
            f"Trend vs 20/50/200d: {d.dist_sma20_pct or 0:+.1f}% / {d.dist_sma50_pct or 0:+.1f}% / "
            f"{d.dist_sma200_pct or 0:+.1f}% · 12-1m {d.mom_12_1_pct or 0:+.1f}% · RSI(2) {d.rsi_2 or 0:.0f}"
        )
    with right:
        st.markdown("**4 · Options & market-implied odds**")
        o1, o2, o3 = st.columns(3)
        o1.metric("ATM IV (30d)", f"{d.atm_iv:.1f}%" if d.atm_iv is not None else "—",
                  f"IV/RV {d.iv_vs_rv_ratio:.2f}" if d.iv_vs_rv_ratio is not None else None, delta_color="off")
        o2.metric("P(>+5%) 30d", f"{d.rnd_prob_up_5pct*100:.0f}%" if d.rnd_prob_up_5pct is not None else "—")
        o3.metric("P(<−5%) 30d", f"{d.rnd_prob_dn_5pct*100:.0f}%" if d.rnd_prob_dn_5pct is not None else "—")
        st.caption(card.prob_label)
        st.markdown("**5 · Events, revisions & news**")
        earn = (f"in {d.days_to_earnings}d ({d.earnings_date})" if d.days_to_earnings is not None
                else "no upcoming date on file")
        lines = [f"- Earnings: {earn}", f"- Next macro: {d.next_macro_event}"]
        if d.rev_breadth_30d is not None:
            lines.append(f"- EPS revision breadth: {d.rev_breadth_30d:+.2f} (TESTING — readout ≥ 2027-01-15)")
        for h in d.headlines[:3]:
            lines.append(f"- {h.get('date', '')} {h['title']}")
        if d.cdr_ticker:
            lines.append(f"- TFSA: Canadian CDR {d.cdr_ticker}")
        st.markdown("\n".join(lines))
    st.markdown("**3 · Peers** — " + card.peers.method)
    pct = st.column_config.NumberColumn(format="%+.1f%%")
    st.dataframe(peers_frame(card), width="stretch", hide_index=True, column_config={
        "1m %": pct, "3m %": pct, "12-1m %": pct, "vs 200d %": pct,
        "Vol 20d %": st.column_config.NumberColumn(format="%.0f%%"),
        "Beta vs SPY": st.column_config.NumberColumn(format="%.2f"),
        f"Corr w/ {card.ticker}": st.column_config.NumberColumn(format="%.2f"),
        "IV/RV": st.column_config.NumberColumn(format="%.2f"),
    })
    st.markdown("**6 · Portfolio fit** — what this adds to what you already hold (descriptive, never a gate)")
    fit = card.fit
    if fit is None:
        st.caption("Holdings not checked.")
    else:
        by_src: Dict[str, List[str]] = {}
        for h in fit.holdings:
            by_src.setdefault(h.source, []).append(h.ticker + (f" {h.weight * 100:.0f}%" if h.weight else ""))
        lines = ["- You hold: " + (" · ".join(f"{k}: {', '.join(v)}" for k, v in sorted(by_src.items()))
                                  or "nothing recorded")]
        lines += [f"- <span style='color:#d6a452'>⚠ {w}</span>" for w in fit.warnings]
        lines += [f"- {n}" for n in fit.notes]
        if fit.correlations:
            lines.append("- Co-movement (126d): " + ", ".join(f"{tk} {r:+.2f}" for tk, r in fit.correlations[:5]))
        if fit.diversifiers:
            lines.append("- Least correlated sleeve-universe ETFs: "
                         + ", ".join(f"{tk} {r:+.2f}" for tk, r in fit.diversifiers) + " (information, not a recommendation)")
        st.markdown("\n".join(lines), unsafe_allow_html=True)
        st.caption(fit.method)
    st.markdown("**7 · Where every number came from**")
    st.dataframe(provenance_frame(card), width="stretch", hide_index=True)


__all__ = [
    "chip_html", "board_frame", "ratios_frame", "lens_frame", "members_frame", "peers_frame",
    "provenance_frame", "null_ledger_frame", "dte_strip_frame", "render_discover",
]
