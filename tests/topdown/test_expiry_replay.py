from __future__ import annotations

import json
import datetime as dt
import sqlite3
from contextlib import closing
from dataclasses import replace
from zoneinfo import ZoneInfo

import pytest
from streamlit.testing.v1 import AppTest

from src.chain_archive import archive_symbols
from src.topdown.expiry_replay import (
    ArchivedEarnings, archive_days, expiry_outcome, load_archived_earnings, load_replay_quotes, load_replay_timeline, mark_outcome,
    read_replay_record, replay_record, replay_sessions,
)


@pytest.fixture
def archive(tmp_path):
    path = str(tmp_path / "chain with spaces.db")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE chain_snapshots (symbol, snap_date, contract, type, expiration, strike, bid, ask, "
            "source, snapshot_ts, last_trade_time, spot)"
        )
        conn.executemany("INSERT INTO chain_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("ABC", "2026-11-02", "ABC261116C00100000", "call", "2026-11-16", 100, 1.8, 2.2,
             "yfinance", "2026-11-02T20:00:00Z", "2026-11-02T19:55:00Z", 100),
            ("ABC", "2026-11-02", "ABC261214C00100000", "call", "2026-12-14", 100, 3.7, 3.8,
             "yfinance", "2026-11-02T20:00:00Z", None, 100),
            ("ABC", "2026-11-02", "ABC261214P00100000", "put", "2026-12-14", 100, 1.9, 2.0,
             "yfinance", "2026-11-02T20:00:00Z", None, 100),
            ("ABC", "2026-11-02", "ABC261221C00100000", "call", "2026-12-21", 100, 0, 4.2,
             "yfinance", None, None, 100),
            ("ABC", "2026-11-02", "ABC1261228C00100000", "call", "2026-12-28", 100, 1.9, 2.0,
             "yfinance", None, None, 100),
            ("ABC", "2026-11-02", "ABC261130C00100000", "call", "2026-11-30", 100, 2.9, 3.0,
             "cboe", None, None, 101),
            ("ABC", "2026-11-02", "ABC261026C00100000", "call", "2026-10-26", 100, 2.9, 3.0,
             "yfinance", None, None, 100),
            ("ABC", "2026-11-01", "ABC261116C00100000", "call", "2026-11-16", 100, 2.9, 3.0,
             "yfinance", None, None, 99),
        ])
        conn.commit()
    return path


def test_replay_reads_exact_session_and_keeps_inputs(archive):
    assert replay_sessions(" abc ", chain_db=archive) == ["2026-11-02", "2026-11-01"]
    quotes = load_replay_quotes("ABC", "2026-11-02", "call", chain_db=archive)
    assert [quote.days_to_expiration for quote in quotes] == [14, 28, 42]
    assert quotes[0].last_trade_at == "2026-11-02T19:55:00Z"
    assert quotes[0].stored_timestamp == "2026-11-02T20:00:00Z"
    outcome = expiry_outcome(quotes[0], 102.50, 1.0)
    assert (outcome.pnl_at_ask, outcome.pnl_at_midpoint, outcome.max_loss, outcome.break_even) == (29, 49, 221, 102.21)
    record = replay_record([quotes[0], quotes[2]], 102.50, 1.0)
    saved = json.loads(json.dumps(record, allow_nan=False))
    assert saved["schema_version"] == 1
    assert saved["comparisons"][1]["outcome_usd"]["pnl_at_ask"] == -131
    assert saved["comparisons"][1]["quote"]["source"] == "yfinance"


def test_put_break_even_and_legacy_provenance(archive, tmp_path):
    quote, = load_replay_quotes("ABC", "2026-11-02", "put", chain_db=archive)
    outcome = expiry_outcome(quote, 96.0, 1.0)
    assert (outcome.pnl_at_ask, outcome.max_loss, outcome.break_even) == (199, 201, 97.99)
    assert expiry_outcome(replace(quote, bid=99, ask=100), 0, 1).break_even is None
    with pytest.raises(ValueError):
        expiry_outcome(quote, -1, 0)

    legacy = str(tmp_path / "legacy.db")
    with closing(sqlite3.connect(legacy)) as conn:
        conn.execute("CREATE TABLE chain_snapshots (symbol, snap_date, contract, type, expiration, strike, bid, ask)")
        conn.execute("INSERT INTO chain_snapshots VALUES ('ABC', '2026-11-02', 'ABC261116C00100000', "
                     "'CALL', '2026-11-16', 100, 1.8, 2.2)")
        conn.commit()
    old, = load_replay_quotes("ABC", "2026-11-02", "call", chain_db=legacy)
    assert old.source is None and old.stored_timestamp is None and old.recorded_stock_price is None


def test_streamlit_replay_controls_and_empty_states(archive):
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.topdown.web import render_expiry_replay\n"
        f"render_expiry_replay(st, default_ticker='ABC', chain_db={archive!r})\n"
    ).run()
    assert not app.exception
    provider = next(select for select in app.selectbox if select.label == "Archived provider")
    provider.set_value("yfinance").run()
    assert not app.exception
    app.number_input[0].set_value(102.50).run()
    app.number_input[1].set_value(1.0).run()
    assert not app.exception
    scenario = next(frame.value for frame in app.dataframe if "Simulated gain or loss" in frame.value.columns)
    assert list(scenario["Simulated gain or loss"]) == ["+29.00 USD", "-131.00 USD"]
    assert any("Most you could lose" in line.value for line in app.markdown)
    assert len(app.get("download_button")) == 1

    app.slider[0].set_value((14, 20)).run()
    assert "No strike appears on two usable expiration dates" in app.info[0].value
    app.text_input[0].set_value("XYZ").run()
    assert app.info[0].value == "No archive sessions were found for XYZ."


def test_replay_import_recomputes_outcomes_and_rejects_mixed_dates(archive):
    first, _, second = load_replay_quotes("ABC", "2026-11-02", "call", chain_db=archive)
    event = ArchivedEarnings("2026-12-01", "2026-11-02T18:00:00+00:00", "DoltHub earnings cache")
    saved = replay_record((first, second), 102.5, 1, earnings=event)
    saved["comparisons"][0]["outcome_usd"]["pnl_at_ask"] = 999999
    read = read_replay_record(json.dumps(saved).encode())
    assert (read.quotes, read.expiration_price, read.fee, read.earnings) == ((first, second), 102.5, 1, event)
    assert expiry_outcome(read.quotes[0], read.expiration_price, read.fee).pnl_at_ask == 29
    saved["comparisons"][1]["quote"]["source"] = "cboe"
    with pytest.raises(ValueError, match="one comparison"):
        read_replay_record(json.dumps(saved).encode())


def test_common_date_timeline_keeps_gaps_and_provider_changes(archive, tmp_path):
    calendar = str(tmp_path / "calendar.db")
    with closing(sqlite3.connect(calendar)) as conn, conn:
        conn.execute("CREATE TABLE ohlcv (ticker TEXT, date TEXT)")
        conn.executemany("INSERT INTO ohlcv VALUES ('SPY', ?)", [
            (d,) for d in ("2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-17")
        ])
    with closing(sqlite3.connect(archive)) as conn, conn:
        conn.execute("CREATE TABLE archive_attempts (symbol, snap_date, status)")
        conn.execute("INSERT INTO archive_attempts VALUES ('ABC', '2026-11-03', 'feed_down')")
        conn.executemany("INSERT INTO chain_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
            ("ABC", "2026-11-04", "ABC261116C00100000", "call", "2026-11-16", 100, 2.4, 2.6, "yfinance", None, None, 102),
            ("ABC", "2026-11-04", "ABC261214C00100000", "call", "2026-12-14", 100, 0, 4.0, "yfinance", None, None, 102),
            ("ABC", "2026-11-05", "ABC261116C00100000", "call", "2026-11-16", 100, 2.4, 2.6, "cboe", None, None, 103),
            ("ABC", "2026-11-05", "ABC261214C00100000", "call", "2026-12-14", 100, 3.4, 3.6, "yfinance", None, None, 103),
            ("ABC", "2026-11-17", "ABC261214C00100000", "call", "2026-12-14", 100, 3.6, 3.8, "yfinance", None, None, 103),
        ])
    first, _, second = load_replay_quotes("ABC", "2026-11-02", "call", chain_db=archive)
    quotes = first, second
    days = archive_days("ABC", "2026-11-02", "2026-11-05", chain_db=archive, equity_db=calendar)
    assert [(d.day, d.status) for d in days][1] == ("2026-11-03", "Feed unavailable")
    marks = load_replay_timeline(quotes, chain_db=archive, equity_db=calendar)
    assert max(m.day for m in marks) == "2026-11-05"
    assert [m.status for m in marks[2:]] == [
        "No archived quote", "No archived quote", "Archived quote", "No two-sided quote",
        "Provider unverified or changed", "Archived quote",
    ]
    result = mark_outcome(first, marks[4], 1.0)
    assert result is not None
    assert (result.midpoint_change, result.spread_cost, result.fees, result.net) == (50, 30, 2, 18)
    assert mark_outcome(second, marks[5], 1.0) is None
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.topdown.web import render_expiry_replay\n"
        f"render_expiry_replay(st, default_ticker='ABC', chain_db={archive!r}, equity_db={calendar!r})\n"
    ).run()
    next(widget for widget in app.selectbox if widget.label == "Archived session").set_value("2026-11-02").run()
    next(widget for widget in app.selectbox if widget.label == "Archived provider").set_value("yfinance").run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == "Session to compare").set_value("2026-11-04").run()
    table = next(df.value for df in app.dataframe if "Quote status" in df.value.columns)
    assert list(table["Quote status"]) == ["Archived quote", "No two-sided quote"]


def test_breakout_research_does_not_build_market_hub():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from src.topdown.web import render_discover\n"
        "def load(): raise AssertionError('Market hub should be lazy')\n"
        "render_discover(st, None, get_card=lambda _: None, load_hub=load)\n"
    ).run()
    assert not app.exception
    app.radio[0].set_value("Breakout research").run()
    assert not app.exception
    assert "None of the tested EMA windows met the threshold" in app.info[0].value
    assert len(app.dataframe[0].value) == 6


def test_archive_freezes_calendar_and_separates_quote_and_fetch_times(tmp_path):
    chain = str(tmp_path / "chain.db")
    calendar = str(tmp_path / "earnings.db")
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    expiry = today + dt.timedelta(days=20)
    event = today + dt.timedelta(days=5)
    with closing(sqlite3.connect(calendar)) as conn, conn:
        conn.execute("CREATE TABLE earnings_cal (symbol TEXT, date TEXT)")
        conn.execute("INSERT INTO earnings_cal VALUES ('ABC', ?)", (event.isoformat(),))
    quote = {
        "contract": f"ABC{expiry:%y%m%d}C00100000", "type": "call", "strike": 100,
        "expiration": expiry.isoformat(), "bid": 1, "ask": 2, "open_interest": 20, "spot": 100,
        "source": "alpaca_indicative", "quote_at": "2026-01-01T16:00:00Z", "last_trade_time": "2026-01-01T15:50:00Z",
    }
    assert archive_symbols(["ABC"], db_path=chain, today=today.isoformat(), fetcher=lambda _: [quote], earnings_db=calendar) == 1
    with closing(sqlite3.connect(calendar)) as conn, conn:
        conn.execute("UPDATE earnings_cal SET date=?", (expiry.isoformat(),))
    assert archive_symbols(["ABC"], db_path=chain, today=today.isoformat(), fetcher=lambda _: [quote], earnings_db=calendar) == 0
    earnings = load_archived_earnings("ABC", today.isoformat(), chain_db=chain)
    assert earnings is not None and earnings.day == event.isoformat()
    archived, = load_replay_quotes("ABC", today.isoformat(), "call", chain_db=chain)
    assert archived.quote_at == quote["quote_at"] and archived.last_trade_at == quote["last_trade_time"]
    assert archived.fetched_at and archived.fetched_at != archived.quote_at
