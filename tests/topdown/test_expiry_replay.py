from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from streamlit.testing.v1 import AppTest

from src.topdown.expiry_replay import expiry_outcome, load_replay_quotes, replay_record, replay_sessions


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
    assert list(app.dataframe[0].value["Simulated gain or loss"]) == ["+29.00 USD", "-131.00 USD"]
    assert len(app.get("download_button")) == 1

    app.slider[0].set_value((14, 20)).run()
    assert "No strike appears on two usable expiration dates" in app.info[0].value
    app.text_input[0].set_value("XYZ").run()
    assert app.info[0].value == "No archive sessions were found for XYZ."
