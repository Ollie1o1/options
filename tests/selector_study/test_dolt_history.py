"""Historical hold-to-expiry companion: parity spot, settlement basis, EOD capture."""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from src.selector_study import capture, dolt_history as h, outcomes
from src.selector_study.db import connect, default_settings


def _chain(spot, day="2023-03-15", exp="2023-03-29", r=0.04, t=14 / 365):
    out = []
    for k in range(int(spot * 0.9), int(spot * 1.1), 5):
        fwd_gap = spot - k * math.exp(-r * t)           # C - P = S - K e^{-rT}
        p_mid = max(0.5, k * 0.02)
        c_mid = p_mid + fwd_gap
        if c_mid <= 0.05:
            continue
        out += [{"expiration": exp, "strike": float(k), "type": "call", "bid": c_mid - 0.05, "ask": c_mid + 0.05},
                {"expiration": exp, "strike": float(k), "type": "put", "bid": p_mid - 0.05, "ask": p_mid + 0.05}]
    return out


def test_parity_spot_recovers_the_forward_without_a_rate():
    est = h.parity_spot(_chain(400.0), "2023-03-15")
    assert est == pytest.approx(400.0 * math.exp(0.04 * 14 / 365), rel=1e-4)
    assert h.parity_spot([], "2023-03-15") is None


def test_weekly_days_skips_holidays():
    days = h.weekly_days("2026-11-23", "2026-12-31")
    assert days[0] == "2026-11-25" and "2026-12-23" in days and all(d >= "2026-11-23" for d in days)


def test_hold_to_expiry_ignores_interim_marks_and_settles_at_intrinsic():
    st = h.settings(default_settings())
    e = {"session": "2023-03-15", "expiration": "2023-04-21", "width": 10.0, "short_strike": 380.0,
         "long_strike": 370.0, "credit_mid": 2.0, "credit_cross": 1.8}
    interim = {"mark_session": "2023-03-22", "status": "marked", "short_bid": 0.1, "short_ask": 0.2,
               "long_bid": 0.0, "long_ask": 0.05}
    assert outcomes.resolve(e, [interim], st, "cross")["status"] == "open"
    settled = {"mark_session": "2023-04-21", "status": "settled", "underlying_close": 375.0}
    out = outcomes.resolve(e, [interim, settled], st, "cross")
    assert out["exit_reason"] == "expired" and out["pnl"] == pytest.approx(1.8 - 5.0)
    assert outcomes.resolve(e, [interim, {**settled, "underlying_close": None}], st, "mid")["status"] == "missing"


def _study_with_entry(tmp_path, sym, session, exp):
    conn = connect(str(tmp_path / "s.db"), settings=h.settings(default_settings()))
    conn.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 ("e1", "p1", "selector", session, sym, "bull_put", exp, 30, 100.0, 95.0, 5.0, 0.25,
                  1.0, 0.9, 0.1, "{}", "x"))
    conn.commit()
    return conn


def _ohlcv(path, rows):
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL, high REAL, low REAL, volume REAL)")
        con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?)", [(t, d, c, c, c, 1) for t, d, c in rows])


def test_settlement_unadjusts_and_keeps_the_entry_basis_across_a_split(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # Ratio 1.10 = dividends since; a 4:1 split between entry and expiry makes the later ratio 4x smaller.
    (out / "spot_ratio.json").write_text(json.dumps(
        {"AAA": {"2023-03-15": 1.10, "2023-04-19": 1.10}, "BBB": {"2023-03-15": 4.0, "2023-04-19": 1.0}}))
    _ohlcv(tmp_path / "o.db", [("AAA", "2023-04-21", 90.0), ("BBB", "2023-04-21", 24.0)])
    conn = _study_with_entry(tmp_path, "AAA", "2023-03-15", "2023-04-21")
    conn.execute("INSERT INTO entries SELECT 'e2','p2',arm,session,'BBB',structure,expiration,dte,short_strike,"
                 "long_strike,width,short_delta,credit_mid,credit_cross,rel_spread,frozen_json,created_at "
                 "FROM entries WHERE entry_id='e1'")
    res = h.settle(conn, out_dir=str(out), ohlcv_db=str(tmp_path / "o.db"), today="2023-05-01")
    closes = dict(conn.execute("SELECT entry_id, underlying_close FROM marks").fetchall())
    assert closes["e1"] == pytest.approx(99.0)          # 90 x 1.10
    assert closes["e2"] == pytest.approx(96.0)          # pre-split basis: 24 x 4
    assert res["split_basis"] == 1 and res["settled"] == 2


def test_eod_capture_skips_the_time_check_and_limits_symbols(tmp_path):
    from src import chain_archive
    db = str(tmp_path / "c.db")
    chain_archive.ensure_db(db)
    rows = []
    for k, d, b, a in ((88, -0.12, 0.46, 0.54), (94, -0.26, 1.32, 1.48), (95, -0.30, 1.66, 1.74),
                       (90, -0.16, 0.68, 0.72)):
        rows.append(["SPY", "2023-03-15", f"P{k}", "put", float(k), "2023-04-14", b, a, None, None, 0.2, d,
                     None, None, None, None, None, None, None, 100.0, None, "dolthub_eod"])
    with sqlite3.connect(db) as con:
        con.executemany(f"INSERT INTO chain_snapshots VALUES ({','.join('?' * 22)})", rows)
    conn = connect(str(tmp_path / "s.db"), settings=h.settings(default_settings()))
    res = capture.capture_session(conn, "2023-03-15", chain_db=db, symbols=["SPY"])
    assert res["status"] == {"captured": 1}
    pair = conn.execute("SELECT status FROM pairs").fetchone()[0]
    assert pair in ("matched", "identical")             # no OI in Dolt: baseline still eligible
