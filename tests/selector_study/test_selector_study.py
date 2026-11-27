"""Selector-vs-liquidity paper study: capture, matching, marks, outcomes and readout."""
from __future__ import annotations

import sqlite3

import pytest

from src import chain_archive
from src.selector_study import baseline as bl
from src.selector_study import capture, marks, outcomes, readout, synthetic, workflow
from src.selector_study.db import connect, default_settings
from src.structure.chain_data import ChainContract, ChainSlice, load_archived_chain_slice

S = "2026-11-24"          # Tuesday; next sessions 11-25, 11-27 (11-26 is Thanksgiving), 11-30, 12-01
EXP = "2026-12-24"        # 30 DTE: inside the 14-58 window
TS = f"{S}T20:30:00Z"     # 15:30 ET

# strike: (delta, bid, ask). Selector takes 94 (closest to 0.25) over 88 (nearest 5% below, 94-5=89).
# The tighter 95/90 vertical is inside the delta and width tolerances, so the baseline takes it.
PUTS = {85: (-0.08, 0.28, 0.32), 88: (-0.12, 0.46, 0.54), 90: (-0.16, 0.68, 0.72),
        92: (-0.21, 0.96, 1.04), 94: (-0.26, 1.32, 1.48), 95: (-0.30, 1.66, 1.74), 96: (-0.34, 1.95, 2.05)}


def _row(sym, day, strike, delta, bid, ask, *, ts=None, exp=EXP, oi=100.0, spot=100.0):
    vals = {"symbol": sym, "snap_date": day, "contract": f"{sym}{exp}P{strike}", "type": "put",
            "strike": float(strike), "expiration": exp, "bid": bid, "ask": ask, "iv": 0.25, "delta": delta,
            "open_interest": oi, "volume": 10.0, "spot": spot, "snapshot_ts": ts or f"{day}T20:30:00Z",
            "source": "fixture"}
    return [vals.get(c) for c in chain_archive._COLUMNS]


def _archive(path, rows, attempts=()):
    chain_archive.ensure_db(str(path))
    with sqlite3.connect(path) as con:
        con.executemany(f"INSERT OR REPLACE INTO chain_snapshots VALUES ({','.join('?' * 22)})", rows)
        con.executemany("INSERT OR REPLACE INTO archive_attempts VALUES (?,?,?,?,?,?)", attempts)


def _chain(sym, day=S, **kw):
    return [_row(sym, day, k, d, b, a, **kw) for k, (d, b, a) in PUTS.items()]


def _study(tmp_path, symbols, *, synthetic_db=False, **overrides):
    st = default_settings()
    st["symbols"] = list(symbols)
    st.update(overrides)
    return connect(str(tmp_path / "study.db"), settings=st, synthetic=synthetic_db)


# ── capture ──────────────────────────────────────────────────────────────────

def test_capture_builds_both_arms_on_identical_quotes(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, _chain("AAA"))
    conn = _study(tmp_path, ["AAA"])
    res = capture.capture_session(conn, S, chain_db=str(db))
    assert res["run_status"] == "complete" and res["status"] == {"captured": 1}
    pair = dict(conn.execute("SELECT * FROM pairs").fetchone())
    assert pair["status"] == "matched"
    legs = {r["arm"]: (r["short_strike"], r["long_strike"]) for r in conn.execute("SELECT * FROM entries")}
    assert legs == {"selector": (94.0, 88.0), "baseline": (95.0, 90.0)}
    obs = dict(conn.execute("SELECT * FROM observations").fetchone())
    assert obs["quote_time"] is None and obs["observed_at"] == TS and obs["slice_json"]


def test_duplicate_runs_and_symbols_are_idempotent(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, _chain("AAA"))
    conn = _study(tmp_path, ["AAA", "aaa ", "AAA"])
    capture.capture_session(conn, S, chain_db=str(db))
    again = capture.capture_session(conn, S, chain_db=str(db))
    assert again["skipped"] is True
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM pairs").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM capture_runs").fetchone()[0] == 1


def test_stale_and_off_window_inputs_are_kept_but_build_nothing(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, _chain("OLD", ts="2026-11-25T20:30:00Z") + _chain("EARLY", ts=f"{S}T13:00:00Z")
             + [r[:19] + [None] + r[20:] for r in _chain("NOSPOT")])
    conn = _study(tmp_path, ["OLD", "EARLY", "NOSPOT", "GONE"])
    res = capture.capture_session(conn, S, chain_db=str(db))
    assert res["status"] == {"stale": 1, "off_window": 1, "missing_spot": 1, "missing_chain": 1}
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0
    assert res["run_status"] == "partial"  # GONE can still arrive


def test_partial_provider_failure_is_visible_and_retry_completes_it(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, _chain("AAA"), attempts=[(S, "BBB", "error", "ConnectionError: down", 1, TS)])
    conn = _study(tmp_path, ["AAA", "BBB"])
    first = capture.capture_session(conn, S, chain_db=str(db))
    assert first["status"] == {"captured": 1, "provider_failure": 1} and first["run_status"] == "partial"
    detail = conn.execute("SELECT detail FROM observations WHERE symbol='BBB'").fetchone()[0]
    assert "ConnectionError" in detail

    _archive(db, _chain("BBB"))  # the archive's own retry filled the symbol later that day
    second = capture.capture_session(conn, S, chain_db=str(db))
    assert second["already"] == 1 and second["status"] == {"captured": 2}
    assert second["run_status"] == "complete"
    assert conn.execute("SELECT attempts FROM observations WHERE symbol='BBB'").fetchone()[0] == 2
    rep = readout.build(conn, today="2026-12-01")
    assert rep["coverage"]["runs"] == {"partial": 1, "complete": 1}


def test_archive_records_why_each_symbol_is_missing(tmp_path):
    db = str(tmp_path / "chain.db")

    def fetch(sym):
        if sym == "ERR":
            raise TimeoutError("feed timeout")
        if sym == "EMPTY":
            return []
        return [{"contract": "OKP1", "type": "put", "strike": 100.0, "expiration": "2026-12-24",
                 "bid": 1.0, "ask": 1.1, "spot": 100.0, "open_interest": 5}]

    chain_archive.archive_symbols(["OK", "ERR", "EMPTY"], db_path=db, today=S, fetcher=fetch)
    chain_archive.archive_symbols(["OK", "ERR"], db_path=db, today=S, fetcher=lambda s: [])
    with sqlite3.connect(db) as con:
        got = {s: (st, n) for s, st, n in con.execute("SELECT symbol, status, attempts FROM archive_attempts")}
    assert got["OK"] == ("archived", 1)         # already archived: not re-attempted, never downgraded
    assert got["ERR"] == ("no_rows", 2)
    assert got["EMPTY"] == ("no_rows", 1)
    chain_archive.archive_symbols(["D1", "D2", "D3"], db_path=db, today=S,
                                  fetcher=lambda s: [], max_consecutive_failures=1)
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT status FROM archive_attempts WHERE symbol='D3'").fetchone()[0] == "feed_down"


def test_slice_loader_reads_a_chosen_session(tmp_path):
    db = tmp_path / "chain.db"
    later = [_row("AAA", "2026-11-25", k, d, b + 0.5, a + 0.5) for k, (d, b, a) in PUTS.items()]
    _archive(db, _chain("AAA") + later)
    old = load_archived_chain_slice("AAA", target_dte=35, min_dte=14, max_dte=58, chain_db=str(db),
                                    build_rnd=False, snap_date=S)
    assert old is not None and old.snap_date == S and old.puts[0].bid == 0.28


# ── baseline ─────────────────────────────────────────────────────────────────

def _slice(puts):
    return ChainSlice(symbol="AAA", expiration=EXP, dte=30, snap_date=S, source="fixture", spot=100.0,
                      puts=[ChainContract(strike=float(k), opt_type="put", bid=b, ask=a, mid=(a + b) / 2,
                                          delta=d, open_interest=oi) for k, (d, b, a, oi) in puts.items()])


def _sel_row(short, long):
    return {"legs": ({"side": "SELL", "strike": short}, {"side": "BUY", "strike": long})}


def test_baseline_has_no_eligible_match_when_liquidity_rule_excludes_every_vertical():
    puts = {k: (d, b, a, 0) for k, (d, b, a) in PUTS.items()}  # zero open interest everywhere
    best, status, n = bl.match(_slice(puts), _sel_row(94, 88), rules=default_settings()["baseline"],
                               max_friction_pct=0.28)
    assert best is None and n == 0 and status.startswith("no_match:no eligible vertical")


def test_baseline_tie_breaks_are_deterministic():
    puts = {k: (d, b, a, 100) for k, (d, b, a) in PUTS.items()}
    puts[95] = (-0.30, 1.66, 1.74, 100)
    puts[92] = (-0.22, 1.21, 1.29, 100)   # 92/86-ish not available; 92/88 width 4 is outside tolerance
    sl = _slice(puts)
    rules = default_settings()["baseline"]
    a = bl.match(sl, _sel_row(94, 88), rules=rules, max_friction_pct=0.28)
    b = bl.match(_slice(dict(reversed(list(puts.items())))), _sel_row(94, 88), rules=rules, max_friction_pct=0.28)
    assert (a[0].short.strike, a[0].long.strike) == (b[0].short.strike, b[0].long.strike)


def test_no_match_pairs_stay_visible_in_the_readout(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, _chain("AAA"))
    conn = _study(tmp_path, ["AAA"], synthetic_db=True)
    conn.execute("UPDATE study_meta SET value=? WHERE key='settings'", (
        __import__("json").dumps({**default_settings(), "symbols": ["AAA"],
                                  "baseline": {**default_settings()["baseline"], "min_open_interest": 1000}}),))
    conn.commit()
    capture.capture_session(conn, S, chain_db=str(db))
    rep = readout.build(conn, today="2026-12-01")
    assert rep["matching"]["no_match"] == 1 and rep["matching"]["comparisons_possible"] == 0
    assert rep["matching"]["no_match_reasons"] == {"no eligible vertical with comparable exposure": 1}


# ── outcomes ─────────────────────────────────────────────────────────────────

SET = default_settings()


def _entry(**kw):
    e = {"session": S, "expiration": "2026-12-14", "width": 5.0, "short_strike": 95.0, "long_strike": 90.0,
         "credit_mid": 1.00, "credit_cross": 0.80}
    e.update(kw)
    return e


def _mark(day, status="marked", sb=0.30, sa=0.40, lb=0.05, lab=0.10, close=None):
    return {"mark_session": day, "status": status, "short_bid": sb, "short_ask": sa, "long_bid": lb,
            "long_ask": lab, "underlying_close": close}


def test_missing_marks_beyond_the_gap_limit_leave_no_outcome():
    ms = [_mark("2026-11-25", "missing_session"), _mark("2026-11-27", "missing_leg"),
          _mark("2026-11-30", "unpriceable"), _mark("2026-12-01", "missing_session"), _mark("2026-12-02")]
    out = outcomes.resolve(_entry(), ms, SET, "cross")
    assert out["status"] == "missing" and "4 sessions" in out["exit_reason"]
    assert outcomes.resolve(_entry(), ms[1:], SET, "cross")["status"] == "closed"


def test_min_hold_then_time_exit_and_expiry_settlement():
    out = outcomes.resolve(_entry(), [_mark("2026-11-25"), _mark("2026-11-27")], SET, "mid")
    assert out["exit_session"] == "2026-11-27" and out["exit_reason"] == "take_profit"
    far = _entry(expiration="2026-12-04", credit_mid=0.50, credit_cross=0.45)
    settled = outcomes.resolve(far, [_mark("2026-12-04", "settled", close=92.0)], SET, "cross")
    assert settled["exit_reason"] == "expired"
    assert settled["pnl"] == pytest.approx(0.45 - 3.0)
    assert settled["ror"] == pytest.approx((0.45 - 3.0) / (5.0 - 0.45))
    nos = outcomes.resolve(far, [_mark("2026-12-04", "settled", close=None)], SET, "cross")
    assert nos["status"] == "missing"


def test_non_fill_and_fees_are_identical_for_both_cost_cases():
    st = {**SET, "fee_per_contract": 1.0}
    out = outcomes.resolve(_entry(credit_cross=0.015), [], st, "cross")
    assert out["status"] == "not_filled"
    assert outcomes.resolve(_entry(credit_cross=0.015), [], st, "mid")["entry_credit"] == pytest.approx(0.98)


def _insert_pair(conn, pid, session, sel, base, sel_marks, base_marks):
    for arm, e, ms in (("selector", sel, sel_marks), ("baseline", base, base_marks)):
        eid = f"{pid}|{arm}"
        conn.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (eid, pid, arm, session, "AAA", "bull_put", e["expiration"], 20, e["short_strike"],
                      e["long_strike"], e["width"], 0.25, e["credit_mid"], e["credit_cross"], 0.1, "{}", "x"))
        for m in ms:
            conn.execute("INSERT INTO marks (entry_id, mark_session, status, short_bid, short_ask, long_bid, "
                         "long_ask) VALUES (?,?,?,?,?,?,?)",
                         (eid, m["mark_session"], m["status"], m["short_bid"], m["short_ask"],
                          m["long_bid"], m["long_ask"]))
    conn.execute("INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?)",
                 (pid, session, "AAA", sel["expiration"], "matched", None, 3, f"{pid}|selector", f"{pid}|baseline"))


def test_costs_reverse_an_apparent_midpoint_advantage(tmp_path):
    conn = _study(tmp_path, ["AAA"], synthetic_db=True)
    sel = _entry(credit_mid=1.00, credit_cross=0.70)     # wide quotes, more mid credit
    base = _entry(credit_mid=0.90, credit_cross=0.85)    # tight quotes
    _insert_pair(conn, "p1", S, sel, base,
                 [_mark("2026-11-30", sb=0.30, sa=0.60, lb=0.05, lab=0.15)],
                 [_mark("2026-11-30", sb=0.38, sa=0.42, lb=0.09, lab=0.11)])
    conn.commit()
    rep = readout.build(conn, today="2026-12-10")
    mid, cross = rep["paired"]["mid"], rep["paired"]["cross"]
    assert mid["mean_diff_weekly"] > 0 > cross["mean_diff_weekly"]
    assert rep["cost_reversal"] is True
    assert rep["verdict"] == readout.INSUFFICIENT
    assert "Costs reverse the sign" in readout.render(rep)


def test_real_study_hides_returns_before_the_read_date(tmp_path):
    conn = _study(tmp_path, ["AAA"])
    _insert_pair(conn, "p1", S, _entry(), _entry(), [_mark("2026-11-30")], [_mark("2026-11-30")])
    conn.commit()
    rep = readout.build(conn, today="2026-12-10")
    assert rep["locked"] and "mean_diff_weekly" not in rep["paired"]["cross"]
    assert rep["paired"]["cross"]["resolved_pairs"] == 1
    assert rep["verdict"].startswith("Locked until 2027-03-01") and "Insufficient data" in rep["verdict"]


# ── marks against an archive ─────────────────────────────────────────────────

def test_marker_writes_gap_rows_and_retries_them(tmp_path):
    db = tmp_path / "chain.db"
    later_other = [_row("ZZZ", d, 90, -0.2, 1.0, 1.1) for d in ("2026-11-25", "2026-11-27", "2026-11-30")]
    _archive(db, _chain("AAA") + later_other)
    conn = _study(tmp_path, ["AAA"])
    capture.capture_session(conn, S, chain_db=str(db))
    counts = marks.mark_entries(conn, chain_db=str(db), ohlcv_db=str(tmp_path / "none.db"), today="2026-11-30")
    assert counts["missing_session"] == 6  # 2 entries x 3 sessions, holiday skipped
    _archive(db, [_row("AAA", "2026-11-27", k, d, b * 0.4, a * 0.4) for k, (d, b, a) in PUTS.items()])
    marks.mark_entries(conn, chain_db=str(db), ohlcv_db=str(tmp_path / "none.db"), today="2026-11-30")
    st = {r[0]: r[1] for r in conn.execute(
        "SELECT mark_session, status FROM marks WHERE entry_id LIKE '%selector%'")}
    assert st == {"2026-11-25": "missing_session", "2026-11-27": "marked", "2026-11-30": "missing_session"}


# ── end to end ───────────────────────────────────────────────────────────────

def test_synthetic_multi_session_workflow(tmp_path):
    data = synthetic.build(str(tmp_path / "syn"), sessions=40, symbols=("SYNA", "SYNB", "SYNC"))
    st = default_settings()
    st["symbols"] = data["symbols"]
    study_db = str(tmp_path / "syn" / "study.db")
    connect(study_db, settings=st, synthetic=True).close()
    out = workflow.run(study_db=study_db, chain_db=data["chain_db"], ohlcv_db=data["ohlcv_db"],
                       today=data["sessions"][-1])
    rep = out["readout"]
    assert rep["synthetic"] and not rep["locked"]
    assert rep["coverage"]["status"]["provider_failure"] == 1
    assert rep["coverage"]["status"]["missing_chain"] == 1
    assert rep["paired"]["cross"]["weeks"] >= 5 and rep["paired"]["cross"]["ci95"] is not None
    assert rep["matching"]["comparisons_possible"] > 50
    again = workflow.run(study_db=study_db, chain_db=data["chain_db"], ohlcv_db=data["ohlcv_db"],
                         today=data["sessions"][-1])
    assert all(c.get("skipped") or c["run_status"] == "partial" for c in again["captures"])
    assert again["readout"]["paired"] == rep["paired"]


def test_settlement_close_is_put_back_on_the_unadjusted_basis(tmp_path):
    db = tmp_path / "chain.db"
    _archive(db, [_row("AAA", "2026-12-23", 95, -0.3, 1.0, 1.1, spot=110.0)])
    o = tmp_path / "o.db"
    with sqlite3.connect(o) as con:
        con.execute("CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL, high REAL, low REAL, volume REAL)")
        con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?)",
                        [("AAA", "2026-12-23", 100.0, 0, 0, 0), ("AAA", "2026-12-24", 99.0, 0, 0, 0)])
    with sqlite3.connect(db) as arch, sqlite3.connect(o) as oh:
        assert marks._close_on(oh, "AAA", "2026-12-24", arch) == pytest.approx(108.9)
        assert marks._close_on(oh, "AAA", "2027-02-01", arch) is None   # no archive spot within a week
