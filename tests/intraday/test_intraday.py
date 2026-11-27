"""Synthetic-bar tests for src/intraday and scripts/intraday_tests (no network)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.intraday import alpaca_bars
from src.intraday.sessions import build_tables, full_sessions, hhmm, price_at, stack, with_clock
from src.intraday.stats import block_bootstrap_means, holm, summarise
from scripts import intraday_tests as it

ET = "America/New_York"


def bars_for(day: dt.date, closes: dict, opens: dict | None = None) -> pd.DataFrame:
    """One bar per given 'HH:MM' start with close c (and open o if given)."""
    rows = []
    for clock, c in closes.items():
        h, m = (int(x) for x in clock.split(":"))
        t = pd.Timestamp(dt.datetime(day.year, day.month, day.day, h, m), tz=ET).tz_convert("UTC")
        o = (opens or {}).get(clock, c)
        rows.append({"t": t, "o": o, "h": max(o, c), "l": min(o, c), "c": c, "v": 100, "n": 1, "vw": c})
    return pd.DataFrame(rows)


D1, D2, D3 = dt.date(2017, 3, 6), dt.date(2017, 3, 7), dt.date(2017, 3, 8)


def full_day(day, p1029, p1529, p1559, extra=None):
    closes = {"09:30": 100.0, "10:29": p1029, "15:29": p1529, "15:59": p1559}
    closes.update(extra or {})
    return bars_for(day, closes)


def test_price_at_uses_bar_starting_one_minute_before():
    b = with_clock(bars_for(D1, {"10:28": 1.0, "10:29": 2.0, "10:30": 3.0}))
    assert price_at(b, hhmm("10:30"))[D1] == 2.0


def test_price_at_falls_back_within_five_minutes_only():
    b = with_clock(bars_for(D1, {"10:25": 5.0}))
    assert price_at(b, hhmm("10:30"))[D1] == 5.0
    b = with_clock(bars_for(D1, {"10:24": 5.0}))
    assert D1 not in price_at(b, hhmm("10:30")).index


def test_full_session_requires_late_bar_and_prev_close_requires_full_prev_day():
    spy = pd.concat([full_day(D1, 100, 101, 102),
                     bars_for(D2, {"09:30": 100.0, "12:59": 100.0}),   # early close
                     full_day(D3, 100, 101, 102)])
    fs = full_sessions(with_clock(spy))
    assert list(fs.values) == [True, False, True]
    tables = build_tables({"SPY": spy}, it.TARGETS)
    t = tables["SPY"]
    assert list(t.index) == [D1, D3]
    assert np.isnan(t.loc[D3, "prev_close"])  # previous trading day was not full
    assert np.isnan(t.loc[D1, "prev_close"])


def test_calendar_excludes_early_close_even_with_after_hours_bars():
    # D2 is a 13:00 early close but has after-hours prints at 15:55-15:59.
    spy = pd.concat([full_day(D1, 100, 101, 102), full_day(D2, 100, 101, 102), full_day(D3, 100, 101, 102)])
    cal = pd.DataFrame({"date": [D1, D2, D3], "open": "09:30", "close": ["16:00", "13:00", "16:00"]})
    assert list(full_sessions(with_clock(spy)).values) == [True, True, True]      # bar rule alone fails
    fs = full_sessions(with_clock(spy), cal)
    assert list(fs.values) == [True, False, True]
    t = build_tables({"SPY": spy}, it.TARGETS, calendar=cal)["SPY"]
    assert list(t.index) == [D1, D3] and np.isnan(t.loc[D3, "prev_close"])


def test_calendar_day_with_no_bars_breaks_prev_close():
    # D2 is a trading day per calendar but SPY has no data: D3 must not use D1's close.
    spy = pd.concat([full_day(D1, 100, 101, 102), full_day(D3, 100, 101, 102)])
    cal = pd.DataFrame({"date": [D1, D2, D3], "open": "09:30", "close": "16:00"})
    t = build_tables({"SPY": spy}, it.TARGETS, calendar=cal)["SPY"]
    assert np.isnan(t.loc[D3, "prev_close"])


def test_prev_close_is_prior_day_1600():
    spy = pd.concat([full_day(D1, 100, 101, 102.5), full_day(D2, 100, 101, 102)])
    t = build_tables({"SPY": spy}, it.TARGETS)["SPY"]
    assert t.loc[D2, "prev_close"] == 102.5


def test_i1_fades_the_move_and_charges_two_cents_round_trip():
    # 10:30 -> 15:30 up (100 -> 110): position short; 15:30 -> 16:00 falls 110 -> 109.
    spy = full_day(D1, 100.0, 110.0, 109.0)
    long = stack(build_tables({"SPY": spy}, it.TARGETS))
    tr = it.rule_i1(long)
    assert len(tr) == 1
    assert tr["gross"].iloc[0] == pytest.approx(1.0 / 110.0)
    net = it.daily(tr, 1)
    assert net.iloc[0] == pytest.approx(1.0 / 110.0 - 0.02 / 110.0)


def test_daily_requires_min_count():
    rows = pd.DataFrame({"date": [D1, D1, D2], "symbol": ["SPY", "QQQ", "SPY"],
                         "gross": [0.01, 0.03, 0.02], "inv_px": [0.0, 0.0, 0.0], "cents": 0.01})
    out = it.daily(rows, 2)
    assert list(out.index) == [D1] and out.iloc[0] == pytest.approx(0.02)


def test_s5_only_trades_gaps_over_half_percent_in_gap_direction():
    d1 = full_day(D1, 100, 100, 100.0)
    d2 = bars_for(D2, {"09:30": 101.0, "09:34": 101.0, "10:29": 101, "15:29": 101, "15:59": 102.0},
                  opens={"09:30": 101.0})
    long = stack(build_tables({"SPY": pd.concat([d1, d2])}, it.TARGETS))
    tr = it.rule_s5(long)
    assert list(tr["date"]) == [D2]
    assert tr["gross"].iloc[0] == pytest.approx(102.0 / 101.0 - 1)


def test_big_move_mask_uses_only_prior_sessions():
    n = 200
    dates = [dt.date(2017, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    df = pd.DataFrame({"date": dates, "symbol": "SPY"})
    move = pd.Series(np.r_[np.full(n - 1, 0.001), 0.05])
    mask = it.big_move_mask(df, move)
    assert not mask.iloc[:120].any()          # fewer than 120 prior values
    assert mask.iloc[-1]                     # today's move is not in its own threshold


def test_block_bootstrap_is_reproducible_and_centred():
    x = np.random.default_rng(0).normal(0.001, 0.01, 500)
    a = block_bootstrap_means(x, draws=2000, seed=1)
    b = block_bootstrap_means(x, draws=2000, seed=1)
    assert np.array_equal(a, b)
    assert abs(a.mean() - x.mean()) < 2e-4
    s = summarise(x, draws=2000)
    assert s["lo"] < s["mean"] < s["hi"]


def test_holm_step_down():
    out = holm({"a": 0.001, "b": 0.03, "c": 0.04})
    # thresholds 0.05/3, 0.05/2, 0.05/1: a passes, b (0.03 > 0.025) stops the procedure
    assert out["a"]["reject"] and not out["b"]["reject"] and not out["c"]["reject"]
    assert out["a"]["p_holm"] == pytest.approx(0.003)
    assert out["b"]["p_holm"] == pytest.approx(0.06)
    assert out["c"]["p_holm"] == pytest.approx(0.06)  # monotone: max(0.06, 0.04)


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def test_fetch_bars_follows_pages_and_retries_429(monkeypatch):
    monkeypatch.setattr(alpaca_bars.time, "sleep", lambda s: None)
    t0 = "2017-03-06T14:30:00Z"
    pages = [FakeResp(429, {}),
             FakeResp(200, {"bars": {"SPY": [{"t": t0, "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "n": 1, "vw": 1}]},
                            "next_page_token": "abc"}),
             FakeResp(200, {"bars": {"SPY": [{"t": "2017-03-06T14:31:00Z", "o": 2, "h": 2, "l": 2, "c": 2,
                                              "v": 1, "n": 1, "vw": 2}]}, "next_page_token": None})]
    seen = []

    def get(url, headers, params, timeout):
        seen.append(params.get("page_token"))
        return pages.pop(0)

    df = alpaca_bars.fetch_bars(get, {}, "SPY", "a", "b", pause=0)
    assert list(df["c"]) == [1, 2]
    assert seen == [None, None, "abc"]


def test_clamp_end_caps_recent_data():
    now = dt.datetime(2026, 11, 27, 18, 0, 30, tzinfo=dt.timezone.utc)
    assert alpaca_bars.clamp_end("2026-12-01T00:00:00Z", now) == "2026-11-27T17:44:00Z"
    assert alpaca_bars.clamp_end("2026-10-01T00:00:00Z", now) == "2026-10-01T00:00:00Z"


def test_keep_session_window_drops_after_hours():
    df = pd.DataFrame({"t": pd.to_datetime(["2017-03-06T08:59:00Z", "2017-03-06T09:00:00Z",
                                            "2017-03-06T20:59:00Z", "2017-03-06T21:00:00Z"], utc=True)})
    kept = alpaca_bars.keep_session_window(df)
    assert len(kept) == 2  # 04:00 and 15:59 ET kept; 03:59 and 16:00 dropped


def test_load_credentials_reads_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text("# c\nAPCA_API_KEY_ID=kid\nAPCA_API_SECRET_KEY='sec'\n", encoding="utf-8")
    assert alpaca_bars.load_credentials(p) == ("kid", "sec")
