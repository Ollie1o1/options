"""Synthetic tests for src/intraday/realized.py (V1 sleeve vol-forecast test)."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from src.intraday import realized as rz

ET = "America/New_York"
D1, D2, D3 = dt.date(2019, 3, 4), dt.date(2019, 3, 5), dt.date(2019, 3, 6)


def day_bars(day, closes, drop=()):
    rows = []
    for k, c in enumerate(closes):
        if k in drop:
            continue
        m = rz.SLOTS[k]
        t = pd.Timestamp(dt.datetime(day.year, day.month, day.day, m // 60, m % 60), tz=ET).tz_convert("UTC")
        rows.append({"t": t, "o": c, "h": c, "l": c, "c": c, "v": 1, "n": 1, "vw": c})
    return pd.DataFrame(rows)


def sessions(*days, full=None):
    return pd.Series(full if full is not None else [True] * len(days), index=list(days), dtype=bool)


def test_first_interval_absorbs_overnight_and_rest_are_consecutive():
    a = [100.0] * 78
    b = [101.0] + [101.0 * (1.001 ** k) for k in range(1, 78)]
    r = rz.interval_returns(pd.concat([day_bars(D1, a), day_bars(D2, b)]), sessions(D1, D2))
    assert r.loc[D1].isna().all()                      # no previous close
    assert r.loc[D2, 0] == pytest.approx(0.01)         # 100 (15:55 prev) -> 101 (09:30)
    assert r.loc[D2, 5] == pytest.approx(0.001)


def test_validity_rules():
    a = [100.0] * 78
    # D2 misses 4 bars (74 < 75): invalid; D3 misses the 09:30 bar: invalid
    bars = pd.concat([day_bars(D1, a), day_bars(D2, a, drop=(10, 11, 12, 13)), day_bars(D3, a, drop=(0,))])
    r = rz.interval_returns(bars, sessions(D1, D2, D3))
    assert r.loc[D2].isna().all() and r.loc[D3].isna().all()
    # 3 missing bars are forward-filled: valid
    r2 = rz.interval_returns(pd.concat([day_bars(D1, a), day_bars(D2, a, drop=(10, 11, 12))]), sessions(D1, D2))
    assert r2.loc[D2].notna().all()


def test_previous_session_must_be_full():
    a = [100.0] * 78
    bars = pd.concat([day_bars(D1, a), day_bars(D2, a), day_bars(D3, a)])
    r = rz.interval_returns(bars, sessions(D1, D2, D3, full=[True, False, True]))
    assert list(r.index) == [D1, D3]
    assert r.loc[D3].isna().all()


def test_basket_rv_combines_weighted_interval_returns():
    idx = [D1, D2]
    ra = pd.DataFrame([[0.01] + [0.0] * 77, [np.nan] * 78], index=idx)
    rb = pd.DataFrame([[-0.01] + [0.0] * 77, [0.0] * 78], index=idx)
    rv = rz.basket_daily_rv({"A": ra, "B": rb}, {"A": 1 / 3, "B": 1 / 3})
    assert rv[D1] == pytest.approx(0.0)    # offsetting returns
    assert np.isnan(rv[D2])                # A invalid that day


def test_e1_requires_18_of_20_valid():
    idx = [D1 + dt.timedelta(days=i) for i in range(20)]
    rv = pd.Series([0.0001] * 20, index=idx)
    assert rz.e1_intraday_rv20(rv, idx[-1]) == pytest.approx(math.sqrt(252 * 0.0001))
    rv.iloc[:3] = np.nan
    assert rz.e1_intraday_rv20(rv, idx[-1]) is None


def test_har_target_is_forward_mean_and_fit_respects_as_of():
    n = 200
    idx = [D1 + dt.timedelta(days=i) for i in range(n)]
    v = pd.Series(np.linspace(1e-4, 3e-4, n), index=idx)
    f = rz.har_frame(v)
    s = f.index[0]
    i = idx.index(s)
    assert f.loc[s, "y"] == pytest.approx(v.iloc[i + 1:i + 22].mean())
    assert f.loc[s, "end"] == idx[i + 21]
    as_of = idx[100]
    assert all(e <= as_of for e in f[[e <= as_of for e in f["end"]]]["end"])
    assert rz.fit_har([f], as_of, min_pairs=10) is not None
    assert rz.fit_har([f], as_of, min_pairs=10_000) is None


def test_qlike_and_ewma():
    assert rz.qlike(0.04, 0.04) == pytest.approx(0.0)
    assert rz.qlike(0.04, 0.02) > 0 and rz.qlike(0.02, 0.04) > 0
    r = np.full(100, 0.01)
    assert rz.e3_ewma(r) == pytest.approx(math.sqrt(252) * 0.01)
    assert rz.e3_ewma(r[:30]) is None
