import datetime as dt

import pandas as pd
import pytest

from src.raw_closes import settlement_close as settlement_truth, unsplit

D = dt.date


def _series():
    # split-adjusted closes around a 4-for-1 split on 2020-08-31
    s = pd.Series({D(2020, 8, 28): 125.0, D(2020, 8, 31): 129.0, D(2020, 9, 4): 121.0})
    return unsplit(s, pd.Series({D(2020, 8, 31): 4.0}))


def test_unsplit_restores_pre_split_raw_prices_only():
    raw = _series()
    assert raw[D(2020, 8, 28)] == pytest.approx(500.0)      # before ex-date: x4
    assert raw[D(2020, 8, 31)] == pytest.approx(129.0)      # ex-date onward: unchanged


def test_settlement_truth_uses_contract_basis_when_split_during_hold():
    raw = _series()
    raw.attrs["splits"] = {"2020-08-31": 4.0}
    # entered before the split, expires after: contract was adjusted -> pre-split equivalent
    assert settlement_truth(raw, "2020-08-26", "2020-09-04") == pytest.approx(484.0)
    # entered after the split: plain raw close
    assert settlement_truth(raw, "2020-08-31", "2020-09-04") == pytest.approx(121.0)
