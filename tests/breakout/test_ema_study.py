import numpy as np
import pandas as pd

from scripts.study_ema_breakouts import observation_frame


def test_breakout_outcome_uses_later_closes_and_features_ignore_them():
    market = pd.Series(np.full(140, 100.0))
    anchors = np.arange(len(market)) == 119
    start = observation_frame(market, anchors, 5).loc[119]
    later = market.copy()
    later.iloc[120] = 102
    hit = observation_frame(later, anchors, 5).loc[119]
    assert not start.hit and hit.hit
    assert (start.bucket, start["5"], start["10"]) == (hit.bucket, hit["5"], hit["10"])
    later.iloc[124] = np.nan
    unresolved = observation_frame(later, anchors, 5).loc[119]
    assert not unresolved.observed
