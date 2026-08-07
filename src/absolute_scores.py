"""Absolute, batch-independent mappings for two components that were
within-chain ranks.

``calculate_scores`` runs once per symbol, so ``rank_norm`` ranked each
contract against its own chain and nothing else.  The composite it feeds is
then compared *across* symbols, where a within-chain rank carries no level
information: a chain whose every contract decays fast still produced a
contract at ``theta_score`` 1.0.  Measured on 908 ledger rows, only
(0.079 / 0.179)^2 ~ 19% of stored ``theta_score`` variance was between-ticker,
and that residue exists only because top-N selection filtered which rank got
logged.  Together the two components here carry 18.5% of the live IC-blended
composite.

Both mappings are logistic in ``log10`` of the raw quantity.  The raw
quantities are right-skewed and DTE-driven -- archived theta pressure runs p5
0.0014, median 0.0186, p95 0.2313 -- so a linear sigmoid saturates.

The constants are **frozen**, not refitted per scan: a mapping that moved with
the batch would reintroduce exactly the defect this removes.  They are
calibrated on 293,343 archived chain snapshots at DTE 7-180
(``data/chain_archive.db``), centred on the median of ``log10`` and scaled so
the interquartile range spans the central half of the sigmoid.  Calibrating on
the paper ledger instead would fit the *selected* picks, whose log10 IQR is
0.265 against the chain population's 0.915 -- 3.5x narrower, which saturates
most of a real chain.

See ``docs/ABSOLUTE_SCORES_20260807.md`` and
``scripts/measure_absolute_scores.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Frozen from data/chain_archive.db, n = 293,343 at DTE 7-180.
# Mapped population lands 3.4% below 0.05 and 2.3% above 0.95 for theta,
# 5.4% / 0.0% for vega -- i.e. the sigmoid is used across its live range.
THETA_LOG_CENTRE = -1.7314
THETA_LOG_SCALE = 2.4021
VEGA_LOG_CENTRE = 1.5239
VEGA_LOG_SCALE = 3.5672

_NEUTRAL = 0.5


def _logistic_log10(vals: pd.Series, centre: float, scale: float) -> pd.Series:
    """Logistic in log10 space, NaN-safe and bounded to [0, 1].

    A non-positive or missing input cannot be ranked on a log scale and yields
    the neutral 0.5 rather than an extreme -- the same convention the rank
    version used when it filled NaNs with the median.
    """
    v = pd.to_numeric(vals, errors="coerce").astype(float)
    v = v.replace([np.inf, -np.inf], np.nan)
    lg = np.log10(v.where(v > 0))
    out = 1.0 / (1.0 + np.exp(-scale * (lg - centre)))
    return pd.Series(out, index=vals.index).fillna(_NEUTRAL).clip(0.0, 1.0)


def theta_pressure_score(theta: pd.Series, premium: pd.Series,
                         is_seller: bool) -> pd.Series:
    """Score decay pressure ``|theta| / premium`` on an absolute scale.

    Sellers are paid by decay and buyers pay for it, so the sign flips exactly
    as it did in the rank version at ``options_screener.py:1746-1749``.
    """
    t = pd.to_numeric(theta, errors="coerce").abs()
    p = pd.to_numeric(premium, errors="coerce").clip(lower=0.01)
    pressure = (t / p).replace([np.inf, -np.inf], np.nan)
    scored = _logistic_log10(pressure, THETA_LOG_CENTRE, THETA_LOG_SCALE)
    return scored if is_seller else (1.0 - scored).clip(0.0, 1.0)


def vega_risk_score_absolute(vega: pd.Series,
                             iv_percentile: pd.Series) -> pd.Series:
    """High vega while IV is already elevated is mean-reversion risk.

    Same shape as the original -- dollar vega times IV percentile, inverted --
    but the vega term is an absolute mapping rather than ``.rank(pct=True)``.
    A missing IV percentile falls back to neutral 0.5, never 0.
    """
    vd = pd.to_numeric(vega, errors="coerce").abs() * 100.0
    ivp = (pd.to_numeric(iv_percentile, errors="coerce")
           .fillna(_NEUTRAL).clip(0.0, 1.0))
    scaled = _logistic_log10(vd, VEGA_LOG_CENTRE, VEGA_LOG_SCALE) * ivp
    return (1.0 - scaled).clip(0.0, 1.0)
