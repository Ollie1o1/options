"""Per-contract expected return on risk, and the entry choice built on it.

WHY THIS EXISTS. The auto-log allowlist selects by strategy NAME. That is the
wrong instrument twice over.

It is wrong statistically, because a name is not an edge. Measured 2026-08-24
on return on capital at risk:

    Bear Call   wins 59.3% at +30.20%, loses at -55.54%  =>  -4.73%
    Long Call   wins 38.6% at +82.79%, loses at -52.42%  =>  -0.18%
    Bull Put    wins 65.0% at +52.13%, loses at -48.72%  =>  +16.80%

The hit rate and the edge point different ways. Only the product decides, and
the product varies contract by contract inside every one of those rows.

It is wrong methodologically, because it is self-sealing. Under a Bull-Put-only
allowlist no other structure can ever accumulate evidence, so a rule justified
by "absence of evidence" guarantees that absence forever. Bear Call, Short Put
and Iron Condor last entered the book on 2026-07-30/31 and Long Put on
2026-07-13; nothing since, and nothing ever again while the list stands.

WHAT REPLACES IT. A number every structure can be compared on:

    E[return on risk] = p * W_s + (1 - p) * L_s

`p` is the calibrated probability from `pop_calibration`, which is validated
out-of-sample and is already cross-structure. `W_s` and `L_s` are the mean win
and loss magnitudes of structure `s` — properties of its payoff geometry,
MEASURED rather than learned.

The decomposition is the point. Predicting return whole failed its guard on
this book (walk-forward slope 0.368, 95% CI [-0.252, 0.987]): option returns
are heavy-tailed and five features cannot pin a conditional mean from 609
points. Imposing the geometry we already know, and asking the model only for
the probability it has been shown to get right, is a variance reduction rather
than another fit.

This module does not decide policy. `choose` returns the best candidate or
None; whether the book acts on it is the caller's business, and the existing
gates — friction, EV, earnings, sizing — still run first and still refuse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import pop_calibration as pc

log = logging.getLogger(__name__)

#: Below this many closed trades a structure does not get to speak for itself
#: and borrows the pooled geometry. Its own mean would otherwise be one or two
#: trades wide.
MIN_ROWS_FOR_OWN_GEOMETRY = 20


@dataclass(frozen=True)
class Geometry:
    """What a structure pays when it wins and costs when it loses."""
    win: float
    loss: float
    n: int = 0
    #: False when `loss` was borrowed because this structure has no losses of
    #: its own yet. A structure that has only ever won has an UNKNOWN loss,
    #: never a costless one — zero there would make its expected return
    #: unbeatable by construction.
    own_losses: bool = True


def _pooled(df: pd.DataFrame) -> Geometry:
    r = pd.to_numeric(df["ret_on_risk"], errors="coerce").dropna()
    wins, losses = r[r > 0], r[r <= 0]
    return Geometry(
        win=float(wins.mean()) if len(wins) else 0.10,
        loss=float(losses.mean()) if len(losses) else -0.10,
        n=int(len(r)))


def magnitudes(df: pd.DataFrame) -> Dict[str, Geometry]:
    """Mean win and mean loss on capital at risk, per structure.

    Return on CAPITAL AT RISK, not on premium or credit. The three coincide
    only for long premium, and on the wrong one the same trades give the
    opposite sign.
    """
    if df is None or len(df) == 0 or "ret_on_risk" not in df:
        return {}
    frame = df.dropna(subset=["ret_on_risk"])
    if len(frame) == 0:
        return {}

    pooled = _pooled(frame)
    out: Dict[str, Geometry] = {"": pooled}
    for name, g in frame.groupby("strategy"):
        r = pd.to_numeric(g["ret_on_risk"], errors="coerce").dropna()
        wins, losses = r[r > 0], r[r <= 0]
        if len(r) < MIN_ROWS_FOR_OWN_GEOMETRY:
            out[str(name)] = Geometry(pooled.win, pooled.loss, int(len(r)),
                                      own_losses=False)
            continue
        out[str(name)] = Geometry(
            win=float(wins.mean()) if len(wins) else pooled.win,
            loss=float(losses.mean()) if len(losses) else pooled.loss,
            n=int(len(r)),
            own_losses=bool(len(losses)))
    return out


def geometry_for(mags: Dict[str, Geometry], strategy: Optional[str]) -> Geometry:
    """This structure's geometry, or the pooled one for an unseen structure.

    An unseen structure must NOT be unrepresentable — that would rebuild the
    allowlist by the back door, as an absent key instead of a missing name.
    """
    if not mags:
        return Geometry(0.10, -0.10, 0)
    key = str(strategy or "")
    return mags.get(key) or mags.get("") or Geometry(0.10, -0.10, 0)


def expected_return_for(row: Dict[str, Any], model: Optional[pc.Model],
                        mags: Dict[str, Geometry]) -> Optional[float]:
    """`p * W + (1 - p) * L` for one candidate, or None without a model."""
    p = pc.probability_for(row, model)
    if p is None:
        return None
    g = geometry_for(mags, row.get("strategy") or row.get("strategy_name"))
    return float(p * g.win + (1.0 - p) * g.loss)


def expected_returns(df: pd.DataFrame, model: Optional[pc.Model],
                     mags: Dict[str, Geometry]) -> np.ndarray:
    """Vectorised `expected_return_for` over a board."""
    if model is None or len(df) == 0:
        return np.full(len(df), np.nan)
    p = pc.predict(model, df)
    wins = np.array([geometry_for(mags, s).win
                     for s in df.get("strategy", pd.Series([""] * len(df)))])
    losses = np.array([geometry_for(mags, s).loss
                       for s in df.get("strategy", pd.Series([""] * len(df)))])
    return p * wins + (1.0 - p) * losses


def walk_forward(df: pd.DataFrame,
                 features: Sequence[str] = pc.DEFAULT_FEATURES,
                 seed_n: int = 300, step: int = 50,
                 min_train: int = 100) -> pd.DataFrame:
    """Out-of-sample expected returns, strictly time-ordered.

    BOTH halves are re-estimated on the training window at every fold — the
    probability model AND the magnitudes. Measuring `W_s` on the full history
    and predicting the middle of it would leak the outcome being predicted,
    which is the quiet version of the same mistake random k-fold makes.
    """
    cols = ["entry_date", "strategy", "predicted", "actual", "trained_through"]
    if len(df) <= seed_n:
        return pd.DataFrame(columns=cols)

    ordered = df.sort_values("entry_date", kind="mergesort").reset_index(drop=True)
    dates = ordered["entry_date"].astype(str)

    out: List[pd.DataFrame] = []
    i = seed_n
    while i < len(ordered):
        boundary = dates.iloc[i]
        train = ordered[dates < boundary]
        block = ordered.iloc[i:i + step]
        i += step
        if len(train) < min_train or block.empty:
            continue
        model = pc.fit(train, features=features)
        mags = magnitudes(train)
        out.append(pd.DataFrame({
            "entry_date": block["entry_date"].astype(str).to_numpy(),
            "strategy": block["strategy"].astype(str).to_numpy(),
            "predicted": expected_returns(block, model, mags),
            "actual": pd.to_numeric(block["ret_on_risk"],
                                    errors="coerce").to_numpy(),
            "trained_through": model.trained_through,
        }))

    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True).dropna(subset=["predicted"])


def choose(board: pd.DataFrame,
           column: str = "expected_return") -> Optional[Dict[str, Any]]:
    """The best candidate on the board, or None if none is worth taking.

    Refuse, do not rank: the best of an all-negative board is still negative,
    and a selector that always returns something turns a screen into a
    requirement to trade.

    Structure name is not consulted. That is the entire change.
    """
    if board is None or len(board) == 0 or column not in board:
        return None
    vals = pd.to_numeric(board[column], errors="coerce")
    if vals.isna().all() or float(vals.max()) <= 0.0:
        return None
    return dict(board.loc[vals.idxmax()])
