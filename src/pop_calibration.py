"""A calibrated, per-contract probability of profit.

The probability already on the boards, `pop_score`, identifies the STRUCTURE
and not the contract: measured 2026-08-23 it spans 0.53-0.58 across 117 Bull
Puts, 0.25-0.32 across 307 Long Calls, and never exceeds 0.61 anywhere in 909
trades. A quantity that does not vary between contracts cannot order them, and
a median split inside each strategy confirms it — four of six point the wrong
way, all intervals overlapping.

This module fits a different number: P(this position closes green under the
exit rules actually in force), trained on realised outcomes rather than on
Black-Scholes. Outcome here is dominated by the exit rule — Take Profit (50%
of credit) wins 93.6% of 187, Stop Loss (-50%) wins 0% of 103 — so the
quantity being modelled is whether the position reaches take-profit before
stop-loss before the clock.

THE MODEL SHIPS ONLY IF IT EARNS IT. `walk_forward` produces out-of-sample
predictions under a strict time order, `reliability` turns them into a curve
against realised outcomes, and `ship_check` refuses the model unless that
curve's slope clears zero. `load_model` returns None for an unshipped
artifact, so the guard holds at the read side too. Handing back "it is flat"
is a successful outcome of this module, not a failed one.

Deliberately small: at most five features plus a strategy term, fit by IRLS in
numpy. 909 closed trades carry ~430 wins, and Bull Put alone has 134 rows —
enough for about four parameters before a fit begins memorising. No sklearn,
no statsmodels; neither is in the venv and neither may be installed.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: The five contract-level features, in fit order. `credit_to_width` is
#: inactive (zero) for single-leg strategies rather than dropping them: that
#: would throw away 383 of the 909 closed trades.
DEFAULT_FEATURES: Tuple[str, ...] = (
    "abs_delta", "dte", "entry_iv", "iv_rank_score", "credit_to_width",
)

TRAINING_COLUMNS: Tuple[str, ...] = DEFAULT_FEATURES + (
    "strategy", "entry_date", "won", "ret_on_risk", "is_short",
)

#: Structures where the account is NET SHORT premium. The distinction is not
#: cosmetic: a high |delta| makes a long call MORE likely to win and a short
#: put LESS likely to, so a single shared coefficient lets whichever family has
#: more rows set the sign for both. With 383 long-premium and 526 short-premium
#: closed trades, that is exactly what happened — a TLT short put at delta 0.05
#: scored 27% while an NVDA short put at delta 0.36 scored 47%.
SHORT_STRATEGIES = frozenset({
    "Bull Put", "Bear Call", "Short Put", "Short Call", "Iron Condor",
    "Iron Butterfly", "Credit Spread", "Short Strangle", "Short Straddle",
})

DEFAULT_MODEL_PATH = os.path.join("data", "pop_calibration.json")

BUCKET_WIDTH = 0.10
MIN_BUCKET_N = 20
MIN_QUALIFYING_BUCKETS = 3


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
@dataclass
class Model:
    """Fitted coefficients plus everything needed to reproduce a prediction.

    `beta` is in STANDARDISED units — features are centred and scaled before
    the fit so that a single ridge penalty means the same thing to a delta of
    0.3 and a DTE of 45. `coefficient()` undoes the scaling, so what it
    returns is per original unit, which is the only form worth reading.
    """
    features: List[str]
    strategies: List[str]
    beta: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_train: int
    trained_through: str
    base_rate: float = 0.5
    kind: str = "logistic"
    #: True when the fit carried short/long interaction terms. Only possible
    #: when BOTH families were present in training — an interaction with no
    #: variation to fit is perfectly collinear with its own main effect.
    interacted: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def coefficient(self, name: str) -> float:
        """The fitted coefficient for `name`, per ORIGINAL unit."""
        i = self.features.index(name)
        return float(self.beta[1 + i] / self.std[i])

    @property
    def intercept(self) -> float:
        return float(self.beta[0])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _continuous(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    cols = [pd.to_numeric(df[f], errors="coerce").astype(float).to_numpy()
            for f in features]
    x = np.column_stack(cols) if cols else np.zeros((len(df), 0))
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def short_flags(df: pd.DataFrame) -> np.ndarray:
    """1.0 where the account is net short premium, 0.0 where it is long."""
    if "is_short" in df:
        vals = pd.to_numeric(df["is_short"], errors="coerce")
        if vals.notna().any():
            return vals.fillna(0.0).astype(float).to_numpy()
    labels = (df["strategy"].astype(str) if "strategy" in df
              else pd.Series([""] * len(df), index=df.index))
    return labels.isin(SHORT_STRATEGIES).astype(float).to_numpy()


def _design(df: pd.DataFrame, features: Sequence[str],
            strategies: Sequence[str], mean: np.ndarray, std: np.ndarray,
            interacted: bool) -> np.ndarray:
    """Intercept, standardised features, short/long interactions, strategies.

    The first strategy level is dropped as the baseline. An unseen strategy
    encodes as all-zero, which puts it on the baseline rather than raising —
    a new structure should read as "no strategy-specific adjustment", not as
    a crash on a live board.

    When `interacted`, every continuous feature appears twice: once as its own
    main effect and once multiplied by the short flag. That lets |delta|, DTE
    and IV carry a DIFFERENT sign for a seller than for a buyer, which they
    genuinely do. A strategy one-hot cannot express this on its own — it moves
    the intercept per structure, not the slope.
    """
    x = (_continuous(df, features) - mean) / std
    n = len(df)
    parts: List[np.ndarray] = [np.ones((n, 1)), x]
    if interacted:
        # Named `short`, not `s`: the strategy loop below binds `s` to a
        # string, and one name meaning two things in one function is how the
        # column and the label get crossed.
        short = short_flags(df).reshape(-1, 1)
        parts.extend([short, x * short])
    if len(strategies) > 1:
        labels = df["strategy"].astype(str).to_numpy() if "strategy" in df \
            else np.array([""] * n)
        onehot = np.zeros((n, len(strategies) - 1))
        for j, name in enumerate(strategies[1:]):
            onehot[:, j] = (labels == name).astype(float)
        parts.append(onehot)
    return np.hstack(parts)


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(a, b, rcond=None)[0]


def _fit_core(df: pd.DataFrame, features: Sequence[str], target: str,
              kind: str, ridge: float, max_iter: int) -> Model:
    """Ridge-penalised fit, standardised design, slopes penalised only.

    The ridge is not decoration. Near-constant columns are the NORMAL case
    here — that is the whole diagnosis this module exists to answer — and an
    unpenalised fit on a constant column is singular.
    """
    features = list(features)
    # A row with no recorded outcome is DROPPED, never zeroed. Filling it
    # would invent a data point at exactly the value that flattens a slope —
    # halving a planted coefficient of 1.0 to 0.48 in the test that guards
    # this. NULL means not recorded.
    values = pd.to_numeric(df[target], errors="coerce")
    df = df.loc[values.notna()]
    y = values.dropna().astype(float).to_numpy()

    raw = _continuous(df, features)
    mean = raw.mean(axis=0) if len(raw) else np.zeros(len(features))
    std = raw.std(axis=0) if len(raw) else np.ones(len(features))
    std = np.where(std < 1e-12, 1.0, std)

    strategies = (sorted(df["strategy"].astype(str).unique().tolist())
                  if "strategy" in df else [])
    # An interaction needs both families present; with only one, the term is
    # perfectly collinear with its own main effect and the ridge would simply
    # halve the coefficient.
    flags = short_flags(df)
    interacted = bool(len(flags)) and 0 < float(flags.mean()) < 1
    x = _design(df, features, strategies, mean, std, interacted)

    k = x.shape[1]
    penalty = np.eye(k) * ridge
    penalty[0, 0] = 0.0  # never penalise the intercept

    if kind == "linear":
        beta = _solve(x.T @ x + penalty, x.T @ y)
    else:
        beta = np.zeros(k)
        for _ in range(max_iter):
            eta = x @ beta
            p = _sigmoid(eta)
            w = np.clip(p * (1.0 - p), 1e-6, None)
            z = eta + (y - p) / w
            xtw = x.T * w
            new = _solve(xtw @ x + penalty, xtw @ z)
            if not np.all(np.isfinite(new)):
                break
            if np.max(np.abs(new - beta)) < 1e-9:
                beta = new
                break
            beta = new

    dates = df["entry_date"].astype(str) if "entry_date" in df else pd.Series([""])
    return Model(features=features, strategies=strategies, beta=beta,
                 mean=mean, std=std, n_train=int(len(df)),
                 trained_through=str(dates.max()) if len(df) else "",
                 base_rate=float(y.mean()) if len(y) else 0.5, kind=kind,
                 interacted=interacted)


def fit(df: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
        ridge: float = 1.0, max_iter: int = 100) -> Model:
    """P(closes green). Logistic regression by IRLS."""
    return _fit_core(df, features, "won", "logistic", ridge, max_iter)


def fit_return(df: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
               target: str = "ret_on_risk", ridge: float = 1.0) -> Model:
    """Expected return on capital at risk. Ridge linear regression.

    A separate model rather than a transform of the probability, because in
    this book the two disagree: measured out-of-sample, the 0.4-0.5 win-rate
    bucket returns -2.32% on risk while the 0.3-0.4 bucket returns +0.87%.
    Win rate rises monotonically; money does not follow it.
    """
    return _fit_core(df, features, target, "linear", ridge, 1)


def predict(model: Model, df: pd.DataFrame) -> np.ndarray:
    """The model's output for each row.

    A probability in [0, 1] for a logistic model; an expected return on
    capital at risk, unbounded and signed, for a linear one.
    """
    x = _design(df, model.features, model.strategies, model.mean, model.std,
                model.interacted)
    eta = x @ model.beta
    if model.kind == "linear":
        return np.asarray(eta, dtype=float)
    return np.clip(_sigmoid(eta), 0.0, 1.0)


#: Expected return on risk, for symmetry with `fit_return` at call sites.
predict_return = predict


# --------------------------------------------------------------------------
# Out-of-sample validation
# --------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, features: Sequence[str] = DEFAULT_FEATURES,
                 seed_n: int = 300, step: int = 50,
                 min_train: int = 100, target: str = "won") -> pd.DataFrame:
    """Expanding-window out-of-sample predictions, strictly time-ordered.

    A fold trains ONLY on rows dated strictly before the boundary row's day.
    Rows sharing that day sit in neither set for this fold. That strictness is
    the point: options entered on the same day on the same underlying are not
    independent, and a model that has seen one of them has effectively seen
    the others. Random k-fold here would report a skill the model does not
    have.
    """
    features = list(features)
    cols = ["entry_date", "strategy", "predicted", "actual", "trained_through"]
    if target == "won":
        cols.append("won")
    if len(df) <= seed_n:
        return pd.DataFrame(columns=cols)

    ordered = df.sort_values("entry_date", kind="mergesort").reset_index(drop=True)
    dates = ordered["entry_date"].astype(str)
    kind = "logistic" if target == "won" else "linear"

    out: List[pd.DataFrame] = []
    i = seed_n
    while i < len(ordered):
        boundary = dates.iloc[i]
        train = ordered[dates < boundary]
        block = ordered.iloc[i:i + step]
        i += step
        if len(train) < min_train or block.empty:
            continue
        model = _fit_core(train, features, target, kind, 1.0, 100)
        actual = pd.to_numeric(block[target], errors="coerce").to_numpy()
        frame = pd.DataFrame({
            "entry_date": block["entry_date"].astype(str).to_numpy(),
            "strategy": block["strategy"].astype(str).to_numpy()
            if "strategy" in block else "",
            "predicted": predict(model, block),
            "actual": actual,
            "trained_through": model.trained_through,
        })
        if target == "won":
            frame["won"] = actual
        out.append(frame)

    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)


def wilson(k: int, n: int) -> Tuple[float, float]:
    """95% Wilson interval. Normal-approximation intervals go outside [0, 1]
    on the thin buckets, which is exactly where they get read."""
    if n <= 0:
        return (0.0, 1.0)
    z = 1.96
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability(oos: pd.DataFrame, bucket_width: float = BUCKET_WIDTH,
                min_n: int = MIN_BUCKET_N) -> pd.DataFrame:
    """Predicted probability against realised win rate, bucketed.

    Thin buckets are REPORTED with their n but do not qualify for the slope
    test. Dropping them silently would hide where the model has no support.
    """
    cols = ["bucket_lo", "bucket_hi", "n", "wins", "mean_predicted",
            "realised", "ci_lo", "ci_hi", "qualifies"]
    if len(oos) == 0:
        return pd.DataFrame(columns=cols)

    p = pd.to_numeric(oos["predicted"], errors="coerce").astype(float)
    won = pd.to_numeric(oos["won"], errors="coerce").fillna(0).astype(int)
    lo = np.floor(np.clip(p, 0.0, 1.0 - 1e-12) / bucket_width) * bucket_width
    lo = np.round(lo, 10)

    rows: List[Dict[str, Any]] = []
    for edge in sorted(pd.unique(lo)):
        sel = lo == edge
        n = int(sel.sum())
        k = int(won[sel].sum())
        ci_lo, ci_hi = wilson(k, n)
        rows.append({
            "bucket_lo": float(edge),
            "bucket_hi": float(edge) + bucket_width,
            "n": n,
            "wins": k,
            "mean_predicted": float(p[sel].mean()),
            "realised": k / n if n else float("nan"),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "qualifies": n >= min_n,
        })
    return pd.DataFrame(rows, columns=cols)


def ship_check(rel: pd.DataFrame,
               min_buckets: int = MIN_QUALIFYING_BUCKETS) -> Tuple[bool, str]:
    """Does the reliability curve rise with its own prediction?

    Weighted least squares of realised on mean-predicted across qualifying
    buckets, weights = bucket n. Ships only if the slope's 95% interval lies
    ENTIRELY above zero.

    A slope of zero means every predicted level wins at the same rate — the
    number orders nothing, and belongs on no board.
    """
    if len(rel) == 0:
        return (False, "no out-of-sample predictions to check")

    q = rel[rel["qualifies"].astype(bool)]
    if len(q) < min_buckets:
        return (False, f"only {len(q)} qualifying bucket(s) of at least "
                       f"{MIN_BUCKET_N} predictions; {min_buckets} needed")

    return _slope_clears_zero(q["mean_predicted"].to_numpy(dtype=float),
                              q["realised"].to_numpy(dtype=float),
                              q["n"].to_numpy(dtype=float))


def return_reliability(oos: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Predicted expected return against realised mean return, by quantile.

    Quantile buckets rather than fixed edges: a predicted return has no
    natural scale the way a probability does, and fixed edges would put every
    prediction in one bin.

    The interval is a t-interval on the bucket MEAN. Option returns are
    heavy-tailed, so this understates the tails — it is a statement about
    where the average sits, not about what any single trade can do.
    """
    cols = ["bucket", "n", "mean_predicted", "mean_actual", "ci_lo", "ci_hi",
            "qualifies"]
    if len(oos) == 0:
        return pd.DataFrame(columns=cols)

    p = pd.to_numeric(oos["predicted"], errors="coerce").astype(float)
    a = pd.to_numeric(oos["actual"], errors="coerce").astype(float)
    keep = p.notna() & a.notna()
    p, a = p[keep], a[keep]
    if len(p) == 0:
        return pd.DataFrame(columns=cols)

    try:
        labels = pd.qcut(p.rank(method="first"), n_buckets, labels=False)
    except ValueError:
        labels = pd.Series(np.zeros(len(p), dtype=int), index=p.index)

    rows: List[Dict[str, Any]] = []
    for b in sorted(pd.unique(labels)):
        sel = labels == b
        n = int(sel.sum())
        vals = a[sel]
        mean = float(vals.mean())
        if n > 1:
            se = float(vals.std(ddof=1)) / math.sqrt(n)
            t = 1.96 if n - 1 >= 30 else _t95(n - 1)
            ci_lo, ci_hi = mean - t * se, mean + t * se
        else:
            ci_lo, ci_hi = float("-inf"), float("inf")
        rows.append({
            "bucket": int(b), "n": n,
            "mean_predicted": float(p[sel].mean()), "mean_actual": mean,
            "ci_lo": ci_lo, "ci_hi": ci_hi, "qualifies": n >= MIN_BUCKET_N,
        })
    return pd.DataFrame(rows, columns=cols)


def ship_check_return(rel: pd.DataFrame,
                      min_buckets: int = MIN_QUALIFYING_BUCKETS
                      ) -> Tuple[bool, str]:
    """Does realised return rise with predicted return?

    Same weighted-least-squares slope test as `ship_check`, on the return
    curve. A slope that does not clear zero means the expected-return number
    orders nothing, and a number that orders nothing must not be displayed as
    though it does.
    """
    if len(rel) == 0:
        return (False, "no out-of-sample predictions to check")
    q = rel[rel["qualifies"].astype(bool)]
    if len(q) < min_buckets:
        return (False, f"only {len(q)} qualifying bucket(s) of at least "
                       f"{MIN_BUCKET_N} predictions; {min_buckets} needed")
    return _slope_clears_zero(q["mean_predicted"].to_numpy(dtype=float),
                              q["mean_actual"].to_numpy(dtype=float),
                              q["n"].to_numpy(dtype=float))


def _slope_clears_zero(x: np.ndarray, y: np.ndarray,
                       w: np.ndarray) -> Tuple[bool, str]:
    """Weighted least squares; ships only if the slope's 95% CI is above 0."""
    sw = w.sum()
    xb = float((w * x).sum() / sw)
    yb = float((w * y).sum() / sw)
    sxx = float((w * (x - xb) ** 2).sum())
    if sxx <= 0:
        return (False, "slope undefined: every qualifying bucket carries the "
                       "same prediction")

    slope = float((w * (x - xb) * (y - yb)).sum() / sxx)
    intercept = yb - slope * xb
    dof = len(x) - 2
    if dof <= 0:
        return (False, f"slope {slope:.3f} but only {len(x)} buckets — too "
                       f"few to put an interval on it")

    resid = y - (intercept + slope * x)
    ss_res = float((w * resid ** 2).sum())
    se = math.sqrt((ss_res / dof) / sxx) if ss_res > 0 else 0.0
    t = 1.96 if dof >= 30 else _t95(dof)
    lo, hi = slope - t * se, slope + t * se

    if lo > 0:
        return (True, f"slope {slope:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] — "
                      f"clears zero on {len(x)} buckets, n={int(sw)}")
    return (False, f"slope {slope:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] — "
                   f"does not clear zero; the number orders nothing")


_T95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}


def _t95(dof: int) -> float:
    return _T95.get(dof, 1.96)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_training_set(db_path: str) -> pd.DataFrame:
    """Closed trades with a realised outcome. THE ONLY I/O IN THIS MODULE.

    `won` is `pnl_usd > 0` — profit at the exit that actually happened, not
    profit at expiry. The operator does not hold to expiry.
    """
    sql = ("SELECT date, expiration, strategy_name, pnl_usd, entry_delta, "
           "       entry_iv, iv_rank_score, net_credit, spread_width, "
           "       capital_at_risk "
           "FROM trades WHERE status = 'CLOSED' AND pnl_usd IS NOT NULL")
    conn: Optional[sqlite3.Connection] = None
    try:
        # `with sqlite3.connect(...)` commits but does NOT close.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql(sql, conn)
    except Exception as exc:
        log.warning("training set unreadable at %s: %s", db_path, exc)
        return pd.DataFrame(columns=list(TRAINING_COLUMNS))
    finally:
        if conn is not None:
            conn.close()

    if len(df) == 0:
        return pd.DataFrame(columns=list(TRAINING_COLUMNS))

    entry = pd.to_datetime(df["date"], errors="coerce")
    expiry = pd.to_datetime(df["expiration"], errors="coerce")
    credit = pd.to_numeric(df["net_credit"], errors="coerce")
    width = pd.to_numeric(df["spread_width"], errors="coerce").replace(0, np.nan)
    pnl = pd.to_numeric(df["pnl_usd"], errors="coerce")
    # NULL means NOT RECORDED, never zero. A missing denominator has to stay
    # missing: filling it with 0 would invent a 0% return and pull every
    # average it touches toward the middle.
    risk = pd.to_numeric(df["capital_at_risk"], errors="coerce").replace(0, np.nan)

    out = pd.DataFrame({
        "abs_delta": pd.to_numeric(df["entry_delta"], errors="coerce").abs(),
        "dte": (expiry - entry).dt.days.astype(float),
        "entry_iv": pd.to_numeric(df["entry_iv"], errors="coerce"),
        "iv_rank_score": pd.to_numeric(df["iv_rank_score"], errors="coerce"),
        # Single-leg rows have no credit and no width. Zero means "this term
        # does not apply", which is why it is not dropped and not imputed.
        "credit_to_width": (credit / width).fillna(0.0),
        "strategy": df["strategy_name"].astype(str),
        "entry_date": entry.dt.strftime("%Y-%m-%d"),
        "won": (pnl > 0).astype(int),
        # Return on CAPITAL AT RISK. Premium, credit and capital at risk
        # coincide only for long premium; on the wrong denominator the same
        # trades give the opposite sign.
        "ret_on_risk": pnl / risk,
        # Which side of the premium the account is on. |delta|, DTE and IV all
        # carry the OPPOSITE sign for a seller, so the fit needs to know.
        "is_short": df["strategy_name"].astype(str).isin(SHORT_STRATEGIES).astype(int),
    })

    for col in ("abs_delta", "dte", "entry_iv", "iv_rank_score"):
        out[col] = out[col].fillna(out[col].median())
    out = out.dropna(subset=["entry_date"])
    return out.sort_values("entry_date", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Artifact
# --------------------------------------------------------------------------
def save_model(model: Model, path: str, *, shipped: bool, reason: str,
               reliability_table: Optional[pd.DataFrame] = None) -> None:
    """Persist coefficients plus the verdict that authorises their display."""
    payload: Dict[str, Any] = {
        "shipped": bool(shipped),
        "reason": reason,
        "features": model.features,
        "strategies": model.strategies,
        "beta": [float(b) for b in model.beta],
        "mean": [float(m) for m in model.mean],
        "std": [float(s) for s in model.std],
        "n_train": model.n_train,
        "interacted": model.interacted,
        "trained_through": model.trained_through,
        "base_rate": model.base_rate,
        "meta": model.meta,
    }
    if reliability_table is not None:
        payload["reliability"] = reliability_table.to_dict(orient="records")
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_model(path: str = DEFAULT_MODEL_PATH) -> Optional[Model]:
    """The fitted model, or None if it is absent or was refused.

    None for an unshipped artifact is deliberate. The guard has to hold at the
    READ side as well as the write side, or a model that failed its own
    reliability check still reaches a board. There is no fallback to
    `pop_score`: a silent substitution is exactly how `quality_score` ended up
    ranking this system's boards unnoticed.
    """
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception:
        return None
    if not payload.get("shipped"):
        return None
    try:
        return Model(
            features=list(payload["features"]),
            strategies=list(payload["strategies"]),
            beta=np.asarray(payload["beta"], dtype=float),
            mean=np.asarray(payload["mean"], dtype=float),
            std=np.asarray(payload["std"], dtype=float),
            n_train=int(payload.get("n_train", 0)),
            interacted=bool(payload.get("interacted", False)),
            trained_through=str(payload.get("trained_through", "")),
            base_rate=float(payload.get("base_rate", 0.5)),
            meta=dict(payload.get("meta", {})),
        )
    except Exception:
        log.warning("model artifact at %s is unreadable", path, exc_info=True)
        return None


#: Scan rows do not carry ledger column names. Each feature lists the aliases
#: it will answer to, most specific first. A mapping that quietly missed would
#: read every feature as zero and render a confident constant — which is the
#: defect this whole module exists to correct.
_ALIASES: Dict[str, Tuple[str, ...]] = {
    "abs_delta": ("abs_delta", "entry_delta", "delta", "net_delta"),
    "dte": ("dte", "days_to_expiry", "days_to_expiration", "dte_at_entry"),
    "entry_iv": ("entry_iv", "impliedVolatility", "implied_volatility", "iv"),
    "iv_rank_score": ("iv_rank_score", "iv_rank", "iv_percentile_30",
                      "iv_percentile"),
}


def row_features(row: Dict[str, Any]) -> Dict[str, float]:
    """Feature values for one scan row, whatever the board calls its columns."""
    def pick(names: Sequence[str]) -> float:
        for n in names:
            v = row.get(n)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                return f
        return 0.0

    out = {k: pick(v) for k, v in _ALIASES.items()}
    out["abs_delta"] = abs(out["abs_delta"])

    credit = pick(("net_credit", "credit", "premium_received"))
    width = pick(("spread_width", "width"))
    out["credit_to_width"] = credit / width if width else 0.0
    return out


def strategy_reliability(oos: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Per-strategy median split of the out-of-sample predictions.

    The aggregate reliability curve is what `ship_check` tests, and it can be
    clean while an individual cell runs backwards. This is the diagnostic that
    shows it. `gap` is the high half's win rate minus the low half's: positive
    means the model orders that structure, negative means it inverts it.

    Reported, never gated on. Picking which strategies to display using the
    same data that measured them would fit the display to noise — at ~50 rows
    a side these cells are worth about 1.5 standard errors.
    """
    cols = ["strategy", "n", "low_win", "high_win", "gap", "sufficient"]
    if len(oos) == 0 or "strategy" not in oos:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for name, g in oos.groupby("strategy"):
        n = int(len(g))
        entry: Dict[str, Any] = {"strategy": str(name), "n": n,
                                 "low_win": float("nan"),
                                 "high_win": float("nan"),
                                 "gap": float("nan"),
                                 "sufficient": n >= min_n}
        if n >= min_n:
            med = g["predicted"].median()
            lo, hi = g[g["predicted"] <= med], g[g["predicted"] > med]
            if len(lo) and len(hi):
                entry["low_win"] = float(lo["won"].mean())
                entry["high_win"] = float(hi["won"].mean())
                entry["gap"] = entry["high_win"] - entry["low_win"]
        rows.append(entry)
    return pd.DataFrame(rows, columns=cols).sort_values("n", ascending=False)


def probability_for(row: Dict[str, Any],
                    model: Optional[Model]) -> Optional[float]:
    """The calibrated probability for one row, or None without a model."""
    if model is None:
        return None
    frame = pd.DataFrame([{**row_features(row),
                           "strategy": str(row.get("strategy")
                                           or row.get("strategy_name") or "")}])
    return float(predict(model, frame)[0])


def provenance(path: str = DEFAULT_MODEL_PATH) -> Optional[str]:
    """One line naming what licences the number, or None if nothing does.

    The closing clause is not boilerplate. Measured out-of-sample on this
    book, win rate rises monotonically with the calibrated probability while
    money does not follow it — the 0.4-0.5 bucket wins 44.3% at PF 0.66 and
    the 0.3-0.4 bucket wins 36.5% at PF 1.13. A reader who takes the
    probability as a profitability claim is reading it wrong, and the board
    has to say so where the number is.
    """
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception:
        return None
    if not payload.get("shipped"):
        return None
    reason = str(payload.get("reason", ""))
    slope = reason.split("—")[0].strip() if "—" in reason else reason.strip()
    return (f"Calibrated on {payload.get('n_train', 0)} closed trades through "
            f"{payload.get('trained_through', '?')} · walk-forward {slope} · "
            f"probability of closing green, not expected profit")


def describe_row(row: Dict[str, Any], model: Optional[Model],
                 rel: pd.DataFrame) -> Optional[str]:
    """One line: the prediction, and the evidence that makes it readable.

    Returns None when there is no shipped model, or when the prediction lands
    in a bucket the model was never checked in. A bare probability with no
    support behind it is precisely the number this module replaces, so it is
    better to draw nothing.
    """
    if model is None or rel is None or len(rel) == 0:
        return None

    frame = pd.DataFrame([{**row_features(row),
                           "strategy": str(row.get("strategy")
                                           or row.get("strategy_name") or "")}])
    p = float(predict(model, frame)[0])

    bucket = bucket_for(rel, p)
    if bucket is None or not bool(bucket.get("qualifies")):
        return None

    n = int(bucket["n"])
    return (f"CalPoP {p:.0%} — at this level {float(bucket['realised']):.0%} "
            f"of {n} out-of-sample analogues closed green "
            f"[{float(bucket['ci_lo']):.0%}, {float(bucket['ci_hi']):.0%}]")


def bucket_for(rel: pd.DataFrame, p: float,
               bucket_width: float = BUCKET_WIDTH) -> Optional[Dict[str, Any]]:
    """The reliability row a prediction falls in, for display beside it.

    A predicted probability with no evidence attached is the number this
    module was written to replace.
    """
    if len(rel) == 0:
        return None
    edge = round(math.floor(min(max(p, 0.0), 1.0 - 1e-12) / bucket_width)
                 * bucket_width, 10)
    hit = rel[np.isclose(rel["bucket_lo"].astype(float), edge)]
    if hit.empty:
        return None
    row: Dict[str, Any] = hit.iloc[0].to_dict()
    return row
