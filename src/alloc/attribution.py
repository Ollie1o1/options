"""Which entry-time features separated the winners from the losers.

This module answers "what made the profitable trades profitable" — but it is
also the most dangerous thing in the backtest, because ranking k features by
correlation and reporting the best one IS a k-way search, and a search finds
something in noise every single time. `docs/ALLOCATION_BACKTEST_FINDINGS.md`
records that mistake twice: a configuration that read DSR 0.921 alone and 0.432
once deflated by the 18 configurations actually tried.

So everything here is built to make the honest read the easy one:

  * `n_trials` is carried on every row. Ranking 8 features is an 8-way search
    and the caller must be able to deflate by it.
  * t-statistics are reported BOTH naive and clustered by entry day. Positions
    opened on the same day share that day's move; clustering roughly halves t
    on this data.
  * A feature that cannot be computed is reported as `ic=None, n=0` —
    "could not measure" and "measured, no effect" are different claims.
  * `split_by_time` is chronological and never shuffles. Shuffling would leak
    the future into the training half through overlapping option positions.

Nothing here decides anything. It measures, and a measurement that fails
deflation is a reason NOT to act.
"""
from __future__ import annotations

import collections
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # scipy is present in this venv
    from scipy import stats as _sps
except ImportError:                     # pragma: no cover
    _sps = None                         # type: ignore[assignment]

MIN_TRADES = 8


def _closed(trades: Sequence[Any]) -> List[Any]:
    return [t for t in trades
            if getattr(t, "exit_date", None) and getattr(t, "capital_at_risk", 0)]


def _roc(t: Any) -> float:
    return float(t.pnl or 0.0) / float(t.capital_at_risk)


def paired(trades: Sequence[Any], feature: str
           ) -> Tuple[List[Any], List[float], List[float]]:
    """(trades, feature values, RoCs) for closed trades that HAVE the feature.

    Trades missing the feature are dropped, never zero-filled: a zero is a
    value, and inventing one would move the correlation toward whatever zero
    happens to mean for that feature.
    """
    keep, xs, ys = [], [], []
    for t in _closed(trades):
        v = (getattr(t, "features", None) or {}).get(feature)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv:                    # NaN
            continue
        keep.append(t)
        xs.append(fv)
        ys.append(_roc(t))
    return keep, xs, ys


def _clustered_t(trades: Sequence[Any], xs: Sequence[float],
                 ys: Sequence[float]) -> float:
    """t on the per-entry-day mean of the rank-demeaned product.

    The naive t treats every trade as an independent draw. Trades sharing an
    entry day share that day's market move, so the naive figure overstates
    significance — by roughly a factor of two on this data.
    """
    if _sps is None or len(xs) < 3:
        return 0.0
    rx = _sps.rankdata(xs)
    ry = _sps.rankdata(ys)
    rx = (rx - rx.mean()) / (rx.std() or 1.0)
    ry = (ry - ry.mean()) / (ry.std() or 1.0)
    prod = rx * ry
    by_day: Dict[str, List[float]] = collections.defaultdict(list)
    for t, p in zip(trades, prod):
        by_day[str(t.entry_date)].append(float(p))
    days = np.array([np.mean(v) for v in by_day.values()], dtype=float)
    if days.size < 3 or days.std(ddof=1) == 0:
        return 0.0
    return float(days.mean() / (days.std(ddof=1) / np.sqrt(days.size)))


def feature_ic(trades: Sequence[Any], feature: str,
               n_trials: int = 1) -> Dict[str, Any]:
    """Spearman IC of one entry feature against realized return on capital.

    Rank correlation rather than Pearson, matching the rest of this repo and
    because option returns are violently non-normal — one max-loss tail would
    otherwise dominate the statistic.
    """
    keep, xs, ys = paired(trades, feature)
    out: Dict[str, Any] = {"feature": feature, "n": len(xs), "ic": None,
                           "p": None, "t": 0.0, "t_clustered": 0.0,
                           "q_top": None, "q_bot": None, "q_spread": None,
                           "mono": None, "tail_auc": None, "tail_p": None,
                           "n_disasters": 0, "n_trials": n_trials,
                           "ic_resid": None, "p_resid": None}
    if len(xs) < MIN_TRADES or _sps is None:
        return out
    # The fifth screen, and on this book the decisive one: an IC that does not
    # survive removing credit richness is arithmetic. See `residual_ic`.
    _res = residual_ic(trades, feature)
    out["ic_resid"], out["p_resid"] = _res["ic"], _res["p"]
    # The other two of the four screens. Neither is derivable from the IC:
    # `mono` asks whether the effect is ORDERED across the parameter, and
    # `tail_auc` asks whether it flags the rare total losses that are this
    # book's entire P&L. See docs/LEADING_INDICATORS_20260809.md §2.
    out["mono"] = monotonicity(trades, feature)
    tail = disaster_auc(trades, feature)
    out["tail_auc"] = tail["auc"]
    out["tail_p"] = tail["p"]
    out["n_disasters"] = tail["n_disasters"]
    # Mean outcome in the top vs bottom quintile of the feature.
    #
    # This exists because the IC alone is BLIND to the effect this book cares
    # about most. On 2026-08-08 `iv_rank` scored a Spearman IC of -0.029 and
    # was written up as flat, while the same data measured as a conditional
    # mean was monotone and strong (DSR 0.797). Both were right: a rank
    # correlation asks whether a feature orders EVERY trade, and short-premium
    # returns have skew -1.7 to -2.0, so a feature whose value is avoiding the
    # rare total loss barely moves the median ordering while moving the mean a
    # long way. Any tail-avoidance signal looks flat to an IC. Read them
    # together, never the IC alone.
    if len(xs) >= 5 * MIN_TRADES // 2:
        order = np.argsort(np.asarray(xs, dtype=float), kind="stable")
        k = max(1, len(order) // 5)
        bot = np.asarray([ys[i] for i in order[:k]], dtype=float)
        top = np.asarray([ys[i] for i in order[-k:]], dtype=float)
        out["q_bot"] = round(float(bot.mean()), 4)
        out["q_top"] = round(float(top.mean()), 4)
        out["q_spread"] = round(float(top.mean() - bot.mean()), 4)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return out                      # constant: no ranking to correlate
    ic, p = _sps.spearmanr(xs, ys)
    if ic != ic:                        # NaN
        return out
    n = len(xs)
    denom = max(1.0 - ic * ic, 1e-12)
    out.update({
        "ic": round(float(ic), 4),
        "p": round(float(p), 4),
        "t": round(float(ic * np.sqrt((n - 2) / denom)), 3),
        "t_clustered": round(_clustered_t(keep, xs, ys), 3),
    })
    return out


# The standing controls. Both are close to mechanical drivers of return on a
# held-to-expiry credit spread, so a feature correlated with either inherits
# their IC without carrying any information of its own.
RESIDUAL_CONTROLS = ("credit_pct_width", "atm_iv")
# A control has to be known for at least this share of the sample to be worth
# honouring. Below it, the control is dropped rather than most of the evidence.
MIN_CONTROL_COVERAGE = 0.80


def residual_ic(trades: Sequence[Any], feature: str,
                controls: Sequence[str] = RESIDUAL_CONTROLS) -> Dict[str, Any]:
    """Spearman IC of the feature against RoC, with the controls regressed out.

    THE screen this book was missing. For a held-to-expiry credit spread the
    return on capital is close to a function of the credit received, and the
    credit is close to a function of implied vol — so any entry feature
    correlated with implied vol posts a large IC that is arithmetic rather than
    a discovery. On optionsDX SPY 2010-2023 (n=1,598) `credit_pct_width` scores
    +0.6991 and `atm_iv` +0.5455, and both hypotheses tested on 2026-08-11
    collapsed once controlled:

        term_slope_1m3m   raw +0.3654  ->  +0.0422  p=0.09
        entry_depth       raw -0.1907  ->  -0.0099  p=0.69

    Both had already passed sign-holds-out-of-sample AND
    clustered-t-significant-in-both-windows. Neither survived this.

    Done on RANKS, matching the rest of the module: the relationships here are
    monotone but not linear, and a least-squares fit on raw values would leave
    most of the control's influence in the residual.

    A control is never used against itself — the residual would be identically
    zero, which reads as "disproven" when it means "not asked". A feature whose
    only control is itself reports None for the same reason.
    """
    out: Dict[str, Any] = {"feature": feature, "n": 0, "ic": None, "p": None,
                           "controls": []}
    keep, xs, ys = paired(trades, feature)
    if len(xs) < MIN_TRADES or _sps is None:
        return out

    def _value(t: Any, name: str) -> Optional[float]:
        v = (getattr(t, "features", None) or {}).get(name)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return None if fv != fv else fv     # NaN is not a value

    # A control is kept if it is known for MOST of the sample, and the trades
    # where it is unknown are then dropped from the residual.
    #
    # The first version dropped the CONTROL whenever any single trade lacked it,
    # which on real output meant one absent value silently disabled the control
    # across a 3,000-trade table while the column went on printing a number.
    # Dropping the affected trades instead keeps the control, and keeps the
    # residualised and non-residualised trades from being mixed into one figure
    # — which was the original worry and is still the right one.
    usable: List[str] = []
    for name in controls:
        if name == feature:
            continue
        have = [t for t in keep if _value(t, name) is not None]
        # If honouring the control would cost most of the evidence, the control
        # goes rather than the sample.
        if len(have) < max(MIN_TRADES, MIN_CONTROL_COVERAGE * len(keep)):
            continue
        if len({_value(t, name) for t in have}) < 2:
            continue                    # constant, so a no-op
        usable.append(name)

    if usable:
        rows_ok = [i for i, t in enumerate(keep)
                   if all(_value(t, n) is not None for n in usable)]
        keep = [keep[i] for i in rows_ok]
        xs = [xs[i] for i in rows_ok]
        ys = [ys[i] for i in rows_ok]
        cols = [[float(_value(t, n)) for t in keep] for n in usable]  # type: ignore[arg-type]
    else:
        cols = []
    out["controls"] = usable
    out["n"] = len(xs)

    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return out                      # constant: no ranking to correlate
    rx = _sps.rankdata(xs)
    if not usable:
        # No control could be applied, so there is no controlled IC. The first
        # version returned the RAW one here, which puts an uncontrolled number
        # under a heading that says otherwise — and it reads as "the control
        # made no difference" when it means "no control ran". `n` stays at the
        # trades available and `ic` stays None, matching `feature_ic`: "could
        # not measure" is a different claim from "measured, no effect".
        return out

    design = np.column_stack([_sps.rankdata(c) for c in cols]
                             + [np.ones(len(rx))])
    beta, *_ = np.linalg.lstsq(design, rx, rcond=None)
    resid = rx - design @ beta
    # A feature that is a RENAMED control leaves a residual of pure
    # floating-point noise, and ranking that noise yields a large spurious
    # correlation — 0.97 in the test that caught this — because the arithmetic
    # error pattern still follows the data. Anything this far below the
    # feature's own scale is collinear, and the honest report is "not
    # measurable", not a number.
    if float(np.std(resid)) <= 1e-8 * (float(np.std(rx)) or 1.0):
        return out                      # the feature IS the controls
    ic, p = _sps.spearmanr(resid, ys)
    if ic != ic:
        return out
    out["ic"], out["p"] = round(float(ic), 4), round(float(p), 4)
    return out


MONO_BUCKETS = 5        # quintiles, so the shape and `q_spread` describe one split
DISASTER_ROC = -0.5     # "more than half the capital", ATTRIBUTION_20260808 §5


def monotonicity(trades: Sequence[Any], feature: str,
                 n_buckets: int = MONO_BUCKETS) -> Optional[float]:
    """Rank correlation between bucket index and mean outcome: the SHAPE, as one number.

    +1 is a cleanly graded effect, -1 the same inverted, and anything near 0 is
    a feature whose good cell is somewhere in the middle — which is what an
    artifact looks like.

    `bucket_table` has produced this shape all along and nothing summarised it,
    so judging it stayed a manual read. It is not a decoration on the IC: §3 of
    `docs/ATTRIBUTION_20260808.md` discarded `credit_pct_width` on exactly this
    basis (Q1 -0.011, Q2 +0.019, Q3 +0.090, Q4 -0.068 — up then down) in spite
    of an IC of 0.56, and §4f kept `width` partly because its shape survived
    the holdout when its IC never reached significance in-sample.

    Coarse by construction: five buckets means this can only take a handful of
    values. That is the point — it asks whether the effect is ordered, not how
    large it is.
    """
    rows = bucket_table(trades, feature, n_buckets=n_buckets)
    if len(rows) < 3 or _sps is None:
        return None
    means = [r["mean_roc"] for r in rows]
    if len(set(means)) < 2:
        return None
    rho, _p = _sps.spearmanr(list(range(len(means))), means)
    return None if rho != rho else round(float(rho), 3)


def disaster_auc(trades: Sequence[Any], feature: str,
                 threshold: float = DISASTER_ROC) -> Dict[str, Any]:
    """Mann-Whitney AUC of the feature, trades that lost most of the capital vs the rest.

    For a short-premium book this is THE question, and it is not the question
    the IC answers. §5 of `docs/ATTRIBUTION_20260808.md`: bull_put's 25
    disasters were 13.4% of trades and **1,577% of total absolute RoC**. The
    entire P&L is the tail, so a feature that shifts the middle and misses the
    tail is worth very little, and one that flags the tail is worth a lot even
    with an IC of zero.

    AUC is P(a disaster scored higher than a survivor). 0.5 is a coin flip;
    far from 0.5 in EITHER direction is informative, since a feature that
    reliably marks the safe trades is as useful as one that marks the doomed.

    §5 ran this by hand once. Running it every time is the difference between
    a screen and a one-off.
    """
    _keep, xs, ys = paired(trades, feature)
    out: Dict[str, Any] = {"feature": feature, "n": len(xs),
                           "n_disasters": 0, "auc": None, "p": None}
    if not xs:
        return out
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    hit = arr_y < threshold
    out["n_disasters"] = int(hit.sum())
    # Both groups must be non-empty: with no disasters there is nothing to
    # separate, and with nothing BUT disasters the same is true. Reporting 0.5
    # in either case would read as "measured, no warning" when the truth is
    # "this window contained no tail to warn about".
    if out["n_disasters"] == 0 or out["n_disasters"] == len(xs):
        return out
    if _sps is None or len(set(xs)) < 2:
        return out
    u, p = _sps.mannwhitneyu(arr_x[hit], arr_x[~hit], alternative="two-sided")
    out["auc"] = round(float(u) / float(hit.sum() * (~hit).sum()), 4)
    out["p"] = round(float(p), 4)
    return out


def bucket_table(trades: Sequence[Any], feature: str,
                 n_buckets: int = 4) -> List[Dict[str, Any]]:
    """Outcome by quantile of the feature.

    The SHAPE across buckets is the evidence, not any single cell. A monotone
    progression across a graded parameter is much harder to produce by chance
    than one good bucket, and a non-monotone series is the signature of an
    artifact — the width study in ALLOCATION_BACKTEST_FINDINGS 4b went
    $5 -> $10 worse -> $20 worse -> $25 positive, which is noise wearing a
    result's clothes.
    """
    _keep, xs, ys = paired(trades, feature)
    if len(xs) < max(MIN_TRADES, n_buckets * 2):
        return []
    order = np.argsort(np.asarray(xs, dtype=float), kind="stable")
    chunks = np.array_split(order, n_buckets)
    rows: List[Dict[str, Any]] = []
    for i, idx in enumerate(chunks):
        if idx.size == 0:
            continue
        vals = np.asarray([xs[j] for j in idx], dtype=float)
        rocs = np.asarray([ys[j] for j in idx], dtype=float)
        rows.append({
            "bucket": i + 1,
            "n": int(idx.size),
            "lo": round(float(vals.min()), 4),
            "hi": round(float(vals.max()), 4),
            "mean_roc": round(float(rocs.mean()), 4),
            "median_roc": round(float(np.median(rocs)), 4),
            "win_rate": round(float((rocs > 0).mean()), 4),
        })
    return rows


def split_by_time(trades: Sequence[Any], frac: float = 0.7
                  ) -> Tuple[List[Any], List[Any]]:
    """Chronological train/test split. Never shuffles.

    A random split would put trades from the same week on both sides. Option
    positions overlap in time, so that leaks the outcome of the test half into
    the training half and every feature looks predictive.
    """
    ordered = sorted(trades, key=lambda t: str(t.entry_date))
    cut = int(round(len(ordered) * frac))
    return ordered[:cut], ordered[cut:]


def benjamini_hochberg(pvalues: Sequence[Optional[float]]
                       ) -> List[Optional[float]]:
    """False-discovery-rate q-values for a family of tests, in input order.

    `n_trials` has been carried on every ranking row since the module was
    written and nothing ever divided by it, so every multi-feature sweep in
    this repo has been read at raw p. That is the gap between "this feature is
    significant" and "the best of eighteen features looks significant", and it
    is the most likely remaining manufacturer of findings that then fail their
    holdout.

    Benjamini-Hochberg rather than Bonferroni deliberately. Bonferroni controls
    the chance of ANY false positive and at k=18 is punishing enough that a real
    but modest effect cannot clear it; BH controls the expected FRACTION of
    claims that are false, which is the right question for a screen whose
    output is "which of these deserve a holdout test".

    Read a q-value as: if you accept everything down to this row, that is the
    share of accepted rows you should expect to be noise.

    None passes through as None and is excluded from the family size — a
    feature that could not be measured was never a trial, and counting it would
    penalise the survivors for a test that did not run.
    """
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    n = len(idx)
    out: List[Optional[float]] = [None] * len(pvalues)
    if n == 0:
        return out
    order = sorted(idx, key=lambda i: float(pvalues[i]))  # type: ignore[arg-type]
    # Step UP: walk from the largest p down, carrying the running minimum. That
    # enforced monotonicity is the whole procedure — without it a middling
    # p-value can score a smaller q than one below it.
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, float(pvalues[i]) * n / rank)  # type: ignore[arg-type]
        out[i] = round(min(running, 1.0), 6)
    return out


def split_at_date(trades: Sequence[Any], boundary: str
                  ) -> Tuple[List[Any], List[Any]]:
    """Chronological split at a fixed DATE. The boundary day is holdout.

    `split_by_time` cuts at a fraction of the trade COUNT, so its boundary
    moves whenever the number of trades moves — widen the universe or loosen a
    filter and the "same" 70/30 split lands in a different year. That is fine
    for an exploratory read and useless for a pre-registered one.

    `docs/PREREG_OPTIONSDX_20260811.md` fixes the optionsDX boundary at
    2017-01-01 for a specific reason: it puts Volmageddon, Q4 2018, COVID and
    the 2022 bear in the holdout, where the stress belongs. Only a date can
    express that, and a boundary that drifts with the data was never
    pre-registered at all.
    """
    closed = sorted(_closed(trades), key=lambda t: str(t.entry_date))
    train = [t for t in closed if str(t.entry_date) < boundary]
    test = [t for t in closed if str(t.entry_date) >= boundary]
    return train, test


def rank_features(trades: Sequence[Any],
                  features: Optional[Sequence[str]] = None,
                  by: str = "ic",
                  ) -> List[Dict[str, Any]]:
    """Every feature's IC and quantile spread, each carrying the trial count.

    `by` selects the sort, and the four answer different questions:

      "ic"        features that rank-order EVERY trade
      "q_spread"  threshold effects an IC cannot see — this is what caught the
                  IV-rank effect after the IC missed it, and what found a
                  20.7-point `width` effect sitting at p=0.58
      "mono"      effects that are ORDERED across the parameter, which is the
                  shape that is hard to produce by chance
      "tail_auc"  features that flag the rare total losses, ranked by distance
                  from a coin flip in either direction

    Run more than one. A feature that leads on all four is a different animal
    from one that leads on a single screen, and this repo has been burned
    exactly once per screen it was missing.

    `n_trials` is the number of features examined, because that is exactly the
    size of the search that produced the winner.
    """
    if features is None:
        seen: List[str] = []
        for t in _closed(trades):
            for k in (getattr(t, "features", None) or {}):
                if k not in seen:
                    seen.append(k)
        features = seen
    k = len(features)
    rows = [feature_ic(trades, f, n_trials=k) for f in features]
    # Deflate here rather than leaving it to the caller. `n_trials` sat on
    # every row unused for exactly as long as this was somebody else's job.
    for row, q in zip(rows, benjamini_hochberg([r["p"] for r in rows])):
        row["q"] = q

    def _strength(r: Dict[str, Any]) -> float:
        v = r.get(by)
        if v is None:
            return -1.0
        # An AUC is strong when it is FAR FROM 0.5 in either direction, not
        # when it is large: 0.20 and 0.80 separate the tail equally well, and
        # ranking them by magnitude would bury half the warnings.
        return abs(float(v) - 0.5) if by == "tail_auc" else abs(float(v))

    rows.sort(key=_strength, reverse=True)
    return rows


def format_ranking(rows: Sequence[Dict[str, Any]], title: str = "") -> str:
    """One line per feature. Unmeasurable rows say so rather than showing 0."""
    out = []
    if title:
        out.append(title)
    out.append(f"  {'feature':<22} {'n':>5} {'IC':>8} {'IC|ctl':>8} {'t':>7} "
               f"{'t_clust':>8} {'p':>8} {'q(BH)':>8} {'q_bot':>8} "
               f"{'q_top':>8} {'q_spread':>9} {'mono':>6} {'tailAUC':>8}")
    for r in rows:
        if r["ic"] is None:
            out.append(f"  {r['feature']:<22} {r['n']:>5}   "
                       f"(not measurable — no variation or too few trades)")
            continue
        quint = ("" if r.get("q_spread") is None else
                 f" {r['q_bot']:>8.4f} {r['q_top']:>8.4f} {r['q_spread']:>9.4f}")
        # A dash is "could not be measured here", which is a different claim
        # from a number — the same distinction the IC column already makes.
        mono = "     -" if r.get("mono") is None else f"{r['mono']:>6.2f}"
        auc = "       -" if r.get("tail_auc") is None else f"{r['tail_auc']:>8.3f}"
        # Sits immediately after p because that is the comparison to make: a
        # raw p that survives and a q that does not is a search artifact.
        bh = "       -" if r.get("q") is None else f"{r['q']:>8.4f}"
        # Sits immediately beside the raw IC because that is the comparison to
        # make: a large IC next to a collapsed IC|ctl is credit richness under
        # another name, which is how BOTH optionsDX hypotheses failed.
        ctl = ("       -" if r.get("ic_resid") is None
               else f"{r['ic_resid']:>8.4f}")
        out.append(f"  {r['feature']:<22} {r['n']:>5} {r['ic']:>8.4f} {ctl} "
                   f"{r['t']:>7.2f} {r['t_clustered']:>8.2f} "
                   f"{r['p']:>8.4f} {bh}{quint} {mono} {auc}")
    return "\n".join(out)
