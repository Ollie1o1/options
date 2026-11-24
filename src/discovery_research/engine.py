"""Chronological incremental prediction tests, isolated from the trading path."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def predict_ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                  alpha: float = 1.0) -> np.ndarray:
    """Fit/standardize only on the training slice; intercept is not penalized."""
    mean, sd = train_x.mean(axis=0), train_x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1)
    x = np.column_stack([np.ones(len(train_x)), (train_x - mean) / sd])
    test = np.column_stack([np.ones(len(test_x)), (test_x - mean) / sd])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ train_y)
    return test @ coef


def block_mean_interval(values: np.ndarray, repetitions: int = 2000, seed: int = 20261124,
                        block: int = 4) -> dict:
    """Circular moving-block bootstrap of held-out DATE-level paired losses.

    Each date has already averaged all its names, and holding windows do not
    overlap. Blocks retain additional short-range dependence. The p-value uses
    a centered-null bootstrap and the add-one finite-simulation correction.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return {"mean": None, "ci": [None, None], "p_value": 1.0}
    rng = np.random.default_rng(seed)
    n = len(values)
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n, size=(repetitions, math.ceil(n / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(repetitions, -1)[:, :n]
    means = values[indices].mean(axis=1)
    observed = float(values.mean())
    centered = means - observed
    p = float((np.count_nonzero(centered >= observed) + 1) / (repetitions + 1))
    lo, hi = np.quantile(means, [.025, .975])
    return {"mean": observed, "ci": [float(lo), float(hi)], "p_value": p}


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    out = [1.0] * len(p_values)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[i]))
        out[i] = previous
    return out


def evaluate(dataset: pd.DataFrame, calendar: list[str], trial: dict, settings: dict) -> dict:
    result: dict[str, Any] = {"id": trial["id"], "question": trial["question"],
                              "target": trial["target"], "horizon": trial["horizon"],
                              "status": "INSUFFICIENT", "p_value": 1.0, "adjusted_p": 1.0,
                              "test_dates": 0, "test_rows": 0, "test_years": 0,
                              "folds": [], "reason": "No matched, point-in-time observations.",
                              "candidate_features": trial["candidate"], "baseline_features": trial["baseline"]}
    if dataset.empty:
        return result
    result["available_dates"] = int(dataset["date"].nunique())
    result["available_rows"] = len(dataset)
    result["available_first"] = str(dataset["date"].min())
    result["available_last"] = str(dataset["date"].max())
    import bisect
    first = bisect.bisect_left(calendar, settings["first_test"])
    step = int(trial["horizon"]) + 2
    selected_dates = set(calendar[first::step])
    columns = trial["baseline"] + trial["candidate"]
    records: list[dict[str, Any]] = []
    block = int(settings["test_block_sessions"])
    for start_i in range(first, len(calendar), block):
        stop_i = min(start_i + block, len(calendar))
        start, stop = calendar[start_i], calendar[stop_i - 1]
        train_start = calendar[max(0, start_i - int(settings["train_window_sessions"]))]
        train = dataset.loc[(dataset["date"] >= train_start) & (dataset["date"] < start)
                            & (dataset["end_date"] < start)]
        test = dataset.loc[(dataset["date"] >= start) & (dataset["date"] <= stop)
                           & dataset["date"].isin(selected_dates)]
        fold: dict[str, Any] = {"test_start": start, "test_end": stop,
                                "train_dates": int(train["date"].nunique()),
                                "test_dates": int(test["date"].nunique()), "test_rows": len(test)}
        if fold["train_dates"] < settings["min_train_dates"] or test.empty:
            fold["status"] = "insufficient training dates or no held-out samples"
            result["folds"].append(fold)
            continue
        fold["train_feature_end"] = str(train["date"].max())
        fold["train_label_end"] = str(train["end_date"].max())
        # An assertion here protects against a later refactor silently removing
        # the LABEL-end purge while leaving a convincing test report.
        if fold["train_label_end"] >= start:
            raise ValueError("Training labels overlap the held-out feature window")
        x, y = train[columns].to_numpy(dtype=float), train["target"].to_numpy(dtype=float)
        tx = test[columns].to_numpy(dtype=float)
        k = len(trial["baseline"])
        base = predict_ridge(x[:, :k], y, tx[:, :k], settings["ridge_alpha"])
        candidate = predict_ridge(x, y, tx, settings["ridge_alpha"])
        fold["status"] = "evaluated"
        result["folds"].append(fold)
        for rec, b, c in zip(test.to_dict("records"), base, candidate):
            truth = float(rec["target"])
            records.append({"date": rec["date"], "symbol": rec["symbol"],
                            "entry_date": rec["entry_date"], "end_date": rec["end_date"],
                            "truth": truth, "baseline": float(b), "candidate": float(c),
                            "base_loss": (truth - float(b)) ** 2,
                            "candidate_loss": (truth - float(c)) ** 2})
    if not records:
        result["reason"] = "No fold had enough independent training dates and matched held-out observations."
        return result
    forecasts = pd.DataFrame(records)
    by_date = forecasts.groupby("date")[["base_loss", "candidate_loss"]].mean().sort_index()
    improvement = (by_date["base_loss"] - by_date["candidate_loss"]).to_numpy(dtype=float)
    stats = block_mean_interval(improvement, settings["bootstrap_repetitions"],
                                settings["bootstrap_seed"], settings.get("bootstrap_block_dates", 4))
    base_mse, candidate_mse = float(by_date["base_loss"].mean()), float(by_date["candidate_loss"].mean())
    by_date["improvement"] = improvement
    years = {str(year): {"dates": len(group), "mean_improvement": float(group["improvement"].mean())}
             for year, group in by_date.groupby(by_date.index.str.slice(0, 4))}
    result.update({"test_dates": len(by_date), "test_rows": len(forecasts), "test_years": len(years),
                   "test_first": str(by_date.index.min()), "test_last": str(by_date.index.max()),
                   "baseline_rmse": math.sqrt(base_mse), "candidate_rmse": math.sqrt(candidate_mse),
                   "mse_improvement_pct": 100 * (base_mse - candidate_mse) / base_mse if base_mse > 0 else None,
                   "mean_error_improvement": stats["mean"], "error_improvement_ci": stats["ci"],
                   "p_value": stats["p_value"], "per_year": years,
                   "date_results": [{"date": str(d), "baseline_mse": float(rec["base_loss"]),
                                     "candidate_mse": float(rec["candidate_loss"]),
                                     "improvement": float(rec["improvement"])}
                                    for d, rec in by_date.iterrows()],
                   "predictions": records})
    if trial["target"] != "log_future_realized_vol":
        hits = forecasts.assign(base_hit=(forecasts["baseline"] > 0) == (forecasts["truth"] > 0),
                                candidate_hit=(forecasts["candidate"] > 0) == (forecasts["truth"] > 0))
        # Equal weight per date here too, rather than counting correlated names.
        result["baseline_hit_rate"] = float(hits.groupby("date")["base_hit"].mean().mean())
        result["candidate_hit_rate"] = float(hits.groupby("date")["candidate_hit"].mean().mean())
    if result["test_dates"] >= settings["min_test_dates"] and result["test_years"] >= settings["min_test_years"]:
        result["status"] = "AWAITING_FAMILY_ADJUSTMENT"
        result["reason"] = "Matched held-out evaluation complete; family-wise adjustment required."
    else:
        result["reason"] = (f"Requires {settings['min_test_dates']} non-overlapping test dates across "
                            f"{settings['min_test_years']} years; has {result['test_dates']} dates across {len(years)} years.")
    return result


def finalize_family(results: list[dict], settings: dict) -> list[dict]:
    eligible = [r["status"] == "AWAITING_FAMILY_ADJUSTMENT" for r in results]
    p_values = [r["p_value"] if ok else 1.0 for r, ok in zip(results, eligible)]
    adjusted = holm_adjust(p_values)
    for r, p, ok in zip(results, adjusted, eligible):
        r["adjusted_p"] = p
        if not ok:
            continue
        years = r["per_year"]
        majority = sum(y["mean_improvement"] > 0 for y in years.values()) > len(years) / 2
        support = p < settings["family_alpha"] and r["error_improvement_ci"][0] > 0 and majority
        r["status"] = "RESEARCH_SUPPORT" if support else "NOT_ESTABLISHED"
        r["reason"] = ("Incremental forecasting evidence in this conditional universe; not an option-profit or deployment finding."
                       if support else "Added channel did not clear all preregistered uncertainty, multiplicity and year-consistency checks.")
    return results
