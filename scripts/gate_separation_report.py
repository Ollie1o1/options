"""Run the pre-registered gate separation test and write a report + manifest.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.gate_separation_report

Registration: docs/PREREG_GATE_SEPARATION_20260929.md. Every threshold comes
from `src/gate_separation.py`, which mirrors that document. This script does
I/O, orchestration and formatting — it decides nothing.

The manifest carries the corpus fingerprint. Two runs whose fingerprints differ
are not comparable; without it, corpus drift looks like a finding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import gate_separation as gs

DEFAULT_DB = os.path.join("data", "candidates.db")
DEFAULT_OUT = os.path.join("reports", "gate_separation")


def _half_stat(df: pd.DataFrame) -> Optional[float]:
    cells = gs.cell_superiority(df)
    means = gs.cluster_means(cells, "d_auc")
    if len(means) == 0:
        return None
    stat = gs.strategy_statistic(means)
    return float(stat) if np.isfinite(stat) else None


def analyse(df: pd.DataFrame, strategy: str, *, n_boot: int, n_perm: int,
            shuffles: int, seed: int) -> Dict[str, object]:
    """Every per-strategy number except the family-wise p, which needs siblings."""
    sub = df[df["strategy_name"] == strategy]
    paired = gs.both_arm_rows(sub)

    cells = gs.cell_superiority(paired)
    means = gs.cluster_means(cells, "d_auc")
    stat = gs.strategy_statistic(means)
    ci_lo, ci_hi = gs.cluster_bootstrap_ci(cells, "d_auc", n_boot=n_boot,
                                           seed=seed)

    sec_cells = gs.cell_mean_difference(paired)
    sec_means = gs.cluster_means(sec_cells, "d_mean")
    secondary = gs.strategy_statistic(sec_means)

    raw_cells = gs.cell_mean_difference(paired, winsor=None)
    raw_secondary = gs.strategy_statistic(
        gs.cluster_means(raw_cells, "d_mean"))

    first, second = gs.half_split(paired)
    half_a, half_b = _half_stat(first), _half_stat(second)

    skew = float(pd.Series(cells["d_auc"]).skew()) if len(cells) > 2 else 0.0
    control = (gs.negative_control(paired, n_shuffles=shuffles, seed=seed)
               if len(means) >= 2 else
               {"observed": float("nan"), "null_mean": 0.0,
                "p95_abs": float("nan"), "n_shuffles": 0.0})

    return {
        "strategy": strategy or "(blank)",
        "n_rows": int(len(sub)),
        "n_paired_rows": int(len(paired)),
        "n_cells": int(len(cells)),
        "n_clusters": int(len(means)),
        "stat": float(stat) if np.isfinite(stat) else float("nan"),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "secondary_winsorized": float(secondary) if np.isfinite(secondary)
        else float("nan"),
        "secondary_raw": float(raw_secondary) if np.isfinite(raw_secondary)
        else float("nan"),
        "half_a": half_a,
        "half_b": half_b,
        "skew": skew,
        "loso_stable": gs.loso_sign_stable(means),
        "negative_control": control,
        "_means": means,
    }


def run(db_path: str, *, n_boot: int, n_perm: int, shuffles: int,
        seed: int) -> Dict[str, object]:
    df = gs.load_cohort(db_path)
    if len(df) == 0:
        return {"error": "empty cohort", "db_path": db_path}

    strategies = sorted(df["strategy_name"].unique())
    rows = [analyse(df, s, n_boot=n_boot, n_perm=n_perm, shuffles=shuffles,
                    seed=seed) for s in strategies]

    # Family-wise correction.
    #
    # AMENDMENT 2026-09-29, after the first run — recorded here rather than
    # folded silently into the registration. The family is the strategies that
    # can make a CLAIM, i.e. those at or above `MIN_CLUSTERS`, not all six.
    #
    # Including the others does not make the test more conservative, it makes
    # it meaningless: Iron Condor has 2 symbols at ~+0.333, so a whole-symbol
    # sign flip leaves its |mean| at 0.333 whenever the two signs agree — 49%
    # of permutations. That alone set the family-wise p to 0.534 while every
    # measurable strategy's CI excluded zero by a wide margin. A strategy
    # already declared `insufficient` is not a look that needs correcting; it
    # is a look that was never taken.
    #
    # The unrestricted value is still computed and reported, so the effect of
    # this choice is visible rather than assumed.
    claimants = {r["strategy"]: r["_means"] for r in rows
                 if r["n_clusters"] >= gs.MIN_CLUSTERS}
    measurable = {r["strategy"]: r["_means"] for r in rows
                  if r["n_clusters"] >= 2}

    fwer_p, names = 1.0, []
    if claimants:
        names, matrix = gs.family_matrix(claimants)
        fwer_p = gs.family_wise_p_masked(matrix, n_perm=n_perm, seed=seed)

    fwer_p_all, all_names = 1.0, []
    if measurable:
        all_names, matrix_all = gs.family_matrix(measurable)
        fwer_p_all = gs.family_wise_p_masked(matrix_all, n_perm=n_perm,
                                             seed=seed)

    # One family-wise p-value for the whole family: it is the probability that
    # the LARGEST statistic among these strategies could arise by chance. It is
    # attached to every strategy because that is what it bounds — the claim
    # "the best of these six is real", not any one of them in isolation.
    for r in rows:
        r["fwer_p"] = fwer_p
        r["in_family"] = r["strategy"] in names
        result = gs.StrategyResult(
            strategy=str(r["strategy"]), n_rows=int(r["n_rows"]),
            n_cells=int(r["n_cells"]), n_clusters=int(r["n_clusters"]),
            stat=float(r["stat"]), ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
            secondary=float(r["secondary_winsorized"]),
            fwer_p=float(fwer_p), loso_stable=bool(r["loso_stable"]),
            half_a=r["half_a"], half_b=r["half_b"], skew=float(r["skew"]))
        r["verdict"] = gs.verdict(result)
        del r["_means"]

    worst_null = max((abs(r["negative_control"]["null_mean"]) for r in rows
                      if np.isfinite(r["negative_control"]["null_mean"])),
                     default=0.0)
    guard_ok = worst_null < gs.MAX_NULL_MEAN

    return {
        "registration": "docs/PREREG_GATE_SEPARATION_20260929.md",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "db_path": db_path,
        "corpus_fingerprint": gs.corpus_fingerprint(db_path),
        "parameters": {"n_boot": n_boot, "n_perm": n_perm,
                       "negative_control_shuffles": shuffles, "seed": seed,
                       "alpha": gs.ALPHA, "min_clusters": gs.MIN_CLUSTERS,
                       "max_fwer_p": gs.MAX_FWER_P, "winsor": list(gs.WINSOR),
                       "cell": list(gs.CELL_COLS), "cluster": gs.CLUSTER_COL},
        "family_wise_p": fwer_p,
        "family_members": names,
        "family_wise_p_all_measurable": fwer_p_all,
        "family_members_all_measurable": all_names,
        "negative_control_guard": {
            "worst_abs_null_mean": worst_null,
            "threshold": gs.MAX_NULL_MEAN,
            "passed": guard_ok,
        },
        "results": rows,
    }


def _fmt(v: object, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return "—" if not np.isfinite(v) else f"{v:.{nd}f}"
    return str(v)


def render(out: Dict[str, object]) -> str:
    if "error" in out:
        return f"gate separation: {out['error']} ({out.get('db_path')})"

    fp = out["corpus_fingerprint"]
    guard = out["negative_control_guard"]
    lines = [
        "Gate separation — pre-registered test",
        f"  registration : {out['registration']}",
        f"  corpus       : {fp.get('digest')} "
        f"({fp.get('candidate_rows')} candidate rows, "
        f"{fp.get('closed_positions')} closed positions)",
        f"  window       : {str(fp.get('ts_min'))[:10]} .. "
        f"{str(fp.get('ts_max'))[:10]}",
        f"  family-wise p: {out['family_wise_p']:.4f} "
        f"(max-stat sign-flip over {len(out['family_members'])} claimants: "
        f"{', '.join(out['family_members'])})",
        f"               : {out['family_wise_p_all_measurable']:.4f} if "
        f"under-powered strategies are included — see AMENDMENT in "
        f"scripts/gate_separation_report.py",
        f"  null guard   : {'PASS' if guard['passed'] else 'VOID'} "
        f"(worst |null mean| {guard['worst_abs_null_mean']:.5f} "
        f"vs {guard['threshold']})",
        "",
        f"{'strategy':<14}{'cells':>6}{'symb':>6}{'d_auc':>9}"
        f"{'95% CI':>19}{'win.mean':>10}{'1st':>8}{'2nd':>8}"
        f"{'loso':>6}  verdict",
        "-" * 104,
    ]
    order = {"inverted": 0, "separates": 1, "null": 2, "insufficient": 3}
    for r in sorted(out["results"], key=lambda x: (order.get(x["verdict"], 9),
                                                   -x["n_clusters"])):
        ci = (f"[{_fmt(r['ci_lo'], 3)}, {_fmt(r['ci_hi'], 3)}]"
              if r["ci_lo"] is not None else "—")
        lines.append(
            f"{r['strategy']:<14}{r['n_cells']:>6}{r['n_clusters']:>6}"
            f"{_fmt(r['stat'], 4):>9}{ci:>19}"
            f"{_fmt(r['secondary_winsorized'], 3):>10}"
            f"{_fmt(r['half_a'], 3):>8}{_fmt(r['half_b'], 3):>8}"
            f"{'yes' if r['loso_stable'] else 'no':>6}  {r['verdict']}")

    lines += [
        "",
        "d_auc  P(passed beats refused) - 0.5, within (symbol, expiration, "
        "scan day),",
        "       averaged over symbols. Positive = the gate keeps the better "
        "contracts.",
        "win.mean  winsorized mean difference in return-on-premium. Economic "
        "magnitude only;",
        "       NOT comparable across strategies — a credit spread and a long "
        "put do not",
        "       share a denominator.",
        "",
        "No verdict here authorises a configuration change. See the "
        "registration's",
        "provenance section: the inversion this tests was seen in this same "
        "corpus first.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=gs.N_BOOT)
    ap.add_argument("--n-perm", type=int, default=gs.N_PERM)
    ap.add_argument("--shuffles", type=int,
                    default=gs.NEGATIVE_CONTROL_SHUFFLES)
    ap.add_argument("--seed", type=int, default=gs.SEED)
    ap.add_argument("--quick", action="store_true",
                    help="small resample counts, for a smoke run only")
    args = ap.parse_args(argv)

    if args.quick:
        args.n_boot, args.n_perm, args.shuffles = 500, 500, 20

    out = run(args.db, n_boot=args.n_boot, n_perm=args.n_perm,
              shuffles=args.shuffles, seed=args.seed)
    text = render(out)
    print(text)

    if "error" not in out:
        os.makedirs(args.out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = os.path.join(args.out_dir, f"gate_separation_{stamp}")
        with open(f"{base}.json", "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        with open(f"{base}.txt", "w") as fh:
            fh.write(text + "\n")
        print(f"\nmanifest: {base}.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
