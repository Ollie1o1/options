"""`python -m src.policy_lab` — run one sweep and emit a report.

Read-only. This command never writes to any database and never touches
config.json; promoting a finding is a separate, later decision requiring its
own PR against the bar in the design doc's section 7.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.policy_lab.calibrate import CorpusUnusable
from src.policy_lab.costs import CostModel
from src.policy_lab.paths import load_corpus_a, load_corpus_b
from src.policy_lab.policies import LONG_PREMIUM_GRID, SHORT_PREMIUM_GRID
from src.policy_lab.report import (
    bounded_walk_benchmark, calibrate_for_run, calibration_max_fail_rate,
    corpus_a_baseline, corpus_b_baseline, render_markdown, run_manifest,
    spread_imputed_fraction, sweep,
)
from src.policy_lab.stats import policy_verdict

_LONG = {"Long Call", "Long Put"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.policy_lab")
    p.add_argument("--corpus", choices=("a", "b"), default="a")
    p.add_argument("--strategy", default="Bull Put")
    p.add_argument("--costs", choices=CostModel.SETTINGS, default="cross")
    p.add_argument("--candidates-db", default="data/candidates.db")
    p.add_argument("--ledger-db", default="paper_trades.db")
    p.add_argument("--archive-db", default="data/chain_archive.db")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--no-corpus-b-check", action="store_true",
                   help="skip loading Corpus B for the independent sign "
                        "check (faster, but every policy's "
                        "corpus_b_sign_agrees is then conservatively False)")
    p.add_argument("--out", default=None,
                   help="write the report here instead of stdout")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    costs = CostModel(args.costs)
    is_long = args.strategy in _LONG
    grid = LONG_PREMIUM_GRID if is_long else SHORT_PREMIUM_GRID

    if args.corpus == "a":
        paths, load = load_corpus_a(args.candidates_db, args.strategy)
        baseline = corpus_a_baseline(is_long)
    else:
        paths, load = load_corpus_b(args.ledger_db, args.archive_db,
                                    args.strategy)
        baseline = corpus_b_baseline(is_long)

    if not paths:
        print(f"no usable paths for {args.strategy}: {load.dropped}",
              file=sys.stderr)
        return 2
    print(f"[policy_lab] paths loaded: {load.loaded} usable "
          f"(corpus {args.corpus.upper()}, {args.strategy})", file=sys.stderr)

    max_fail_rate = calibration_max_fail_rate(args.corpus)
    # `calibrate_for_run` always checks reproduction at MID, never at `costs`
    # (this run's own cost setting) — see its docstring in report.py for why
    # (the recorder priced its exits at mid, so a mid replay is the only
    # like-for-like comparison; a cross/surface calibration measures the
    # crossing cost, not the harness).
    try:
        cal = calibrate_for_run(paths, baseline, costs, max_fail_rate)
    except CorpusUnusable as exc:
        print(f"corpus refused: {exc}", file=sys.stderr)
        return 3
    print(f"[policy_lab] calibration: {cal.passed}/{cal.checked} reproduce "
          f"({100 * cal.fail_rate:.1f}% fail, max {100 * max_fail_rate:.1f}%)",
          file=sys.stderr)

    # Corpus B is loaded as an INDEPENDENT check regardless of which corpus
    # is primary — a promotion from Corpus A alone, or from Corpus B alone
    # with nothing to corroborate it, cannot claim corpus_b_sign_agrees.
    corpus_b_paths = None
    if args.corpus == "a" and not args.no_corpus_b_check:
        try:
            corpus_b_paths, _ = load_corpus_b(
                args.ledger_db, args.archive_db, args.strategy)
        except Exception:
            corpus_b_paths = None

    results = sweep(paths, grid, baseline, costs,
                    corpus_b_paths=corpus_b_paths, seed=args.seed,
                    n_boot=args.n_boot, n_perm=args.n_perm)

    grid_by_name = {p.name: p for p in grid}
    benchmark = {}
    for r in results:
        if policy_verdict(r) != "promote":
            continue
        policy = grid_by_name.get(r.policy_name)
        if policy is None:
            continue
        synth_d = bounded_walk_benchmark(paths, baseline, policy, costs,
                                         seed=args.seed)
        diff = (r.mean_d_cross - synth_d)
        benchmark[r.policy_name] = (r.mean_d_cross, synth_d, diff)
    print(f"[policy_lab] benchmark done: {len(benchmark)} promoted policies",
          file=sys.stderr)

    row_counts = {
        "loaded": load.loaded,
        "dropped": dict(load.dropped),
        "spread_imputed_frac": spread_imputed_fraction(paths),
    }
    manifest = run_manifest(args.corpus.upper(), grid, args.costs, args.seed,
                            row_counts)
    body = render_markdown(results, manifest, cal, benchmark=benchmark)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print(f"wrote {args.out}")
    else:
        print(body)
    return 0
