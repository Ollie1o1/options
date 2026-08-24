"""Fit, validate and — only if it earns it — ship the calibrated probability.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m scripts.pop_calibration_report

Prints two reliability curves and two verdicts:

  POP     P(this position closes green under the exit rules in force)
  RETURN  E[return on capital at risk]

They are separate because on this book they disagree. Measured out-of-sample
2026-08-23, win rate rises cleanly with predicted probability while money does
not follow it — the 0.4-0.5 bucket wins 44.3% at PF 0.66, the 0.3-0.4 bucket
wins 36.5% at PF 1.13. A number that is honest about winning can still mislead
about profit, so each carries its own guard and each ships or is refused on
its own evidence.

The artifact is written either way. `shipped: false` in it is not a failure to
be retried until it passes — it is the finding, and `pop_calibration.load_model`
returns None for it so that a refused model cannot reach a board.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

import pandas as pd

from src import pop_calibration as pc

DEFAULT_DB = "paper_trades.db"


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def run(db_path: str = DEFAULT_DB, model_path: str = pc.DEFAULT_MODEL_PATH,
        seed_n: int = 300, step: int = 50) -> Dict[str, Any]:
    """Fit both heads, validate both out of sample, persist the verdict."""
    df = pc.load_training_set(db_path)
    out: Dict[str, Any] = {"n_train": int(len(df))}

    # ---- P(closes green) -------------------------------------------------
    oos = pc.walk_forward(df, seed_n=seed_n, step=step)
    rel = pc.reliability(oos)
    ok, reason = pc.ship_check(rel)
    out["pop"] = {"shipped": ok, "reason": reason, "n_oos": int(len(oos)),
                  "reliability": rel}

    model = pc.fit(df) if len(df) else None
    if model is not None:
        model.meta = {"n_oos": int(len(oos)), "guard": reason,
                      "target": "P(closes green at the real exit)"}
        pc.save_model(model, model_path, shipped=ok, reason=reason,
                      reliability_table=rel)

    # ---- E[return on capital at risk] ------------------------------------
    roos = pc.walk_forward(df, seed_n=seed_n, step=step, target="ret_on_risk")
    rrel = pc.return_reliability(roos)
    rok, rreason = pc.ship_check_return(rrel)
    out["ret"] = {"shipped": rok, "reason": rreason, "n_oos": int(len(roos)),
                  "reliability": rrel}
    return out


def _print(out: Dict[str, Any]) -> None:
    pd.set_option("display.width", 200)
    print(f"training set: {out['n_train']} closed trades\n")

    for key, title, note in (
        ("pop", "P(CLOSES GREEN)", "predicted probability vs realised win rate"),
        ("ret", "E[RETURN ON CAPITAL AT RISK]",
         "predicted return vs realised mean return"),
    ):
        block = out[key]
        print("=" * 78)
        print(f"{title} — {note}")
        print(f"out-of-sample predictions: {block['n_oos']}")
        print("=" * 78)
        rel = block["reliability"]
        if len(rel):
            print(rel.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        else:
            print("(no out-of-sample predictions)")
        verdict = "SHIPS" if block["shipped"] else "REFUSED"
        print(f"\n  {verdict}: {block['reason']}\n")

    if not out["ret"]["shipped"] and out["pop"]["shipped"]:
        print("Read this carefully: the probability is validated, the return "
              "is not.\nA calibrated win rate is NOT a claim that the trade "
              "is profitable.\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--model-path", default=pc.DEFAULT_MODEL_PATH)
    ap.add_argument("--seed-n", type=int, default=300)
    ap.add_argument("--step", type=int, default=50)
    args = ap.parse_args(argv)

    out = run(args.db, args.model_path, seed_n=args.seed_n, step=args.step)
    _print(out)
    # Exit 0 whether or not it ships: a refusal is a successful measurement,
    # and a non-zero exit would make a scheduler treat the finding as a fault.
    return 0


if __name__ == "__main__":
    sys.exit(main())
