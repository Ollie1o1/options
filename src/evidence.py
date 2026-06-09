"""
Model-evidence loader: surfaces the ranking model's out-of-sample track record
to the UI so predictive outputs are labeled with their actual evidence.

Pure file-parsing, no network. Reads the latest walk-forward report and the
forward-cohort checkpoint history, both written by the validation pipeline.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Dict, Optional

# Forward-cohort gate target: the checkpoint job needs >= this many closed
# cohort trades before the validation gate can fire (see reports/checkpoint_*.md).
GATE_TARGET_N = 50


def _latest_by_mtime(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def load_model_evidence(reports_dir: str = "reports") -> Dict[str, Any]:
    """
    Return the ranking model's evidence with safe defaults when artifacts are
    missing:

        {
          "pooled_ic":     float | None,   # walk-forward pooled IC
          "p_value":       float | None,   # its p-value
          "n_oos":         int,            # trades behind the walk-forward
          "cohort_n":      int,            # forward-cohort size (latest checkpoint)
          "gate_decision": str,            # e.g. "GATHERING" / "READY" / "UNKNOWN"
          "as_of":         str | None,     # most recent artifact timestamp/date
        }
    """
    ev: Dict[str, Any] = {
        "pooled_ic": None,
        "p_value": None,
        "n_oos": 0,
        "cohort_n": 0,
        "gate_decision": "UNKNOWN",
        "as_of": None,
    }

    # --- walk-forward report -------------------------------------------------
    wf_path = _latest_by_mtime(os.path.join(reports_dir, "walk_forward_*.json"))
    if wf_path:
        try:
            with open(wf_path) as f:
                wf = json.load(f)
            if wf.get("pooled_ic") is not None:
                ev["pooled_ic"] = float(wf["pooled_ic"])
            if wf.get("pooled_pvalue") is not None:
                ev["p_value"] = float(wf["pooled_pvalue"])
            if wf.get("n_total_trades") is not None:
                ev["n_oos"] = int(wf["n_total_trades"])
            if wf.get("generated_at"):
                ev["as_of"] = str(wf["generated_at"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # --- forward-cohort checkpoint history (TSV) -----------------------------
    tsv_path = os.path.join(reports_dir, "checkpoint_history.tsv")
    if os.path.exists(tsv_path):
        try:
            with open(tsv_path, newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            if rows:
                last = rows[-1]
                if last.get("n") not in (None, ""):
                    ev["cohort_n"] = int(float(last["n"]))
                if last.get("decision"):
                    ev["gate_decision"] = str(last["decision"]).strip()
                # Prefer the checkpoint date as as_of when it is more recent.
                if last.get("date"):
                    if not ev["as_of"] or str(last["date"]) > str(ev["as_of"])[:10]:
                        ev["as_of"] = str(last["date"])
        except (OSError, ValueError, KeyError):
            pass

    return ev


def format_evidence_banner(ev: Optional[Dict[str, Any]] = None) -> str:
    """
    One-line, honest evidence label for the ranking model, e.g.:
      'Ranking model: EXPERIMENTAL — OOS IC +0.10 (p=0.48, n=94) | gate: GATHERING (n=2/50)'

    Reads from load_model_evidence() when ``ev`` is not supplied.
    """
    if ev is None:
        ev = load_model_evidence()

    ic = ev.get("pooled_ic")
    p = ev.get("p_value")
    n = ev.get("n_oos") or 0
    if ic is None or p is None:
        oos = "OOS IC n/a (no walk-forward report yet)"
    else:
        oos = f"OOS IC {ic:+.2f} (p={p:.2f}, n={n})"

    gate = ev.get("gate_decision", "UNKNOWN") or "UNKNOWN"
    cohort_n = ev.get("cohort_n") or 0
    return (
        f"Ranking model: EXPERIMENTAL — {oos} | "
        f"gate: {gate} (n={cohort_n}/{GATE_TARGET_N})"
    )
