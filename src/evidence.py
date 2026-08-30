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
from datetime import date
from typing import Any, Dict, Optional

from src.formatting import truncate

# Forward-cohort gate target: the checkpoint job needs >= this many closed
# cohort trades before the validation gate can fire (see reports/checkpoint_*.md).
GATE_TARGET_N = 50

# The walk-forward artifact is only re-run monthly (src/maintenance.py
# due_walk_forward); past this many days since its own generated_at, the
# banner flags it so a stale OOS number is never read as fresh evidence.
WALK_FORWARD_STALE_DAYS = 30


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
          "wf_as_of":      str | None,     # walk-forward artifact's OWN date,
                                            # never overridden by the (more
                                            # frequent) checkpoint date, so the
                                            # banner can flag it going stale
          "cohort_ic_pearson":  float | None,  # gate statistic, latest checkpoint
          "cohort_ic_spearman": float | None,  # rank IC beside it (None pre-2026-07)
          "wf_refused":        bool,           # latest walk-forward run refused to
                                                # report (too few folds survived
                                                # purging) rather than measuring
          "wf_refused_reason": str | None,     # why, verbatim from the artifact
        }
    """
    ev: Dict[str, Any] = {
        "pooled_ic": None,
        "p_value": None,
        "n_oos": 0,
        "cohort_n": 0,
        "gate_decision": "UNKNOWN",
        "as_of": None,
        "wf_as_of": None,
        "cohort_ic_pearson": None,
        "cohort_ic_spearman": None,
        "fold_ic_mean": None,
        "fold_ic_ci_95": None,
        "folds_ic_positive": None,
        "n_folds": None,
        "wf_refused": False,
        "wf_refused_reason": None,
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
                ev["wf_as_of"] = str(wf["generated_at"])
            # The interval, not just the point estimate. The pooled IC and the
            # fold mean can disagree in SIGN (-0.119 vs +0.067 on 2026-08-17),
            # and the fold CI can straddle zero, so a banner showing only the
            # pooled number reads as a verdict the data does not support.
            for _k in ("fold_ic_mean", "folds_ic_positive", "n_folds"):
                if wf.get(_k) is not None:
                    ev[_k] = wf[_k]
            if isinstance(wf.get("fold_ic_ci_95"), (list, tuple)):
                ev["fold_ic_ci_95"] = list(wf["fold_ic_ci_95"])
            # A refusal (Tasks 1-2, src/walk_forward.py::_refused_summary)
            # writes every statistic above as None rather than 0.0, so the
            # is-not-None guards above leave them at their safe defaults.
            # Without this flag that reads as "not computed yet" in the
            # banner; with it, the banner names the refusal instead.
            if wf.get("refused"):
                ev["wf_refused"] = True
                if wf.get("refused_reason"):
                    ev["wf_refused_reason"] = str(wf["refused_reason"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    # --- forward-cohort checkpoint history (TSV) -----------------------------
    tsv_path = os.path.join(reports_dir, "checkpoint_history.tsv")
    if os.path.exists(tsv_path):
        try:
            with open(tsv_path, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t", restkey=_EXTRA)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
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
                ev["cohort_ic_pearson"] = _num(last.get("ic"))
                ev["cohort_ic_spearman"] = _rank_ic_field(last, fields)
        except (OSError, ValueError, KeyError):
            pass

    return ev


# csv.DictReader parks fields beyond the header under this key.
_EXTRA = "_extra"


def _num(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_ic_field(row: Dict[str, Any], fields: list) -> Optional[float]:
    """Spearman IC from a checkpoint-history row, whichever width it is.

    The history holds 6-field rows written before the rank IC was recorded and
    8-field rows written after. Under the current header the value is a named
    column; under an older header (a checkout that has not run a checkpoint
    since the column was added) it arrives as the first overflow field.
    """
    if "spearman" in fields:
        return _num(row.get("spearman"))
    extra = row.get(_EXTRA)
    if isinstance(extra, list) and extra:
        return _num(extra[0])
    return None


def format_evidence_banner(ev: Optional[Dict[str, Any]] = None,
                           today: Optional[date] = None) -> str:
    """
    Honest evidence label for the ranking model. One line when there is
    nothing to say about walk-forward age or the cohort rank IC, two when
    there is (each line kept under ui.banner's 100-char budget rather than
    growing one line without bound), e.g.:

      'Ranking model: EXPERIMENTAL — OOS IC +0.10 (p=0.48, n=94) | gate:
       GATHERING (n=2/50)
       OOS walk-forward as of 2026-05-29 (63d old, STALE >30d) | cohort IC
       +0.048 pearson / -0.020 rank'

    The cohort IC is shown as both statistics because on floored option returns
    they routinely disagree, and reporting only the Pearson one overstates the
    evidence. The walk-forward artifact is only regenerated monthly (see
    src/maintenance.py due_walk_forward), so its own age is surfaced and
    flagged past WALK_FORWARD_STALE_DAYS — reading it as fresh past that point
    would overstate how current the OOS number is.

    A third line appears when the latest walk-forward run refused to report
    (too few folds survived purging — see src/walk_forward.py) rather than
    leaving line 1's OOS slot merely empty, which would read as "not computed
    yet" instead of "computed and refused":

      'Ranking model: EXPERIMENTAL — OOS IC REFUSED | gate: GATHERING (n=2/50)
       OOS walk-forward as of 2026-08-29 (0d old) | cohort IC +0.048 pearson...
       refused: only 0 of 15 folds kept 54+ training trades after purging...'

    Reads from load_model_evidence() when ``ev`` is not supplied. ``today``
    is injectable for deterministic tests; defaults to date.today().
    """
    if ev is None:
        ev = load_model_evidence()

    ic = ev.get("pooled_ic")
    p = ev.get("p_value")
    n = ev.get("n_oos") or 0
    if ev.get("wf_refused"):
        # A refusal (too few folds survived purging) is a completed run that
        # measured nothing — distinct from "no walk-forward report yet" below,
        # which means the job has never run at all. Collapsing the two would
        # make a refusal read as "not computed yet" instead of "computed and
        # refused". The reason itself goes on its own line (see
        # _wf_refusal_segment) since it can run past the line budget alone —
        # sharing a line with the age/cohort segments already fills it.
        oos = "OOS IC REFUSED"
    elif ic is None or p is None:
        oos = "OOS IC n/a (no walk-forward report yet)"
    else:
        oos = f"OOS IC {ic:+.2f} (p={p:.2f}, n={n})"

    gate = ev.get("gate_decision", "UNKNOWN") or "UNKNOWN"
    cohort_n = ev.get("cohort_n") or 0
    line1 = (
        f"Ranking model: EXPERIMENTAL — {oos}{_fold_interval_segment(ev)} | "
        f"gate: {gate} (n={cohort_n}/{GATE_TARGET_N})"
    )

    line2_parts = [seg for seg in (
        _walk_forward_age_segment(ev, today),
        _cohort_ic_segment(ev).lstrip(" |"),
    ) if seg]

    lines = [line1]
    if line2_parts:
        lines.append(" | ".join(line2_parts))
    # The refusal reason gets its own line rather than folding into line2:
    # age + cohort already fill that line's 100-char budget on their own, and
    # a refused_reason from walk_forward can run past 100 chars by itself.
    refusal_line = _wf_refusal_segment(ev)
    if refusal_line:
        lines.append(refusal_line)
    return "\n".join(lines)


def _fold_interval_segment(ev: Dict[str, Any]) -> str:
    """The fold-level estimate and its 95% interval, and whether that interval
    contains zero.

    The pooled IC is one number and reads as a verdict. On 2026-08-17 it was
    -0.119 while the fold mean was +0.067 with a 95% CI of [-0.099, +0.239] and
    11 of 18 folds positive — i.e. the two estimators disagreed on SIGN and the
    interval straddled zero. Showing only the pooled figure told the operator
    the ranking model was mildly anti-predictive when the honest statement is
    that it is not distinguishable from zero.

    "Not distinguishable" is read off the interval containing zero, not from a
    threshold anybody picked. Absent or malformed intervals render nothing, so
    artifacts written before these fields existed still produce a banner.
    """
    ci = ev.get("fold_ic_ci_95")
    mean = _num(ev.get("fold_ic_mean"))
    if mean is None or not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return ""
    lo, hi = _num(ci[0]), _num(ci[1])
    if lo is None or hi is None:
        return ""
    seg = f", folds {mean:+.2f} [95% CI {lo:+.2f}..{hi:+.2f}]"
    pos, n_f = ev.get("folds_ic_positive"), ev.get("n_folds")
    if pos is not None and n_f:
        seg += f", {pos}/{n_f} positive"
    if lo <= 0.0 <= hi:
        seg += " — NOT distinguishable from zero"
    return seg


# A refused_reason from walk_forward can run well past 100 chars on its own
# (e.g. "only 0 of 15 folds kept 54+ training trades after purging (minimum
# 3); widen train_size or wait for more closed trades" is 117). The segment
# gets its own banner line (see format_evidence_banner) so it never has to
# share the 100-char budget with the age/cohort segments, but even alone a
# raw reason can exceed it, so it is still summarised here; the full text
# stays in the artifact (reports/walk_forward_*.json), which has room for it.
_REFUSED_REASON_MAX_CHARS = 85


def _wf_refusal_segment(ev: Dict[str, Any]) -> str:
    """'refused: only 0 of 15 folds kept 54+ training trades after purg...',
    or '' when the latest walk-forward artifact was not a refusal."""
    if not ev.get("wf_refused"):
        return ""
    reason = ev.get("wf_refused_reason")
    if not reason:
        return "refused (no reason recorded)"
    return "refused: " + truncate(str(reason), _REFUSED_REASON_MAX_CHARS)


def _walk_forward_age_days(wf_as_of: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """Calendar days between the walk-forward artifact's own date and today."""
    if not wf_as_of:
        return None
    try:
        as_of_date = date.fromisoformat(str(wf_as_of)[:10])
    except ValueError:
        return None
    return ((today or date.today()) - as_of_date).days


def _walk_forward_age_segment(ev: Dict[str, Any], today: Optional[date] = None) -> str:
    """'OOS walk-forward as of 2026-05-29 (63d old, STALE >30d)', or '' when
    the walk-forward artifact's own date is unrecorded."""
    wf_as_of = ev.get("wf_as_of")
    if not wf_as_of:
        return ""
    date_str = str(wf_as_of)[:10]
    age = _walk_forward_age_days(wf_as_of, today)
    if age is None:
        return f"OOS walk-forward as of {date_str}"
    flag = f", STALE >{WALK_FORWARD_STALE_DAYS}d" if age > WALK_FORWARD_STALE_DAYS else ""
    return f"OOS walk-forward as of {date_str} ({age}d old{flag})"


def _cohort_ic_segment(ev: Dict[str, Any]) -> str:
    """' | cohort IC +0.048 pearson / -0.020 rank', or '' when unrecorded."""
    pearson = ev.get("cohort_ic_pearson")
    if pearson is None:
        return ""
    rank = ev.get("cohort_ic_spearman")
    rank_txt = "rank n/a" if rank is None else f"{rank:+.3f} rank"
    return f" | cohort IC {pearson:+.3f} pearson / {rank_txt}"
