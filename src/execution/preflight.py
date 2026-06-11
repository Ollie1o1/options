"""One-command go-live preflight: machine-checks docs/GO_LIVE_RUNBOOK.md.

    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.execution.preflight

Prints each runbook item with a pass/fail and a final CLEARED / NOT CLEARED
verdict. Every check is a pure function over injected inputs so the logic is
trivially testable; the CLI only gathers the real inputs.

Design rule: preflight *reads* the existing sources of truth — the gate
(phase1_checkpoint), arming (execution.pipeline.arm_status), automation
health (src.health), the sizing defaults (execution.sizing) — and never
defines its own version of any of them. NOT CLEARED with reasons is a
correct, expected answer until the gate fires.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


# ── Pure checks ──────────────────────────────────────────────────────────────

def gate_check(decision: str) -> CheckResult:
    """Runbook step 1: the gate must read READY."""
    ok = decision == "READY"
    detail = "gate=READY" if ok else f"gate={decision} (need READY) — do not trade"
    return CheckResult("gate", ok, detail)


def arming_check(arm: dict) -> CheckResult:
    """Runbook step 3: arm_status must report ARMED (gate + live flag)."""
    ok = bool(arm.get("armed"))
    detail = "ARMED" if ok else "DISARMED: " + "; ".join(arm.get("blockers") or ["unknown"])
    return CheckResult("arming", ok, detail)


def risk_caps_check(max_risk_pct: Optional[float] = None,
                    max_position_pct: Optional[float] = None) -> CheckResult:
    """Runbook step 4/6: sizing caps must still be <=2% risk / <=10% cost.

    Defaults are read from the real ``size_position`` signature so an
    accidental loosening of the caps fails preflight.
    """
    if max_risk_pct is None or max_position_pct is None:
        from src.execution.sizing import size_position
        params = inspect.signature(size_position).parameters
        if max_risk_pct is None:
            max_risk_pct = params["max_risk_pct"].default
        if max_position_pct is None:
            max_position_pct = params["max_position_pct"].default
    ok = max_risk_pct <= 0.02 and max_position_pct <= 0.10
    detail = (f"risk cap {max_risk_pct:.0%} (max 2%), "
              f"position cap {max_position_pct:.0%} (max 10%)")
    if not ok:
        detail += " — caps loosened beyond the runbook limits"
    return CheckResult("risk caps", ok, detail)


def checkpoint_freshness_check(last_checkpoint: Optional[str],
                               today: Optional[str] = None,
                               max_age_days: int = 8) -> CheckResult:
    """The gate read must be current: last weekly checkpoint within max_age_days."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    if not last_checkpoint:
        return CheckResult("checkpoint freshness", False,
                           "no checkpoint recorded — run the screener first")
    try:
        age = (datetime.strptime(today, "%Y-%m-%d")
               - datetime.strptime(last_checkpoint[:10], "%Y-%m-%d")).days
    except ValueError:
        return CheckResult("checkpoint freshness", False,
                           f"unparseable checkpoint date {last_checkpoint!r}")
    ok = age <= max_age_days
    detail = f"last checkpoint {last_checkpoint} ({age}d ago, max {max_age_days}d)"
    if not ok:
        detail += " — stale; run the screener to refresh the gate"
    return CheckResult("checkpoint freshness", ok, detail)


def automation_health_check(warnings: List[str]) -> CheckResult:
    """Auto-log / exit-enforcer / checkpoint must not be stale (src.health)."""
    if not warnings:
        return CheckResult("automation health", True, "all automation fresh")
    return CheckResult("automation health", False, "; ".join(warnings))


def slippage_db_check(data_dir: str = "data") -> CheckResult:
    """Runbook step 8: fills are recorded to data/fills.db — dir must be writable."""
    target = data_dir if os.path.isdir(data_dir) else os.path.dirname(data_dir) or "."
    ok = os.access(target, os.W_OK) if os.path.isdir(target) else False
    if not os.path.isdir(data_dir):
        # creatable counts: parent writable
        parent = os.path.dirname(os.path.abspath(data_dir))
        ok = os.access(parent, os.W_OK)
    detail = f"{os.path.join(data_dir, 'fills.db')} ({'writable' if ok else 'NOT writable'})"
    return CheckResult("slippage db", ok, detail)


# ── Aggregation / rendering ──────────────────────────────────────────────────

def aggregate(checks: List[CheckResult]) -> dict:
    failed = [c.name for c in checks if not c.ok]
    return {"cleared": not failed, "failed": failed}


def render(result: dict, checks: List[CheckResult]) -> str:
    lines = ["Go-live preflight — docs/GO_LIVE_RUNBOOK.md, machine-checked", ""]
    for c in checks:
        mark = "✅" if c.ok else "❌"
        lines.append(f"  {mark} {c.name}: {c.detail}")
    lines.append("")
    if result["cleared"]:
        lines.append("CLEARED ✅ — every runbook precondition holds. Follow the "
                     "runbook for ticket generation and fills.")
    else:
        lines.append(f"NOT CLEARED 🔒 — blocked by: {', '.join(result['failed'])}.")
        lines.append("This is the honest answer until the gate fires; do not "
                     "override it.")
    return "\n".join(lines)


# ── Real-input gathering ─────────────────────────────────────────────────────

def _last_checkpoint_date(history_path: str = "reports/checkpoint_history.tsv") -> Optional[str]:
    try:
        with open(history_path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if len(lines) < 2:
        return None
    return lines[-1].split("\t", 1)[0].strip()


def run_preflight(db_path: str = "paper_trades.db",
                  config_path: str = "config.json") -> Tuple[dict, List[CheckResult]]:
    import json

    from src import health
    from src.execution import pipeline

    with open(config_path) as f:
        cfg = json.load(f)
    phase1_start = (cfg.get("auto_log") or {}).get("phase1_start_date")

    # An unreadable gate (missing/empty DB: fresh checkout, wrong cwd) is a
    # blocked gate, not a crash — preflight must always deliver a verdict.
    try:
        arm = pipeline.arm_status(db_path, cfg, phase1_start)
        gate_c, arm_c = gate_check(arm["gate"]), arming_check(arm)
    except Exception as e:
        gate_c = CheckResult("gate", False, f"gate unavailable ({e}) — do not trade")
        arm_c = CheckResult("arming", False, "DISARMED: gate unavailable")
    checks = [
        gate_c,
        arm_c,
        risk_caps_check(),
        checkpoint_freshness_check(_last_checkpoint_date()),
        automation_health_check(health.automation_health_warnings(db_path=db_path)),
        slippage_db_check("data"),
    ]
    return aggregate(checks), checks


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Go-live preflight (runbook, machine-checked)")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    result, checks = run_preflight(db_path=args.db, config_path=args.config)
    print(render(result, checks))
    raise SystemExit(0 if result["cleared"] else 1)


if __name__ == "__main__":
    main()
