"""Glue between the validation gate and the execution stack.

Reads the *live* gate decision and the live-execution flag, then sizes/exits/renders
a ticket. Because it reads the real gate and the real flag (default false), tickets
are DRY-RUN until both genuinely open — the safety property is enforced by data, not
by remembering to pass the right arguments.

WHICH gate is now explicit (`config.gate.authorising_gate`) rather than implied by
this module being the only caller of the Long Call one. There are two gates as of
2026-07-31 and they disagree: the Long Call gate resolved STOP, and the
short-premium gate — the family the book actually earns on — reads Arm A READY.
Every gate is reported on every status line; only the authorising one can arm.
Selecting a different one is a real-money decision and deliberately requires a
config edit that says so.
"""
from __future__ import annotations

from typing import Optional

from src import phase1_checkpoint
from src.execution import exits as exits_mod
from src.execution import sizing as sizing_mod
from src.execution import ticket as ticket_mod
from src.paths import repo_path

# The gates that can authorise execution. Names match config.gate.authorising_gate.
LONG_CALL = "long_call"
SHORT_PREMIUM = "short_premium"


def live_enabled(config: dict) -> bool:
    return bool((config or {}).get("live_execution", {}).get("enabled", False))


def authorising_gate(config: dict) -> str:
    """Which gate's verdict arms the pipeline. Default is the historical one."""
    return str((config or {}).get("gate", {}).get("authorising_gate")
               or LONG_CALL)


def gate_readings(db_path: str, phase1_start: str,
                  config: Optional[dict] = None) -> dict:
    """Every gate's current verdict, in one checkpoint pass.

    Both are returned whichever one authorises, because the failure this
    prevents is silent: on 2026-08-01 the pipeline printed `gate: STOP` while
    the short-premium family — the one the book actually earns on, promoted to
    validation evidence that same day — read Arm A READY. Neither number was
    wrong; the report simply did not say which question it had answered.

    The cohort is defined by config, not by this function's defaults. The
    short-premium gate reads a cohort capped at `auto_log.max_capital_at_risk`,
    and evaluating it uncapped drags in the $32k-$83k cash-secured puts the
    account could never have held — a different cohort, a different verdict,
    silently disagreeing with the checkpoint the operator reads.
    """
    cfg = config or {}
    cap = (cfg.get("auto_log") or {}).get("max_capital_at_risk")
    gate_cfg = cfg.get("gate") or {}
    r = phase1_checkpoint.compute_checkpoint(
        db_path, phase1_start,
        max_capital_at_risk=float(cap) if cap else None,
        gate_version=int(gate_cfg.get("version", 2)),
        extensions_used=int(gate_cfg.get("extensions_used", 0)),
        short_premium_extensions_used=int(
            gate_cfg.get("short_premium_extensions_used", 0)))
    sp = r.get("short_premium_gate") or {}
    return {
        LONG_CALL: r.get("decision"),
        # Arm A is the arm that would authorise capital; Arm B only decides
        # whether the scorer picks the contracts (docs/GATE_REDESIGN_SPEC.md).
        SHORT_PREMIUM: sp.get("arm_a"),
        "short_premium_arm_b": sp.get("arm_b"),
    }


def current_gate(db_path: str, phase1_start: str,
                 config: Optional[dict] = None) -> str:
    """The live verdict of whichever gate authorises (always current)."""
    cfg = config or {}
    verdict = gate_readings(db_path, phase1_start, cfg).get(authorising_gate(cfg))
    # An unknown or unevaluable gate must never read as permission.
    return str(verdict or "GATHERING")


def build_ticket(pick: dict,
                 account_value: float,
                 db_path: str,
                 config: dict,
                 phase1_start: str,
                 win_prob: Optional[float] = None,
                 payoff_ratio: Optional[float] = None) -> dict:
    """Size + exits + render, gated by the real gate decision and live flag."""
    entry_price = float(pick.get("entry_price")
                        or ticket_mod._limit_price(pick))
    e = exits_mod.compute_exits(entry_price=entry_price,
                                expiration=pick["expiration"], config=config)
    s = sizing_mod.size_position(account_value=account_value,
                                 entry_price=entry_price,
                                 stop_price=e.stop_price,
                                 win_prob=win_prob, payoff_ratio=payoff_ratio)
    gate = current_gate(db_path, phase1_start, config)
    return ticket_mod.render_ticket(pick, s, e, gate_decision=gate,
                                    live_enabled=live_enabled(config))


def arm_status(db_path: str, config: dict, phase1_start: str) -> dict:
    """Is live execution armed? Reports every gate, the flag, and the verdict.

    Only the authorising gate can arm. The others are reported so a READY
    sitting on a gate that does not authorise is visible as exactly that,
    rather than looking like permission the pipeline is ignoring.
    """
    which = authorising_gate(config)
    readings = gate_readings(db_path, phase1_start, config)
    gate = str(readings.get(which) or "GATHERING")
    flag = live_enabled(config)
    return {
        "gate": gate,
        "authorising_gate": which,
        "gate_readings": readings,
        "live_enabled": flag,
        "armed": (gate == "READY" and flag),
        "blockers": [b for b in (
            None if gate == "READY" else f"gate={gate} (need READY)",
            None if flag else "config.live_execution.enabled=false",
        ) if b],
    }


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Execution arming status (Phase 3)")
    ap.add_argument("--db", default="paper_trades.db")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    with open(repo_path(args.config)) as f:
        cfg = json.load(f)
    p1 = (cfg.get("auto_log") or {}).get("phase1_start_date")
    st = arm_status(args.db, cfg, p1)
    print("Phase 3 execution —", "ARMED ✅" if st["armed"] else "DISARMED 🔒")
    print(f"  authorising gate: {st['authorising_gate']} → {st['gate']}")
    for name, verdict in (st.get("gate_readings") or {}).items():
        if name != st["authorising_gate"]:
            mark = "" if name.startswith("short_premium_arm") else "  (does not authorise)"
            print(f"    {name}: {verdict}{mark}")
    print(f"  live_execution.enabled: {st['live_enabled']}")
    if st["blockers"]:
        print("  blockers:")
        for b in st["blockers"]:
            print(f"    • {b}")
    print("\nThis is mirror-mode only: the system prints a ticket, you place it. "
          "No broker API.")


if __name__ == "__main__":
    main()
