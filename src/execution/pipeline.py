"""Glue between the validation gate and the execution stack.

Reads the *live* gate decision and the live-execution flag, then sizes/exits/renders
a ticket. Because it reads the real gate (currently GATHERING) and the real flag
(default false), tickets are DRY-RUN until both genuinely open — the safety property
is enforced by data, not by remembering to pass the right arguments.
"""
from __future__ import annotations

from typing import Optional

from src import phase1_checkpoint
from src.execution import exits as exits_mod
from src.execution import sizing as sizing_mod
from src.execution import ticket as ticket_mod


def live_enabled(config: dict) -> bool:
    return bool((config or {}).get("live_execution", {}).get("enabled", False))


def current_gate(db_path: str, phase1_start: str) -> str:
    """The live gate decision from the checkpoint logic (always current)."""
    return phase1_checkpoint.compute_checkpoint(db_path, phase1_start)["decision"]


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
    gate = current_gate(db_path, phase1_start)
    return ticket_mod.render_ticket(pick, s, e, gate_decision=gate,
                                    live_enabled=live_enabled(config))


def arm_status(db_path: str, config: dict, phase1_start: str) -> dict:
    """Is live execution armed? Reports gate, flag, and the combined verdict."""
    gate = current_gate(db_path, phase1_start)
    flag = live_enabled(config)
    return {
        "gate": gate,
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
    with open(args.config) as f:
        cfg = json.load(f)
    p1 = (cfg.get("auto_log") or {}).get("phase1_start_date")
    st = arm_status(args.db, cfg, p1)
    print("Phase 3 execution —", "ARMED ✅" if st["armed"] else "DISARMED 🔒")
    print(f"  gate: {st['gate']}")
    print(f"  live_execution.enabled: {st['live_enabled']}")
    if st["blockers"]:
        print("  blockers:")
        for b in st["blockers"]:
            print(f"    • {b}")
    print("\nThis is mirror-mode only: the system prints a ticket, you place it. "
          "No broker API.")


if __name__ == "__main__":
    main()
