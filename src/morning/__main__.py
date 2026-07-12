"""Build (or re-render) the morning briefing.

    python -m src.morning              # fetch, write reports/briefings/<date>.{html,json}
    python -m src.morning --open       # ...and open in the browser
    python -m src.morning --from reports/briefings/2026-07-10.json   # offline re-render
"""
import argparse
import json
import os
import subprocess
import sys


def _open_file(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:  # pragma: no cover
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(path))


def _mark_maintenance_state(date_s: str) -> None:
    """A fresh build satisfies the daily job regardless of who triggered it
    (heartbeat, INTEL menu, CLI) — record it so the health banner stays quiet.
    Offline --from re-renders don't count. Never raises."""
    try:
        from src.maintenance import DEFAULT_STATE_PATH, load_state, save_state
        state = load_state(DEFAULT_STATE_PATH)
        state["last_morning_briefing"] = date_s
        save_state(DEFAULT_STATE_PATH, state)
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.morning")
    ap.add_argument("--open", action="store_true", help="open the page when written")
    ap.add_argument("--from", dest="src", default=None,
                    help="re-render offline from an existing JSON sidecar")
    ap.add_argument("--out-dir", default="reports/briefings")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="fetch budget in seconds")
    args = ap.parse_args(argv)

    from src.morning import write_briefing
    if args.src:
        try:
            with open(args.src) as f:
                data = json.load(f)
        except OSError as exc:
            print(f"cannot read sidecar: {exc}", file=sys.stderr)
            return 2
    else:
        from src.morning.collect import build
        data = build(slow=True, budget_s=args.budget)

    html_path, json_path = write_briefing(data, out_dir=args.out_dir)
    if not args.src:
        _mark_maintenance_state(data["meta"]["date"])
    print(html_path)
    print(json_path)
    if args.open:
        _open_file(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
