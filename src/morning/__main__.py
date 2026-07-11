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
    print(html_path)
    print(json_path)
    if args.open:
        _open_file(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
