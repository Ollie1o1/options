"""Build (or re-render) the research desk.

    python -m src.research              # market-only desk, opens in browser
    python -m src.research NVDA         # + ticker deep-dive tab
    python -m src.research --no-open    # build without opening
    python -m src.research --json reports/research/research_20260712_0930_NVDA.json
                                        # offline re-render from a sidecar
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
    ap = argparse.ArgumentParser(prog="python -m src.research")
    ap.add_argument("symbol", nargs="?", default=None,
                    help="optional ticker for the deep-dive tab")
    ap.add_argument("--no-open", action="store_true",
                    help="do not open the page when written")
    ap.add_argument("--json", dest="src", default=None,
                    help="re-render offline from an existing JSON sidecar")
    ap.add_argument("--out-dir", default="reports/research")
    ap.add_argument("--budget", type=float, default=25.0,
                    help="fetch budget in seconds")
    args = ap.parse_args(argv)

    from src.research import write_desk
    if args.src:
        try:
            with open(args.src) as f:
                data = json.load(f)
        except OSError as exc:
            print("cannot read sidecar: {}".format(exc), file=sys.stderr)
            return 2
    else:
        from src.research.collect import build
        data = build(symbol=args.symbol, slow=True, budget_s=args.budget)

    html_path, json_path = write_desk(data, out_dir=args.out_dir)
    print(html_path)
    print(json_path)
    if not args.no_open:
        _open_file(html_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
