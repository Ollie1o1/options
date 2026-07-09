"""CLI: re-render a saved tearsheet snapshot offline.

    python -m src.tearsheet --from reports/tearsheets/NVDA_190C_20260717.json

It deliberately takes no --pick: a pick exists only inside a live scan, and
selecting one belongs to the options_screener CLI.
"""
import argparse
import json
import os
import sys
import webbrowser

from .render import render


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.tearsheet")
    ap.add_argument("--from", dest="src", required=True,
                    help="path to a tearsheet sidecar .json")
    ap.add_argument("--no-open", action="store_true", help="write only, do not open")
    args = ap.parse_args(argv)

    with open(args.src) as f:
        data = json.load(f)
    out = os.path.splitext(args.src)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(data))
    print("wrote {}".format(out))
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
