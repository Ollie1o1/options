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

from .collect import SCHEMA
from .render import render


def _check_schema(data) -> None:
    """Say so when the sidecar was not written by this renderer.

    `render` reads its keys directly; a sidecar from another version can still
    produce a page, but one with holes in it. A quiet partial render is the one
    failure mode this package exists to prevent.
    """
    found = (data.get("meta") or {}).get("schema")
    if found == SCHEMA:
        return
    where = "no schema" if found is None else "schema {}".format(found)
    print("warning: sidecar declares {}, this renderer writes schema {} — "
          "panels may be missing or stale".format(where, SCHEMA))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.tearsheet")
    ap.add_argument("--from", dest="src", required=True,
                    help="path to a tearsheet sidecar .json")
    ap.add_argument("--no-open", action="store_true", help="write only, do not open")
    args = ap.parse_args(argv)

    with open(args.src) as f:
        data = json.load(f)
    _check_schema(data)
    out = os.path.splitext(args.src)[0] + ".html"
    with open(out, "w") as f:
        f.write(render(data))
    print("wrote {}".format(out))
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
