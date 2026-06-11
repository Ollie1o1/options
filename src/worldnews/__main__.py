"""CLI: python -m src.worldnews [--topics "..." ...]"""
from __future__ import annotations

import argparse


def main() -> None:
    from src.worldnews import panel, scoring, sources

    ap = argparse.ArgumentParser(description="World-news market pulse")
    ap.add_argument("--topics", nargs="*", help="override Google News topics")
    ap.add_argument("--line", action="store_true",
                    help="print only the one-line dashboard summary")
    args = ap.parse_args()

    items = sources.fetch_all(args.topics)
    crowd = sources.fetch_crowd()
    agg = scoring.aggregate(items)
    if args.line:
        print(panel.pulse_line(agg, crowd))
    else:
        print(panel.render_full(agg, crowd))


if __name__ == "__main__":
    main()
