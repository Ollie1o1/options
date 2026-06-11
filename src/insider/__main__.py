"""CLI: python -m src.insider TICKER [TICKER ...] | --archive-symbols"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    from src.insider import edgar
    from src.insider.parse import parse_form4
    from src.insider.signal import cluster_score

    ap = argparse.ArgumentParser(
        description="EDGAR Form 4 insider cluster-buy scan (90d window)")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--archive-symbols", action="store_true",
                    help="scan the config.json data_archive symbols")
    ap.add_argument("--window", type=int, default=90)
    args = ap.parse_args()

    tickers = [t.upper() for t in args.tickers]
    if args.archive_symbols:
        try:
            with open("config.json") as f:
                tickers += (json.load(f).get("data_archive") or {}).get("symbols") or []
        except (OSError, ValueError):
            pass
    if not tickers:
        ap.error("give tickers or --archive-symbols")

    results = []
    for t in dict.fromkeys(tickers):          # dedupe, keep order
        cik = edgar.cik_for(t)
        if not cik:
            print(f"  {t:6s} — no CIK found, skipped")
            continue
        txs = []
        for filing in edgar.recent_form4(cik, since_days=args.window + 30):
            xml = edgar.fetch_form4_xml(cik, filing["accession"], filing["document"])
            if xml:
                txs.extend(parse_form4(xml))
        s = cluster_score(txs, window_days=args.window)
        results.append((t, s))

    results.sort(key=lambda r: -r[1]["score"])
    print(f"\nInsider activity — open-market Form 4 buys, trailing {args.window}d "
          f"(sells shown, never scored)\n")
    for t, s in results:
        line = (f"  {t:6s} {s['label']:<12s} score {s['score']:.2f} | "
                f"buyers {s['n_buyers']} | bought ${s['buy_value']:,.0f}")
        if s["sell_value"]:
            line += f" | sold ${s['sell_value']:,.0f}"
        print(line)
    print("\nCluster buys (>=2 insiders) carry multi-month evidence "
          "(Cohen-Malloy-Pomorski). Overlay only — not in scoring while the "
          "gate gathers.")


if __name__ == "__main__":
    main()
