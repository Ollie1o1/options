"""Run the fixed channel study from local data; no network or trading writes.

    python -m src.discovery_research --no-open
    python -m src.discovery_research --view
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.paths import repo_path
from . import data as D
from . import engine as E
from .report import render


def run(config_path: str, prices_path: str, news_path: str, analysts_path: str,
        vix_path: str, output_dir: str) -> dict:
    spec_bytes = Path(config_path).read_bytes()
    spec = json.loads(spec_bytes)
    settings = spec["evaluation"]
    prices, price_coverage = D.load_prices(prices_path, spec["universe"], settings["start"])
    if prices.empty:
        raise ValueError("No price observations available; cannot run this study")
    vix, vix_coverage = D.load_cboe_csv(vix_path)
    news, news_coverage = D.load_news(news_path, spec["universe"])
    analysts, analyst_coverage = D.load_analysts(analysts_path, spec["universe"])
    features = D.make_features(prices, vix, news, analysts, spec["peer_groups"])
    results = []
    for trial in spec["trials"]:
        dataset = D.make_dataset(prices, features, trial, spec["peer_groups"], settings["entry_lag_sessions"])
        results.append(E.evaluate(dataset, list(prices.index), trial, settings))
    results = E.finalize_family(results, settings)
    price_coverage["selected_data_sha256"] = hashlib.sha256(prices.to_csv().encode()).hexdigest()
    coverage = {"prices": price_coverage, "vix": vix_coverage, "news": news_coverage, "analysts": analyst_coverage}
    for name in ("VIX3M", "VVIX", "SKEW"):
        _, coverage[name] = D.load_cboe_csv(str(Path(vix_path).parent / f"{name}_History.csv"))
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    stamp = datetime.now(timezone.utc).isoformat()
    report = {"schema": 1, "generated_at": stamp, "preregistration_id": spec["id"],
              "preregistration_sha256": hashlib.sha256(spec_bytes).hexdigest(),
              "source_code_sha256": code_hashes, "evaluation": settings,
              "coverage": coverage, "limitations": spec["limitations"], "results": results,
              "relationships": D.relationship_snapshot(prices, spec["peer_groups"]),
              "deployment": "Research only; no scoring, gate or execution changes."}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Save all trials and audits, not just the attractive result.
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    html = render(report)
    base = "discovery_" + stamp[:10].replace("-", "")
    for name in (base, "latest"):
        (output / f"{name}.json").write_text(encoded)
        (output / f"{name}.html").write_text(html)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=repo_path("configs/discovery_research_v1.json"))
    parser.add_argument("--prices", default=repo_path("data/equity_ohlcv.db"))
    parser.add_argument("--news", default=repo_path("data/news_archive.db"))
    parser.add_argument("--analysts", default=repo_path("data/estimates_archive.db"))
    parser.add_argument("--vix", default=repo_path("data/discovery/VIX_History.csv"))
    parser.add_argument("--output", default=repo_path("reports/discovery"))
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--view", action="store_true", help="open saved findings and audit local coverage; no hypothesis evaluation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--coverage", action="store_true", help="audit local coverage only; no hypothesis evaluation")
    mode.add_argument("--update-prices", action="store_true", help="refresh the existing equity archive, then report coverage only")
    args = parser.parse_args(argv)
    if args.view and (args.coverage or args.update_prices):
        parser.error("--view cannot be combined with --coverage or --update-prices")
    if args.coverage or args.update_prices:
        from .collection import collect
        try:
            result = collect(config_path=args.config, prices_path=args.prices, news_path=args.news,
                             analysts_path=args.analysts, output_dir=args.output, update=args.update_prices)
        except (OSError, ValueError, sqlite3.Error) as exc:
            print(f"Coverage unavailable: {exc}")
            return 1
        for row in result["coverage"]["prices"]["symbols"]:
            print(f"{row['symbol']}: {row['status']} | {row['first']} to {row['last']} | "
                  f"{len(row['missing_spy_dates'])} missing SPY dates | {len(row['large_moves'])} large moves")
            if (row.get("last_attempt") or {}).get("error"):
                print("  Collection: " + row["last_attempt"]["error"])
        path = Path(args.output) / "coverage.html"
        print(path)
        if not args.no_open:
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
        return 0 if result["complete"] else 1
    path = Path(args.output) / "latest.html"
    if args.view:
        # Refresh only the linked coverage audit. Saved findings keep their
        # original sample and remain available even when local archives are absent.
        from .collection import collect
        try:
            collect(config_path=args.config, prices_path=args.prices, news_path=args.news,
                    analysts_path=args.analysts, output_dir=args.output, update=False)
        except (OSError, ValueError, sqlite3.Error) as exc:
            print(f"Coverage audit unavailable: {exc}")
        if not path.exists():
            snapshot = Path(repo_path("docs/discovery/research_20261124.json"))
            if not snapshot.exists():
                print("No saved discovery research yet. Run python -m src.discovery_research --no-open first.")
                return 1
            saved = json.loads(snapshot.read_text())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(saved))
            print("Opening the dated, packaged research snapshot; this is not a live market update.")
    else:
        try:
            report = run(args.config, args.prices, args.news, args.analysts, args.vix, args.output)
        except (OSError, ValueError, sqlite3.Error) as exc:
            print(f"Research unavailable: {exc}")
            return 1
        for r in report["results"]:
            gain = r.get("mse_improvement_pct")
            text = "not measurable" if gain is None else f"MSE improvement {gain:+.2f}%"
            print(f"{r['id']}: {r['status']} | {r['test_dates']} non-overlapping test dates | {text}")
    print(path)
    if not args.no_open:
        import webbrowser
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
