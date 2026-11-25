"""Offline Click CliRunner tests for the `opt` subcommand CLI (`src/cli/`)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner

from src.cli.main import cli


def _make_mock_ohlcv_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE ohlcv (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)"
    )
    d0 = dt.date(2025, 10, 1)
    dates = []
    cur = d0
    while len(dates) < 270:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    rows = []
    for sym, drift in [("SPY", 0.0005), ("XLK", 0.0009), ("SMH", 0.0011), ("GLD", 0.0007), ("QQQ", 0.0008)]:
        px = 100.0
        for i, d in enumerate(dates):
            px *= 1.0 + drift + (0.003 if i % 2 == 0 else -0.002)
            rows.append((sym, d, px, px * 1.01, px * 0.99, px, 20_000_000.0))
    px = 150.0
    for i, d in enumerate(dates):
        px *= 1.0012 if i < 267 else 0.975
        rows.append(("AAPL", d, px, px * 1.01, px * 0.99, px, 5_000_000.0))
    con.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


class TestOptCli(unittest.TestCase):
    def test_all_primary_commands_json_and_text(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            cache = Path(td) / "radar.json"
            picks_db = Path(td) / "own_picks.db"
            fills_db = Path(td) / "fills.db"
            _make_mock_ohlcv_db(db)
            obj = {
                "ohlcv_db": str(db),
                "cache_path": str(cache),
                "own_picks_db": str(picks_db),
                "fills_db": str(fills_db),
            }

            # 1. opt today --refresh --json
            r1 = runner.invoke(cli, ["--no-color", "today", "--refresh", "--json"], obj=dict(obj))
            self.assertEqual(r1.exit_code, 0, r1.output)
            p1 = json.loads(r1.output)
            self.assertIn("snapshot", p1)
            self.assertGreater(len(p1["snapshot"]["core_allocation"]), 0)

            # 2. opt radar --tier lead --json
            r2 = runner.invoke(cli, ["radar", "--tier", "lead", "--json"], obj=dict(obj))
            self.assertEqual(r2.exit_code, 0, r2.output)
            p2 = json.loads(r2.output)
            self.assertGreaterEqual(p2["count"], 1)

            # 3. opt check AAPL --json
            r3 = runner.invoke(cli, ["check", "AAPL", "--json"], obj=dict(obj))
            self.assertEqual(r3.exit_code, 0, r3.output)
            p3 = json.loads(r3.output)
            self.assertEqual(p3["ticker"], "AAPL")
            self.assertIsNotNone(p3["spot"])

            # 4. opt express AAPL --dir bull --target +8% --days 30 --owns-shares --json
            r4 = runner.invoke(
                cli,
                ["express", "AAPL", "--dir", "bull", "--target", "+8%", "--days", "30", "--owns-shares", "--json"],
                obj=dict(obj),
            )
            self.assertEqual(r4.exit_code, 0, r4.output)
            p4 = json.loads(r4.output)
            self.assertGreaterEqual(len(p4["vehicles"]), 5)

            # 5. opt log AAPL --price 200 --shares 5 --playbook equity_pullback --json
            r5 = runner.invoke(
                cli,
                ["log", "AAPL", "--price", "200", "--shares", "5", "--playbook", "equity_pullback", "--json"],
                obj=dict(obj),
            )
            self.assertEqual(r5.exit_code, 0, r5.output)
            p5 = json.loads(r5.output)
            self.assertTrue(p5["ok"])

            # 6. opt book --json
            r6 = runner.invoke(cli, ["book", "--json"], obj=dict(obj))
            self.assertEqual(r6.exit_code, 0, r6.output)
            p6 = json.loads(r6.output)
            self.assertEqual(len(p6["equity_picks"]), 1)

            # 7. opt lab health & playbooks & log-fill & fills
            r7 = runner.invoke(cli, ["lab", "health", "--json"], obj=dict(obj))
            self.assertEqual(r7.exit_code, 0, r7.output)
            self.assertIn("disk_free_gb", json.loads(r7.output))

            r8 = runner.invoke(cli, ["lab", "playbooks", "--json"], obj=dict(obj))
            self.assertEqual(r8.exit_code, 0, r8.output)
            self.assertGreaterEqual(len(json.loads(r8.output)), 8)

            r9 = runner.invoke(
                cli,
                [
                    "lab", "log-fill", "--broker", "ibkr", "--ticker", "SPY",
                    "--structure", "Bull Put", "--side", "SELL", "--bid", "1.00",
                    "--ask", "1.20", "--limit", "1.08", "--filled",
                ],
                obj=dict(obj),
            )
            self.assertEqual(r9.exit_code, 0, r9.output)
            r10 = runner.invoke(cli, ["lab", "fills", "--json"], obj=dict(obj))
            self.assertEqual(r10.exit_code, 0, r10.output)
            self.assertEqual(json.loads(r10.output)["total_filled"], 1)


if __name__ == "__main__":
    unittest.main()
