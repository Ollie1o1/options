"""Offline unit tests for src.radar (playbooks, scanners, confluence engine, cache, dossier)."""
from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.radar import (
    Opportunity,
    build_dossier,
    load_cached_radar,
    load_playbooks,
    run_radar,
    save_radar_cache,
)
from src.radar.engine import merge_confluence


def _make_mock_ohlcv_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE ohlcv (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    # Build 270 synthetic trading days
    import datetime as dt

    d0 = dt.date(2025, 10, 1)
    dates = []
    cur = d0
    while len(dates) < 270:
        if cur.weekday() < 5:
            dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)

    rows = []
    # Steady uptrending SPY + 4 ETFs
    for sym, drift in [("SPY", 0.0005), ("XLK", 0.0009), ("SMH", 0.0011), ("GLD", 0.0007), ("QQQ", 0.0008), ("TLT", -0.0002)]:
        px = 100.0
        for i, d in enumerate(dates):
            px *= 1.0 + drift + (0.003 if i % 2 == 0 else -0.002)
            rows.append((sym, d, px, px * 1.01, px * 0.99, px, 20_000_000.0))

    # AAPL: strong uptrend then sharp 3-day pullback -> triggers strict Playbook 1!
    px = 150.0
    for i, d in enumerate(dates):
        if i < 267:
            px *= 1.0012
        else:
            px *= 0.975  # 3-day ~ -7.3% pullback, RSI(2) near 0, still > SMA200
        rows.append(("AAPL", d, px, px * 1.01, px * 0.99, px, 5_000_000.0))

    con.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()


class TestRadarPackage(unittest.TestCase):
    def test_load_playbooks_excludes_closed_from_default(self) -> None:
        reg = load_playbooks("configs/playbooks.json")
        self.assertIn("vol_managed_outlook", reg)
        self.assertEqual(reg["vol_managed_outlook"].evidence_tier, "VALIDATED")
        self.assertTrue(reg["vol_managed_outlook"].is_default_radar_eligible)
        self.assertEqual(reg["squeeze_equity"].evidence_tier, "CLOSED")
        self.assertFalse(reg["squeeze_equity"].is_default_radar_eligible)

    def test_merge_confluence_and_deterministic_sort(self) -> None:
        a = Opportunity(
            ticker="AAPL", asset_class="stock", playbook_id="equity_pullback",
            playbook_label="Pullback", evidence_tier="LEAD", direction="LONG",
            spot=200.0, chg_1d_pct=-2.0, chg_5d_pct=-4.0, horizon_label="3-10d",
            setup_summary="RSI2=4", badges=(">200MA",), risk_flags=(),
            friction_label="0.10%", friction_value=0.001,
        )
        b = Opportunity(
            ticker="AAPL", asset_class="stock", playbook_id="holdings_buy_zone",
            playbook_label="Holdings", evidence_tier="CONTEXT", direction="LONG",
            spot=200.0, chg_1d_pct=-2.0, chg_5d_pct=-4.0, horizon_label="60-365d",
            setup_summary="IN_ZONE", badges=("HOLDING",), risk_flags=(),
            friction_label="0.05%", friction_value=0.0005,
        )
        v = Opportunity(
            ticker="SMH", asset_class="etf", playbook_id="vol_managed_outlook",
            playbook_label="VM Outlook", evidence_tier="VALIDATED", direction="LONG",
            spot=250.0, chg_1d_pct=1.0, chg_5d_pct=3.0, horizon_label="21-63d",
            setup_summary="Rank #1", badges=("FAVOR:SMH",), risk_flags=(),
            friction_label="0.06%", friction_value=0.0006,
        )
        merged = merge_confluence([b, a, v])
        # VALIDATED first, then LEAD (with confluence=2), then CONTEXT (confluence=2)
        self.assertEqual([x.evidence_tier for x in merged], ["VALIDATED", "LEAD", "CONTEXT"])
        self.assertEqual(merged[1].confluence_count, 2)
        self.assertIn("holdings_buy_zone", merged[1].confluence_tags)

    def test_run_radar_timeout_isolation_and_cache_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "eq.db"
            cache = Path(td) / "radar.json"
            _make_mock_ohlcv_db(db)

            def _exploding_adapter():
                raise RuntimeError("boom")

            def _slow_adapter():
                time.sleep(0.5)
                return []

            snap = run_radar(
                ohlcv_db=str(db),
                cache_path=str(cache),
                timeout_s=0.1,
                extra_adapters={"bad": _exploding_adapter, "slow": _slow_adapter},
            )
            self.assertGreater(len(snap.core_allocation), 0)
            self.assertTrue(any(o.ticker == "AAPL" and o.playbook_id == "equity_pullback" for o in snap.opportunities))
            self.assertTrue(any("bad: boom" in d for d in snap.degraded_scanners))
            self.assertTrue(any("slow: timeout" in d for d in snap.degraded_scanners))
            # Never leak CLOSED playbooks
            self.assertTrue(all(o.evidence_tier != "CLOSED" for o in snap.opportunities))

            reloaded = load_cached_radar(str(cache))
            self.assertIsNotNone(reloaded)
            self.assertTrue(reloaded.from_cache)
            self.assertEqual(len(reloaded.opportunities), len(snap.opportunities))

            # Dossier check
            dos = build_dossier("AAPL", ohlcv_db=str(db), cache_path=str(cache), fetch_chain=False)
            self.assertEqual(dos.ticker, "AAPL")
            self.assertIsNotNone(dos.spot)
            self.assertLess(dos.rsi_2, 10.0)
            self.assertGreater(len(dos.active_signals), 0)


if __name__ == "__main__":
    unittest.main()
