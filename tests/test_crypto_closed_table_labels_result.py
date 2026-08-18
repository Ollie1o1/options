"""The crypto closed-positions table prints a column its header does not name.

The row ends `{reason:<22}  {result}`, where `result` is WIN / LOSS / FLAT. The
header stops at `Reason`. The text happens to label itself, so this is the mild
end of the defect — but it is the same split, and a reader scanning for column
boundaries has no anchor for the last one.

Rows arrive as `sqlite3.Row`; plain dicts satisfy the same `r["key"]` access.
"""
import io
import unittest
from contextlib import redirect_stdout

from src.crypto.check_pnl import _print_closed


def _closed(pnl_usd):
    return {
        "ticker": "btc", "type": "call", "strike": 90000.0,
        "expiration": "2026-09-26", "date": "2026-08-01",
        "exit_date": "2026-08-15 14:00:00", "entry_price": 1200.0,
        "exit_price": 1500.0, "pnl_usd": pnl_usd, "pnl_pct": 25.0,
        "exit_reason": "target hit",
    }


def _render(rows):
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_closed(rows, width=120)
    return buf.getvalue()


class ClosedTableTest(unittest.TestCase):
    def test_the_result_column_is_named_in_the_header(self):
        out = _render([_closed(300.0)])
        header = next(ln for ln in out.splitlines() if "Ticker" in ln)
        self.assertIn("Result", header,
                      "the row prints WIN/LOSS/FLAT under no header")

    def test_a_winner_still_renders_its_result(self):
        self.assertIn("WIN", _render([_closed(300.0)]))

    def test_a_loser_still_renders_its_result(self):
        self.assertIn("LOSS", _render([_closed(-300.0)]))

    def test_the_existing_columns_survive(self):
        out = _render([_closed(300.0)])
        for label in ("Ticker", "Strike", "Expiry", "P/L $", "Reason"):
            self.assertIn(label, out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
