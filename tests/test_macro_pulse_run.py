"""End-to-end orchestration test for macro_pulse — fully offline."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.macro_pulse import orchestrator as O


_ITEMS = [
    {"title": "Iran strikes escalate", "source": "reuters.com",
     "published": None, "url": "u1"},
    {"title": "Powell signals patience on rate cuts", "source": "wsj.com",
     "published": None, "url": "u2"},
    {"title": "Nvidia unveils new AI chip", "source": "cnbc.com",
     "published": None, "url": "u3"},
]


class _FakeScorer:
    calls = 0

    def safe_chat_complete(self, system, user, max_tokens=400):
        _FakeScorer.calls += 1
        return None  # force deterministic, but lets us count AI attempts


class RunTest(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.db = os.path.join(d, "hist.db")
        self.cache = os.path.join(d, "cache.db")
        _FakeScorer.calls = 0

    def test_run_renders_and_persists(self):
        out = O.run(fetch_fn=lambda: list(_ITEMS), scorer=_FakeScorer(),
                    db_path=self.db, cache_db=self.cache, today="2026-06-18")
        self.assertIn("MACRO PULSE", out.upper())
        self.assertIn("geopolitics", out)
        # one reading persisted
        from src.macro_pulse import context as C
        self.assertEqual(len(C.load_history(self.db)), 1)

    def test_second_run_same_day_hits_cache_no_ai(self):
        O.run(fetch_fn=lambda: list(_ITEMS), scorer=_FakeScorer(),
              db_path=self.db, cache_db=self.cache, today="2026-06-18")
        first_calls = _FakeScorer.calls
        out2 = O.run(fetch_fn=lambda: list(_ITEMS), scorer=_FakeScorer(),
                     db_path=self.db, cache_db=self.cache, today="2026-06-18")
        self.assertIn("MACRO PULSE", out2.upper())
        self.assertEqual(_FakeScorer.calls, first_calls)  # no new AI call
        # cache hit must NOT persist a second reading
        from src.macro_pulse import context as C
        self.assertEqual(len(C.load_history(self.db)), 1)

    def test_run_ticker_with_prebuilt_ctx_costs_no_ai(self):
        # Build once (one AI attempt), then a ticker read reusing that ctx
        ctx = O.build_context(fetch_fn=lambda: list(_ITEMS), scorer=_FakeScorer(),
                              db_path=self.db, cache_db=self.cache,
                              today="2026-06-18")
        calls_after_build = _FakeScorer.calls
        out = O.run_ticker("AAPL", sector="Technology", ctx=ctx)
        self.assertIn("AAPL", out)
        self.assertIn("Technology", out)
        self.assertEqual(_FakeScorer.calls, calls_after_build)  # no extra AI

    def test_run_ticker_discovery_is_general(self):
        ctx = O.build_context(fetch_fn=lambda: list(_ITEMS), scorer=_FakeScorer(),
                              db_path=self.db, cache_db=self.cache,
                              today="2026-06-18")
        out = O.run_ticker(None, ctx=ctx)
        self.assertIn("MARKET", out.upper())


if __name__ == "__main__":
    unittest.main()
