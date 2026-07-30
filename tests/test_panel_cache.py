"""Cache-first rendering for startup panels.

Time-to-menu was ~11s, almost all of it live market fetches: the world pulse
(3.54s) and the VIX/regime fetch (1.09s) re-fetch on every launch. The
sector-outlook box already solves this by serving from cache and refreshing
behind the user; this generalises that pattern.

Two rules the design must not break:

1. **Never present stale market data as live.** Every cached render carries the
   time it was produced so the caller can stamp "as of HH:MM".
2. **Never render a panel from a background thread.** `redirect_stdout` swaps
   the *global* `sys.stdout`, so a thread capturing a panel while the main
   thread prints the menu swallows the menu — the blank-UI race this repo
   already hit once. Background refresh is therefore a detached subprocess, and
   these tests inject the spawn so nothing is launched under test.
"""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta

from src.panel_cache import render_cached, load_panel, store_panel


class PanelCacheTestBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.now = datetime(2026, 7, 30, 14, 30, 0)
        self.spawned = []

    def spawn(self, key, width):
        self.spawned.append((key, width))


class ColdCache(PanelCacheTestBase):
    def test_cold_cache_renders_synchronously_and_returns_the_text(self):
        text, asof, from_cache = render_cached(
            "pulse", 100, lambda: print("PULSE BODY"), ttl=300,
            cache_dir=self.dir, now=self.now, spawn_refresh=self.spawn,
        )
        self.assertIn("PULSE BODY", text)
        self.assertEqual(asof, self.now)
        self.assertFalse(from_cache)

    def test_cold_render_is_stored_for_the_next_launch(self):
        render_cached("pulse", 100, lambda: print("BODY"), ttl=300,
                      cache_dir=self.dir, now=self.now, spawn_refresh=self.spawn)
        cached = load_panel("pulse", 100, cache_dir=self.dir)
        self.assertIsNotNone(cached)
        self.assertIn("BODY", cached[0])

    def test_cold_cache_does_not_spawn_a_refresh(self):
        # It just rendered live; refreshing again would be pure waste.
        render_cached("pulse", 100, lambda: print("BODY"), ttl=300,
                      cache_dir=self.dir, now=self.now, spawn_refresh=self.spawn)
        self.assertEqual(self.spawned, [])

    def test_a_renderer_that_raises_yields_empty_text_and_no_crash(self):
        def boom():
            raise RuntimeError("feed down")
        text, asof, from_cache = render_cached(
            "pulse", 100, boom, ttl=300, cache_dir=self.dir,
            now=self.now, spawn_refresh=self.spawn,
        )
        self.assertEqual(text, "")
        self.assertIsNone(asof)

    def test_a_failed_render_is_not_cached(self):
        def boom():
            raise RuntimeError("feed down")
        render_cached("pulse", 100, boom, ttl=300, cache_dir=self.dir,
                      now=self.now, spawn_refresh=self.spawn)
        self.assertIsNone(load_panel("pulse", 100, cache_dir=self.dir))


class FreshCache(PanelCacheTestBase):
    def setUp(self):
        super().setUp()
        store_panel("pulse", 100, "CACHED BODY", self.now, cache_dir=self.dir)

    def test_fresh_cache_never_calls_the_renderer(self):
        calls = []
        render_cached("pulse", 100, lambda: calls.append(1), ttl=300,
                      cache_dir=self.dir, now=self.now + timedelta(seconds=60),
                      spawn_refresh=self.spawn)
        self.assertEqual(calls, [])

    def test_fresh_cache_returns_the_stored_text_and_its_asof(self):
        text, asof, from_cache = render_cached(
            "pulse", 100, lambda: print("LIVE"), ttl=300, cache_dir=self.dir,
            now=self.now + timedelta(seconds=60), spawn_refresh=self.spawn,
        )
        self.assertEqual(text, "CACHED BODY")
        self.assertEqual(asof, self.now)
        self.assertTrue(from_cache)

    def test_fresh_cache_does_not_spawn_a_refresh(self):
        render_cached("pulse", 100, lambda: print("LIVE"), ttl=300,
                      cache_dir=self.dir, now=self.now + timedelta(seconds=60),
                      spawn_refresh=self.spawn)
        self.assertEqual(self.spawned, [])

    def test_age_exactly_at_the_ttl_still_counts_as_fresh(self):
        text, _, from_cache = render_cached(
            "pulse", 100, lambda: print("LIVE"), ttl=300, cache_dir=self.dir,
            now=self.now + timedelta(seconds=300), spawn_refresh=self.spawn,
        )
        self.assertTrue(from_cache)
        self.assertEqual(text, "CACHED BODY")


class StaleCache(PanelCacheTestBase):
    def setUp(self):
        super().setUp()
        store_panel("pulse", 100, "OLD BODY", self.now, cache_dir=self.dir)
        self.later = self.now + timedelta(seconds=301)

    def test_stale_cache_still_returns_instantly_from_cache(self):
        # The whole point: the user never waits, even when the data has aged.
        text, asof, from_cache = render_cached(
            "pulse", 100, lambda: print("LIVE"), ttl=300, cache_dir=self.dir,
            now=self.later, spawn_refresh=self.spawn,
        )
        self.assertEqual(text, "OLD BODY")
        self.assertEqual(asof, self.now)
        self.assertTrue(from_cache)

    def test_stale_cache_does_not_render_inline(self):
        calls = []
        render_cached("pulse", 100, lambda: calls.append(1), ttl=300,
                      cache_dir=self.dir, now=self.later, spawn_refresh=self.spawn)
        self.assertEqual(calls, [])

    def test_stale_cache_spawns_exactly_one_background_refresh(self):
        render_cached("pulse", 100, lambda: print("LIVE"), ttl=300,
                      cache_dir=self.dir, now=self.later, spawn_refresh=self.spawn)
        self.assertEqual(self.spawned, [("pulse", 100)])


class CacheKeying(PanelCacheTestBase):
    def test_a_different_width_is_a_separate_entry(self):
        # A panel rendered at width 100 is garbage when replayed at width 80.
        store_panel("pulse", 100, "WIDE", self.now, cache_dir=self.dir)
        self.assertIsNone(load_panel("pulse", 80, cache_dir=self.dir))

    def test_a_different_panel_key_is_a_separate_entry(self):
        store_panel("pulse", 100, "PULSE", self.now, cache_dir=self.dir)
        self.assertIsNone(load_panel("regime", 100, cache_dir=self.dir))

    def test_ansi_escapes_survive_the_round_trip(self):
        colored = "\x1b[32mGREEN\x1b[0m"
        store_panel("regime", 100, colored, self.now, cache_dir=self.dir)
        self.assertEqual(load_panel("regime", 100, cache_dir=self.dir)[0], colored)


class CorruptCache(PanelCacheTestBase):
    def test_unparseable_cache_file_is_treated_as_a_miss(self):
        store_panel("pulse", 100, "BODY", self.now, cache_dir=self.dir)
        path = [os.path.join(self.dir, f) for f in os.listdir(self.dir)][0]
        with open(path, "w") as f:
            f.write("{not json")
        self.assertIsNone(load_panel("pulse", 100, cache_dir=self.dir))

    def test_corrupt_cache_falls_back_to_a_live_render(self):
        store_panel("pulse", 100, "BODY", self.now, cache_dir=self.dir)
        path = [os.path.join(self.dir, f) for f in os.listdir(self.dir)][0]
        with open(path, "w") as f:
            f.write("{not json")
        text, _, from_cache = render_cached(
            "pulse", 100, lambda: print("LIVE"), ttl=300, cache_dir=self.dir,
            now=self.now, spawn_refresh=self.spawn,
        )
        self.assertIn("LIVE", text)
        self.assertFalse(from_cache)

    def test_cache_entry_missing_its_timestamp_is_a_miss(self):
        path = os.path.join(self.dir, "pulse_100.json")
        with open(path, "w") as f:
            json.dump({"text": "BODY"}, f)
        self.assertIsNone(load_panel("pulse", 100, cache_dir=self.dir))


class RefreshSubprocessRegistry(PanelCacheTestBase):
    """The refresh subprocess runs `python -m src.panel_cache`, which imports
    ONLY this module. A registry populated by importers is therefore empty
    there, and every background refresh silently does nothing — the cache would
    freeze at its first render forever. Renderers must be resolvable from
    src.panel_cache alone."""

    def test_every_panel_the_app_caches_is_resolvable_here(self):
        from src.panel_cache import panel_renderer
        for key in ("regime_dashboard",):
            self.assertIsNotNone(
                panel_renderer(key),
                f"{key!r} has no renderer reachable from src.panel_cache; "
                "its background refresh would be a silent no-op",
            )

    def test_an_unknown_key_has_no_renderer(self):
        from src.panel_cache import panel_renderer
        self.assertIsNone(panel_renderer("not_a_panel"))

    def test_refresh_with_an_unknown_key_writes_nothing(self):
        from src.panel_cache import main
        main(["--refresh", "not_a_panel", "--width", "100"])
        self.assertEqual(os.listdir(self.dir), [])


class AtomicWrite(PanelCacheTestBase):
    def test_store_leaves_no_temp_files_behind(self):
        # A refresh subprocess can die mid-write; the write must be rename-based
        # so a half-written file never becomes the cache.
        store_panel("pulse", 100, "BODY", self.now, cache_dir=self.dir)
        leftovers = [f for f in os.listdir(self.dir) if not f.endswith(".json")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
