"""Tests for src/worldnews — trust-weighted market pulse (all offline)."""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.worldnews import scoring, sources


def _item(title, domain="reuters.com", age_hours=1.0, now=None):
    now = now or datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    return {"title": title, "source": domain,
            "published": now - timedelta(hours=age_hours), "url": "http://x"}


NOW = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


class ThemeTest(unittest.TestCase):
    def test_fed_theme(self):
        self.assertEqual(scoring.classify_theme("Federal Reserve holds rates steady"),
                         "fed_rates")

    def test_geopolitics(self):
        self.assertEqual(scoring.classify_theme("Strikes escalate in Middle East war"),
                         "geopolitics")

    def test_tariffs(self):
        self.assertEqual(scoring.classify_theme("New tariffs on chip imports announced"),
                         "trade_tariffs")

    def test_other(self):
        self.assertEqual(scoring.classify_theme("Local sports team wins title"), "other")


class TrustTest(unittest.TestCase):
    def test_wire_services_highest(self):
        self.assertGreater(scoring.source_trust("reuters.com"),
                           scoring.source_trust("cnbc.com"))

    def test_social_lowest(self):
        self.assertLess(scoring.source_trust("stocktwits.com"),
                        scoring.source_trust("marketwatch.com"))

    def test_unknown_mid(self):
        self.assertAlmostEqual(scoring.source_trust("randomblog.biz"), 0.5)


class RecencyTest(unittest.TestCase):
    def test_halves_per_day(self):
        w0 = scoring.recency_weight(_item("x", age_hours=0)["published"], NOW)
        w24 = scoring.recency_weight(_item("x", age_hours=24)["published"], NOW)
        self.assertAlmostEqual(w24 / w0, 0.5, places=2)

    def test_missing_timestamp_gets_low_weight(self):
        self.assertLessEqual(scoring.recency_weight(None, NOW), 0.5)


class AggregateTest(unittest.TestCase):
    def test_positive_news_positive_pulse(self):
        items = [_item("Stocks rally as growth beats expectations"),
                 _item("Markets surge on strong earnings")]
        r = scoring.aggregate(items, now=NOW)
        self.assertGreater(r["pulse"], 0)
        self.assertGreater(r["bull_pct"], 0.5)

    def test_negative_news_negative_pulse(self):
        items = [_item("Stocks plunge as recession fears mount"),
                 _item("Crash deepens amid war escalation")]
        r = scoring.aggregate(items, now=NOW)
        self.assertLess(r["pulse"], 0)
        self.assertGreater(r["bear_pct"], 0.5)

    def test_trusted_source_outweighs_social(self):
        items = [_item("Markets collapse in historic crash", "reuters.com"),
                 _item("stocks mooning, rally rally rally", "stocktwits.com")]
        r = scoring.aggregate(items, now=NOW)
        self.assertLess(r["pulse"], 0)

    def test_confidence_grows_with_items_and_diversity(self):
        few = scoring.aggregate([_item("Stocks rally hard")], now=NOW)
        many = scoring.aggregate(
            [_item(f"Stocks rally on day {i}", d) for i, d in enumerate(
                ["reuters.com", "cnbc.com", "marketwatch.com", "apnews.com",
                 "wsj.com", "bloomberg.com"] * 5)], now=NOW)
        self.assertGreater(many["confidence"], few["confidence"])

    def test_empty_items(self):
        r = scoring.aggregate([], now=NOW)
        self.assertEqual(r["n_items"], 0)
        self.assertEqual(r["pulse"], 0.0)
        self.assertEqual(r["confidence"], 0)

    def test_themes_populated(self):
        items = [_item("Fed signals rate cut optimism"),
                 _item("Tariff war escalates, markets fall")]
        r = scoring.aggregate(items, now=NOW)
        self.assertIn("fed_rates", r["themes"])
        self.assertIn("trade_tariffs", r["themes"])


class InflationInversionTest(unittest.TestCase):
    def test_hot_inflation_is_market_negative(self):
        s = scoring.headline_sentiment(
            "Producer prices rise at fastest pace in three years",
            theme="inflation")
        self.assertLess(s, 0)

    def test_cooling_inflation_is_market_positive(self):
        s = scoring.headline_sentiment(
            "CPI cools more than expected in May", theme="inflation")
        self.assertGreater(s, 0)

    def test_non_inflation_theme_untouched(self):
        up = scoring.headline_sentiment("Stocks rise on strong earnings",
                                        theme="earnings_tech")
        self.assertGreater(up, 0)


GOOGLE_RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Fed holds rates - Reuters</title>
<link>https://news.google.com/x</link>
<pubDate>Thu, 11 Jun 2026 10:00:00 GMT</pubDate>
<source url="https://www.reuters.com">Reuters</source></item>
<item><title>Markets slip on tariff fears</title>
<link>https://news.google.com/y</link>
<pubDate>Thu, 11 Jun 2026 09:00:00 GMT</pubDate>
<source url="https://www.cnbc.com">CNBC</source></item>
</channel></rss>"""

STOCKTWITS_JSON = {
    "messages": [
        {"entities": {"sentiment": {"basic": "Bullish"}}},
        {"entities": {"sentiment": {"basic": "Bearish"}}},
        {"entities": {"sentiment": {"basic": "Bullish"}}},
        {"entities": {"sentiment": None}},
    ]
}


class SourceParseTest(unittest.TestCase):
    def test_parse_rss_extracts_true_source(self):
        items = sources.parse_rss(GOOGLE_RSS)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["source"], "reuters.com")
        self.assertEqual(items[1]["source"], "cnbc.com")
        self.assertIn("Fed holds rates", items[0]["title"])
        self.assertIsNotNone(items[0]["published"])

    def test_parse_rss_garbage(self):
        self.assertEqual(sources.parse_rss("<not-xml"), [])

    def test_parse_rss_rejects_dtd(self):
        self.assertEqual(sources.parse_rss("<!DOCTYPE foo []><rss/>"), [])

    def test_stocktwits_bull_ratio(self):
        r = sources.parse_stocktwits(STOCKTWITS_JSON)
        self.assertEqual(r["tagged"], 3)
        self.assertAlmostEqual(r["bull_ratio"], 2 / 3)


class TopicsTest(unittest.TestCase):
    def test_policy_topics_present(self):
        topics = [t.lower() for t in sources.GOOGLE_NEWS_TOPICS]
        # geopolitics already covered; assert the new policy/political coverage
        self.assertIn("white house policy", topics)
        self.assertTrue(any("tariff" in t for t in topics))
        self.assertTrue(any("middle east" in t for t in topics))


if __name__ == "__main__":
    unittest.main()
