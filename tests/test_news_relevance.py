"""Headline relevance matching — pure, offline.

The original matcher was a plain substring test (`symbol.lower() in
headline.lower()`), which inverted the ranking for the most-scanned names:

    F    "Fed cuts rates as inflation cools"  -> 1.0  (noise, top tier)
    T    "Tesla recalls vehicles"             -> 1.0  (wrong company)
    MSFT "Microsoft earnings beat"            -> 0.6  (real news, demoted)

Relevance feeds both the display sort (_rank_key) and the relevance-weighted
aggregate sentiment, so a false 1.0 corrupts what the panel reports.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_news_relevance -v
"""
from __future__ import annotations

import unittest

from src.news_fetcher import _boost_relevance, _name_tokens


class SingleLetterTickerTests(unittest.TestCase):
    """The failure that motivated this: one-letter tickers matched everything."""

    def test_ford_does_not_match_the_word_fed(self):
        self.assertLess(_boost_relevance("Fed cuts rates as inflation cools", "F"), 1.0)

    def test_att_does_not_match_the_word_tesla(self):
        self.assertLess(_boost_relevance("Tesla recalls vehicles", "T"), 1.0)

    def test_citi_does_not_match_the_word_chip(self):
        self.assertLess(_boost_relevance("Chip stocks slide on China fears", "C"), 1.0)

    def test_ford_matches_its_own_standalone_symbol(self):
        self.assertEqual(_boost_relevance("Ford Motor (F) recalls trucks", "F"), 1.0)


class TickerTokenTests(unittest.TestCase):
    def test_standalone_uppercase_ticker_matches(self):
        self.assertEqual(_boost_relevance("AAPL upgraded at Morgan Stanley", "AAPL"), 1.0)

    def test_ticker_inside_a_larger_word_does_not_match(self):
        # "CAT" must not match "CATALYST"
        self.assertLess(_boost_relevance("CATALYST trial results due", "CAT"), 1.0)

    def test_lowercase_prose_word_does_not_match_ticker(self):
        # "IT" (Gartner) must not match the ordinary word "it"
        self.assertLess(_boost_relevance("Analysts say it will recover", "IT"), 1.0)


class CompanyNameTests(unittest.TestCase):
    """Headlines name the company, not the symbol — that was the demotion bug."""

    def test_company_name_matches_even_without_the_symbol(self):
        self.assertEqual(
            _boost_relevance("Microsoft earnings beat", "MSFT", "Microsoft Corporation"),
            1.0)

    def test_company_name_match_is_case_insensitive(self):
        self.assertEqual(
            _boost_relevance("apple unveils new chip", "AAPL", "Apple Inc."), 1.0)

    def test_unrelated_headline_with_name_supplied_stays_low(self):
        self.assertLess(
            _boost_relevance("Fed cuts rates", "MSFT", "Microsoft Corporation"), 1.0)

    def test_generic_suffixes_alone_do_not_match(self):
        # "Inc"/"Corp"/"Holdings" are not distinctive; a headline mentioning
        # some other "Inc" must not match.
        self.assertLess(
            _boost_relevance("Acme Inc files for bankruptcy", "MSFT",
                             "Microsoft Corporation"),
            1.0)


class NameTokenTests(unittest.TestCase):
    def test_strips_corporate_suffixes(self):
        self.assertEqual(_name_tokens("Microsoft Corporation"), ["microsoft"])

    def test_keeps_multiword_distinctive_names(self):
        self.assertIn("goldman", _name_tokens("The Goldman Sachs Group, Inc."))
        self.assertIn("sachs", _name_tokens("The Goldman Sachs Group, Inc."))

    def test_drops_short_tokens(self):
        self.assertNotIn("co", _name_tokens("3M Co"))

    def test_handles_empty_name(self):
        self.assertEqual(_name_tokens(""), [])
        self.assertEqual(_name_tokens(None), [])


class BackCompatTests(unittest.TestCase):
    def test_two_argument_call_still_works(self):
        self.assertEqual(_boost_relevance("AAPL rises", "AAPL"), 1.0)

    def test_irrelevant_headline_keeps_the_old_floor(self):
        self.assertAlmostEqual(_boost_relevance("Markets open higher", "AAPL"), 0.6)
