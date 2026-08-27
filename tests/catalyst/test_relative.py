"""Board-relative superlatives. Descriptive only — never a ranking."""
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import relative


@dataclass
class FakeEvent:
    ticker: str


@dataclass
class FakeRunway:
    quarters: Optional[float] = None
    cash_generative: bool = False


@dataclass
class FakeImplied:
    move_pct: Optional[float] = None


@dataclass
class FakeAmendments:
    versions: int = 0
    available: bool = False


@dataclass
class FakeRow:
    event: FakeEvent
    runway: FakeRunway
    implied: FakeImplied
    amendments: FakeAmendments


def row(ticker, quarters=None, cash_generative=False, move_pct=None,
        versions=0, amend_available=True):
    return FakeRow(event=FakeEvent(ticker),
                   runway=FakeRunway(quarters, cash_generative),
                   implied=FakeImplied(move_pct),
                   amendments=FakeAmendments(versions, amend_available))


class TestCompute(unittest.TestCase):
    def test_finds_the_shortest_and_longest_runway(self):
        sup = relative.compute([row("AAA", quarters=2.0),
                                row("BBB", quarters=9.0),
                                row("CCC", quarters=15.0)])
        self.assertEqual(sup.shortest_runway, "AAA")
        self.assertEqual(sup.longest_runway, "CCC")

    def test_finds_the_widest_implied_move(self):
        sup = relative.compute([row("AAA", move_pct=0.08),
                                row("BBB", move_pct=0.40),
                                row("CCC", move_pct=0.15)])
        self.assertEqual(sup.widest_implied, "BBB")

    def test_finds_the_most_amended(self):
        sup = relative.compute([row("AAA", versions=4),
                                row("BBB", versions=43),
                                row("CCC", versions=17)])
        self.assertEqual(sup.most_amended, "BBB")

    def test_an_unmeasured_runway_is_never_the_shortest(self):
        # NULL means NOT RECORDED, never zero. A missing runway winning
        # "shortest" would be the exact defect this repo has paid for before.
        # Four rows so that dropping the None still leaves MIN_N measured —
        # otherwise this would pass for the wrong reason.
        sup = relative.compute([row("AAA", quarters=None),
                                row("BBB", quarters=9.0),
                                row("CCC", quarters=15.0),
                                row("DDD", quarters=4.0)])
        self.assertEqual(sup.shortest_runway, "DDD")

    def test_a_cash_generative_company_has_no_runway_to_rank(self):
        # It has no burn limit at all — it is not "the longest runway", it is
        # a different kind of object.
        sup = relative.compute([row("AAA", quarters=2.0),
                                row("BBB", quarters=9.0),
                                row("CCC", quarters=None, cash_generative=True),
                                row("DDD", quarters=5.0)])
        self.assertEqual(sup.longest_runway, "BBB")

    def test_unavailable_amendments_are_not_ranked(self):
        sup = relative.compute([row("AAA", versions=99, amend_available=False),
                                row("BBB", versions=4),
                                row("CCC", versions=9),
                                row("DDD", versions=2)])
        self.assertEqual(sup.most_amended, "CCC")

    def test_fewer_than_three_measurements_claims_nothing(self):
        # "The shortest of two" is noise, not a description of a spread.
        sup = relative.compute([row("AAA", quarters=2.0),
                                row("BBB", quarters=9.0)])
        self.assertIsNone(sup.shortest_runway)
        self.assertIsNone(sup.longest_runway)

    def test_a_tie_claims_nothing(self):
        sup = relative.compute([row("AAA", versions=43),
                                row("BBB", versions=43),
                                row("CCC", versions=9)])
        self.assertIsNone(sup.most_amended)

    def test_an_empty_board_claims_nothing(self):
        sup = relative.compute([])
        self.assertIsNone(sup.shortest_runway)
        self.assertIsNone(sup.widest_implied)
        self.assertIsNone(sup.most_amended)


class TestNoteFor(unittest.TestCase):
    def setUp(self):
        self.sup = relative.compute([row("AAA", quarters=2.0, move_pct=0.08, versions=4),
                                     row("BBB", quarters=9.0, move_pct=0.40, versions=43),
                                     row("CCC", quarters=15.0, move_pct=0.15, versions=9)])

    def test_names_the_shortest_runway(self):
        self.assertEqual(relative.note_for("AAA", "runway", self.sup),
                         "shortest runway shown")

    def test_names_the_longest_runway(self):
        self.assertEqual(relative.note_for("CCC", "runway", self.sup),
                         "longest runway shown")

    def test_names_the_widest_implied_move(self):
        self.assertEqual(relative.note_for("BBB", "implied", self.sup),
                         "widest implied move shown")

    def test_names_the_most_amended(self):
        self.assertEqual(relative.note_for("BBB", "amend", self.sup),
                         "most-amended on this board")

    def test_an_unremarkable_row_gets_no_note(self):
        self.assertIsNone(relative.note_for("BBB", "runway", self.sup))

    def test_an_unknown_field_gets_no_note(self):
        self.assertIsNone(relative.note_for("AAA", "design", self.sup))

    def test_the_wording_makes_no_claim_about_outcome(self):
        # Guards the line this module must never cross: these describe the
        # spread of THIS board, they do not predict or recommend.
        for ticker, field in (("AAA", "runway"), ("BBB", "implied"),
                              ("BBB", "amend")):
            note = relative.note_for(ticker, field, self.sup)
            lowered = note.lower()
            for banned in ("best", "worst", "top", "buy", "avoid", "risk",
                           "opportunity", "attractive"):
                self.assertNotIn(banned, lowered)


if __name__ == "__main__":
    unittest.main()
