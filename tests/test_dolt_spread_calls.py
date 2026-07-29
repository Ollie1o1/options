"""Call credit spreads (Bear Call) on real DoltHub marks.

The put side has been backtestable all along; the call side never was. That is
the line the 2026-07-29 cost measurement promoted — Bear Call's real half-spread
is $0.025 against the $0.05 it was being charged, so it was the structure the
flat cost model penalised hardest. Testing it against real bid/ask needs no
slippage assumption at all.

A bear call spread is the mirror of a bull put: the long wing sits ABOVE the
short strike, not below, and width is long minus short. Getting that backwards
produces a debit spread wearing a credit spread's name.
"""
import unittest

from src import dolt_spread as sp
from src.paper_manager import _normalize_exit_rules

RULES = _normalize_exit_rules({})


def _call(strike, bid, ask, delta, expiration="2024-04-19", date="2024-03-01"):
    return {"symbol": "X", "date": date, "expiration": expiration, "strike": strike,
            "type": "call", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.3,
            "delta": delta, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": 0.02}


def _put(strike, bid, ask, delta, expiration="2024-04-19", date="2024-03-01"):
    return {"symbol": "X", "date": date, "expiration": expiration, "strike": strike,
            "type": "put", "bid": bid, "ask": ask, "mid": (bid + ask) / 2, "iv": 0.3,
            "delta": delta, "gamma": 0.01, "theta": -0.03, "vega": 0.1, "rho": -0.02}


class TestCallSpreadSelection(unittest.TestCase):
    def test_long_wing_sits_above_the_short_strike(self):
        chain = [_call(105, 2.0, 2.2, 0.25), _call(110, 1.0, 1.1, 0.15),
                 _call(115, 0.4, 0.5, 0.10)]
        short, long = sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                             min_dte=7, side="call")
        self.assertEqual(short["strike"], 105)
        self.assertGreater(long["strike"], short["strike"])

    def test_picks_the_strike_closest_to_the_target_delta(self):
        chain = [_call(105, 2.0, 2.2, 0.24), _call(107, 1.5, 1.6, 0.31),
                 _call(115, 0.4, 0.5, 0.10)]
        short, _ = sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                          min_dte=7, side="call")
        self.assertEqual(short["strike"], 105)

    def test_no_spread_when_nothing_sits_above_the_short(self):
        chain = [_call(105, 2.0, 2.2, 0.25)]
        self.assertIsNone(sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                                 min_dte=7, side="call"))

    def test_puts_are_not_selectable_on_the_call_side(self):
        chain = [_put(95, 2.0, 2.2, -0.25), _put(90, 1.0, 1.1, -0.10)]
        self.assertIsNone(sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                                 min_dte=7, side="call"))

    def test_both_legs_share_an_expiry(self):
        chain = [_call(105, 2.0, 2.2, 0.25, expiration="2024-04-19"),
                 _call(110, 1.0, 1.1, 0.10, expiration="2024-05-17"),
                 _call(112, 0.8, 0.9, 0.09, expiration="2024-04-19")]
        short, long = sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                             min_dte=7, side="call")
        self.assertEqual(short["expiration"], long["expiration"])


class TestPutSideUnchanged(unittest.TestCase):
    """The put path is live in the existing backtest; adding calls must not
    move it."""

    def test_put_wing_still_sits_below_the_short_strike(self):
        chain = [_put(95, 2.0, 2.2, -0.25), _put(90, 1.0, 1.1, -0.15),
                 _put(85, 0.4, 0.5, -0.10)]
        short, long = sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                             min_dte=7, side="put")
        self.assertEqual(short["strike"], 95)
        self.assertLess(long["strike"], short["strike"])

    def test_the_old_entry_point_still_works(self):
        chain = [_put(95, 2.0, 2.2, -0.25), _put(90, 1.0, 1.1, -0.10)]
        short, long = sp._pick_put_spread(chain, "2024-03-01", 0.25, 0.10, min_dte=7)
        self.assertEqual((short["strike"], long["strike"]), (95, 90))

    def test_calls_are_not_selectable_on_the_put_side(self):
        chain = [_call(105, 2.0, 2.2, 0.25), _call(110, 1.0, 1.1, 0.10)]
        self.assertIsNone(sp._pick_credit_spread(chain, "2024-03-01", 0.25, 0.10,
                                                 min_dte=7, side="put"))


class TestLegLookup(unittest.TestCase):
    def test_finds_a_call_leg_by_strike_and_expiry(self):
        chain = [_call(105, 2.0, 2.2, 0.25), _put(105, 1.0, 1.1, -0.25)]
        leg = sp._leg(chain, 105, "2024-04-19", side="call")
        self.assertEqual(leg["type"], "call")

    def test_does_not_confuse_a_put_for_a_call_at_the_same_strike(self):
        chain = [_put(105, 1.0, 1.1, -0.25)]
        self.assertIsNone(sp._leg(chain, 105, "2024-04-19", side="call"))


class TestWidthAndRisk(unittest.TestCase):
    def test_call_spread_width_is_long_minus_short(self):
        self.assertAlmostEqual(sp._spread_width(105.0, 110.0, "call"), 5.0)

    def test_put_spread_width_is_short_minus_long(self):
        self.assertAlmostEqual(sp._spread_width(95.0, 90.0, "put"), 5.0)

    def test_width_is_never_negative_for_a_valid_credit_spread(self):
        for side, s, l in (("call", 105.0, 110.0), ("put", 95.0, 90.0)):
            self.assertGreater(sp._spread_width(s, l, side), 0)


class TestExpirySettlement(unittest.TestCase):
    """A spread carried to expiry settles at intrinsic on the expiration date.

    Two defects lived here. The fallback branch reported exit_date as the last
    date in the entire price history and days_held as the distance to it, so a
    45-day spread entered in March 2023 came back claiming 823 days held and a
    2026 exit. And it valued the position at the last quote it happened to see
    rather than at expiry intrinsic, which for a chain carrying only a few
    expirations per date can be weeks early — precisely the window in which a
    short strike gets breached.
    """

    def test_worthless_expiry_keeps_the_whole_credit(self):
        # Bull put 95/90, spot finishes at 100: both legs expire worthless.
        self.assertAlmostEqual(
            sp._expiry_close_cost(95.0, 90.0, "put", spot=100.0), 0.0
        )

    def test_full_loss_expiry_costs_the_width(self):
        # Spot below both strikes: the spread is worth its full width.
        self.assertAlmostEqual(
            sp._expiry_close_cost(95.0, 90.0, "put", spot=80.0), 5.0
        )

    def test_partial_itm_costs_the_breach(self):
        # Short put breached by 2, long wing still out of the money.
        self.assertAlmostEqual(
            sp._expiry_close_cost(95.0, 90.0, "put", spot=93.0), 2.0
        )

    def test_call_spread_settles_on_the_upside(self):
        # Bear call 105/110, spot 107: short breached by 2, wing worthless.
        self.assertAlmostEqual(
            sp._expiry_close_cost(105.0, 110.0, "call", spot=107.0), 2.0
        )

    def test_call_spread_worthless_below_the_short_strike(self):
        self.assertAlmostEqual(
            sp._expiry_close_cost(105.0, 110.0, "call", spot=100.0), 0.0
        )

    def test_call_spread_loss_is_capped_at_the_width(self):
        self.assertAlmostEqual(
            sp._expiry_close_cost(105.0, 110.0, "call", spot=200.0), 5.0
        )


if __name__ == "__main__":
    unittest.main()
