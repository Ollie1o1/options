import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import unittest

import numpy as np

from src import exit_model as X
from src.paper_manager import (_evaluate_long_single_leg_exit,
                               _evaluate_short_single_leg_exit,
                               _normalize_exit_rules)

RULES = _normalize_exit_rules({})   # config defaults — the book's own fallbacks


def _sim(**over):
    kw = dict(spot=100.0, strike=105.0, opt_type="call", entry_price=None,
              t_years=45 / 365.0, sigma_real=0.35, iv_mark=0.40,
              rules=RULES, spread_pct=0.03, r=0.04, n_paths=2000, seed=7)
    kw.update(over)
    if kw["entry_price"] is None:
        # self-consistent fixture: on real rows the IV is solved FROM the
        # premium, so entry == BS(iv_mark) by construction
        from src.utils import bs_price
        kw["entry_price"] = float(bs_price(
            kw["opt_type"], kw["spot"], kw["strike"], kw["t_years"],
            kw["r"], kw["iv_mark"]))
    return X.simulate_exits(**kw)


class TestSimulateExits(unittest.TestCase):
    def test_outcome_probabilities_partition(self):
        out = _sim()
        total = out["p_tp"] + out["p_time"] + out["p_sl"] + out["p_expiry"]
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_deterministic_given_seed(self):
        self.assertEqual(_sim(), _sim())

    def test_no_hardcoded_rules_config_drives_the_sim(self):
        # doubling the TP target must reduce how often TP fires first
        low_tp = _normalize_exit_rules({"exit_rules": {"long_option":
                                        {"take_profit": 0.25, "stop_loss": -0.5}}})
        hi_tp = _normalize_exit_rules({"exit_rules": {"long_option":
                                       {"take_profit": 3.0, "stop_loss": -0.5}}})
        self.assertGreater(_sim(rules=low_tp)["p_tp"], _sim(rules=hi_tp)["p_tp"])

    def test_time_exit_dominates_short_dte_entries(self):
        # entered inside the 21-DTE window → time exit at min_days_held, the
        # same behaviour the cohort time-exit bias postmortem documented
        out = _sim(t_years=15 / 365.0)
        self.assertGreater(out["p_time"], 0.5)

    def test_peak_ladder_is_monotone(self):
        p = _sim()["peak"]["p_ge"]
        self.assertGreaterEqual(p["1.5"], p["2"])
        self.assertGreaterEqual(p["2"], p["3"])
        self.assertGreaterEqual(p["3"], p["5"])

    def test_peak_is_if_held_not_rule_truncated(self):
        # with an aggressive TP (exit at +10%) the ladder must still see 2x
        # touches at a similar rate to the wide-TP sim (same seed, same paths)
        tight = _normalize_exit_rules({"exit_rules": {"long_option":
                                       {"take_profit": 0.10, "stop_loss": -0.5}}})
        p_tight = _sim(rules=tight)["peak"]["p_ge"]["2"]
        p_wide = _sim()["peak"]["p_ge"]["2"]
        self.assertAlmostEqual(p_tight, p_wide, delta=0.02)

    def test_put_direction(self):
        out = _sim(opt_type="put", strike=95.0)
        self.assertAlmostEqual(
            out["p_tp"] + out["p_time"] + out["p_sl"] + out["p_expiry"],
            1.0, places=3)

    def test_short_premium_uses_tiered_tp_and_premium_stop(self):
        out = _sim(is_short=True, entry_delta=-0.30, opt_type="put",
                   strike=95.0)
        self.assertIsNotNone(out)
        self.assertEqual(out["rules"]["tiered_tp"],
                         [RULES["short"]["tp_ge_21"],
                          RULES["short"]["tp_7_21"],
                          RULES["short"]["tp_lt_7"]])

    def test_garbage_inputs_return_none(self):
        self.assertIsNone(_sim(entry_price=0))
        self.assertIsNone(_sim(entry_price=2.0, sigma_real=None, iv_mark=None))
        self.assertIsNone(_sim(entry_price=2.0, t_years=-1))

    def test_json_safe(self):
        import json
        json.dumps(_sim())


class TestTriggerParityWithPaperManager(unittest.TestCase):
    """The sim's trigger logic must agree with the book's enforcement
    functions on the same market states — one rule semantics, two engines."""

    def test_long_triggers_match_enforcement(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            spot = float(rng.uniform(50, 150))
            strike = 100.0
            entry = 3.0
            iv = 0.4
            dte = int(rng.integers(1, 60))
            held = int(rng.integers(0, 30))
            from src.utils import bs_price
            cur = float(bs_price("call", spot, strike,
                                 max(dte / 365.0, 1 / 365.0), 0.04, iv))
            should, reason, _ = _evaluate_long_single_leg_exit(
                RULES, "call", strike, spot, entry, cur, iv, dte, held, 0.04)
            # replicate the sim's long trigger cascade
            pnl = (cur - entry) / entry
            from src.utils import bs_delta
            d = abs(float(bs_delta("call", spot, strike,
                                   max(dte / 365.0, 1 / 365.0), 0.04, iv)))
            sim_fire = (pnl >= RULES["long"]["tp"]
                        or (0 < dte <= RULES["time_exit_dte"]
                            and held >= RULES["min_days_held"])
                        or d >= RULES["long"]["tp_delta"]
                        or pnl <= RULES["long"]["sl"])
            self.assertEqual(bool(should), bool(sim_fire),
                             f"spot={spot:.1f} dte={dte} held={held} "
                             f"cur={cur:.2f} reason={reason}")

    def test_short_triggers_match_enforcement(self):
        rng = np.random.default_rng(4)
        for _ in range(300):
            spot = float(rng.uniform(60, 140))
            strike = 100.0
            entry = 2.0
            iv = 0.35
            entry_delta = -0.30
            dte = int(rng.integers(1, 60))
            held = int(rng.integers(0, 30))
            from src.utils import bs_delta, bs_price
            cur = float(bs_price("put", spot, strike,
                                 max(dte / 365.0, 1 / 365.0), 0.04, iv))
            should, reason, _ = _evaluate_short_single_leg_exit(
                RULES, "put", strike, spot, entry, cur, entry_delta, iv,
                dte, held, 0.04)
            pnl = (entry - cur) / entry
            tp = (RULES["short"]["tp_ge_21"] if dte >= 21 else
                  RULES["short"]["tp_7_21"] if dte >= 7 else
                  RULES["short"]["tp_lt_7"])
            d = abs(float(bs_delta("put", spot, strike,
                                   max(dte / 365.0, 1 / 365.0), 0.04, iv)))
            breach = (RULES["short"]["sl_strike"]
                      and spot <= strike * (1 - RULES["short"]["sl_strike_buf"]))
            sim_fire = (pnl >= tp
                        or (0 < dte <= RULES["time_exit_dte"]
                            and held >= RULES["min_days_held"])
                        or breach
                        or pnl <= -(RULES["short"]["sl_prem_mult"] - 1.0)
                        or d >= RULES["short"]["sl_delta_mult"] * abs(entry_delta))
            self.assertEqual(bool(should), bool(sim_fire),
                             f"spot={spot:.1f} dte={dte} held={held} "
                             f"cur={cur:.2f} reason={reason}")


class TestRowConvenience(unittest.TestCase):
    def test_simulate_for_row(self):
        row = {"underlying": 182.4, "strike": 190.0, "type": "call",
               "premium": 4.2, "T_years": 36 / 365.0, "hv_ewma": 0.38,
               "impliedVolatility": 0.42, "spread_pct": 0.021,
               "strategy_name": "long_call", "delta": 0.45,
               "symbol": "NVDA", "expiration": "2026-07-17"}
        out = X.simulate_for_row(row, config={}, rfr=0.043)
        self.assertIsNotNone(out)
        self.assertFalse(out["hv_fallback"])
        self.assertIn("peak", out)

    def test_row_missing_fields_is_none(self):
        self.assertIsNone(X.simulate_for_row({"underlying": 100}, config={}))


if __name__ == "__main__":
    unittest.main()
