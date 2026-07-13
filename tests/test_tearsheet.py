"""Ticker tearsheet — theme, charts, render, collect."""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tearsheet import theme  # noqa: E402


class TestTheme(unittest.TestCase):
    def test_light_and_dark_define_identical_keys(self):
        # A token present in light but missing in dark is an invisible-text bug
        # that no eyeball test reliably catches.
        self.assertEqual(set(theme.LIGHT), set(theme.DARK))

    def test_every_token_is_a_hex_colour(self):
        for name, table in (("LIGHT", theme.LIGHT), ("DARK", theme.DARK)):
            for k, v in table.items():
                self.assertRegex(v, r"^#[0-9a-fA-F]{6}$", f"{name}.{k}={v}")

    def test_dark_uses_the_terminal_palette(self):
        self.assertEqual(theme.DARK["good"], "#5ec98d")
        self.assertEqual(theme.DARK["bad"], "#e06c75")
        self.assertEqual(theme.DARK["warn"], "#d6a452")
        self.assertEqual(theme.DARK["muted"], "#626870")

    def test_heat_inks_returns_two_hexes(self):
        lo, hi = theme.heat_inks(500.0, 1000.0)
        self.assertRegex(lo, r"^#[0-9a-f]{6}$")
        self.assertRegex(hi, r"^#[0-9a-f]{6}$")
        self.assertNotEqual(lo, hi)

    def test_heat_sign_is_preserved_in_both_inks(self):
        # Loss must never render as the gain ink in either theme.
        gain_l, gain_d = theme.heat_inks(1000.0, 1000.0)
        loss_l, loss_d = theme.heat_inks(-1000.0, 1000.0)
        self.assertNotEqual(gain_l, loss_l)
        self.assertNotEqual(gain_d, loss_d)

    def test_zero_span_does_not_crash(self):
        self.assertEqual(len(theme.heat_inks(5.0, 0.0)), 2)

    def test_css_tokens_defines_both_themes(self):
        css = theme.css_tokens()
        self.assertIn(":root", css)
        self.assertIn('[data-theme="dark"]', css)
        self.assertIn("--good", css)


from src.tearsheet import charts  # noqa: E402


class TestCharts(unittest.TestCase):
    def test_line_chart_is_svg_and_uses_css_vars(self):
        svg = charts.line_chart([1.0, 3.0, 2.0, 5.0])
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("var(--ink)", svg)
        self.assertNotRegex(svg, r"stroke=\"#[0-9a-f]{6}\"")

    def test_line_chart_is_deterministic(self):
        s = [1.0, 2.0, 3.0]
        self.assertEqual(charts.line_chart(s), charts.line_chart(s))

    def test_line_chart_too_short_returns_empty(self):
        self.assertEqual(charts.line_chart([1.0]), "")
        self.assertEqual(charts.line_chart([]), "")

    def test_flat_series_does_not_divide_by_zero(self):
        svg = charts.line_chart([4.0, 4.0, 4.0])
        self.assertIn("<svg", svg)

    def test_price_with_bands_marks_levels(self):
        svg = charts.price_with_bands(
            [10.0, 11.0, 12.0], [{"label": "50d MA", "level": 9.5, "pct": -0.05}],
            [{"label": "swing", "level": 13.0, "pct": 0.08}])
        self.assertIn("var(--good)", svg)   # support band
        self.assertIn("var(--bad)", svg)    # resistance band

    def test_vol_cone_marks_current_iv(self):
        cone = [{"window": 30, "p25": .2, "median": .3, "p75": .4, "current": .32, "pctile": .78}]
        svg = charts.vol_cone(cone, current_iv=0.42)
        self.assertIn("<svg", svg)
        self.assertIn("var(--bad)", svg)

    def test_vol_cone_empty_returns_empty(self):
        self.assertEqual(charts.vol_cone([], None), "")

    def test_term_curve_labels_each_expiry(self):
        svg = charts.term_curve([[9, 0.30], [44, 0.37]])
        self.assertIn("9d", svg)
        self.assertIn("44d", svg)

    def test_waterfall_bars_totals(self):
        html = charts.waterfall_bars([["Gross edge", 6.0], ["Spread", -18.0]])
        self.assertIn("Gross edge", html)
        self.assertIn("var(--bad)", html)
        self.assertIn("var(--good)", html)


from src.tearsheet import render as R  # noqa: E402


def _fixture(**over):
    data = {
        "meta": {"generated_at": "2026-07-09T14:22:00", "ticker": "NVDA", "strike": 190.0,
                 "opt_type": "call", "expiration": "2026-07-17", "dte": 36,
                 "mode": "Discovery scan", "rank": 3, "n_picks": 9, "spot": 182.40,
                 "rfr": 0.043, "vix": 17.2, "vix_regime": "normal",
                 "config_sha": "4f1a9c", "sidecar": "nvda.json"},
        "verdict": {"decision": "SKIP", "reason": "", "net_ev": -12.0,
                    "gross_ev": 6.0, "cost": 18.0},
        "stats": {"pop": 0.62, "max_loss": 420.0, "breakeven": 194.20},
        "cost_waterfall": [["Gross edge", 6.0], ["Entry spread", -9.0],
                           ["Exit spread", -9.0]],
        "greeks": {"delta": 0.45, "gamma": 0.012, "vega": 0.31, "theta": -0.08},
        "liquidity": {"spread_pct": 0.021, "oi": 8450, "volume": 1240,
                      "quote_freshness": "fresh"},
        "stress": {"moves": [-0.05, 0.0, 0.05],
                   "rows": [{"iv": 0.10, "pnls": [-288.0, -12.0, 180.0]},
                            {"iv": -0.10, "pnls": [-612.0, -240.0, 150.0]}],
                   "worst": "-$1,240 @ -5% spot / +10pp IV"},
        "vol": {"iv": 0.42, "hv": 0.38, "vrp": 0.04, "iv_rank": 0.41,
                "svi_residual": 0.18,
                "cone": [{"window": 30, "p25": .2, "median": .3, "p75": .4,
                          "current": .32, "pctile": .78}],
                "term": [[9, 0.30], [44, 0.37]], "skew_vp": 4.1, "skew_rank": 0.83,
                "expected_move": 11.13, "required_move": 11.80},
        "name": {"closes": [180.0, 181.5, 182.4],
                 "supports": [{"label": "50d MA", "level": 174.2, "pct": -0.045}],
                 "resistances": [{"label": "swing", "level": 191.5, "pct": 0.05}],
                 "rsi": 58.0, "ret_5d": 0.023, "pcr": 0.82, "oi_change": 320,
                 "max_pain": 185.0},
        "narrative": {"thesis": "Momentum is constructive.", "vehicle": "perp > option",
                      "portfolio_fit": ["3rd long-vega NVDA position"],
                      "history": "NVDA calls: 4 trades, -$180"},
        "evidence": {"pooled_ic": 0.03, "p_value": 0.41, "n_oos": 2400,
                     "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-15"},
        "context": [{"label": "Quality score", "value": "0.78",
                     "badge": "IC +0.03", "badge_kind": "bad"}],
        "panels": {k: {"status": "ok", "reason": ""} for k in
                   ("decision", "vol", "name", "narrative", "evidence", "context")},
    }
    for k, v in over.items():
        data[k] = v
    return data


class TestVerdict(unittest.TestCase):
    def test_positive_net_ev_is_take(self):
        d, _ = R.decide_verdict(12.0, 30.0, 18.0, [])
        self.assertEqual(d, "TAKE")

    def test_negative_net_ev_is_skip(self):
        d, _ = R.decide_verdict(-12.0, 6.0, 18.0, [])
        self.assertEqual(d, "SKIP")

    def test_zero_net_ev_is_skip(self):
        # A coin flip that costs a spread is not a trade.
        d, _ = R.decide_verdict(0.0, 18.0, 18.0, [])
        self.assertEqual(d, "SKIP")

    def test_missing_net_ev_is_indeterminate(self):
        # options_screener sets ev_per_contract = NaN on the HV-fallback mask.
        d, why = R.decide_verdict(None, 6.0, 18.0, [])
        self.assertEqual(d, "INDETERMINATE")
        self.assertTrue(why)

    def test_nan_net_ev_is_indeterminate(self):
        d, _ = R.decide_verdict(float("nan"), 6.0, 18.0, [])
        self.assertEqual(d, "INDETERMINATE")

    def test_skip_reason_names_the_largest_negative_term(self):
        _, why = R.decide_verdict(-12.0, 6.0, 18.0,
                                  [["Gross edge", 6.0], ["Entry spread", -9.0],
                                   ["Commission", -1.3]])
        self.assertIn("Entry spread", why)

    def test_negative_gross_edge_is_never_blamed_on_transaction_costs(self):
        # UNH 420P on 2026-07-10: the contract is mispriced against you BEFORE
        # a single cent of cost. Saying "costs ate the edge" misattributes the
        # cause and implies a better fill could rescue it.
        _, why = R.decide_verdict(-240.0, -200.0, 40.0,
                                  [["Gross edge", -200.0], ["Entry spread", -20.0],
                                   ["Exit spread", -20.0]])
        self.assertNotIn("does not survive", why)
        self.assertIn("before", why.lower())

    def test_negative_gross_edge_is_skip(self):
        d, _ = R.decide_verdict(-240.0, -200.0, 40.0, [])
        self.assertEqual(d, "SKIP")

    def test_zero_gross_edge_is_skip_with_no_edge_reason(self):
        d, why = R.decide_verdict(-40.0, 0.0, 40.0, [])
        self.assertEqual(d, "SKIP")
        self.assertNotIn("does not survive", why)

    def test_net_ev_above_the_noise_band_is_take(self):
        d, _ = R.decide_verdict(216.0, 257.0, 41.0, [], noise=45.0)
        self.assertEqual(d, "TAKE")

    def test_net_ev_inside_the_noise_band_is_marginal_not_take(self):
        # WMT 113P on 2026-07-10: net EV +$8 on a contract whose vega is worth
        # ~$10 per IV point. The sign is not resolvable from this data.
        d, why = R.decide_verdict(8.0, 50.0, 42.0, [], noise=25.0)
        self.assertEqual(d, "MARGINAL")
        self.assertIn("noise", why.lower())

    def test_noise_band_never_rescues_a_negative_net_ev(self):
        # Symmetry would say "unresolvable"; a trade you cannot justify is a pass.
        d, _ = R.decide_verdict(-5.0, 40.0, 45.0, [], noise=100.0)
        self.assertEqual(d, "SKIP")

    def test_zero_noise_preserves_the_old_take_boundary(self):
        self.assertEqual(R.decide_verdict(1.0, 42.0, 41.0, [], noise=0.0)[0], "TAKE")


class TestFlipPrice(unittest.TestCase):
    """The single actionable number on a page that says SKIP."""

    def test_fill_to_flip_is_below_the_assumed_fill(self):
        # net EV -28 at an assumed $2.84 fill; each cent of price improvement is
        # worth $1 per contract, so $0.28 better clears zero and $0.53 clears +25.
        self.assertAlmostEqual(R.fill_to_flip(-28.0, 25.0, 2.84), 2.31, places=2)

    def test_fill_to_flip_of_a_winner_is_above_the_assumed_fill(self):
        self.assertAlmostEqual(R.fill_to_flip(216.0, 45.0, 5.15), 6.86, places=2)

    def test_fill_to_flip_is_none_without_a_fill_reference(self):
        self.assertIsNone(R.fill_to_flip(-28.0, 25.0, None))
        self.assertIsNone(R.fill_to_flip(None, 25.0, 2.84))
        self.assertIsNone(R.fill_to_flip(float("nan"), 25.0, 2.84))

    def test_fill_to_flip_is_none_when_the_price_would_be_negative(self):
        # A -$900 net EV on a $1.20 contract cannot be fixed by any fill.
        self.assertIsNone(R.fill_to_flip(-900.0, 0.0, 1.20))

    def test_flip_line_quotes_a_reachable_price_inside_the_quote(self):
        # -$5 net, $10 band: $0.16 of price improvement clears it, and $2.68 is
        # still above the $2.60 bid, so the fill is obtainable.
        line = R._flip_line("SKIP", {"net_ev": -5.0, "noise": 10.0, "assumed_fill": 2.84},
                            {"bid": 2.60, "ask": 3.08})
        self.assertIn("2.68", line)
        self.assertIn("TAKE", line)

    def test_flip_line_says_when_no_fill_inside_the_quote_helps(self):
        # XOM 140C on 2026-07-10 in miniature: the required fill is under the bid.
        line = R._flip_line("SKIP", {"net_ev": -128.0, "noise": 25.0, "assumed_fill": 5.25},
                            {"bid": 5.00, "ask": 5.50})
        self.assertIn("no fill", line.lower())
        self.assertIn("5.00", line)     # names the bid it would have to beat
        self.assertIn("3.71", line)     # ...and the price it would have taken

    def test_flip_line_refuses_to_price_a_contract_with_no_gross_edge(self):
        line = R._flip_line("SKIP", {"net_ev": -900.0, "noise": 0.0, "assumed_fill": 1.20}, {})
        self.assertIn("No entry price", line)

    def test_a_take_page_carries_no_flip_line(self):
        self.assertEqual(R._flip_line("TAKE", {"net_ev": 216.0, "noise": 45.0,
                                               "assumed_fill": 5.15}, {"bid": 5.0}), "")

    def test_flip_line_is_empty_without_a_fill_reference(self):
        self.assertEqual(R._flip_line("SKIP", {"net_ev": -28.0, "noise": 25.0}, {}), "")

    def test_a_marginal_page_carries_a_flip_line(self):
        line = R._flip_line("MARGINAL", {"net_ev": 8.0, "noise": 25.0, "assumed_fill": 2.17},
                            {"bid": 1.90, "ask": 2.29})
        self.assertIn("TAKE", line)
        self.assertIn("1.99", line)

    def test_reachable_flip_at_a_mid_cent_breakeven_is_not_called_unreachable(self):
        # WMT 113P, 2026-07-10: break-even fill ≈ $2.052. Flooring a full cent to
        # $2.04 wrongly reads as below the $2.05 bid; the bid itself clears the
        # band, so the honest answer is "$2.05 or better", a TAKE.
        line = R._flip_line("MARGINAL",
                            {"net_ev": 8.0, "noise": 27.5, "assumed_fill": 2.24715},
                            {"bid": 2.05, "ask": 2.29})
        self.assertIn("TAKE", line)
        self.assertIn("2.05", line)
        self.assertNotIn("no fill", line.lower())

    def test_the_quoted_reachable_price_actually_clears_the_band(self):
        fill, noise, net, bid = 2.24715, 27.5, 8.0, 2.05
        line = R._flip_line("MARGINAL", {"net_ev": net, "noise": noise,
                                         "assumed_fill": fill}, {"bid": bid})
        price = float(re.search(r"\$(\d+\.\d\d)", line).group(1))
        self.assertGreater(net + (fill - price) * 100.0, noise)

    def test_flip_price_clears_the_band_after_rounding_to_the_cent(self):
        # A price that only *equals* the band leaves the page MARGINAL. The quoted
        # fill must still be a TAKE once a human rounds it to a real limit order.
        fill, noise, net = 2.84, 25.0, -28.0
        line = R._flip_line("SKIP", {"net_ev": net, "noise": noise, "assumed_fill": fill}, {})
        price = float(re.search(r"\$(\d+\.\d\d)", line).group(1))
        improved_net = net + (fill - price) * 100.0
        self.assertGreater(improved_net, noise)


class TestRenderShell(unittest.TestCase):
    def test_is_a_full_document(self):
        out = R.render(_fixture())
        self.assertTrue(out.lstrip().startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", out)

    def test_self_contained_no_external_refs(self):
        out = R.render(_fixture())
        self.assertNotIn("http://", out)
        self.assertNotIn("https://", out)

    def test_render_is_pure(self):
        # A stray datetime.now() in the render path would break reproducibility.
        d = _fixture()
        self.assertEqual(R.render(d), R.render(d))

    def test_verdict_is_invariant_to_quality_score(self):
        a = _fixture()
        b = _fixture()
        b["context"] = [{"label": "Quality score", "value": "0.99",
                         "badge": "IC +0.03", "badge_kind": "bad"}]
        self.assertIn("SKIP", R.render(a))
        self.assertIn("SKIP", R.render(b))

    def test_verdict_flips_with_net_ev_sign(self):
        d = _fixture()
        d["verdict"] = dict(d["verdict"], net_ev=25.0)
        self.assertIn("TAKE", R.render(d))

    def test_marginal_verdict_gets_its_own_class_not_the_take_class(self):
        d = _fixture()
        d["verdict"] = dict(d["verdict"], net_ev=5.0, noise=40.0)
        out = R.render(d)
        self.assertIn('<span class="verdict v-marg">MARGINAL</span>', out)
        self.assertNotIn('<span class="verdict v-take">', out)

    def test_skip_page_carries_the_flip_price(self):
        d = _fixture()
        d["verdict"] = dict(d["verdict"], noise=4.0, assumed_fill=4.24)
        d["quote"] = {"bid": 4.00, "ask": 4.30}
        out = R.render(d)
        self.assertIn('class="flip"', out)
        self.assertIn("4.07", out)      # -12 net, $4 band → $0.17 improvement

    def test_take_page_carries_no_flip_price(self):
        d = _fixture()
        d["verdict"] = dict(d["verdict"], net_ev=250.0, noise=4.0, assumed_fill=4.24)
        self.assertNotIn('class="flip"', R.render(d))

    def test_noise_band_is_shown_in_the_headline_strip(self):
        d = _fixture()
        d["verdict"] = dict(d["verdict"], noise=25.0)
        out = R.render(d)
        self.assertIn("Noise band", out)

    def test_ticker_is_escaped(self):
        d = _fixture()
        d["meta"] = dict(d["meta"], ticker='<script>alert(1)</script>')
        out = R.render(d)
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_theme_toggle_and_tokens_present(self):
        out = R.render(_fixture())
        self.assertIn('data-theme', out)
        self.assertIn("prefers-color-scheme", out)
        self.assertIn("--good", out)

    def test_unavailable_panel_renders_placeholder(self):
        d = _fixture()
        d["panels"]["vol"] = {"status": "unavailable", "reason": "SVI fit failed"}
        out = R.render(d)
        self.assertIn("unavailable", out)
        self.assertIn("SVI fit failed", out)

    def test_not_fetched_panel_renders_placeholder(self):
        d = _fixture()
        d["panels"]["narrative"] = {"status": "not_fetched", "reason": "exceeded 2.5s budget"}
        out = R.render(d)
        self.assertIn("not fetched", out)


class TestZonesOneTwo(unittest.TestCase):
    def test_heat_grid_cells_carry_both_inks(self):
        grid = R._heat_grid(_fixture()["stress"])
        cells = re.findall(r'style="--hl:(#[0-9a-f]{6});--hd:(#[0-9a-f]{6})"', grid)
        self.assertEqual(len(cells), 6)  # 2 iv rows x 3 moves

    def test_every_heat_cell_in_document_has_both_inks(self):
        out = R.render(_fixture())
        for style in re.findall(r'class="hc"\s+style="([^"]*)"', out):
            self.assertIn("--hl:", style)
            self.assertIn("--hd:", style)

    def test_decision_zone_shows_cost_wall_and_greeks(self):
        out = R.render(_fixture())
        self.assertIn("Cost wall", out)
        self.assertIn("0.45", out)      # delta
        self.assertIn("8,450", out)     # open interest

    def test_vol_zone_shows_skew_and_rich_cheap(self):
        out = R.render(_fixture())
        self.assertIn("RICH", out.upper())
        self.assertIn("4.1", out)       # skew vol points

    def test_vol_zone_absent_when_panel_unavailable(self):
        d = _fixture()
        d["panels"]["vol"] = {"status": "unavailable", "reason": "no SVI fit"}
        out = R.render(d)
        self.assertIn("no SVI fit", out)
        self.assertNotIn("Vol complex", out)


class TestZonesThreeToFive(unittest.TestCase):
    def test_evidence_panel_always_present(self):
        self.assertIn("Model evidence", R.render(_fixture()))

    def test_low_ic_is_badged_no_edge(self):
        out = R.render(_fixture())
        self.assertIn("no edge", out)

    def test_high_and_significant_ic_is_badged_has_edge(self):
        d = _fixture()
        d["evidence"] = dict(d["evidence"], pooled_ic=0.31, p_value=0.001)
        d["context"] = []          # the context badges carry their own text
        out = R.render(d)
        self.assertNotIn("no edge", out)
        self.assertIn("has edge", out)
        self.assertIn("statistically significant", out)

    def test_high_but_insignificant_ic_is_underpowered_not_an_edge(self):
        # The real walk-forward artifact reports IC +0.10 at p=0.48 on n=94.
        # A large IC with a large p-value is a coin flip wearing a decimal point.
        _txt, badge = R._ic_badge(0.10214, 0.48029)
        self.assertIn("underpowered", badge)
        self.assertNotIn("has edge", badge)

    def test_live_evidence_values_do_not_claim_skill(self):
        d = _fixture()
        d["evidence"] = dict(d["evidence"], pooled_ic=0.10214, p_value=0.48029, n_oos=94)
        out = R.render(d)
        self.assertIn("underpowered", out)
        self.assertIn("not significant", out)
        self.assertNotIn("statistically significant", out)

    def test_missing_p_value_never_claims_an_edge(self):
        _txt, badge = R._ic_badge(0.31, None)
        self.assertIn("underpowered", badge)
        self.assertFalse(R._has_scorer_edge(0.31, None))

    def test_missing_evidence_fails_closed(self):
        # An absent track record is not a good track record.
        d = _fixture()
        d["evidence"] = {"pooled_ic": None, "p_value": None, "n_oos": 0,
                         "cohort_n": 0, "gate_decision": "UNKNOWN", "as_of": None}
        out = R.render(d)
        self.assertIn("unknown", out)
        self.assertIn("no edge", out)

    def test_missing_evidence_is_never_badged_validated_for_the_scorer(self):
        _txt, badge = R._ic_badge(None, None)
        self.assertIn("no edge", badge)
        self.assertNotIn("has edge", badge)

    def test_context_zone_is_demoted_and_badged(self):
        out = R.render(_fixture())
        self.assertIn("no demonstrated edge", out)
        self.assertIn("demote", out)
        self.assertIn("IC +0.03", out)

    def test_name_zone_renders_levels(self):
        out = R.render(_fixture())
        self.assertIn("174.20", out)   # support level
        self.assertIn("Max pain", out)

    def test_narrative_thesis_is_escaped(self):
        d = _fixture()
        d["narrative"] = dict(d["narrative"], thesis="<b>boom</b>")
        out = R.render(d)
        self.assertNotIn("<b>boom</b>", out)

    def test_evidence_survives_a_failed_narrative_panel(self):
        # Evidence is rendered inside zone IV. If a narrative failure could take
        # it down with it, the page would silently drop its own honesty panel.
        d = _fixture()
        d["panels"]["narrative"] = {"status": "unavailable", "reason": "pick_context blew up"}
        out = R.render(d)
        self.assertIn("Model evidence", out)
        self.assertIn("no edge", out)
        self.assertIn("pick_context blew up", out)


from src.tearsheet import collect  # noqa: E402


def _row():
    return {
        "symbol": "NVDA", "type": "call", "strike": 190.0, "expiration": "2026-07-17",
        "T_years": 36 / 365.0, "underlying": 182.40, "premium": 4.20,
        "bid": 4.15, "ask": 4.24, "spread_pct": 0.021, "volume": 1240,
        "openInterest": 8450, "impliedVolatility": 0.42, "hv_30d": 0.38,
        "iv_percentile_30": 0.41, "delta": 0.45, "gamma": 0.012, "vega": 0.31,
        "theta": -0.08, "prob_profit": 0.62, "max_loss": 420.0, "breakeven": 194.20,
        "ev_per_contract": -12.0, "ev_gross_per_contract": 6.0,
        "ev_cost_per_contract": 18.0, "iv_surface_residual": 0.18,
        "iv_skew": 0.041, "iv_skew_rank": 0.83, "expected_move": 11.13,
        "required_move": 11.80, "quality_score": 0.78, "rsi_14": 58.0,
        "ret_5d": 0.023, "pcr": 0.82, "oi_change": 320, "quote_freshness": "fresh",
        "strategy_name": "long_call", "vega_dollar": 31.0, "iv_confidence": "High",
    }


def _ctx():
    return {"mode": "Discovery scan", "rank": 3, "n_picks": 9, "spot": 182.40,
            "rfr": 0.043, "vix": 17.2, "vix_regime": "normal",
            "config": {}, "config_sha": "4f1a9c"}


class TestCollect(unittest.TestCase):
    def test_row_underlying_beats_a_stale_ctx_spot(self):
        # On a cross-ticker scan the ctx spot is whatever ticker the scan loop
        # touched last. A TSLA sheet shipped with spot=59.67 and its whole
        # stress grid degenerated to ±$0 of floating-point dust.
        data = collect.build(_row(), dict(_ctx(), spot=59.67), slow=False)
        self.assertAlmostEqual(data["meta"]["spot"], 182.40)

    def test_ctx_spot_is_still_the_fallback(self):
        row = dict(_row())
        row.pop("underlying")
        data = collect.build(row, _ctx(), slow=False)
        self.assertAlmostEqual(data["meta"]["spot"], 182.40)

    def test_payoff_panel_has_matching_ladder_and_curve(self):
        data = collect.build(_row(), _ctx(), slow=False)
        po = data.get("payoff")
        self.assertIsNotNone(po)
        self.assertEqual(len(po["prices"]), len(po["today_pnl"]))
        self.assertGreater(len(po["prices"]), 10)

    def test_build_produces_json_serialisable_data(self):
        data = collect.build(_row(), _ctx(), slow=False)
        json.dumps(data)   # must not raise

    def test_nan_becomes_none_not_nan(self):
        r = _row()
        r["ev_per_contract"] = float("nan")
        data = collect.build(r, _ctx(), slow=False)
        self.assertIsNone(data["verdict"]["net_ev"])
        self.assertNotIn("NaN", json.dumps(data))

    def test_cost_waterfall_sums_to_net_ev(self):
        data = collect.build(_row(), _ctx(), slow=False)
        total = sum(v for _, v in data["cost_waterfall"])
        self.assertAlmostEqual(total, data["verdict"]["net_ev"], places=2)

    def test_context_zone_lists_the_quality_score(self):
        data = collect.build(_row(), _ctx(), slow=False)
        labels = [c["label"] for c in data["context"]]
        self.assertIn("Quality score", labels)

    def test_every_zone_has_a_panel_status(self):
        data = collect.build(_row(), _ctx(), slow=False)
        for zone in ("decision", "vol", "name", "narrative", "context"):
            self.assertIn(zone, data["panels"])

    def test_failing_builder_marks_panel_unavailable(self):
        panels = {}
        out = collect._safe("vol", lambda: 1 / 0, panels, default={})
        self.assertEqual(out, {})
        self.assertEqual(panels["vol"]["status"], "unavailable")
        self.assertIn("division", panels["vol"]["reason"].lower())

    def test_build_output_renders(self):
        self.assertIn("<!DOCTYPE html>", R.render(collect.build(_row(), _ctx(), slow=False)))

    def test_indeterminate_verdict_when_ev_is_nan(self):
        r = _row()
        r["ev_per_contract"] = float("nan")
        out = R.render(collect.build(r, _ctx(), slow=False))
        self.assertIn("INDETERMINATE", out)

    def test_sidecar_declares_its_schema_version(self):
        data = collect.build(_row(), _ctx(), slow=False)
        self.assertEqual(data["meta"]["schema"], collect.SCHEMA)


class TestNoiseBand(unittest.TestCase):
    """Net EV is a Black-Scholes point estimate. Its error bar is vega-sized."""

    def test_band_is_vega_dollars_times_the_iv_uncertainty(self):
        # $31 per IV point, high-confidence IV -> a one-point band.
        self.assertAlmostEqual(collect._ev_noise(_row()), 31.0, places=6)

    def test_a_less_trusted_iv_widens_the_band(self):
        high = collect._ev_noise(dict(_row(), iv_confidence="High"))
        low = collect._ev_noise(dict(_row(), iv_confidence="Low"))
        self.assertGreater(low, high)

    def test_missing_iv_confidence_assumes_the_worst(self):
        r = _row()
        r.pop("iv_confidence")
        self.assertEqual(collect._ev_noise(r),
                         collect._ev_noise(dict(_row(), iv_confidence="Low")))

    def test_band_falls_back_to_a_fraction_of_cost_without_vega(self):
        r = _row()
        r.pop("vega_dollar")
        self.assertAlmostEqual(collect._ev_noise(r), 0.25 * 18.0, places=6)

    def test_band_is_zero_when_nothing_is_known(self):
        self.assertEqual(collect._ev_noise({}), 0.0)

    def test_build_carries_the_band_and_the_assumed_fill(self):
        v = collect.build(_row(), _ctx(), slow=False)["verdict"]
        self.assertAlmostEqual(v["noise"], 31.0, places=6)
        # The cost model charges a half-spread on entry: it assumes you pay the ask.
        self.assertAlmostEqual(v["assumed_fill"], 4.20 * (1 + 0.021 / 2), places=6)

    def test_a_thinly_traded_contract_lands_marginal_not_take(self):
        # +$8 of net EV against a $31 band is a coin flip, not a trade.
        r = dict(_row(), ev_per_contract=8.0, ev_gross_per_contract=26.0)
        out = R.render(collect.build(r, _ctx(), slow=False))
        self.assertIn("MARGINAL", out)


class TestSlowTier(unittest.TestCase):
    def test_slow_panels_that_exceed_budget_are_marked_not_fetched(self):
        import time
        slow = {"earnings": lambda: time.sleep(5), "news": lambda: "ok"}
        vals, panels = collect.gather_slow("NVDA", budget_s=0.4, _fns=slow)
        self.assertEqual(panels["earnings"]["status"], "not_fetched")
        self.assertIn("budget", panels["earnings"]["reason"])
        self.assertEqual(panels["news"]["status"], "ok")
        self.assertEqual(vals["news"], "ok")

    def test_slow_panel_that_raises_is_unavailable_not_not_fetched(self):
        vals, panels = collect.gather_slow(
            "NVDA", budget_s=1.0, _fns={"insider": lambda: 1 / 0})
        self.assertEqual(panels["insider"]["status"], "unavailable")

    def test_budget_is_respected_overall(self):
        # A `with ThreadPoolExecutor(...)` block joins running threads on exit,
        # which would silently blow the wall-clock budget while looking correct.
        import time
        t0 = time.time()
        collect.gather_slow("NVDA", budget_s=0.3,
                            _fns={"a": lambda: time.sleep(5), "b": lambda: time.sleep(5)})
        self.assertLess(time.time() - t0, 2.5)

    def test_overrun_workers_are_daemons_so_the_process_can_still_exit(self):
        # The budget bounds the PAGE. It only bounds the PROCESS if the abandoned
        # worker cannot hold interpreter shutdown open: `Future.cancel()` is a
        # no-op on an already-running call, and ThreadPoolExecutor registers an
        # atexit hook that joins its workers. EDGAR's own timeout is 20s.
        import threading
        import time
        before = set(threading.enumerate())
        collect.gather_slow("NVDA", budget_s=0.2,
                            _fns={"slow": lambda: time.sleep(3)})
        spawned = [t for t in threading.enumerate() if t not in before and t.is_alive()]
        self.assertTrue(spawned, "expected the overrunning worker to still be alive")
        for t in spawned:
            self.assertTrue(t.daemon, "worker {!r} would block interpreter exit".format(t.name))

    def test_slow_disabled_marks_all_three_not_fetched(self):
        data = collect.build(_row(), _ctx(), slow=False)
        for pid in ("earnings", "insider", "news"):
            self.assertEqual(data["panels"][pid]["status"], "not_fetched")
            self.assertEqual(data["panels"][pid]["reason"], "disabled")


import tempfile  # noqa: E402


class TestWriteAndReRender(unittest.TestCase):
    def test_writes_html_and_sidecar(self):
        from src.tearsheet import write_tearsheet
        with tempfile.TemporaryDirectory() as d:
            html_p, json_p = write_tearsheet(_fixture(), out_dir=d)
            self.assertTrue(os.path.exists(html_p))
            self.assertTrue(os.path.exists(json_p))
            with open(html_p) as f:
                self.assertIn("<!DOCTYPE html>", f.read())

    def test_sidecar_round_trips_to_identical_html(self):
        # Reproducibility: the sidecar must regenerate the exact page.
        from src.tearsheet import write_tearsheet
        with tempfile.TemporaryDirectory() as d:
            html_p, json_p = write_tearsheet(_fixture(), out_dir=d)
            with open(json_p) as f:
                reloaded = json.load(f)
            with open(html_p) as f:
                original = f.read()
            self.assertEqual(R.render(reloaded), original)

    def test_package_does_not_shadow_the_render_submodule(self):
        # Exporting a function named `render` on the package would make
        # `from src.tearsheet import render` yield a function, not the module.
        import src.tearsheet as pkg
        from src.tearsheet import render as mod
        self.assertTrue(hasattr(pkg, "render_html"))
        self.assertTrue(hasattr(mod, "decide_verdict"))

    def test_cli_from_sidecar_writes_html_without_opening(self):
        from src.tearsheet.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            side = os.path.join(d, "x.json")
            with open(side, "w") as f:
                json.dump(dict(_fixture(), meta=dict(_fixture()["meta"], schema=1)), f)
            rc = main(["--from", side, "--no-open"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "x.html")))

    def _render_sidecar(self, meta_over):
        from src.tearsheet.__main__ import main
        d = _fixture()
        d["meta"] = dict(d["meta"], **meta_over)
        with tempfile.TemporaryDirectory() as tmp:
            side = os.path.join(tmp, "x.json")
            with open(side, "w") as f:
                json.dump(d, f)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--from", side, "--no-open"])
        return rc, buf.getvalue()

    def test_cli_warns_when_the_sidecar_predates_the_current_schema(self):
        # A sidecar written before `meta.schema` existed still renders — but the
        # page may quietly lack keys the current renderer expects, and a silent
        # partial render is exactly what this package refuses elsewhere.
        rc, out = self._render_sidecar({})       # no schema key at all
        self.assertEqual(rc, 0)
        self.assertIn("schema", out.lower())

    def test_cli_warns_on_a_sidecar_from_the_future(self):
        rc, out = self._render_sidecar({"schema": 99})
        self.assertEqual(rc, 0)
        self.assertIn("99", out)

    def test_cli_is_quiet_on_a_current_sidecar(self):
        _rc, out = self._render_sidecar({"schema": collect.SCHEMA})
        self.assertNotIn("schema", out.lower())


import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402


class TestScreenerIntegration(unittest.TestCase):
    def test_offer_is_silent_and_none_when_not_interactive(self):
        import pandas as pd
        from src.options_screener import offer_tearsheet
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = offer_tearsheet(pd.DataFrame([_row()]), _ctx(),
                                  interactive=False, preselect=None)
        self.assertIsNone(out)
        self.assertEqual(buf.getvalue(), "")

    def test_offer_never_calls_input_when_not_interactive(self):
        import builtins
        import pandas as pd
        from src.options_screener import offer_tearsheet
        original = builtins.input

        def _boom(*a):
            raise AssertionError("input() reached in non-interactive mode")

        builtins.input = _boom
        try:
            offer_tearsheet(pd.DataFrame([_row()]), _ctx(), interactive=False, preselect=None)
        finally:
            builtins.input = original

    def test_multileg_mode_is_refused(self):
        import pandas as pd
        from src.options_screener import offer_tearsheet
        ctx = dict(_ctx(), mode="Iron Condor")
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = offer_tearsheet(pd.DataFrame([_row()]), ctx, interactive=True, preselect=1)
        self.assertIsNone(out)
        self.assertIn("multi-leg", buf.getvalue())

    def test_invalid_selection_is_treated_as_no(self):
        import pandas as pd
        from src.options_screener import offer_tearsheet
        out = offer_tearsheet(pd.DataFrame([_row()]), _ctx(),
                              interactive=True, preselect=99)
        self.assertIsNone(out)

    def test_empty_picks_returns_none(self):
        import pandas as pd
        from src.options_screener import offer_tearsheet
        self.assertIsNone(offer_tearsheet(pd.DataFrame(), _ctx(),
                                          interactive=True, preselect=1))


def _fixture_full(**over):
    d = _fixture()
    d["greeks_full"] = {
        "first": {"delta": 0.45, "gamma": 0.012, "vega": 0.31, "theta": -0.08},
        "second": {"rho": -0.021, "vanna": 0.0043, "charm": -0.0011},
        "dollar": {"vega_dollar": 31.0, "gamma_theta_ratio": 0.15,
                   "theta_burn_rate": 0.019, "abs_delta": 0.45},
    }
    d["quote"] = {"premium": 4.20, "bid": 4.15, "ask": 4.24, "mid": 4.195,
                  "spread_pct": 0.021, "volume": 1240, "oi": 8450, "oi_change": 320,
                  "liquidity_flag": "GOOD", "spread_flag": "OK",
                  "quote_freshness": "fresh", "iv_confidence": "high",
                  "iv_surface_confidence": 0.91, "prob_touch": 0.78,
                  "rr_ratio": 2.1, "annualized_return": 0.34, "be_dist_pct": 0.052,
                  "max_pain_dist_pct": 0.014, "gamma_pin_dist_pct": 0.028,
                  "strategy_name": "long_call"}
    d["ticket"] = {"entry_price": 4.20, "profit_target": 6.30, "stop_loss": 2.52,
                   "breakeven": 194.20, "max_loss": 420.0, "potential_profit": 210.0,
                   "risk_reward_ratio": 0.5, "guidance": "Use a limit at mid."}
    d["events"] = {"earnings": "2026-07-30",
                   "insider": {"n_buyers": 2, "n_buys": 3, "buy_value": 1.2e6,
                               "sell_value": 8.7e7, "score": 0.1,
                               "label": "net selling", "window_days": 90},
                   "news": ["Apple secures chip deal", "Apple analyst raises target"]}
    d["narrative"] = dict(d["narrative"],
                          thesis_caveat="rests on seasonal — see zone V, no demonstrated edge",
                          history={"n": 4, "win_rate": 0.25, "avg_pnl_pct": -0.09})
    d["name"] = dict(d["name"], uoa={"n_unusual": 12, "net_call_share": 0.61,
                                     "call_oi_added": 53105.0, "put_oi_added": 55789.0,
                                     "date": "2026-07-09"})
    d["panels"] = dict(d["panels"], **{k: {"status": "ok", "reason": ""}
                                       for k in ("earnings", "insider", "news")})
    for k, v in over.items():
        d[k] = v
    return d


def _pane(html, name):
    """Extract one tab pane's markup, so assertions can't be satisfied by the
    Raw tab's JSON dump of the whole snapshot."""
    start = html.index('<div class="tabpane tp-{}">'.format(name))
    end = html.index('<div class="tabpane', start + 10) if '<div class="tabpane' in html[start + 10:] \
        else html.index('</div><div class="demote"', start)
    return html[start:end]


class TestDetailDeck(unittest.TestCase):
    def test_all_five_tabs_render(self):
        out = R.render(_fixture_full())
        for pane in ("tp-greeks", "tp-execution", "tp-chain", "tp-events", "tp-raw"):
            self.assertIn(pane, out)

    def test_tabs_are_pure_css_no_javascript(self):
        out = R.render(_fixture_full())
        self.assertIn('type="radio" name="tsdeck"', out)
        self.assertNotIn('onclick="showTab', out)

    def test_second_order_greeks_are_shown(self):
        out = R.render(_fixture_full())
        for label in ("rho", "vanna", "charm", "Vega $/1% IV"):
            self.assertIn(label, out)

    def test_order_ticket_and_exit_plan_present(self):
        out = R.render(_fixture_full())
        self.assertIn("Order ticket", out)
        self.assertIn("Exit plan", out)
        self.assertIn("Profit target", out)
        self.assertIn("Stop loss", out)

    def test_events_tab_renders_the_slow_tier(self):
        # Assert against the EVENTS pane specifically. Asserting against the whole
        # document would be satisfied by the Raw tab's JSON dump of the sidecar.
        pane = _pane(R.render(_fixture_full()), "events")
        self.assertIn("2026-07-30", pane)                  # earnings
        self.assertIn("net selling", pane)                 # insider label
        self.assertIn("Apple secures chip deal", pane)     # news
        self.assertIn("buyer(s)", pane)                    # formatted, not a dict repr

    def test_no_raw_dict_reprs_reach_the_page(self):
        out = R.render(_fixture_full())
        self.assertNotIn("{'n_buyers'", out)
        self.assertNotIn("{'call_oi_added'", out)
        self.assertNotIn("'win_rate':", out)

    def test_uoa_and_insider_are_sentences(self):
        self.assertIn("call-led", R._fmt_uoa({"n_unusual": 12, "net_call_share": 0.61,
                                              "call_oi_added": 1.0, "put_oi_added": 2.0}))
        self.assertIn("buyer(s)", R._fmt_insider({"n_buyers": 2, "n_buys": 3,
                                                  "buy_value": 1.0, "sell_value": 2.0,
                                                  "label": "x", "window_days": 90}))
        self.assertIn("win rate", R._fmt_history({"n": 4, "win_rate": 0.25}))

    def test_no_caveat_hides_behind_a_tab(self):
        # The rendered evidence panel and the demoted no-edge zone must sit
        # OUTSIDE the deck. (The Raw tab necessarily contains the sidecar's text,
        # so this asserts on structure, not on substrings.)
        out = R.render(_fixture_full())
        deck_start = out.index('<div class="deck">')
        deck_end = out.index('<div class="demote">')
        deck = out[deck_start:deck_end]
        self.assertNotIn('<div class="eye" style="margin-bottom:4px">Model evidence</div>', deck)
        self.assertNotIn('class="demote"', deck)
        # ...and they really are above it, in the always-visible scroll.
        head = out[:deck_start]
        self.assertIn("Model evidence", head)
        self.assertLess(out.index("Model evidence"), deck_start)

    def test_print_css_expands_every_tab(self):
        out = R.render(_fixture_full())
        self.assertIn("@media print", out)
        self.assertIn(".tabpane { display:block !important;", out)

    def test_a_failing_tab_does_not_kill_the_page(self):
        d = _fixture_full()
        d["greeks_full"] = "not-a-dict"     # will blow up _greek_rows
        out = R.render(d)
        self.assertIn("<!DOCTYPE html>", out)
        self.assertIn("unavailable", out)

    def test_raw_tab_drops_the_price_series_and_nothing_else(self):
        # The caption promises "price series omitted". It used to drop the whole
        # `name` block — rsi, pcr, max pain and the flow summary went with it.
        d = _fixture_full()
        d["name"] = dict(d["name"], closes=[float(i) for i in range(130)])
        pane = _pane(R.render(d), "raw")
        self.assertIn("max_pain", pane)
        self.assertIn("rsi", pane)
        self.assertNotIn("127.0", pane)      # a close that appears nowhere else

    def test_events_tab_shows_not_fetched_placeholder(self):
        d = _fixture_full()
        d["panels"]["news"] = {"status": "not_fetched", "reason": "exceeded 2.5s budget"}
        out = R.render(d)
        self.assertIn("not fetched", out)
        self.assertIn("exceeded 2.5s budget", out)


class TestTermStructureHonesty(unittest.TestCase):
    def test_single_expiry_says_why_instead_of_drawing_nothing(self):
        d = _fixture()
        d["vol"] = dict(d["vol"], term=[[14, 0.23]])
        out = R.render(d)
        self.assertIn("no term curve", out)
        self.assertIn("need ≥2", out)

    def test_two_expiries_draw_a_curve(self):
        d = _fixture()
        d["vol"] = dict(d["vol"], term=[[14, 0.23], [44, 0.31]])
        out = R.render(d)
        self.assertNotIn("no term curve", out)
        self.assertIn("44d", out)


class TestCollectDetail(unittest.TestCase):
    def test_jsonable_recurses_into_dicts(self):
        out = collect._jsonable({"a": {"b": [1, float("nan")]}})
        self.assertEqual(out, {"a": {"b": [1, None]}})
        self.assertIsInstance(out["a"], dict)

    def test_term_curve_built_from_sibling_picks(self):
        siblings = [dict(_row(), T_years=14 / 365, impliedVolatility=0.23),
                    dict(_row(), T_years=44 / 365, impliedVolatility=0.31)]
        curve = collect._term_from_siblings(_row(), {"sibling_rows": siblings})
        self.assertEqual([c[0] for c in curve], [14, 44])

    def test_term_curve_ignores_other_tickers(self):
        siblings = [dict(_row(), symbol="MSFT", T_years=99 / 365, impliedVolatility=0.5)]
        curve = collect._term_from_siblings(_row(), {"sibling_rows": siblings})
        self.assertEqual([c[0] for c in curve], [36])   # falls back to the row itself

    def test_thesis_caveat_flags_no_edge_signals(self):
        self.assertIn("seasonal", collect._thesis_caveat("Strong seasonality (100% win)"))
        self.assertIsNone(collect._thesis_caveat("Cheap vs surface, wide gamma"))

    def test_build_populates_the_detail_blocks(self):
        d = collect.build(_row(), _ctx(), slow=False)
        self.assertIn("rho", d["greeks_full"]["second"])
        self.assertIn("mid", d["quote"])
        self.assertIn("entry_price", d["ticket"])
        self.assertEqual(set(d["events"]), {"earnings", "insider", "news"})
        json.dumps(d)


if __name__ == "__main__":
    unittest.main()
