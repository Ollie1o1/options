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

    def test_high_ic_is_not_badged_no_edge(self):
        d = _fixture()
        d["evidence"] = dict(d["evidence"], pooled_ic=0.31, p_value=0.001)
        d["context"] = []          # the context badges carry their own text
        self.assertNotIn("no edge", R.render(d))

    def test_missing_evidence_fails_closed(self):
        # An absent track record is not a good track record.
        d = _fixture()
        d["evidence"] = {"pooled_ic": None, "p_value": None, "n_oos": 0,
                         "cohort_n": 0, "gate_decision": "UNKNOWN", "as_of": None}
        out = R.render(d)
        self.assertIn("unknown", out)
        self.assertIn("no edge", out)

    def test_missing_evidence_is_never_badged_validated_for_the_scorer(self):
        d = _fixture()
        d["evidence"] = dict(d["evidence"], pooled_ic=None)
        _txt, badge = R._ic_badge(None)
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
        "strategy_name": "long_call",
    }


def _ctx():
    return {"mode": "Discovery scan", "rank": 3, "n_picks": 9, "spot": 182.40,
            "rfr": 0.043, "vix": 17.2, "vix_regime": "normal",
            "config": {}, "config_sha": "4f1a9c"}


class TestCollect(unittest.TestCase):
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
                json.dump(_fixture(), f)
            rc = main(["--from", side, "--no-open"])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(os.path.join(d, "x.html")))


if __name__ == "__main__":
    unittest.main()
