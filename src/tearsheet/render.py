"""TearsheetData -> HTML, composed on the desk kit. A pure function: no
network, no DB, no clock.

Purity is load-bearing. It is what makes `--from <sidecar>.json` reproduce a
page byte-for-byte months later, and what lets the tests run without mocks.

Layout: verdict hero first (the lead number proves the thesis in five
seconds), then anchored card sections — payoff, risk, vol, the name,
narrative & evidence — with depth folded into a tab deck. Nothing that
QUALIFIES the trade hides behind a tab.
"""
import html as _html
import math

from src.desk_kit import charts, shell, theme

_ZONES = ("decision", "vol", "name", "narrative", "detail", "context")


def _esc(v) -> str:
    return _html.escape("" if v is None else str(v))


def _num(v, fmt="{:,.2f}", dash="—"):
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(f):
        return dash
    return fmt.format(f)


def _panel_ok(data, pid) -> bool:
    return data.get("panels", {}).get(pid, {}).get("status") == "ok"


def _placeholder(data, pid) -> str:
    p = data.get("panels", {}).get(pid, {})
    status = p.get("status", "unavailable")
    reason = _esc(p.get("reason", ""))
    word = "not fetched" if status == "not_fetched" else "unavailable"
    return ('<div class="ph">{w}{sep}{r}</div>'
            .format(w=word, sep=" — " if reason else "", r=reason))


_CONTRACT_MULTIPLIER = 100.0


def decide_verdict(net_ev, gross_ev, cost, waterfall, noise=0.0):
    """(decision, reason). One of TAKE / MARGINAL / SKIP / INDETERMINATE.

    Never consults the quality score: its out-of-sample IC is ~0.03. When net EV
    is missing or non-finite (the HV-fallback path sets it to NaN) the verdict is
    INDETERMINATE rather than a silent fallback to a signal with no edge.

    Three things this refuses to conflate:

    * A contract with **no gross edge** was mispriced against you before a single
      cent of cost. Blaming the spread there is a misattribution — and it implies
      a better fill could rescue a trade that no fill can.
    * A net EV inside `noise` is a **point estimate smaller than its own error
      bar**. Net EV is Black-Scholes over an implied vol this project's own
      cross-check corrects on most scans; +$1 is not an edge.
    * A negative net EV inside `noise` is still a SKIP. Symmetry would call it
      unresolvable, but a trade you cannot justify is a pass.
    """
    v = _finite_or_none(net_ev)
    if v is None:
        return "INDETERMINATE", (
            "Net expected value is unavailable for this contract, so no verdict "
            "is offered. It is never inferred from the quality score.")

    band = abs(_finite_or_none(noise) or 0.0)
    gross = _finite_or_none(gross_ev)

    if gross is not None and gross <= 0:
        return "SKIP", (
            "This contract is priced above its model-fair value even at mid: the "
            "gross edge is {} per contract before any transaction cost. The "
            "round-trip cost is not what breaks it.".format(_num(gross, "${:+,.0f}")))

    if v > band:
        return "TAKE", (
            "Net expected value is {} per contract after the {} round-trip cost, "
            "clear of the {} band implied by this contract's vol uncertainty."
            .format(_num(v, "${:+,.0f}"), _num(cost, "${:,.0f}"),
                    _num(band, "±${:,.0f}")))

    if v > 0:
        return "MARGINAL", (
            "Net expected value is {} per contract — positive, but inside the {} "
            "noise band this contract's vega implies for a one-sigma error in its "
            "implied vol. The sign of the edge is not resolvable from this data."
            .format(_num(v, "${:+,.0f}"), _num(band, "±${:,.0f}")))

    negatives = [(l, x) for l, x in (waterfall or []) if x < 0]
    worst = min(negatives, key=lambda t: t[1])[0] if negatives else "transaction cost"
    return "SKIP", (
        "Net expected value is {} per contract. The gross edge of {} does not "
        "survive the {} round-trip cost; the largest single drag is {}."
        .format(_num(v, "${:+,.0f}"), _num(gross_ev, "${:+,.0f}"),
                _num(cost, "${:,.0f}"), worst))


def fill_to_flip(net_ev, target, assumed_fill, multiplier=_CONTRACT_MULTIPLIER):
    """The entry fill (per share) at which net EV would reach `target`.

    The cost model charges a half-spread on entry, i.e. it assumes you pay the
    ask. Net EV moves one-for-one with what you actually pay, so every cent of
    price improvement is worth `multiplier` cents of expectation. Returns None
    when no positive price gets there — some contracts are simply not buyable.
    """
    net = _finite_or_none(net_ev)
    fill = _finite_or_none(assumed_fill)
    tgt = _finite_or_none(target) or 0.0
    if net is None or fill is None:
        return None
    price = fill - (tgt - net) / multiplier
    return price if price > 0 else None


def _flip_line(decision, verdict, quote) -> str:
    """The one actionable number on a page that says don't.

    A SKIP is a statement about a price, not about a contract. Say which price.

    The quoted fill is floored to the cent below the exact break-even, so that a
    limit order actually placed at it lands on the TAKE side of the band rather
    than back on its boundary.
    """
    if decision not in ("SKIP", "MARGINAL"):
        return ""
    noise = abs(_finite_or_none(verdict.get("noise")) or 0.0)
    fill = _finite_or_none(verdict.get("assumed_fill"))
    if fill is None:
        return ""
    exact = fill_to_flip(verdict.get("net_ev"), noise, fill)
    if exact is None:
        return ('<div class="flip">No entry price makes this a TAKE &mdash; the '
                "expectation is negative before the contract is even paid for.</div>")
    # Largest whole cent STRICTLY below the exact break-even, so a limit order
    # placed at it clears the band (TAKE is net > band, strict) rather than
    # landing back on its boundary. ceil(x)-1 == floor(x) off a cent boundary and
    # steps down a cent when the break-even sits exactly on one.
    price = (math.ceil(exact * 100.0) - 1) / 100.0
    if price <= 0:
        return ('<div class="flip">No entry price makes this a TAKE &mdash; the '
                "expectation is negative before the contract is even paid for.</div>")
    bid = _finite_or_none((quote or {}).get("bid"))
    if bid is not None and price < bid:
        return ('<div class="flip">No fill inside the current quote makes this a TAKE: '
                "it would take {p} against a {b} bid. Passing is the only price."
                "</div>".format(p=_num(price, "${:,.2f}"), b=_num(bid, "${:,.2f}")))
    return ('<div class="flip">Becomes a <strong>TAKE</strong> at a fill of {p} or '
            "better. The cost model assumes you pay {f}; each cent of price "
            "improvement is worth {m} of expectation.</div>".format(
                p=_num(price, "${:,.2f}"), f=_num(fill, "${:,.2f}"),
                m=_num(_CONTRACT_MULTIPLIER / 100.0, "${:,.0f}")))


# Tearsheet-only styling on top of the kit sheet.
_CSS = """
h1 { font-size:23px; margin:2px 0 0; color:var(--ink-strong); font-weight:650;
  font-family:ui-monospace,"SF Mono",Menlo,monospace; letter-spacing:.01em; }
.hero { padding-top:12px; }
.demote { opacity:.75; background:var(--panel); border:1px solid var(--rule);
  border-radius:5px; padding:12px 14px; margin-top:12px; }
.cols2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; min-width:0; }
@media (max-width:760px){ .cols2{grid-template-columns:1fr} }

/* Detail deck: radio + :checked, no JavaScript — the depth tabs keep working
   in viewers that strip scripts, and print expands every pane. */
.deck { margin-top:4px; }
.tabin { position:absolute; opacity:0; pointer-events:none; }
.deck .tabbar { border-bottom:1px solid var(--rule); margin:0 0 12px; }
.tablab { cursor:pointer; font-family:ui-sans-serif,system-ui,sans-serif;
  font-size:9.5px; letter-spacing:.14em; text-transform:uppercase;
  font-weight:600; color:var(--muted); padding:7px 12px;
  border:1px solid transparent; border-bottom:none;
  border-radius:3px 3px 0 0; }
.tablab:hover { color:var(--ink); }
.tabpane { display:none; }
#tab-greeks:checked ~ .tabbar label[for="tab-greeks"],
#tab-execution:checked ~ .tabbar label[for="tab-execution"],
#tab-chain:checked ~ .tabbar label[for="tab-chain"],
#tab-events:checked ~ .tabbar label[for="tab-events"],
#tab-raw:checked ~ .tabbar label[for="tab-raw"] {
  color:var(--ink-strong); border-color:var(--rule); background:var(--paper); }
#tab-greeks:checked ~ .tp-greeks,
#tab-execution:checked ~ .tp-execution,
#tab-chain:checked ~ .tp-chain,
#tab-events:checked ~ .tp-events,
#tab-raw:checked ~ .tp-raw { display:block; }

/* Paper never hides anything: printing expands every tab. */
@media print {
  .deck .tabbar { display:none; }
  .tabpane { display:block !important; page-break-inside:avoid;
    margin-bottom:14px; }
}
"""


def _masthead(data):
    m = data.get("meta", {})
    title = '<span class="chip">{tkr} {strike:g}{tl} · {exp}</span>'.format(
        tkr=_esc(m.get("ticker")), strike=float(m.get("strike") or 0),
        tl=_esc((m.get("opt_type") or "c")[0].upper()), exp=_esc(m.get("expiration")))
    meta = shell.chipline([
        ("spot", _num(m.get("spot"))), ("rfr", _num(m.get("rfr"), "{:.2%}")),
        ("VIX", "{} ({})".format(_num(m.get("vix"), "{:.1f}"),
                                 _esc(m.get("vix_regime")))),
        ("gen", _esc(m.get("generated_at"))),
    ])
    nav = shell.anchor_nav([("verdict", "Verdict"), ("payoff", "Payoff"),
                            ("risk", "Risk"), ("vol", "Vol"),
                            ("name", "The name"), ("evidence", "Evidence"),
                            ("detail", "Detail")])
    return shell.masthead("TEARSHEET", title, meta_html=meta, nav_html=nav,
                          where="tearsheets")


_VERDICT_CLASS = {"TAKE": "v-take", "SKIP": "v-skip", "MARGINAL": "v-marg",
                  "INDETERMINATE": "v-ind"}


def _sv_tone(v, invert=False):
    """Green/red ink for a signed headline number. Neutral when unknown."""
    f = _finite_or_none(v)
    if f is None or f == 0:
        return ""
    good = (f < 0) if invert else (f > 0)
    return "good" if good else "bad"


def _hero(data) -> str:
    m = data.get("meta", {})
    v = data["verdict"]
    decision, reason = decide_verdict(v.get("net_ev"), v.get("gross_ev"),
                                      v.get("cost"), data.get("cost_waterfall"),
                                      noise=v.get("noise"))
    s = data.get("stats", {})
    cells = (("Net EV", _num(v.get("net_ev"), "${:+,.0f}"), _sv_tone(v.get("net_ev"))),
             ("Noise band", _num(v.get("noise"), "±${:,.0f}"), ""),
             ("Gross EV", _num(v.get("gross_ev"), "${:+,.0f}"), _sv_tone(v.get("gross_ev"))),
             ("RT cost", _num(v.get("cost"), "${:,.0f}"), ""),
             ("POP", _num(s.get("pop"), "{:.0%}"), ""),
             ("Max loss", _num(s.get("max_loss"), "${:,.0f}"), ""),
             ("Breakeven", _num(s.get("breakeven"), "{:,.2f}"), ""))
    eye = ("{mode} &middot; pick {rank} of {n} &middot; {typ} &middot; "
           "{dte} DTE").format(
        mode=_esc(m.get("mode")), rank=_esc(m.get("rank")),
        n=_esc(m.get("n_picks")), typ=_esc(m.get("opt_type")),
        dte=_esc(m.get("dte")))
    return (
        '<section class="hero" id="verdict">'
        '<div class="eye">{eye}</div>'
        "<h1>{tkr} {strike:g}{tl} &mdash; {exp}</h1>"
        '<div style="margin-top:10px"><span class="verdict {c}">{d}</span></div>'
        '<p class="lede">{r}</p>{flip}{s}</section>'
    ).format(eye=eye, tkr=_esc(m.get("ticker")), strike=float(m.get("strike") or 0),
             tl=_esc((m.get("opt_type") or "c")[0].upper()),
             exp=_esc(m.get("expiration")),
             c=_VERDICT_CLASS.get(decision, "v-ind"), d=_esc(decision),
             r=_esc(reason), flip=_flip_line(decision, v, data.get("quote") or {}),
             s=shell.strip(cells))


def _rows(pairs) -> str:
    return "".join('<tr><td>{}</td><td class="n m">{}</td></tr>'.format(_esc(k), _esc(v))
                   for k, v in pairs)


def _payoff_section(data) -> str:
    """Payoff card + cost-waterfall card. The payoff line is arithmetic over
    the contract itself; the dashed today curve appears only when the sidecar
    (schema ≥ 2) carries model repricing points."""
    m = data.get("meta", {})
    q = data.get("quote") or {}
    v = data.get("verdict") or {}
    prem = _finite_or_none(q.get("premium"))
    if prem is None:
        prem = _finite_or_none((data.get("ticket") or {}).get("entry_price"))
    if prem is None:
        prem = _finite_or_none(v.get("assumed_fill"))
    po = data.get("payoff") or {}
    svg = ""
    if prem is not None:
        svg = charts.payoff_chart(
            m.get("spot"), m.get("strike"), m.get("opt_type"), prem,
            breakeven=(data.get("stats") or {}).get("breakeven"),
            today_prices=po.get("prices"), today_pnl=po.get("today_pnl"))
    left = (svg + '<div class="eye" style="margin-top:4px">P&amp;L per contract '
            "at expiry (solid) &middot; model P&amp;L today (dashed)"
            "</div>") if svg else shell.ph(
                "payoff unavailable — no entry premium on this sidecar")
    left += _exit_block(data)
    wf = charts.waterfall(data.get("cost_waterfall"))
    right = (wf or shell.ph("no cost breakdown")) + (
        '<div class="eye" style="margin-top:4px">Gross model edge minus the '
        "round-trip cost of actually trading it</div>")
    return shell.grid([
        shell.card("Payoff — {} {:g}{} @ {}".format(
            _esc(m.get("ticker")), float(m.get("strike") or 0),
            _esc((m.get("opt_type") or "c")[0].upper()),
            _num(prem, "${:,.2f}")), left, span=7, anchor="payoff"),
        shell.card("Cost wall — where the edge goes", right, span=5),
    ])


def _heat_grid(stress) -> str:
    """Spot x IV grid. Each cell carries BOTH inks; CSS picks one."""
    return charts.heat_grid(stress, theme.heat_inks)


def _exit_block(data) -> str:
    """The exit-aware read: hold-to-expiry is a fiction, so show what the
    book's OWN rules do to this contract — which exit fires first, the
    expected P&L under those rules, and the peak-multiple ladder if held.
    Degrades to a labelled placeholder, never silently absent."""
    ex = data.get("exits")
    if not ex:
        if "exits" in (data.get("panels") or {}):
            return ('<div style="margin-top:10px"></div>'
                    + _placeholder(data, "exits"))
        return ""   # pre-exits sidecar (schema ≤2): say nothing, claim nothing
    peak = ex.get("peak") or {}
    pg = peak.get("p_ge") or {}
    cells = (
        ("Rule EV", _num(ex.get("ev_exit_per_contract"), "${:+,.0f}"),
         _sv_tone(ex.get("ev_exit_per_contract"))),
        ("P TP first", _num(ex.get("p_tp"), "{:.0%}"), ""),
        ("P time-exit", _num(ex.get("p_time"), "{:.0%}"), ""),
        ("P stop", _num(ex.get("p_sl"), "{:.0%}"), ""),
        ("Med peak", _num(peak.get("med_mult"), "{:.2f}×"), ""),
        ("P ≥2×", _num(pg.get("2"), "{:.0%}"), ""),
        ("P ≥3×", _num(pg.get("3"), "{:.0%}"), ""),
    )
    rules = ex.get("rules") or {}
    if rules.get("tiered_tp"):
        tps = rules["tiered_tp"]
        rule_txt = "TP {:.0%}/{:.0%}/{:.0%} by DTE".format(*tps)
    else:
        rule_txt = "TP +{:.0%} or Δ≥{:.2f} · SL {:+.0%}".format(
            rules.get("tp") or 0, rules.get("tp_delta") or 0,
            rules.get("sl") or 0)
    note = ("Your exit rules simulated ({rt} · {te}d time-exit · {n:,} paths): "
            "peak ladder is if-held-to-expiry, not rule-truncated. {a}.").format(
        rt=rule_txt, te=rules.get("time_exit_dte"),
        n=int(ex.get("n_paths") or 0), a=_esc(ex.get("assumptions") or ""))
    return ('<div class="eye" style="margin:12px 0 0">Exit odds — the P&amp;L '
            "process you actually run</div>"
            + shell.strip(cells)
            + '<div class="eye" style="margin-top:5px;line-height:1.5">{}</div>'
            .format(note))


def _risk_section(data) -> str:
    g, l = data.get("greeks") or {}, data.get("liquidity") or {}
    heat = _heat_grid(data.get("stress", {}))
    left = ((heat or shell.ph("stress grid unavailable"))
            + '<div class="eye" style="margin-top:5px">worst {}</div>'.format(
                _esc(data.get("stress", {}).get("worst", "n/a"))))
    right = ('<div class="cols2"><div>'
             '<div class="eye" style="margin-bottom:4px">Greeks (per contract)</div>'
             "<table>" + _rows((
                 ("Delta", _num(g.get("delta"))),
                 ("Gamma", _num(g.get("gamma"), "{:.3f}")),
                 ("Vega", _num(g.get("vega"))),
                 ("Theta", _num(g.get("theta")) + " /day"))) + "</table></div>"
             '<div><div class="eye" style="margin-bottom:4px">Liquidity</div>'
             "<table>" + _rows((
                 ("Spread", _num(l.get("spread_pct"), "{:.1%}")),
                 ("Open interest", _num(l.get("oi"), "{:,.0f}")),
                 ("Volume", _num(l.get("volume"), "{:,.0f}")),
                 ("Quote", _esc(l.get("quote_freshness"))))) + "</table></div></div>")
    return shell.grid([
        shell.card("Stress — P&L across spot × IV", left, span=7,
                   anchor="risk"),
        shell.card("Greeks & liquidity", right, span=5),
    ])


def _vol_section(data) -> str:
    if not _panel_ok(data, "vol"):
        return ('<div class="grid">'
                + shell.card("Vol", _placeholder(data, "vol"), span=12,
                             anchor="vol") + "</div>")
    v = data["vol"]
    resid = v.get("svi_residual")
    if resid is None:
        rich = "no surface fit"
    elif float(resid) > 0:
        rich = "RICH +{:.2f}σ vs surface".format(float(resid))
    else:
        rich = "CHEAP {:.2f}σ vs surface".format(float(resid))
    cone = charts.cone_chart(v.get("cone"))
    left = ((cone or shell.ph("vol cone unavailable"))
            + "<table>" + _rows((
                ("IV / HV30", "{} / {}".format(_num(v.get("iv"), "{:.1%}"),
                                               _num(v.get("hv"), "{:.1%}"))),
                ("VRP", _num(v.get("vrp"), "{:+.1%}")),
                ("IV rank", _num(v.get("iv_rank"), "{:.0%}")),
                ("vs SVI", rich))) + "</table>")
    term = v.get("term") or []
    if len(term) >= 2:
        term_html = charts.term_chart(term)
    else:
        # A heading over a blank chart reads as "no term structure". Say why.
        term_html = shell.ph("no term curve — this scan surfaced only "
                             "one expiry for this name (need ≥2)")
    right = (term_html + "<table>" + _rows((
        ("Skew 25Δ", "{}vp".format(_num(v.get("skew_vp"), "{:+.1f}"))),
        ("Skew rank", _num(v.get("skew_rank"), "{:.0%}")),
        ("Expected move", _num(v.get("expected_move"))),
        ("Required move", _num(v.get("required_move"))))) + "</table>")
    return shell.grid([
        shell.card("Vol complex — cone vs current", left, span=6, anchor="vol"),
        shell.card("Term structure & skew", right, span=6),
    ])


_IC_EDGE_THRESHOLD = 0.05
_P_VALUE_THRESHOLD = 0.05


def _badge(text, kind) -> str:
    return shell.badge(text, kind)


def _finite_or_none(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _has_scorer_edge(pooled_ic, p_value) -> bool:
    """An edge requires BOTH magnitude and significance.

    A large IC with p=0.48 is a coin flip wearing a decimal point. The project's
    own walk-forward reports an IC of +0.10 at p=0.48 on n=94 — reporting that as
    'has edge' would make this panel certify the very noise it exists to expose.
    """
    ic = _finite_or_none(pooled_ic)
    p = _finite_or_none(p_value)
    return (ic is not None and ic >= _IC_EDGE_THRESHOLD
            and p is not None and p < _P_VALUE_THRESHOLD)


def _ic_badge(pooled_ic, p_value=None):
    """(display_text, badge_html). Missing evidence fails closed.

    An absent track record is not a good track record. Neither is an
    insignificant one.
    """
    ic = _finite_or_none(pooled_ic)
    if ic is None:
        return "unknown", _badge("no edge", "bad")
    txt = _num(ic, "{:+.2f}")
    if ic < _IC_EDGE_THRESHOLD:
        return txt, _badge("no edge", "bad")
    if _has_scorer_edge(pooled_ic, p_value):
        return txt, _badge("has edge", "ok")
    return txt, _badge("underpowered", "warn")


def _fmt_uoa(uoa) -> str:
    """flow_summary() -> a sentence, not a repr of its dict."""
    if not isinstance(uoa, dict):
        return "n/a"
    share = uoa.get("net_call_share")
    lean = "n/a"
    if isinstance(share, (int, float)):
        lean = "call-led" if share > 0.55 else ("put-led" if share < 0.45 else "balanced")
    return "{} unusual · {} (calls {:+,.0f} / puts {:+,.0f} OI)".format(
        _num(uoa.get("n_unusual"), "{:,.0f}"), lean,
        float(uoa.get("call_oi_added") or 0), float(uoa.get("put_oi_added") or 0))


def _fmt_insider(ins) -> str:
    """cluster_score() -> a sentence, not a repr of its dict."""
    if not isinstance(ins, dict):
        return "n/a"
    return "{} · {} buyer(s) / {} buy(s) over {}d · bought {} vs sold {}".format(
        _esc(ins.get("label") or "no signal"),
        _num(ins.get("n_buyers"), "{:,.0f}"), _num(ins.get("n_buys"), "{:,.0f}"),
        _num(ins.get("window_days"), "{:,.0f}"),
        _num(ins.get("buy_value"), "${:,.0f}"), _num(ins.get("sell_value"), "${:,.0f}"))


def _fmt_history(h) -> str:
    """analog_stats() -> a sentence, not a repr of its dict."""
    if not isinstance(h, dict):
        return "no prior trades"
    n = h.get("n") or h.get("count")
    parts = []
    if n is not None:
        parts.append("{} comparable trade(s)".format(_num(n, "{:,.0f}")))
    for key, label, fmt in (("win_rate", "win rate", "{:.0%}"),
                            ("avg_pnl_pct", "avg", "{:+.1%}"),
                            ("median_pnl_pct", "median", "{:+.1%}")):
        if h.get(key) is not None:
            parts.append("{} {}".format(label, _num(h[key], fmt)))
    return " · ".join(parts) if parts else "no prior trades"


def _name_section(data) -> str:
    if not _panel_ok(data, "name"):
        return ('<div class="grid">'
                + shell.card("The name", _placeholder(data, "name"), span=12,
                             anchor="name") + "</div>")
    n = data["name"]
    sup = (n.get("supports") or [{}])
    res = (n.get("resistances") or [{}])
    price = charts.price_chart(
        n.get("closes"), None, None,
        sup[0] if sup and sup[0].get("level") is not None else None,
        res[0] if res and res[0].get("level") is not None else None,
        None, "px", h=220)
    left = ((price or shell.ph("price history unavailable"))
            + "<table>" + _rows((
                ("RSI(14)", _num(n.get("rsi"), "{:.0f}")),
                ("5d return", _num(n.get("ret_5d"), "{:+.1%}")),
                ("Support", _num(sup[0].get("level"))),
                ("Resistance", _num(res[0].get("level"))))) + "</table>")
    right = ("<table>" + _rows((
        ("Put/call ratio", _num(n.get("pcr"))),
        ("OI change (1d)", _num(n.get("oi_change"), "{:+,.0f}")),
        ("Unusual activity", _fmt_uoa(n.get("uoa"))),
        ("Max pain", _num(n.get("max_pain"))))) + "</table>")
    return shell.grid([
        shell.card("The name — {}".format(_esc(data.get("meta", {}).get("ticker"))),
                   left, span=7, anchor="name"),
        shell.card("Flow & positioning", right, span=5),
    ])


def _evidence_body(data) -> str:
    e = data.get("evidence", {})
    ic_txt, ic_badge = _ic_badge(e.get("pooled_ic"), e.get("p_value"))
    gate = _esc(e.get("gate_decision") or "UNKNOWN")
    p_txt = _num(e.get("p_value"), "{:.3f}")
    cohort_n = _finite_or_none(e.get("cohort_n")) or 0
    rows = (
        '<tr><td>Scorer OOS IC</td><td class="n m">{}</td><td class="n">{}</td></tr>'.format(
            ic_txt, ic_badge),
        '<tr><td>p-value</td><td class="n m">{}</td><td class="n mut">{}</td></tr>'.format(
            p_txt, "significant" if _has_scorer_edge(e.get("pooled_ic"), e.get("p_value"))
            else "not significant"),
        '<tr><td>Walk-forward n</td><td class="n m">{}</td><td class="n mut">{}</td></tr>'.format(
            _num(e.get("n_oos"), "{:,.0f}"), _esc(e.get("as_of") or "n/a")),
        '<tr><td>Cohort gate</td><td class="n m">{} / 50</td><td class="n">{}</td></tr>'.format(
            _num(e.get("cohort_n"), "{:,.0f}"), _badge(gate.lower(), "warn")),
        '<tr><td>Cost model</td><td class="n m">round-trip</td><td class="n">{}</td></tr>'.format(
            _badge("validated", "ok")),
    )
    # Conditional, not hardcoded — and gated on significance, not magnitude alone.
    if _has_scorer_edge(e.get("pooled_ic"), e.get("p_value")):
        caption = ("The ranking that surfaced this pick has statistically significant "
                   "out-of-sample skill. The cost and Greeks arithmetic above is "
                   "independent of it.")
    else:
        caption = ("The ranking that surfaced this pick has no demonstrated "
                   "out-of-sample skill. The cost and Greeks arithmetic above does.")
    return ('<div class="eye" style="margin-bottom:4px">Model evidence</div>'
            "<table>{}</table>"
            + charts.meter(cohort_n, 50)
            + '<div class="eye" style="margin-top:6px;line-height:1.5">{}</div>'
            ).format("".join(rows), _esc(caption))


def _narrative_section(data) -> str:
    """Narrative card + evidence card. ALWAYS rendered, even when the
    narrative panel failed: the evidence card qualifies everything above it
    and must never vanish with a bad thesis fetch."""
    if _panel_ok(data, "narrative"):
        nar = data.get("narrative", {})
        fit = nar.get("portfolio_fit") or []
        left = '<p class="lede">{}</p>'.format(_esc(nar.get("thesis")))
        caveat = nar.get("thesis_caveat")
        if caveat:
            # The thesis may lean on a signal this very page marks as no-edge.
            left += '<div class="eye" style="margin:-4px 0 8px">⚠ {}</div>'.format(_esc(caveat))
        if fit:
            counts = {}
            for f in fit:
                counts[f] = counts.get(f, 0) + 1
            fit_txt = "{} open on this name: {}".format(
                len(fit), ", ".join("{}×{}".format(v, k) for k, v in sorted(counts.items())))
        else:
            fit_txt = "no open positions on this name"
        left += "<table>"
        left += _rows((("Vehicle verdict", nar.get("vehicle") or "n/a"),
                       ("Portfolio fit", fit_txt),
                       ("Your history", _fmt_history(nar.get("history")))))
        left += "</table>"
    else:
        left = _placeholder(data, "narrative")
    return shell.grid([
        shell.card("Narrative & provenance", left, span=6, anchor="evidence"),
        shell.card("Evidence", _evidence_body(data), span=6),
    ])


def _greek_rows(data):
    g = data.get("greeks_full") or {}
    first, second, dollar = g.get("first", {}), g.get("second", {}), g.get("dollar", {})
    return ("<table>" + _rows((
        ("Δ delta", _num(first.get("delta"), "{:+.4f}")),
        ("Γ gamma", _num(first.get("gamma"), "{:+.5f}")),
        ("ν vega", _num(first.get("vega"), "{:+.4f}")),
        ("Θ theta", _num(first.get("theta"), "{:+.4f}") + " /day"),
    )) + "</table>",
        "<table>" + _rows((
            ("ρ rho", _num(second.get("rho"), "{:+.4f}")),
            ("vanna", _num(second.get("vanna"), "{:+.5f}")),
            ("charm", _num(second.get("charm"), "{:+.5f}")),
        )) + "</table>",
        "<table>" + _rows((
            ("Vega $/1% IV", _num(dollar.get("vega_dollar"), "${:+,.2f}")),
            ("Γ/Θ ratio", _num(dollar.get("gamma_theta_ratio"), "{:,.3f}")),
            ("Θ burn rate", _num(dollar.get("theta_burn_rate"), "{:.2%}") + " /day"),
            ("|Δ|", _num(dollar.get("abs_delta"), "{:.4f}")),
        )) + "</table>")


def _tab_greeks(data) -> str:
    a, b, c = _greek_rows(data)
    return ('<div class="grid" style="margin-top:0">'
            '<div class="c4"><div class="eye">First order</div>{}</div>'
            '<div class="c4"><div class="eye">Second order</div>{}</div>'
            '<div class="c4"><div class="eye">Dollar exposure</div>{}</div></div>'
            ).format(a, b, c)


def _tab_execution(data) -> str:
    """Order ticket + exit plan. The stress grid deliberately is NOT repeated
    here: it is decision-grade and stays in the always-visible risk section."""
    t = data.get("ticket") or {}
    left = ('<div class="eye" style="margin-bottom:4px">Order ticket</div><table>'
            + _rows((("Limit price", _num(t.get("entry_price"), "${:,.2f}")),
                     ("Breakeven", _num(t.get("breakeven"), "{:,.2f}")),
                     ("Max loss", _num(t.get("max_loss"), "${:,.0f}")),
                     ("Potential profit", _num(t.get("potential_profit"), "${:,.0f}")),
                     ("Risk / reward", _num(t.get("risk_reward_ratio"), "{:,.2f}x"))))
            + "</table>")
    right = ('<div class="eye" style="margin-bottom:4px">Exit plan</div><table>'
             + _rows((("Profit target", _num(t.get("profit_target"), "${:,.2f}")),
                      ("Stop loss", _num(t.get("stop_loss"), "${:,.2f}"))))
             + "</table>")
    if t.get("guidance"):
        right += '<div class="eye" style="margin-top:6px;line-height:1.5">{}</div>'.format(
            _esc(t["guidance"]))
    return '<div class="cols2"><div>{}</div><div>{}</div></div>'.format(left, right)


def _tab_chain(data) -> str:
    q = data.get("quote") or {}
    m = data.get("meta") or {}
    left = "<table>" + _rows((
        ("Contract", "{} {:g}{} {}".format(m.get("ticker"), float(m.get("strike") or 0),
                                           str(m.get("opt_type", "c"))[0].upper(),
                                           m.get("expiration"))),
        ("Strategy", _esc(q.get("strategy_name") or "n/a")),
        ("DTE", _num(m.get("dte"), "{:,.0f}")),
        ("Premium", _num(q.get("premium"), "${:,.2f}")),
        ("Bid / Ask", "{} / {}".format(_num(q.get("bid"), "${:,.2f}"),
                                       _num(q.get("ask"), "${:,.2f}"))),
        ("Mid", _num(q.get("mid"), "${:,.2f}")),
        ("Spread", _num(q.get("spread_pct"), "{:.2%}")),
    )) + "</table>"
    right = "<table>" + _rows((
        ("Volume", _num(q.get("volume"), "{:,.0f}")),
        ("Open interest", _num(q.get("oi"), "{:,.0f}")),
        ("OI change", _num(q.get("oi_change"), "{:+,.0f}")),
        ("Liquidity", _esc(q.get("liquidity_flag") or "n/a")),
        ("Spread flag", _esc(q.get("spread_flag") or "n/a")),
        ("Quote", _esc(q.get("quote_freshness") or "n/a")),
        ("IV confidence", _esc(q.get("iv_confidence") or "n/a")),
        ("SVI fit quality", _num(q.get("iv_surface_confidence"), "{:.0%}")),
    )) + "</table>"
    third = "<table>" + _rows((
        ("Prob. of touch", _num(q.get("prob_touch"), "{:.0%}")),
        ("Risk / reward", _num(q.get("rr_ratio"), "{:,.2f}x")),
        ("Annualised return", _num(q.get("annualized_return"), "{:+.1%}")),
        ("Breakeven distance", _num(q.get("be_dist_pct"), "{:+.2%}")),
        ("Max-pain distance", _num(q.get("max_pain_dist_pct"), "{:+.2%}")),
        ("Gamma-pin distance", _num(q.get("gamma_pin_dist_pct"), "{:+.2%}")),
    )) + "</table>"
    return ('<div class="grid" style="margin-top:0">'
            '<div class="c4"><div class="eye">Contract</div>{}</div>'
            '<div class="c4"><div class="eye">Liquidity &amp; quote</div>{}</div>'
            '<div class="c4"><div class="eye">Derived</div>{}</div></div>'
            ).format(left, right, third)


def _event_panel(data, pid, title, body_fn) -> str:
    if not _panel_ok(data, pid):
        return '<div class="c4"><div class="eye">{}</div>{}</div>'.format(
            _esc(title), _placeholder(data, pid))
    return '<div class="c4"><div class="eye">{}</div>{}</div>'.format(
        _esc(title), body_fn())


def _tab_events(data) -> str:
    """The slow tier. Collected under a 2.5s budget — and, until now, discarded."""
    ev = data.get("events") or {}

    def _earn():
        val = ev.get("earnings")
        return '<div class="lede">{}</div>'.format(
            _esc(val) if val else "no earnings date before expiry")

    def _ins():
        return '<div class="lede">{}</div>'.format(_fmt_insider(ev.get("insider")))

    def _news():
        items = ev.get("news")
        if isinstance(items, str):
            items = [x.strip() for x in items.split(";") if x.strip()]
        if not items:
            return '<div class="lede">no recent headlines</div>'
        return "<ul style='margin:4px 0;padding-left:16px;font-size:12.5px'>" + "".join(
            "<li>{}</li>".format(_esc(i)) for i in items[:5]) + "</ul>"

    return ('<div class="grid" style="margin-top:0">{}{}{}</div>').format(
        _event_panel(data, "earnings", "Earnings", _earn),
        _event_panel(data, "insider", "Insider (EDGAR, 90d)", _ins),
        _event_panel(data, "news", "News", _news))


def _tab_raw(data) -> str:
    """The sidecar, minus the 130 daily closes that would drown it.

    Only `name.closes` is dropped. Dropping the whole `name` block — as this
    once did — silently took RSI, put/call, max pain and the flow summary with
    it, under a caption that claimed only the price series had gone.
    """
    import json as _json
    safe = dict(data)
    name = safe.get("name")
    if isinstance(name, dict) and name.get("closes"):
        safe["name"] = dict(name, closes="<{} daily closes omitted>".format(
            len(name["closes"])))
    blob = _json.dumps(safe, indent=2, sort_keys=True)
    return ('<div class="eye" style="margin-bottom:4px">Sidecar snapshot '
            "(price series omitted) &mdash; this is exactly what regenerates the page</div>"
            '<pre class="raw">{}</pre>').format(_esc(blob))


_TABS = (("greeks", "Greeks", _tab_greeks),
         ("execution", "Execution", _tab_execution),
         ("chain", "Chain &amp; quote", _tab_chain),
         ("events", "Events", _tab_events),
         ("raw", "Raw", _tab_raw))


def _zone_detail(data) -> str:
    """Tabbed depth. Pure CSS (radio + :checked) so it works offline with no JS.

    Nothing that QUALIFIES the trade lives here — the verdict, cost wall,
    model evidence and the no-edge zone all stay in the always-visible scroll.
    Only depth hides behind a tab. Print CSS expands every tab.
    """
    inputs, labels, panes = [], [], []
    for i, (key, label, fn) in enumerate(_TABS):
        checked = " checked" if i == 0 else ""
        inputs.append('<input type="radio" name="tsdeck" id="tab-{k}" class="tabin"{c}>'
                      .format(k=key, c=checked))
        labels.append('<label for="tab-{k}" class="tablab">{l}</label>'.format(k=key, l=label))
        try:
            body = fn(data)
        except Exception as exc:  # a bad tab must not kill the page
            body = '<div class="ph">unavailable — {}: {}</div>'.format(
                _esc(type(exc).__name__), _esc(exc))
        panes.append('<div class="tabpane tp-{k}">{b}</div>'.format(k=key, b=body))
    deck = ('<div class="deck">{i}<div class="tabbar">{l}</div>{p}</div>').format(
        i="".join(inputs), l="".join(labels), p="".join(panes))
    return ('<div class="grid">'
            + shell.card("Detail", deck, span=12, anchor="detail") + "</div>")


def _zone_context(data) -> str:
    items = data.get("context") or []
    if not items:
        return ""
    rows = "".join(
        '<tr><td>{l}</td><td class="n m">{v}</td><td class="n">{b}</td></tr>'.format(
            l=_esc(i.get("label")), v=_esc(i.get("value")),
            b=_badge(i.get("badge", ""), i.get("badge_kind", "bad")))
        for i in items)
    return ('<div class="demote"><h5 class="mut">Context &mdash; '
            "no demonstrated edge</h5><table>{}</table></div>").format(rows)


def _zone_lottery(data) -> str:
    """Lottery-mode panel: play archetype, edge/crush verdict, honest metrics."""
    lot = data.get("lottery")
    if not lot:
        return ""
    play = lot.get("play") or "LONGSHOT"
    if lot.get("crush_trap"):
        badge = _badge("⚠ CRUSH TRAP", "bad")
        note = _esc(lot["crush_trap"]) + " &mdash; a long here overpays for the event."
    elif lot.get("edge"):
        badge = _badge("✦ EDGE", "ok")
        note = "cheap IV + reachable strike + a real catalyst / aligned momentum."
    else:
        badge = _badge("speculative", "warn")
        note = "no clean edge &mdash; treat as a tiny, uncapped-tail bet."

    def _pct(v):
        return _num(v, "{:.0%}")

    def _x(v):
        return _num(v, "{:.1f}x")

    reach = "n/a"
    if lot.get("breakeven_move_pct") is not None and lot.get("expected_move_pct") is not None:
        reach = "needs {} vs EM {}".format(_pct(lot["breakeven_move_pct"]), _pct(lot["expected_move_pct"]))
        if lot.get("breakeven_vs_em") is not None:
            reach += " ({}x EM)".format(_num(lot["breakeven_vs_em"], "{:.1f}"))

    table = "<table>" + _rows((
        ("Play", play),
        ("IV state", lot.get("iv_state") or "n/a"),
        ("Hit ≥3× (to expiry)", _pct(lot.get("hit_prob"))),
        ("Tail @1EM", _x(lot.get("tail_x_1em"))),
        ("Tail @2EM", _x(lot.get("tail_x_2em"))),
        ("Reachability", reach),
    )) + "</table>"
    body = ('<div style="margin-bottom:6px">{badge}</div>'
            '<div class="eye" style="margin-bottom:6px">{note}</div>{table}'
            '<div class="eye" style="margin-top:6px">Base reality: naked far-OTM is '
            'negative-EV on average &mdash; tiny size, the edge (if any) is breadth + an '
            "uncapped tail.</div>").format(badge=badge, note=note, table=table)
    return ('<div class="grid">'
            + shell.card("Lottery read · {}".format(_esc(play)), body, span=12)
            + "</div>")


def render(data: dict) -> str:
    """The complete HTML document. Pure."""
    body = [_hero(data), _zone_lottery(data), _payoff_section(data)]
    if _panel_ok(data, "decision"):
        body.append(_risk_section(data))
    else:
        body.append('<div class="grid">' + shell.card(
            "Stress", _placeholder(data, "decision"), span=12, anchor="risk")
            + "</div>")
    body.append(_vol_section(data))
    body.append(_name_section(data))
    body.append(_narrative_section(data))   # always: carries the evidence card
    body.append(_zone_detail(data))
    body.append(_zone_context(data))        # demoted: least important, last
    meta = data.get("meta", {})
    body.append('<div class="foot">{}</div>'.format(_esc(
        "{} · config sha {} · regenerate: python -m src.tearsheet --from {}".format(
            meta.get("sidecar", ""), meta.get("config_sha", ""),
            meta.get("sidecar", "")))))
    title = "{} {} tearsheet".format(meta.get("ticker", ""),
                                     meta.get("expiration", ""))
    return shell.page(title, _masthead(data), "".join(body), extra_css=_CSS)
