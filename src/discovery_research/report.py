"""Readable research findings, with all trials and unavailable evidence visible."""
from __future__ import annotations

from src.desk_kit import shell
from .channels import CHANNELS, SECTOR_DRIVERS

_LABELS = {"vix_realized_risk_5": "VIX: next-week volatility",
           "vix_realized_risk_21": "VIX: next-month volatility",
           "vix_direction_5": "VIX: market direction", "peer_incremental_5": "Peer moves: stock direction",
           "news_tone_5": "News tone: stock direction", "news_activity_risk_5": "News volume: movement risk",
           "news_disagreement_risk_5": "Headline disagreement: movement risk",
           "analyst_revisions_5": "Analyst changes: stock direction"}


def _num(value, pattern="{:.2f}") -> str:
    if value is None:
        return "unavailable"
    return pattern.format(value)


def render(data: dict) -> str:
    esc = shell.esc
    rows = []
    for r in data.get("results", []):
        status = r["status"].replace("_", " ")
        # Evidence states are textual; no green buy badges from a forecasting test.
        rows.append('<tr><td><strong>{}</strong><br>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            esc(_LABELS.get(r["id"], r["id"])), esc(r["question"]), esc(status),
            esc(f"{r['test_dates']} dates / {r['test_rows']} rows"),
            esc(_num(r.get("mse_improvement_pct"), "{:+.2f}%")), esc(_num(r.get("adjusted_p"), "{:.4f}"))))
    price_cutoff = ((data.get("coverage") or {}).get("prices") or {}).get("last")
    overview = ('<p>Which information improves an estimate after simple price and risk baselines? '
                'Every registered test is shown. A result about volatility does not establish stock direction '
                'or option profitability.</p><div class="table-scroll"><table><thead><tr>'
                '<th>Question</th><th>Finding</th><th>Non-overlapping dates / rows</th><th>MSE improvement</th>'
                '<th>Adjusted p</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    if price_cutoff:
        overview = '<p><strong>Price history ends {}. These findings are not a current market call.</strong></p>'.format(esc(price_cutoff)) + overview
    body = '<div class="grid">' + shell.card("Information-channel research", overview, span=12, anchor="findings") + '</div>'
    interpretation = (
        '<p><strong>RESEARCH SUPPORT</strong>: the added information cleared this study\'s registered forecasting checks. '
        'It remains a research result in a selected universe, with no trading authorisation.</p>'
        '<p><strong>NOT ESTABLISHED</strong>: sufficient evaluation coverage, but the added channel did not clear every check.</p>'
        '<p><strong>INSUFFICIENT</strong>: too few independent training or test dates/years to decide. '
        'A large number of headlines or correlated stock rows does not repair this.</p>'
        '<p>MSE improvement compares candidate and baseline forecasts on exactly the same held-out observations. '
        'Positive means smaller squared prediction errors. These are not trading returns.</p>')
    body += '<div class="grid">' + shell.card("How to read the findings", interpretation, span=12) + '</div>'
    coverage = '<table><thead><tr><th>Channel</th><th>Available dates</th><th>Coverage</th></tr></thead><tbody>'
    for name, info in data.get("coverage", {}).items():
        count = (f"{info['observations']:,} observations" if "observations" in info else
                 f"{info['rows']:,} archived items, {info.get('dates', 0)} observation dates" if "rows" in info else
                 f"{info.get('symbols', 0)} symbols, {info.get('sessions', 0)} sessions")
        coverage += '<tr><td>{}</td><td>{} to {}</td><td>{}</td></tr>'.format(
            esc(name), esc(info.get("first", "unknown")), esc(info.get("last", "unknown")), esc(info.get("error") or count))
    coverage += '</tbody></table>'
    coverage += '<p>The common price window caps the study even when an index or news feed is newer. '
    coverage += 'No forecast is extrapolated into missing price history.</p>'
    body += '<div class="grid">' + shell.card("Data coverage and limits", coverage, span=12, anchor="coverage") + '</div>'
    relationships = data.get("relationships") or []
    if relationships:
        rel = '<p>These stocks moved together after removing their common market component. '
        rel += 'This can explain shared exposure; the separate peer test above asks whether it predicts future moves.</p>'
        rel += '<div class="table-scroll"><table><thead><tr><th>Group</th><th>Pair</th><th>Market-adjusted correlation</th>'
        rel += '<th>First / second half</th><th>Dates</th></tr></thead><tbody>'
        for pair in relationships:
            rel += '<tr><td>{}</td><td>{} / {}</td><td>{}</td><td>{} / {}</td><td>{} to {} (n={})</td></tr>'.format(
                esc(pair["group"]), esc(pair["a"]), esc(pair["b"]),
                esc(_num(pair["market_adjusted_correlation"])), esc(_num(pair["first_half"])), esc(_num(pair["second_half"])),
                esc(pair["first"]), esc(pair["last"]), pair["observations"])
        rel += '</tbody></table></div>'
        body += '<div class="grid">' + shell.card("Which stocks share exposure?", rel, span=12, anchor="relationships") + '</div>'
    for r in data.get("results", []):
        content = '<p>{}</p>'.format(esc(r["reason"]))
        if "error_improvement_ci" in r:
            lo, hi = r["error_improvement_ci"]
            content += ('<p>Mean paired loss improvement {} · 95% moving-block interval [{}, {}]. '
                        'Baseline RMSE {} · candidate RMSE {}.</p>').format(
                            esc(_num(r.get("mean_error_improvement"), "{:.6g}")),
                            esc(_num(lo, "{:.6g}")), esc(_num(hi, "{:.6g}")),
                            esc(_num(r.get("baseline_rmse"), "{:.6g}")),
                            esc(_num(r.get("candidate_rmse"), "{:.6g}")))
        if r.get("per_year"):
            content += '<table><thead><tr><th>Year</th><th>Dates</th><th>Mean loss improvement</th></tr></thead><tbody>'
            for year, values in r["per_year"].items():
                content += '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                    esc(year), values["dates"], esc(_num(values["mean_improvement"], "{:+.6g}")))
            content += '</tbody></table>'
        content += '<details><summary>Chronological folds and purge audit</summary><ul>'
        for fold in r.get("folds", []):
            content += '<li>{}</li>'.format(esc(str(fold)))
        content += '</ul></details>'
        body += '<div class="grid">' + shell.card(r["id"], content, span=12) + '</div>'
    channel_body = ''
    for c in CHANNELS:
        channel_body += ('<h3>{}</h3><p><strong>{}</strong> {}</p><p>{}</p><p>{}</p>'
                         '<p><a href="{}">{}</a> · {}</p>').format(
            esc(c["name"]), esc(c["inputs"]), esc(c["question"]), esc(c["timing"]), esc(c["state"]),
            esc(c["url"]), esc(c["source"]), "Source documentation; not a claim of predictive skill")
    body += '<div class="grid">' + shell.card("Where to gather information next", channel_body, span=12, anchor="channels") + '</div>'
    sector_body = ''
    for c in SECTOR_DRIVERS:
        sector_body += '<h3>{}</h3><p>{}</p><p>{}</p><p><a href="{}">{}</a></p>'.format(
            esc(c["sector"]), esc(c["drivers"]), esc(c["mechanism"]), esc(c["url"]), esc(c["source"]))
    body += '<div class="grid">' + shell.card("Investigate the actual business drivers", sector_body, span=12, anchor="business") + '</div>'
    limitations = '<ul>' + ''.join('<li>{}</li>'.format(esc(x)) for x in data.get("limitations", [])) + '</ul>'
    body += '<div class="grid">' + shell.card("Research constraints", limitations, span=12) + '</div>'
    masthead = shell.masthead("DISCOVERY RESEARCH", "",
                              meta_html=shell.chipline([("generated", esc(data.get("generated_at"))),
                                                        ("preregistration", esc(data.get("preregistration_id")))]),
                              nav_html='<nav class="tabbar">' + ''.join(
                                  '<a class="tabbtn" href="#{}">{}</a>'.format(key, title)
                                  for key, title in (("findings", "Findings"), ("coverage", "Coverage"),
                                                     ("relationships", "Relationships"), ("channels", "Sources"),
                                                     ("business", "Business drivers"))) + '</nav>', where="research")
    css = '.table-scroll{overflow-x:auto} h3{font-size:14px;margin:20px 0 8px} li{margin:8px 0} '
    css += 'td{vertical-align:top;white-space:normal;overflow-wrap:anywhere} p{line-height:1.65} '
    return shell.page("Discovery research", masthead, body, extra_css=css)
