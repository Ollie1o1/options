"""Source research and economic mechanisms; these are hypotheses, not signals."""

CHANNELS = [
    {"name": "Market-implied risk", "inputs": "VIX level/trend, VIX3M, VVIX, SKEW",
     "question": "Is uncertainty rising, and does that add information beyond recent realized volatility?",
     "source": "Cboe", "url": "https://www.cboe.com/tradable-products/vix/vix-historical-data",
     "timing": "Index observation time; end-of-day tests enter the next session. VIX is not a directional forecast.",
     "state": "VIX tested here; related index files audited for coverage, not searched for winning formulas."},
    {"name": "Sector and peer transmission", "inputs": "Leave-one-out peer returns, market residuals, supplier/customer events",
     "question": "Is the move broad-market exposure, peer leadership, or company-specific information?",
     "source": "SEC filings and dated price history", "url": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
     "timing": "Relationships must come from a filing known then. Contemporaneous correlation is not a lead-lag result.",
     "state": "Fixed peer-group incremental test here; filing-derived business relationships are the next collection stage."},
    {"name": "News novelty, activity and disagreement", "inputs": "First-seen headlines, deduplication, tone, activity, dispersion",
     "question": "Does genuinely new coverage explain direction, or mainly how much a stock moves?",
     "source": "Existing multi-source archive; Federal Reserve research", "url": "https://www.federalreserve.gov/econres/feds/news-versus-sentiment-predicting-stock-returns-from-news-stories.htm",
     "timing": "Use our first archival time, never a backfilled publication time. Syndication does not create independent evidence.",
     "state": "Three declared tests here; collection coverage and independent dates determine whether a result is measurable."},
    {"name": "Analyst revisions", "inputs": "Upgrade/downgrade changes, estimate and target revisions",
     "question": "Is there incremental information after the price has already reacted?",
     "source": "Existing first-seen analyst archive", "url": "https://www.sec.gov/edgar/search/",
     "timing": "Feed first_seen, not historical grade_date. Estimate levels must retain each release vintage.",
     "state": "First-seen upgrade/downgrade test here; target/estimate revisions remain separate untested hypotheses."},
    {"name": "Rates, credit and macro surprises", "inputs": "Yield curve, real yields, credit spreads, dollar, released surprises",
     "question": "Which market changes matter for this company's financing, demand and valuation?",
     "source": "FRED / ALFRED", "url": "https://fred.stlouisfed.org/docs/api/fred/realtime_period.html",
     "timing": "Economic observations require release dates and historical vintages. Revised history must not leak into old decisions.",
     "state": "Source design researched; historical-vintage adapter required before predictive testing."},
    {"name": "Positioning and crowd sentiment", "inputs": "CFTC positioning, put/call activity, OI changes, social crowd tone",
     "question": "Is positioning crowded or changing, and is the change independent of price?",
     "source": "CFTC and existing chain archive", "url": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm",
     "timing": "Trade only after the actual release. Reporting dates and publication dates differ; consult the release schedule.",
     "state": "Existing OI overlays remain descriptive; positioning/social tests need archived release-time coverage."},
    {"name": "Company disclosures and insider activity", "inputs": "8-K, 10-Q/10-K, Form 4, guidance and customer concentration",
     "question": "What changed in cash generation, financing, demand, ownership or business dependencies?",
     "source": "SEC EDGAR", "url": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
     "timing": "Use accepted/publication time, not accounting period end or insider transaction date.",
     "state": "Existing insider collector available; structured filing-change and relationship extraction remains to be validated."},
]

SECTOR_DRIVERS = [
    {"sector": "Semiconductors / AI infrastructure", "drivers": "Customer capex and concentration, foundry capacity, memory pricing, export restrictions, peer guidance",
     "mechanism": "Demand or supply shocks may travel through customers, foundries and equipment vendors; establish each business link from dated disclosures.",
     "source": "SEC filings / company investor relations", "url": "https://www.sec.gov/edgar/search/"},
    {"sector": "Energy producers and refiners", "drivers": "Crude/gas prices, inventories, refinery margins, utilization, production and hedges",
     "mechanism": "An input-price rise can help producers while squeezing refiners or transport customers; do not apply one direction to every energy ticker.",
     "source": "EIA weekly petroleum reports", "url": "https://www.eia.gov/petroleum/supply/weekly/"},
    {"sector": "Banks and lenders", "drivers": "Deposit costs, yield curve, loan growth, delinquencies, provisions, capital and credit spreads",
     "mechanism": "Rates change both asset income and funding costs. Balance-sheet mix determines the exposure; a steeper curve is not uniformly bullish.",
     "source": "Federal Reserve H.8 / SEC disclosures", "url": "https://www.federalreserve.gov/releases/h8/"},
    {"sector": "Biotech / healthcare", "drivers": "Trial endpoints, enrollment, amendments, regulatory dates, cash runway, reimbursement and dilution",
     "mechanism": "Event outcomes and financing can dominate broad-sector trends. Use event base rates and implied-move scenarios, not generic sentiment.",
     "source": "ClinicalTrials.gov / FDA", "url": "https://clinicaltrials.gov/data-api/api"},
    {"sector": "Retail / consumer", "drivers": "Real spending, units versus pricing, inventories, promotions, wage costs, credit stress and guidance",
     "mechanism": "Revenue growth from higher prices can coexist with weaker volumes or margins. Compare like-for-like operating changes.",
     "source": "Census retail releases / company filings", "url": "https://www.census.gov/retail/index.html"},
    {"sector": "Utilities / real estate", "drivers": "Real yields, refinancing schedules, regulated returns, occupancy, power demand and capital spending",
     "mechanism": "Financing exposure and contract structure matter alongside sector duration; higher demand can require expensive new capital.",
     "source": "SEC filings / EIA", "url": "https://www.eia.gov/electricity/"},
]


def driver_profile(symbol: str, sector_etf: str = "") -> dict:
    """Research prompts, never assertions that a business dependency exists."""
    chips = {"NVDA", "AMD", "AVGO", "MU", "INTC", "QCOM", "AMAT", "LRCX", "TSM", "ASML"}
    if symbol.upper() in chips:
        return dict(SECTOR_DRIVERS[0], peers=[s for s in ("NVDA", "AMD", "AVGO", "MU") if s != symbol.upper()])
    mapping = {"XLE": 1, "XLF": 2, "XLV": 3, "XLY": 4, "XLP": 4, "XLU": 5, "XLRE": 5}
    if sector_etf in mapping:
        return dict(SECTOR_DRIVERS[mapping[sector_etf]], peers=[])
    return {"sector": "Company-specific investigation", "drivers": "Guidance, revenue mix, customer concentration, margins, financing and dated catalysts",
            "mechanism": "Verify each claimed driver and business relationship in the company's disclosures before interpreting correlated price moves.",
            "source": "SEC EDGAR / company investor relations", "url": "https://www.sec.gov/edgar/search/", "peers": []}
