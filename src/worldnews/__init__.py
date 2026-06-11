"""Market-wide world-news pulse: multi-source, trust-weighted, honest.

Free sources (validated live 2026-06-11): Google News RSS topic searches
(with the true publisher extracted from each item), CNBC and MarketWatch
RSS, and StockTwits' tagged crowd sentiment. CNN Fear & Greed is bot-blocked;
the regime dashboard's own VIX/PCR gauges cover that role.

Every headline is scored by lexicon sentiment, weighted by source trust
(wire services 1.0 … social 0.3) and recency (24h half-life), and aggregated
into a pulse ∈ [-1, +1] with bull/bear percentages and a confidence figure
that grows with item count, source diversity, and cross-source agreement.

Honest framing: documented value of public news at retail speed is *risk
timing* around events, not directional alpha — the panel says so. Display
overlay only; never touches quality_score while the Phase-2 gate gathers.
"""
