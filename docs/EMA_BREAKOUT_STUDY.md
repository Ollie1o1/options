# Do EMAs improve historical breakout forecasts?

Moving averages and range breaks have a long research history. Results vary across test periods, so this study checks whether an EMA adds information to a fixed baseline on later data. [Brock, Lakonishok and LeBaron](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x) · [Sullivan, Timmermann and White](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=65140)

## Protocol fixed before reading outcomes

- Use the local adjusted daily OHLCV universe. The SPY dates provide the exchange-session grid. Use every fifth session starting at the first available session in 2009. Require 100 consecutive observed closes through the observation date. Report unresolved horizons separately. The cache reflects the current universe and cannot establish survivorship-free or real-time performance.
- Start from observations below or at the highest close of the previous 20 sessions. The outcome is at least one **later close** at least 1% above that fixed level. Measure it over 5, 10 or 20 sessions. An observation day belongs to all of its eligible comparisons.
- The baseline learns breakout frequencies from the distance to the level (0–2%, 2–5%, above 5%) and daily volatility (below 1%, 1–2%, above 2%), using 20 returns through the observation date. The EMA version splits each baseline group by whether the close is above that EMA. Test exactly six combinations: 5-day horizon with EMA 5 or 10, 10-day horizon with EMA 20 or 50, and 20-day horizon with EMA 50 or 100.
- Fit historical group frequencies on 2010–2022 observations with outcomes that mature in that period. Use a fixed add-one estimate and fall back to the baseline group when an EMA subgroup has fewer than 100 training observations. Leave 2023 unused. Evaluate 2024 through the final cached session in 2026.
- Primary measure is improvement in Brier score, the average squared error of the probabilities. Resample four consecutive observation weeks as blocks, keeping all symbols on each date together. Use 3,000 draws and a family-adjusted 99.17% interval for each comparison. A result merits research follow-up only if its lower bound is above zero, its Brier error falls by at least 1%, and it improves in both 2024 and 2025–26. This test measures historical forecasting accuracy. A paper profit study would require a separate design and cost model.

Run `python -m scripts.study_ema_breakouts --out configs/ema_breakout_study.json` against the local database.

## Results with cached data through November 24, 2026

The 375-symbol archive supported between 45,451 and 46,360 held-out observations per horizon. Four
comparisons improved forecast error with adjusted intervals above zero. Every comparison was below the
fixed 1% improvement threshold. The largest estimated improvement was 0.97% for EMA 20 on the 10-session
horizon. These results support keeping the findings in **Discover → Breakout research** as historical
context. A broader, survivorship-controlled panel or a new forward sample would be the next independent
test of these small differences.

| Forecast sessions | EMA sessions | Less forecast error | Family-adjusted interval | Held-out observations |
|---|---|---|---|---|
| 5 | 5 | 0.54% | −0.04% to +1.03% | 46,360 |
| 5 | 10 | 0.57% | +0.16% to +0.92% | 46,360 |
| 10 | 20 | 0.97% | +0.53% to +1.46% | 46,021 |
| 10 | 50 | 0.60% | +0.20% to +1.06% | 46,021 |
| 20 | 50 | 0.75% | +0.08% to +1.40% | 45,451 |
| 20 | 100 | 0.42% | −0.09% to +0.91% | 45,451 |
