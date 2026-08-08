# Can News Sentiment Be Wired Into Scoring? — 2026-08-08

Short answer: **not yet, and a large effect is already ruled out.** The
archive can answer the small-effect question by roughly mid-October 2026 and
the very-small-effect question by late November. Until then the honest weight
is the one it already has: **0.0**.

Run the test:

```bash
PYTHONPATH=$PWD ~/.venvs/options/bin/python -m src.news_signal --horizons 1,3,5,10
```

---

## 1. Why this was untestable before, and is not anymore

`docs/INTEL_BACKTEST_FINDINGS.md` lists news as `not backtestable from price`,
and `DOLT_NEXT_STEPS.md` §12 records the sentiment scorer as deliberately dead:
weight 0.0 in config, and the primary fetch path hardcoding `sentiment_score =
0.0`. Both were correct at the time — the only news available was today's.

`src/news_archive.py` has since been quietly accumulating: **36,400 headlines,
165 symbols, 2026-06-20 to 2026-08-08**, each with a sentiment score, a
relevance score, the raw headline, and — critically — an `archived_at` stamp.

That last column is what makes a test possible. Feeds revise and backfill
publication timestamps, so `published` can move after the fact; `archived_at`
records when *we* first saw the story and cannot leak the future.

## 2. How it is measured

`src/news_signal.py`. The design choice that matters:

**The day is the unit of independence.** There are 3,210 symbol-days of news,
which sounds like a large sample and is not — they sit on ~30 distinct dates,
and every symbol on one date shares that date's market move. Pooling them
would inflate the effective sample by roughly 100x and manufacture
significance out of one week's market direction.

So sentiment is measured the way a factor is measured: rank the cross-section
of names within each day, Spearman against the forward return, then t-test the
resulting series of daily ICs. Horizon counts trading rows, not calendar days.
The forward return starts at the decision date's own close — acting on day D's
news means paying day D's close.

## 3. The result

Prices were refreshed to 2026-08-07 first (they had ended 2026-07-21, which was
the binding constraint, not the news).

```
 horizon   days     obs   mean IC       t        p  daily sd
      1d     26    2862   -0.0139   -0.73   0.4696    0.0962
      3d     24    2645    0.0010    0.05   0.9636    0.1101
      5d     22    2426    0.0070    0.31   0.7570    0.1045
     10d     17    1873   -0.0009   -0.03   0.9750    0.1213
```

Every horizon is indistinguishable from zero, and testing four horizons is a
four-way search, so even a marginal p would not have counted.

## 4. What that does and does not establish

This is the part that is usually got wrong. A null on 26 days is **not**
evidence of no effect — unless the study could have seen one. At the daily IC
standard deviation actually observed (0.108, rather than an assumed value):

| effect | days needed | status |
|---|---:|---|
| mean daily IC 0.10 | 10 | **ALREADY POWERED — a large effect is excluded** |
| mean daily IC 0.05 | 37 | 11 more trading days (~2 weeks) |
| mean daily IC 0.03 | 102 | 76 more trading days (~3.6 months) |

So the defensible claims today are:

- **News sentiment is not a large factor.** That is a real, powered finding.
- **Whether it is a small one is unanswered**, and the archive answers it on a
  known schedule simply by continuing to run.

For scale: this repo's whole scorer has an IC around 0.03, and the intel
harness weights `momentum` at IC +0.027. An IC-0.03 news factor would be a
genuinely useful addition — and that is exactly the size this archive cannot
yet resolve.

## 5. Why not just buy historical news and settle it now

Three paths were checked. All are blocked.

**Alpha Vantage `NEWS_SENTIMENT`** — the free tier **ignores `time_from`**.
Probed directly: asked for AAPL news from 2023-01-03, received items dated
2026-08-06 to 2026-08-08. Asked from 2026-01-01, same three days. Free tier is
current news only, at 25 requests/day. It cannot backfill and it cannot feed a
165-symbol daily archive.

Worth noting for the *live* path though: AV returns a **finance-tuned,
ticker-level** score with its own relevance (`ticker_sentiment_score 0.303,
relevance 0.647` on the AAPL sample), which is a better instrument than what we
compute locally — see §6.

**FNSPID** (Hugging Face, `Zihan1004/FNSPID`) — 15.7M news records, 4,775
S&P500 companies, **1999-2023**, with sentiment. Exactly the corpus needed, and
blocked twice over: **29.6 GB** against ~6 GiB of free disk, and licensed
**CC BY-NC-4.0 — commercial use explicitly prohibited**, which is disqualifying
for a system whose stated destination is real money. Streaming a per-ticker
subset via the HF API would solve the size problem but not the licence.

**GDELT** — free and historical, but it is general-news tone, not
ticker-resolved financial sentiment; mapping stories to tickers is the hard
part and is exactly where this repo's last news attempt went wrong
(`project_news_relevance`: substring matching inverted the ranking).

## 6. The known weakness in what is being scored

Sentiment is `TextBlob` polarity plus a keyword nudge — a general-English
lexicon applied to financial headlines, where "beats estimates" is
lexically neutral and "sells off stake" is lexically negative but
informationally neither.

Two structural issues visible in the raw data:

- **Story duplication.** A single event produces many headlines. The AAPL
  sample for one day is four separate rows about the same Xiao-I patent
  ruling, so a "relevance-weighted mean of 7 headlines" is really one story
  counted four times.
- **A positive bias.** Mean daily sentiment is **+0.095** with sd 0.125; 9.3%
  of symbol-days score exactly zero.

**The archive is nonetheless future-proof: all 36,400 rows retain the raw
`headline`.** Any better scorer — a finance-tuned model, or AV's own
ticker-level score — can be applied retroactively to the entire history
without losing a day. That is the single most important property of this
dataset and it is already in place.

## 6b. Testing the right target: news FLOW vs how far a name moves

Predicting direction was always the weaker hypothesis. This is an options book
— a straddle, a condor and every buy-vs-sell-premium decision turn on **how
far** a name moves, not which way. And "news arrives, volatility follows" is a
far more plausible mechanism than "news tone predicts sign".

Re-run against `|forward return|`, adding two fields that describe the news
*flow* rather than its tone: `flow` (headline count) and `dispersion` (spread
of sentiment across the day's headlines).

```
  field          h  days    obs   mean IC       t        p
  flow           1    26   2862    0.1015    4.31   0.0002
  flow           3    24   2645    0.1287    5.53   0.0000
  flow           5    22   2426    0.1321    4.70   0.0001
  dispersion     1    26   2862    0.0831    3.99   0.0005
  dispersion     3    24   2645    0.0820    3.19   0.0041
  dispersion     5    22   2426    0.0768    3.27   0.0036
  abs_score      1    26   2862   -0.0023   -0.15   0.8841
  score          1    26   2862   -0.0033   -0.19   0.8510
```

At first read this is a find: `flow` is monotone in horizon
(0.1015 -> 0.1287 -> 0.1321), clears Bonferroni on a 12-way search
(0.05/12 = 0.0042), and sits **above the powered threshold** — IC 0.10 needs 10
days and there are 26. Tone is dead in the same table, so whatever this is, it
is about how much news there is, not what it says.

### And then the confound kills most of it

News flow is a proxy for "something just happened", and volatility clusters.
If that is all this is, trailing realized vol already captures it — from prices,
free, no news required.

Partial rank correlation of `flow` against `|forward return|`, controlling for
trailing 20-day realized vol computed from the same price series:

| horizon | raw IC | t | **partial IC** | t | p |
|---|---:|---:|---:|---:|---:|
| 1d | 0.1015 | 4.31 | **0.0303** | 1.22 | 0.2321 |
| 3d | 0.1287 | 5.53 | **0.0641** | 2.84 | 0.0093 |
| 5d | 0.1321 | 4.70 | **0.0662** | 2.17 | 0.0417 |

**Half to two-thirds of the effect was vol clustering.** And the control is
enormous on its own:

```
trailing 20d RV alone -> |forward move|:  IC 0.362 (t 17.4) / 0.361 (t 15.4) / 0.382 (t 16.0)
```

**IC 0.36 at t=16, from prices alone.** That is three times the size of the raw
news-flow effect and it costs nothing.

What survives for news is a residual of **0.03-0.07**, which (a) does not clear
Bonferroni on the now-15-way search, and (b) lands exactly in the underpowered
band — an IC of 0.03-0.07 needs 39-107 days and there are 22-26.

**Verdict: news flow is not established as adding anything over trailing
realized vol.** Recheck it when the archive reaches ~100 days.

### The one thing that is established

Trailing realized vol strongly predicts forward absolute move. That is
textbook vol clustering and it is *not* a new edge — implied vol already
prices it, which is precisely why `iv_minus_rv` measured flat in
`docs/ATTRIBUTION_20260808.md` §4. The two results are consistent: vol is
predictable, and the option market knows.

## 7. The recommendation

**Leave `sentiment` at weight 0.0.** Adding it now would inject a feature that
is measured at zero, cannot yet be shown to be even small, and is computed by a
scorer with known defects. `docs/ADJUSTMENT_STACK_20260807.md` already found
that hand-set constants in the score carry an IC of **-0.096** — adding another
unvalidated term is the documented failure mode of this codebase, not a new
idea.

What to do instead, in order:

1. **Let it run.** The archive gains a day per weekday at no cost. Re-run
   `python -m src.news_signal` in ~2 weeks for the IC-0.05 question and in
   ~3.6 months for IC-0.03. The command prints its own power verdict.
2. **Improve the scorer, not the weight.** Because headlines are retained,
   re-scoring is retroactive. A finance-tuned sentiment model would raise the
   quality of all 36,400 archived rows at once, and would make the eventual
   test a test of sentiment rather than a test of TextBlob.
3. **Deduplicate by story before aggregating.** Four headlines about one
   ruling should count once. `dedup_key` exists; the daily aggregate does not
   use it.
4. **Only then consider a weight** — and only if the IC clears the same bar
   every other feature in this repo is held to.

## 8. If it does eventually work, where it would go

Not into `quality_score`. The score ranks contracts *within* a chain, and
`docs/SCORE_AUDIT_20260807.md` found half of it is already ticker-constant —
a per-ticker sentiment value would add another ticker-constant term to a score
whose job is within-chain ranking, which is the wrong place structurally.

The right home is the layer that chooses **which underlying to trade**, where
`src/outlook` already operates and where the intel harness found its only real
edge (relative IC +0.05-0.08). A validated news factor is a symbol-selection
input, not a contract-selection one.
