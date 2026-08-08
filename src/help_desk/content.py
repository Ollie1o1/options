"""The manual, as data.

Everything here is a literal. Nothing is computed, fetched, or read from the
ledger at display time — a manual that can fail to render because yfinance was
rate-limited is not a manual. The cost is that these numbers go stale; the
mitigation is that every one of them names the document that measured it, so a
reader can always check the source and a maintainer always knows what to
re-derive.

Sources, once, so the chapters can cite them short:
  docs/EXECUTION_TRUTH.md            entry crossing, measured off CBOE quotes
  docs/MULTILEG_COMPOSITE_20260807.md  multi-leg IC + the net-of-cost book
  docs/ADJUSTMENT_STACK_20260807.md    the single-leg score decomposition
  docs/CONDOR_COMPOSITE_20260807.md    the inverted condor weighting
  docs/BROKER_COSTS.md                 Wealthsimple fee schedule
  docs/LAB_FINDINGS.md                 the long-call gate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Chapter:
    """One screenful-or-several of manual. `key` is the stable identifier the
    menu accepts; `blurb` is its one-line index entry."""
    key: str
    title: str
    blurb: str
    body: Tuple[tuple, ...]


# ── [1] START HERE ────────────────────────────────────────────────────────────

_START = Chapter(
    key="start",
    title="START HERE",
    blurb="what this is, and your first ten minutes",
    body=(
        ("h", "What this program is"),
        ("p", "A research desk for equity options that measures its own "
              "decisions. It scans chains, scores candidates, prices what a "
              "fill would really cost, writes every pick to a paper ledger, "
              "and then grades itself against what happened."),
        ("p", "It is not a trading system. Nothing here sends an order. The "
              "ledger is paper, the marks are model-checked, and the gates "
              "that would authorise real capital have not fired."),
        ("callout", "bad",
         "Real money is OFF. No gate authorises it and no live path is "
         "wired. Treat every number in this program as research."),

        ("h", "The six desks"),
        ("p", "The first menu splits the work by asset and by purpose. Most "
              "people only ever need the first one."),
        ("table",
         ({"h": "desk", "w": 12}, {"h": "what it is for", "w": 58}),
         (("1 STOCKS", "equity options — the main screener, 13 modes"),
          ("2 CRYPTO", "BTC/ETH options on Deribit, perp funding and basis"),
          ("3 LEVERAGE", "perp futures — research only, no edge found yet"),
          ("4 RESEARCH", "breakout, vol-intelligence, equity-VRP — read-only"),
          ("5 HOLDINGS", "long-term stock accumulation, buy zones, tranches"),
          ("6 STRATEGIES", "your written setups, with costs attached"))),

        ("h", "Your first ten minutes"),
        ("num", (
            "Open [1] STOCKS. That is the equity screener and it has its own "
            "menu of thirteen scan modes.",
            "Run [10] INTEL first. It is a briefing, not a scan: where the "
            "market is, what the vol regime is, what is happening this week. "
            "Context before candidates.",
            "Then run [3] DISCOVER. It scans the hundred most-traded tickers "
            "with no budget limit, which is the widest honest look you can "
            "take in one command.",
            "Read the VERDICT column before anything else. It answers one "
            "question: does this trade still make money after you pay to get "
            "into it and out of it? Chapter [3] explains it in full.",
            "Ignore the score when ranking. It is hygiene, not an ordering — "
            "chapter [7] gives the measurements behind that sentence.",
        )),

        ("h", "Where things are written down"),
        ("bullet", "This manual — press ? or 8 from either menu, any time."),
        ("bullet", "docs/ in the repo — every finding, with its data and a "
                   "script that reproduces it."),
        ("bullet", "[7] SETTINGS then DOCTOR — checks this install: "
                   "dependencies, network, config, scheduler."),
        ("gap",),
        ("p", "Scans never write to the ledger unless you pass --auto-log. "
              "Looking is always free and always safe."),
    ),
)


# ── [2] PICKING A TRADE ───────────────────────────────────────────────────────

_PICKING = Chapter(
    key="picking",
    title="PICKING A TRADE",
    blurb="the decision ladder, in the order that matters",
    body=(
        ("h", "The order is the content"),
        ("p", "Most option screeners hand you a ranked list and let you take "
              "the top row. This one does not, because the ranking number was "
              "measured against real outcomes and it does not rank. What "
              "follows is the order the evidence supports. Work down it, and "
              "stop at the first step that fails."),

        ("h", "1 · Can it be priced?"),
        ("p", "Both sides quoted, on every leg. A candidate with a missing bid "
              "is not cheap, it is unpriced — and an iron condor built from "
              "three real quotes and one guess is not a price at all. The "
              "screener marks these rather than guessing for you."),

        ("h", "2 · Does it survive crossing?"),
        ("p", "You do not trade at the mid. You cross the spread getting in "
              "and you cross it again getting out. On a two-leg credit spread "
              "that round trip eats a median 18.5% of the credit, and for 8% "
              "of trades it exceeds the entire credit. The ceiling here is "
              "25% of reward, round trip. Above that the trade is refused."),
        ("p", "This single step removes more candidates than every filter in "
              "the config combined. Chapter [4] has the measurements."),

        ("h", "3 · Does the breakeven beat your book?"),
        ("p", "For any credit spread, the win rate you must clear is pure "
              "arithmetic:"),
        ("code", "p* = 1 - credit / width"),
        ("p", "Collect $1.05 on a $2.50-wide spread and you need 58% to break "
              "even. Cross the spread and you collect $0.60 instead, and now "
              "you need 73.3%. Nothing about the underlying changed; you just "
              "moved the bar fifteen points by paying to get in."),
        ("p", "Compare p* against your measured win rate on that structure, "
              "not against a feeling. The screener does this for you and "
              "refuses the trade when your history does not clear the bar."),

        ("h", "4 · Does the structure match your view?"),
        ("p", "A view has a shape, and the shape picks the structure. This is "
              "a judgement, not a score — the screener cannot know what you "
              "think."),
        ("table",
         ({"h": "your view", "w": 30}, {"h": "structure", "w": 40}),
         (("it goes up, soon, hard", "long call, or a call debit spread"),
          ("it goes up, or just not down", "bull put spread, short put"),
          ("it goes nowhere for a month", "iron condor"),
          ("it stops going up", "bear call spread"),
          ("something breaks, direction unknown", "long straddle — rarely worth it"),
          ("vol is too expensive", "sell premium; check IV rank first"))),
        ("p", "If two structures both express the view, take the cheaper one "
              "to trade. That is almost always the one with fewer legs, "
              "because friction is charged per leg and reward is not."),

        ("h", "5 · Can you size it?"),
        ("p", "Size on capital at risk — the most the position can actually "
              "lose — and never on entry price times a hundred. On a short "
              "option that second number is meaningless, and the ledger "
              "carried a loss for weeks that turned out to be nothing but "
              "that mistake."),
        ("p", "One position should not be able to hurt the book. If a single "
              "trade going to max loss would change how you feel about the "
              "next one, it is too big."),

        ("h", "What to do about the score"),
        ("p", "Use quality_score as hygiene: a very low score usually means "
              "something is wrong with the row — thin liquidity, a stale "
              "quote, a strike nobody trades. Do not use it to choose between "
              "two reasonable candidates. It has no measured ability to do "
              "that, and on the long-premium book it has measured ability to "
              "do the opposite. Chapter [7], with numbers."),
        ("gap",),
        ("callout", "warn",
         "If you remember one thing: rank by what survives its costs, and "
         "let the score break ties. Never the other way round."),
    ),
)


# ── [3] THE VERDICT ───────────────────────────────────────────────────────────

_VERDICT = Chapter(
    key="verdict",
    title="THE VERDICT",
    blurb="PASS, REFUSE, unpriced — and why it outranks the score",
    body=(
        ("h", "What the verdict answers"),
        ("p", "One question, and only one: is there anything left of this "
              "trade once you have paid to get into it and out of it? It is "
              "not a forecast and it does not claim to know what the stock "
              "will do. It is an arithmetic check against your own costs."),
        ("p", "Candidates are ordered by verdict first. quality_score breaks "
              "ties and does nothing else."),

        ("h", "The three outcomes"),
        ("kv", "PASS", "priced, survives its costs, breakeven within reach"),
        ("kv", "REFUSE", "priced, and one of the four checks below failed"),
        ("kv", "unpriced", "could not be priced honestly — not a rejection"),
        ("gap",),
        ("p", "An unpriced candidate is a data gap, not a bad trade. It means "
              "a leg had no two-sided quote when the scan ran. Re-run in "
              "market hours before concluding anything about it."),

        ("h", "The four ways a candidate is refused"),
        ("p", "These are the reasons the screener prints, verbatim:"),
        ("bullet", "\"no two-sided quote on every leg\" — one or more legs had "
                   "no bid or no ask, so the cost of trading it is unknown."),
        ("bullet", "\"credit disappears once the spread is crossed\" — you "
                   "would collect nothing, or pay, to open a credit position. "
                   "This happens to 4% of logged credit trades."),
        ("bullet", "\"friction N% of reward exceeds the 25% ceiling\" — the "
                   "round trip costs more than a quarter of everything the "
                   "trade can pay you."),
        ("bullet", "\"needs an N% win rate; your history on this structure is "
                   "M%\" — the breakeven p* is above what you have actually "
                   "achieved on that structure. The comparison is against "
                   "your ledger, not an industry figure."),

        ("h", "Where the 25% ceiling comes from"),
        ("p", "It is set deliberately between two measured burdens. One "
              "crossing costs 0.7-1.7% of a single leg's premium, and 33% of "
              "a two-leg credit spread's credit. A 25% round-trip ceiling "
              "therefore sits well above the single-leg burden and below the "
              "spread burden — so it barely touches single legs and bites "
              "hard on spreads, which is exactly where the evidence says the "
              "problem is."),
        ("p", "Round trip means both crossings. A structure costing 33% per "
              "crossing reads as 67% here, and is refused."),

        ("h", "Why the verdict ranks and the score does not"),
        ("p", "The gate refuses exactly the trades that the net-of-cost "
              "restatement counts as losers: the 8% whose round trip exceeds "
              "their credit, and the 4% whose credit vanishes on entry. That "
              "is a measured overlap, not a hoped-for one."),
        ("p", "Cost is a gate rather than a sort key, though. Among single "
              "legs the round trip sits at 1-4% across the whole board and "
              "separates almost nothing, so sorting on it would put a "
              "negative-EV pick above a good one. Pass the gate first, then "
              "order by net expected value."),
        ("gap",),
        ("p", "Source: src/candidate_verdict.py, docs/EXECUTION_TRUTH.md, "
              "docs/MULTILEG_COMPOSITE_20260807.md."),
    ),
)


# ── [4] FRICTION AND COST ─────────────────────────────────────────────────────

_FRICTION = Chapter(
    key="friction",
    title="FRICTION AND COST",
    blurb="what crossing the spread actually takes from you",
    body=(
        ("h", "The largest single fact in this book"),
        ("p", "For most of this program's life, entries were priced at the "
              "bid/ask midpoint and entry friction was modelled as exactly "
              "zero. Re-pricing thirty logged bull put spreads against the "
              "real CBOE quotes that existed on the days they were logged "
              "showed what that assumption cost."),
        ("table",
         ({"h": "median, $2.50-wide bull put", "w": 34},
          {"h": "at mid", "w": 16, "align": "right"},
          {"h": "crossed", "w": 18, "align": "right"}),
         (("credit collected", "$1.05", "$0.60"),
          ("credit / width", "0.420", "0.267"),
          ("breakeven win rate needed", "58.0%", "73.3%"))),
        ("p", "Crossing costs $0.35 a share — 27% of the credit. Three of the "
              "thirty spreads had no credit left at all. The book's bull put "
              "win rate is 70.4%, which clears 58% comfortably and misses "
              "73.3% entirely. The assumption was the difference between a "
              "winning strategy and a losing one."),

        ("h", "Friction is charged per leg; reward is not"),
        ("p", "This is the whole reason multi-leg structures are expensive. "
              "You pay the spread on every leg, but you are paid on the "
              "difference between them."),
        ("table",
         ({"h": "structure", "w": 22}, {"h": "crossings", "w": 11, "align": "right"},
          {"h": "measured toll", "w": 35}),
         (("single leg", "2", "0.7-1.7% of premium per crossing"),
          ("two-leg vertical", "4", "27-33% of credit per crossing"),
          ("iron condor", "8", "four legs against one credit"))),

        ("h", "What it does to the whole book"),
        ("p", "Across 109 logged multi-leg trades carrying both a mid price "
              "and a crossed price, the entry crossing alone takes a median "
              "9.2% of the credit. Doubled for the round trip:"),
        ("kv", "round trip", "median 18.5% of credit, mean 39.0%"),
        ("kv", "exceeds all", "8% of trades cost more than the whole credit"),
        ("kv", "vanishes", "4% have no credit left once crossed"),
        ("gap",),
        ("p", "Restating the book at the price it would really fill moves it "
              "from profitable to losing:"),
        ("table",
         ({"h": "cohort", "w": 16}, {"h": "n", "w": 5, "align": "right"},
          {"h": "median at mid", "w": 16, "align": "right"},
          {"h": "median net", "w": 14, "align": "right"},
          {"h": "net win", "w": 9, "align": "right"}),
         (("ALL", "105", "+16.4%", "-12.9%", "43%"),
          ("Bull Put", "26", "+40.7%", "-19.1%", "42%"),
          ("Bear Call", "41", "+11.5%", "-20.8%", "29%"),
          ("Iron Condor", "38", "+12.3%", "+5.6%", "58%"))),
        ("callout", "warn",
         "Iron condors are the only multi-leg structure with a positive "
         "median return net of crossing — and the composite ranks them "
         "worst. See chapter [7]."),

        ("h", "What your broker charges"),
        ("p", "Wealthsimple, Ontario, USD account. Verified against the "
              "published schedule."),
        ("kv", "commission", "$0 per equity or ETF option contract, every tier"),
        ("kv", "FX", "$0 while the USD subscription is active"),
        ("kv", "spread", "the real cost — everything above"),
        ("gap",),
        ("callout", "bad",
         "Index options are NOT free. SPX/SPXW and VIX cost US$1.00 per "
         "contract, RUT US$0.50, XSP US$0.25 — and the cost model does not "
         "include them, so every index result you see is understated."),
        ("p", "There is also a $45 fee for early exercise. Close short "
              "options that go deep in the money rather than letting them be "
              "assigned."),
        ("gap",),
        ("p", "Sources: docs/EXECUTION_TRUTH.md, "
              "docs/MULTILEG_COMPOSITE_20260807.md, docs/BROKER_COSTS.md."),
    ),
)


# ── [5] THE DESKS AND THE MODES ───────────────────────────────────────────────

_MODES = Chapter(
    key="modes",
    title="THE DESKS AND THE MODES",
    blurb="every menu entry, one line each",
    body=(
        ("h", "First menu — the desks"),
        ("table",
         ({"h": "", "w": 14}, {"h": "", "w": 56}),
         (("1 STOCKS", "equity options — discover, spreads, iron, sell"),
          ("2 CRYPTO", "BTC/ETH options on Deribit + perp funding/basis"),
          ("3 LEVERAGE", "BTC/ETH perp futures — tagged no edge yet"),
          ("4 RESEARCH", "breakout, vol-intel, equity-VRP — read-only"),
          ("5 HOLDINGS", "long-term accumulation, buy zones, tranches, TFSA"),
          ("6 STRATEGIES", "your setups — display-only, never an order"),
          ("7 SETTINGS", "theme, preferences, doctor"),
          ("8 HELP", "this manual"))),

        ("h", "Inside STOCKS — the thirteen scan modes"),
        ("table",
         ({"h": "", "w": 14}, {"h": "", "w": 56}),
         (("1 TICKER", "single-stock deep analysis, e.g. AAPL"),
          ("2 ALL", "budget-based multi-stock scan"),
          ("3 DISCOVER", "top 100 most-traded tickers, no budget limit"),
          ("4 SELL", "premium selling — income via short puts"),
          ("5 SPREADS", "credit spread analysis"),
          ("6 IRON", "iron condor analysis — range-bound views"),
          ("7 PORTFOLIO", "open position P/L"),
          ("8 MY LIST", "scan your saved watchlist"),
          ("9 LOTTERY", "far-OTM plays on extreme moves"),
          ("10 INTEL", "pre-trade briefing — context before candidates"),
          ("11 SQUEEZE", "high-short-interest squeeze candidates"),
          ("12 PROB LAB", "risk-neutral density + your-view ranking"),
          ("13 STRUCTURE", "view to structure, sized to your account"))),

        ("h", "Watchlist commands"),
        ("p", "Typed at the mode prompt rather than picked from the menu:"),
        ("code", "ADD AAPL   ·   REMOVE AAPL   ·   SHOW LIST"),

        ("h", "Which mode when"),
        ("bullet", "You have no idea what to trade — [10] INTEL, then "
                   "[3] DISCOVER."),
        ("bullet", "You have a ticker in mind — [1] TICKER."),
        ("bullet", "You have a view and want the structure for it — "
                   "[13] STRUCTURE, or [12] PROB LAB if you want to see the "
                   "market's own distribution first."),
        ("bullet", "You want income and the vol is rich — [4] SELL or "
                   "[5] SPREADS. Check IV rank before either."),
        ("bullet", "You think nothing happens for a month — [6] IRON."),
        ("bullet", "You want to know how you are doing — [7] PORTFOLIO."),
        ("gap",),
        ("p", "Modes tagged read-only or display-only cannot write to the "
              "ledger, by construction rather than by convention."),
    ),
)


# ── [6] GLOSSARY ──────────────────────────────────────────────────────────────

_GLOSSARY = Chapter(
    key="glossary",
    title="GLOSSARY",
    blurb="every term that appears on a scan row",
    body=(
        ("h", "The greeks"),
        ("kv", "delta", "how much the option moves per $1 of stock. Also a "
                        "rough probability of finishing in the money."),
        ("kv", "theta", "dollars lost per day to time passing. Negative when "
                        "you are long an option, positive when you are short."),
        ("kv", "vega", "dollars gained per 1 point of implied volatility. "
                       "Long options are long vega."),
        ("kv", "gamma", "how fast delta changes. High near the strike and "
                        "near expiry — this is what makes short options "
                        "dangerous in the last week."),

        ("h", "Volatility"),
        ("kv", "IV", "implied volatility — what the market is charging for "
                     "uncertainty. Not a forecast, a price."),
        ("kv", "IV rank", "where today's IV sits in its own past year, 0-100. "
                          "High rank favours selling premium, low favours "
                          "buying it. More useful than IV itself."),
        ("kv", "HV", "historical volatility — what the stock actually did. "
                     "Trailing, so it describes the past."),
        ("kv", "VRP", "variance risk premium: IV minus HV. Positive means "
                      "options are priced above what realised. Currently "
                      "negative on single-name equity."),

        ("h", "Reward and risk"),
        ("kv", "credit", "what you are paid to open a position."),
        ("kv", "debit", "what you pay to open one."),
        ("kv", "width", "distance between the strikes of a spread. Max loss "
                        "on a credit spread is width minus credit."),
        ("kv", "credit/width", "the fraction of the spread's width you "
                               "collect. The single most useful number on a "
                               "credit spread row."),
        ("kv", "p*", "breakeven win rate, 1 - credit/width. Beat it or the "
                     "structure loses regardless of how good it looked."),
        ("kv", "POP", "probability of profit, model-estimated. Treat with "
                      "suspicion — see chapter [7] on the condor weighting."),
        ("kv", "RoC", "return on capital: profit divided by what the trade "
                      "actually tied up."),
        ("kv", "DTE", "days to expiry. Positions are time-exited at 21 DTE."),

        ("h", "Terms with traps in them"),
        ("kv", "max_loss", "correct on spreads. On a single-leg row it is "
                           "just entry price times 100, which is meaningless "
                           "on a short option — a short put's real risk is "
                           "the strike. Never size off this field."),
        ("kv", "ev_per_contract", "expected value net of costs, but priced "
                                  "against trailing realised vol, so it "
                                  "assumes the future looks like the past. It "
                                  "is 0.68% of the quality score."),
        ("kv", "quality_score", "a 35-metric composite. Hygiene only — it has "
                                "no measured ability to rank outcomes. "
                                "Chapter [7]."),
        ("kv", "capital at risk", "the real maximum loss of a position. The "
                                  "only correct basis for sizing."),
        ("kv", "GEX", "dealer gamma exposure. Positive suppresses moves, "
                      "negative amplifies them. Context, not a signal."),

        ("h", "Ledger and exits"),
        ("kv", "entry_price", "the mid at the time of the scan — NOT what you "
                              "would fill at."),
        ("kv", "entry_price_fill", "the crossed price. This is the honest one; "
                                   "read it, not entry_price."),
        ("gap",),
        ("p", "The book's own exit rules, for reference: short premium takes "
              "profit at 50% of credit above 21 DTE, 35% between 7 and 21, "
              "25% under 7, and stops at twice the credit or a strike breach. "
              "Spreads take 50% and stop at -100%. Long options take +100% or "
              "delta 0.80, and stop at -50%. Everything time-exits at 21 DTE, "
              "with a 3-day minimum hold."),
    ),
)


# ── [7] WHAT IS AND IS NOT TRUSTED ────────────────────────────────────────────

_TRUST = Chapter(
    key="trust",
    title="WHAT IS AND IS NOT TRUSTED",
    blurb="the measurements, including the unflattering ones",
    body=(
        ("h", "Why this chapter exists"),
        ("p", "This program grades itself, and some of the grades are bad. "
              "Hiding them would make the tool feel better and make you "
              "poorer. Every claim below names the document that measured it "
              "and the sample size it rests on."),

        ("h", "The score does not rank"),
        ("p", "quality_score is a 35-metric composite. Measured against "
              "return on capital on the long-premium book, its Spearman rank "
              "correlation is -0.132 — the wrong sign. Its top bucket is the "
              "worst cell in the ledger: 31.6% win rate, -19.9% return on "
              "capital."),
        ("p", "Decomposed, the single-leg composite runs a rank IC of -0.0995 "
              "and is negative in 5 of 5 walk-forward windows. The composite "
              "itself is roughly neutral at +0.004; the negative comes almost "
              "entirely from about twenty hand-set additive constants applied "
              "afterwards, worth -0.096."),
        ("p", "Half of that score is constant per ticker and a fifth of it is "
              "just rank within the chain."),
        ("callout", "bad",
         "Do not rank candidates by quality_score. On the long-premium book "
         "that actively selects the worst ones. Rank by verdict."),

        ("h", "The multi-leg score is a different story"),
        ("p", "Spreads and condors score through a separate composite that "
              "never touches the constant stack — and it is not "
              "anti-predictive. Rank IC -0.009 over 405 trades, positive in 4 "
              "of 5 windows, and mildly positive on verticals (+0.14 bull "
              "put, +0.10 bear call). Nothing here is significant at n≈130, "
              "but the contrast with the single-leg path is not subtle, and "
              "the path without the constants is the one that is not dragged "
              "negative."),

        ("h", "The condor weighting looks inverted"),
        ("p", "In the iron condor composite, POP carries the largest weight "
              "at 0.30 — and POP's own rank IC is -0.31. The mechanism is "
              "visible: a high POP condor is one with a tiny credit, and the "
              "correlation between POP and credit is -0.72. Meanwhile spread "
              "and theta carry 0.05 and 0.08 while scoring +0.53 and +0.39."),
        ("p", "So the structure that survives its own costs best is the one "
              "the score likes least. This is the best-evidenced scoring "
              "defect found so far. It has deliberately NOT been acted on — "
              "n is too small and the condor composite could not be "
              "reconstructed exactly. Treat condor rankings with particular "
              "suspicion."),

        ("h", "The gates"),
        ("kv", "long call", "STOP. n=92, Pearson -0.065, Spearman -0.132, "
                            "posterior 4%. Reproduced independently. Later "
                            "closes push it further negative."),
        ("kv", "short prem", "Arm A READY, Arm B EXTEND. READY carries a real "
                             "caveat: the loss tail is unobserved, so the "
                             "result rests on trades that have not yet had "
                             "the chance to go badly wrong."),
        ("kv", "real money", "OFF. No gate authorises it."),

        ("h", "Known data limits"),
        ("bullet", "Shadow tracking produced no data at all before "
                   "2026-08-05 — a bug swallowed by a bare except. Any "
                   "shadow analysis starts there."),
        ("bullet", "The quote archive covers 194 of 907 ledger rows. Anything "
                   "said about the other 713 is modelled, not measured, and "
                   "the two are never pooled."),
        ("bullet", "Quotes outside market hours are 15+ minutes delayed. Use "
                   "them for planning, never for execution."),
        ("bullet", "Sentiment carries weight 0.0. News is displayed, not "
                   "scored."),
        ("bullet", "The signal overlays — unusual options activity, insider "
                   "buys, the news pulse — are all display-only. UOA cannot "
                   "be backtested at all: no vendor keeps open-interest "
                   "history."),

        ("h", "What has actually held up"),
        ("bullet", "Friction. Measured three separate ways, from unrelated "
                   "data paths, agreeing each time. It is the most reliable "
                   "finding in the repo."),
        ("bullet", "p* as a decision rule. It is arithmetic, so it cannot "
                   "drift."),
        ("bullet", "The short-interest squeeze signal is real and holds out "
                   "of sample — though it is the SI level that carries it, "
                   "not days-to-cover."),
        ("bullet", "BTC short-vol carry clears the cost wall. ETH does not."),
        ("bullet", "Relative sector/asset outlook ranking, IC +0.05 to +0.08."),
        ("gap",),
        ("p", "Sources: docs/ADJUSTMENT_STACK_20260807.md, "
              "docs/CONDOR_COMPOSITE_20260807.md, "
              "docs/MULTILEG_COMPOSITE_20260807.md, docs/LAB_FINDINGS.md, "
              "docs/EXECUTION_TRUTH.md."),
    ),
)


CHAPTERS: Tuple[Chapter, ...] = (
    _START, _PICKING, _VERDICT, _FRICTION, _MODES, _GLOSSARY, _TRUST,
)
