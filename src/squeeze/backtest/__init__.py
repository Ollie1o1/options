"""Point-in-time backtest harness for the squeeze grader.

Answers one question: does an ``assess_squeeze`` SETUP grade actually raise the
probability of a right-tail up-move in the underlying, relative to WATCH/NONE
names drawn from the same high-short-interest universe?

Survivorship-free on both sides — the universe comes from FINRA's consolidated
short-interest file as it stood on each settlement date, and prices come from
the DoltHub ``stocks`` cross-section, which still carries names that later
delisted (BBBY, MULN, ...). Nothing here touches the live screener.
"""

DEFAULT_DB = "data/squeeze_backtest.db"      # FINRA short interest
PRICES_DB = "data/squeeze_prices.db"         # daily closes
# Separate file on purpose: the EDGAR backfill and the FINRA backfill run
# concurrently, and sharing one SQLite file makes them fight for the write lock.
SHARES_DB = "data/squeeze_shares.db"
