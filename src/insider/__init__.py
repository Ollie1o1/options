"""SEC EDGAR Form 4 insider-buying signal (free, slow, documented alpha).

Opportunistic open-market insider *buys* — especially clusters of multiple
insiders buying the same stock within a window — carry multi-month predictive
power (Cohen, Malloy & Pomorski 2012, "Decoding Inside Information"). Sells
are mostly uninformative (diversification, comp plans) and are reported but
never scored.

Modules: ``edgar`` (polite EDGAR I/O), ``parse`` (pure Form 4 XML → trades),
``signal`` (pure cluster scoring). CLI: ``python -m src.insider TICKER...``.

Overlay only — not wired into quality_score while the Phase-2 gate gathers.
"""
