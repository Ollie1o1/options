"""Phase 3 Unexhausted TFSA-Legal Alpha & External Alt-Data Labs (`src/alpha/`).

Every signal here carries an explicit evidence label (`WATCHLIST` or `CONTEXT`) and
exact data provenance — never promoted to `PROVEN` without clearing a preregistered
out-of-sample test.
"""
from .options_for_shares import OptionShareSignal, compute_dealer_gex, scan_option_signals_for_shares
from .edgar_events import EdgarEvent, load_cached_edgar_events, refresh_edgar_events
from .macro_plumbing import MacroPlumbingSnapshot, load_macro_plumbing, refresh_macro_plumbing
from .premortem import PreMortemReport, build_premortem

__all__ = [
    "OptionShareSignal",
    "compute_dealer_gex",
    "scan_option_signals_for_shares",
    "EdgarEvent",
    "load_cached_edgar_events",
    "refresh_edgar_events",
    "MacroPlumbingSnapshot",
    "load_macro_plumbing",
    "refresh_macro_plumbing",
    "PreMortemReport",
    "build_premortem",
]
