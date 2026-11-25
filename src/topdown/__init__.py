"""Top-down discovery: Macro → Sector board → Lenses → Evidence card.

Every number here is descriptive and stamped with its source and date. The
only ordering with validated evidence behind it is the Outlook rank of the 16
sector/asset ETFs (`vol_managed_outlook`, VALIDATED); every other lens says
what it is, what tested it (or that nothing has), and how it is sorted.
"""
from __future__ import annotations

from .card import EvidenceCard, Provenance, build_evidence_card, probability_label
from .hub import DiscoveryHub, build_hub
from .lenses import (
    LENS_BY_ID,
    LENSES,
    NULL_LEDGER,
    Lens,
    LensContext,
    LensResult,
    LensRow,
    build_context,
    run_lens,
    run_lenses,
)
from .macro import (
    ROTATION_PAIRS,
    MacroSnapshot,
    RatioReading,
    build_macro_snapshot,
    load_fred_cache,
    refresh_fred_cache,
    vol_regime,
)
from .peers import PeerRow, PeerStrip, peer_strip
from .sectors import SECTOR_NAMES, SectorBoard, SectorRow, build_sector_board, sector_members

__all__ = [
    "EvidenceCard", "Provenance", "build_evidence_card", "probability_label",
    "DiscoveryHub", "build_hub",
    "LENS_BY_ID", "LENSES", "NULL_LEDGER", "Lens", "LensContext", "LensResult", "LensRow",
    "build_context", "run_lens", "run_lenses",
    "ROTATION_PAIRS", "MacroSnapshot", "RatioReading", "build_macro_snapshot",
    "load_fred_cache", "refresh_fred_cache", "vol_regime",
    "PeerRow", "PeerStrip", "peer_strip",
    "SECTOR_NAMES", "SectorBoard", "SectorRow", "build_sector_board", "sector_members",
]
