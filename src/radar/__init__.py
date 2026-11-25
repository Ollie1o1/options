"""Unified multi-asset Opportunity Radar & Ticker Dossier package."""
from .dossier import build_dossier
from .engine import load_cached_radar, run_radar, save_radar_cache
from .playbooks import TIER_RANK, PlaybookMeta, get_playbook, load_playbooks
from .schema import Opportunity, RadarSnapshot, TickerDossier

__all__ = [
    "TIER_RANK",
    "PlaybookMeta",
    "Opportunity",
    "RadarSnapshot",
    "TickerDossier",
    "load_playbooks",
    "get_playbook",
    "run_radar",
    "load_cached_radar",
    "save_radar_cache",
    "build_dossier",
]
