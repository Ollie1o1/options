"""CDR (Canadian Depositary Receipt) lookup — a curated, user-maintained
map of US ticker -> CDR ticker symbol. Not derivable from the ticker
itself (CDR symbols often differ from the underlying, e.g. KO -> COLA,
MCD -> MCDS, BAC -> BOFA), so this is a flat, editable JSON file, same
spirit as watchlist.json.

Annotation only: a CDR existing says nothing about whether the stock is
worth owning, and no price is fetched for the CDR itself (it tracks the
underlying via CIBC's own currency hedge) — this module never claims
otherwise, matching the non-predictive stance of the rest of this
package (see discover.py's module docstring).
"""
import json
import os
from typing import Dict, Optional

DEFAULT_PATH = "cdr_map.json"

_META_KEYS = {"_SOURCE"}


def load_cdr_map(path: str = DEFAULT_PATH) -> Dict[str, str]:
    """{US_TICKER: CDR_TICKER}, uppercase keys and values. Missing file,
    unreadable file, malformed JSON, or JSON that isn't an object all
    degrade to {} — never raises, matching every other function in this
    package's failure philosophy."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k).upper(): str(v).upper()
            for k, v in raw.items() if str(k).upper() not in _META_KEYS}


def cdr_for(ticker: str, path: str = DEFAULT_PATH) -> Optional[str]:
    """The CDR ticker for `ticker`, or None if it has no known CDR."""
    return load_cdr_map(path).get(ticker.upper())
