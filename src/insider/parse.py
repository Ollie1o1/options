"""Pure Form 4 XML parsing — no I/O, fixture-testable."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List


def _text(node, path: str) -> str:
    el = node.find(path)
    return (el.text or "").strip() if el is not None and el.text else ""


def _flag(node, path: str) -> bool:
    return _text(node, path) in ("1", "true", "True")


def parse_form4(xml_text: str) -> List[Dict[str, Any]]:
    """Form 4 XML → list of non-derivative transactions:
    {owner, is_officer, is_director, code, shares, price, value, date}.
    Never raises; malformed documents return []."""
    # XML hardening without a new dependency: legitimate Form 4 XML carries no
    # DTD, so any entity/DOCTYPE declaration is rejected outright (blocks
    # billion-laughs / XXE vectors; stdlib etree already never fetches
    # external entities).
    if not xml_text or "<!DOCTYPE" in xml_text or "<!ENTITY" in xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, ValueError, TypeError):
        return []

    owners = []
    for ro in root.findall("reportingOwner"):
        owners.append({
            "owner": _text(ro, "reportingOwnerId/rptOwnerName"),
            "is_officer": _flag(ro, "reportingOwnerRelationship/isOfficer"),
            "is_director": _flag(ro, "reportingOwnerRelationship/isDirector"),
        })
    who = owners[0] if owners else {"owner": "", "is_officer": False,
                                    "is_director": False}

    out: List[Dict[str, Any]] = []
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        try:
            shares = float(_text(tx, "transactionAmounts/transactionShares/value") or 0)
            price = float(_text(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        out.append({
            **who,
            "code": _text(tx, "transactionCoding/transactionCode"),
            "shares": shares,
            "price": price,
            "value": shares * price,
            "date": _text(tx, "transactionDate/value")[:10],
        })
    return out
