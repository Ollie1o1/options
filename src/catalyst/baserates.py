"""Published phase-transition success rates, used strictly as a REFERENCE PRIOR.

READ THIS BEFORE USING ANY NUMBER HERE. These are historical frequencies for
OTHER drugs that reached the same phase. They say nothing about the specific
asset on the board. They are not a forecast, they must never be multiplied into
anything, and they must never be summed into a score. `describe()` bakes the
caveat into the rendered string precisely so the number cannot travel without
it.

The area map is deliberately coarse and deliberately incomplete: an unmapped
condition returns None and falls back to the all-areas rate, rather than being
forced into the nearest-looking bucket. A wrong therapeutic area silently
swaps in a prior that is off by tens of percentage points.

Figures are round because the underlying literature disagrees at the margin;
they are indicative magnitudes, not precise estimates.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

CITATION = ("Indicative industry phase-transition frequencies (BIO/Informa/QLS "
            "-style analyses of historical development programmes). Figures are "
            "rounded; sources disagree at the margin. Reference prior only.")

RATES: Dict[Tuple[str, str], float] = {
    ("PHASE2", "all"): 0.30,
    ("PHASE2", "oncology"): 0.25,
    ("PHASE2", "neurology"): 0.25,
    ("PHASE2", "infectious_disease"): 0.40,
    ("PHASE2", "ophthalmology"): 0.35,
    ("PHASE3", "all"): 0.58,
    ("PHASE3", "oncology"): 0.40,
    ("PHASE3", "neurology"): 0.50,
    ("PHASE3", "infectious_disease"): 0.70,
    ("PHASE3", "ophthalmology"): 0.55,
}

_AREA_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("oncology", ("cancer", "carcinoma", "tumor", "tumour", "lymphoma",
                  "leukemia", "leukaemia", "myeloma", "sarcoma", "melanoma",
                  "glioblastoma", "neoplasm")),
    ("ophthalmology", ("macular", "retina", "retinal", "geographic atrophy",
                       "glaucoma", "uveitis", "ophthalm")),
    # Keywords are matched as SUBSTRINGS, so every entry must be long enough
    # to be unambiguous. "als" was removed from this list deliberately: it
    # fires on "trials", "false", "signals", and "physicals". "amyotrophic"
    # already covers ALS with no such risk.
    ("neurology", ("alzheimer", "parkinson", "epilep", "multiple sclerosis",
                   "muscular dystrophy", "amyotrophic", "migraine",
                   "neuropath", "huntington")),
    ("infectious_disease", ("influenza", "hiv", "hepatitis", "tuberculosis",
                            "malaria", "covid", "sars-cov", "rsv", "infection")),
)


def area_for(conditions: Sequence[str]) -> Optional[str]:
    """Coarse therapeutic area, or None when nothing matches.

    None is a real answer here — it routes to the all-areas prior instead of
    inventing a specificity the mapping does not have."""
    text = " ".join(conditions or ()).lower()
    if not text:
        return None
    for area, keywords in _AREA_KEYWORDS:
        if any(word in text for word in keywords):
            return area
    return None


def prior(phase: str, area: Optional[str]) -> Optional[float]:
    """Historical success frequency for ``phase``, area-specific if known."""
    if area and (phase, area) in RATES:
        return RATES[(phase, area)]
    return RATES.get((phase, "all"))


def describe(phase: str, area: Optional[str]) -> Optional[str]:
    """One-line rendering that cannot be quoted without its caveat."""
    value = prior(phase, area)
    if value is None:
        return None
    label = phase.replace("PHASE", "Ph")
    where = area.replace("_", " ") if area and (phase, area) in RATES else "all areas"
    return (f"{label} -> approval, {where} ~{value * 100:.0f}% "
            f"(what happened to other drugs, not a forecast for this one)")
