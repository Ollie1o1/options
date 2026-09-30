"""Counterfactual policy lab: replay recorded price paths under alternative
exit and universe policies.

See docs/superpowers/specs/2026-09-22-counterfactual-policy-lab-design.md.
Read-only by construction; nothing here opens a database for write.
"""
from src.policy_lab.types import Outcome, PathPoint, PricePath

__all__ = ["Outcome", "PathPoint", "PricePath"]
