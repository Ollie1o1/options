"""Long-term stock accumulation desk (HOLDINGS).

Buy-zone watcher for long-term holdings: curated tranche ladders per name
(longterm_plan.json), zone-state context (drawdown / sigma-distance — NEVER
predictive scoring), real-fill tracking (data/longterm.db), terminal board,
and a desk-kit HTML report. Display/checklist only — executes nothing.
"""

from .board import banner, menu  # noqa: F401
from .plan import Plan, PlanName, Tranche, load_plan, save_plan  # noqa: F401
from .report import write_report  # noqa: F401
