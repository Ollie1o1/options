"""Phase 3 execution stack (Sub-project C).

Mirror-mode only: the system sizes a position, computes exit levels, and renders
an order *ticket* for a human to place. There is NO broker API and no order
placement anywhere in this package.

All live output is gated behind BOTH:
  - the validation gate reading READY, AND
  - config ``live_execution.enabled`` being true (default false).
Until both hold, ``ticket.render_ticket`` returns a DRY-RUN refusal.

Modules:
  - sizing.py    — account-aware, capped position sizing (pure).
  - exits.py     — long-call exit levels, reusing paper_manager exit rules (pure).
  - ticket.py    — mirror-mode order ticket + the hard switch.
  - slippage.py  — record intended-vs-actual fills; report real-vs-paper drift.
"""
