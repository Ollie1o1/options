# Status Folder — Real-Money Readiness

This folder preserves historical plans and decisions. STATUS.md and ROADMAP.md
are older snapshots and must not be read as current readiness guidance.

For the latest available measurements, start with the dated
[generated gate status](../reports/GATE_STATUS.md) and
[paper track record](../reports/TRACK_RECORD.md). A generated report is still
a snapshot: check its date, and do not assume it reflects today's state.

## The files

| File | What it answers |
|------|-----------------|
| [STATUS.md](STATUS.md) | Historical readiness snapshot, with a supersession notice |
| [ROADMAP.md](ROADMAP.md) | Historical plan, with a supersession notice |
| [DECISIONS.md](DECISIONS.md) | "What judgment calls did we make, and why?" |

## How it stays current

- Generated reports supply the measured gate decision and track record. Historical
  prose does not override them or authorise execution.
- The weekly checkpoint (`scripts/phase1_checkpoint.sh`, Sundays 18:00) writes machine
  reports to `reports/checkpoint_*.md`. STATUS.md is the human summary of those.
- When you want a refresh, just ask: "update the status folder."

## Research objective

Measure candidate strategies on unseen data, preserve failed findings, and keep
research observations distinct from execution authorisation.
Last updated: 2026-11-24.
