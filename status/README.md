# Status Folder — Real-Money Readiness

This folder is our shared, plain-English supervision dashboard. Read it any time to see
**where we are on the path to deploying real money** without digging through code or git.

## The files

| File | What it answers |
|------|-----------------|
| [STATUS.md](STATUS.md) | "Where are we *right now*, and what's the one number that matters?" |
| [ROADMAP.md](ROADMAP.md) | "What's the full path to real money, and what gate are we waiting on?" |
| [DECISIONS.md](DECISIONS.md) | "What judgment calls did we make, and why?" |

## How it stays current

- **STATUS.md** is the living one — it should be refreshed whenever something material
  changes (new gate decision, cohort milestone, a strategy turned on/off).
- The weekly checkpoint (`scripts/phase1_checkpoint.sh`, Sundays 18:00) writes machine
  reports to `reports/checkpoint_*.md`. STATUS.md is the human summary of those.
- When you want a refresh, just ask: "update the status folder."

## The goal (never lose sight of it)

Deploy **real money** on Long-Call options — but only once the screener has *proven*,
on trades it has never seen, that its ranking beats chance. No proof, no real money.
Last updated: 2026-05-29.
