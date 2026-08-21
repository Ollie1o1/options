# Equity scheduler agents

The two jobs that put trades into the book and take them out. **launchd, not
cron** — the reason is measured, not stylistic.

## Why these moved off cron (2026-08-21)

The three equity entry windows and the exit enforcer ran from `crontab`. macOS
cron **silently skips a job scheduled while the Mac is asleep and never runs it
late**. Measured over twelve weekdays:

```
weekday      10:30  12:30  14:15
2026-08-07     ok     ok     ok
2026-08-10    ---     ok     ok
2026-08-11    ---    ---     ok
2026-08-13    ---    ---     ok
2026-08-14    ---     ok     ok
2026-08-17    ---     ok     ok
2026-08-18    ---    ---    ---
2026-08-19    ---     ok     ok
2026-08-20    ---     ok    ---
2026-08-21    ---    ---    ---

14 of 36 scheduled entry windows ran — 39%
```

The 10:30 slot fired **once in twelve weekdays**. Nothing was broken: cron was
running, the script exited `rc=0` whenever it was invoked, and the runs that did
happen logged correctly. The machine was simply asleep.

`launchd` runs a missed `StartCalendarInterval` job once, on wake. That single
difference is the whole change. The crypto jobs stay on cron because they are
hourly and catch whatever hour the machine is awake for; the weekly calibration
snapshots stay on cron for the same reason.

This matters beyond tidiness: the Bull Put sample is the only evidence that will
settle whether this book has an edge, and at 39% window uptime it accumulates at
well under half the assumed rate.

## The agents

| label | schedule | script |
|---|---|---|
| `com.ollie.options.auto-log-equity` | Mon–Fri 10:30, 12:30, 14:15 | `scripts/auto_log_equity.sh` |
| `com.ollie.options.enforce-exits` | Mon–Fri 14:07 | `scripts/enforce_exits.sh` |

**`RunAtLoad` is `false` on both, deliberately.** The auto-logger writes real
paper trades; loading or reloading an agent must never fire an entry window as a
side effect.

The label prefix must stay `com.ollie.options.*`. `maintenance_health.py`
matches on that prefix, and a health check looking at the wrong prefix reports
"all clear" for dead jobs — which is exactly how three agents sat dead for 46
days once already.

## Install / reinstall

```bash
cp scripts/launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ollie.options.auto-log-equity.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ollie.options.enforce-exits.plist
launchctl list | grep ollie.options        # expect 5 agents, exit 0
```

macOS may ask for a one-time background-item approval: **System Settings >
General > Login Items & Extensions > Allow in the Background**. Nothing runs
until that is granted, and no error is printed if it is not — check
`launchctl list` rather than assuming.

## Verifying it worked

Do NOT verify by kickstarting `auto-log-equity`; that logs real trades. Check
that a scheduled window produced a line:

```bash
grep -a "auto_log_equity starting" logs/auto_log_equity.log | tail -5
```

The crontab lines these replaced were removed on 2026-08-21; the crypto and
calibration entries were left alone.
