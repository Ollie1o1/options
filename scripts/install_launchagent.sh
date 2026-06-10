#!/bin/bash
# Install the headless cohort-filling LaunchAgent.
#
# Usage:   bash scripts/install_launchagent.sh
# Remove:  launchctl bootout gui/$UID/com.ollie.options.maintenance \
#            && rm ~/Library/LaunchAgents/com.ollie.options.maintenance.plist
#
# After installing you MUST approve the job once:
#   System Settings -> General -> Login Items & Extensions -> Allow in the Background
# Until approved, launchctl shows status 78 (EX_CONFIG) and nothing runs.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/.venvs/options/bin/python"
LABEL="com.ollie.options.maintenance"
TEMPLATE="$REPO/scripts/$LABEL.plist"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found - the options venv must exist first." >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs"
sed -e "s|__REPO__|$REPO|g" -e "s|__PYTHON__|$PYTHON|g" "$TEMPLATE" > "$TARGET"

# Reload if already present (ignore errors on first install)
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$TARGET"

echo "Installed $TARGET"
echo
launchctl print "gui/$UID/$LABEL" 2>/dev/null | grep -E "state|last exit" || true
echo
echo "NEXT (required, one time): System Settings -> General -> Login Items & Extensions"
echo "  -> under 'Allow in the Background', enable the entry for this agent/python."
echo "  Until then the job exits 78 (EX_CONFIG) and will NOT run."
echo
echo "Verify after the next window (10:20 / 12:20 / 14:05 on weekdays):"
echo "  tail logs/launchagent.log   # and watch the 'Forward cohort: X/50' line"
echo "The screener's startup automation-health check will warn if it goes stale."
