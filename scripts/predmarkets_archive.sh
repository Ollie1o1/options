#!/bin/bash
# Daily prediction-market archive. Snapshots Kalshi macro series into
# data/predmarkets.db with an `archived_at` stamp.
#
# ARCHIVE ONLY. Nothing in the screener reads this database. The point is to
# accumulate a point-in-time record now so "does this explain anything?" is
# answerable in a few months, rather than arguing about it today. This repo
# already measured news sentiment at IC ~0 and weight 0.0; the prior on a
# prediction-market feed as alpha is poor, and archiving first is how you find
# out cheaply.
set -uo pipefail

REPO="/Users/ollie/Projects/options"
PYTHON="$HOME/.venvs/options/bin/python"

cd "$REPO" || exit 1
mkdir -p logs

echo "=== $(date '+%Y-%m-%d %H:%M:%S') predmarkets archive ==="
PYTHONPATH="$REPO" "$PYTHON" -m src.predmarkets
status=$?
echo "exit=$status"
exit $status
