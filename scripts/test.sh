#!/usr/bin/env bash
# Canonical local test command. See scripts/run_tests.py for why the obvious
# alternatives under-collect. Optional argument filters modules by substring.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${OPTIONS_PYTHON:-$HOME/.venvs/options/bin/python}"
PYTHONPATH="$PWD" exec "$PYTHON" scripts/run_tests.py "$@"
