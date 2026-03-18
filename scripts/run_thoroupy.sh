#!/usr/bin/env bash
# Run thoroupy policy scheduler on a harness binary with a trace plan.
#
# Usage:
#   ./scripts/run_thoroupy.sh <binary> <plan.json> [seed_file]
#
# Example:
#   ./scripts/run_thoroupy.sh harnesses/zlibstatic/gzputc.ucsan harnesses/zlibstatic/plan_gzputc.json

set -euo pipefail

THOROUPY_DIR="$HOME/fuzzing/ucsan/thoroupy"
VENV_DIR="$HOME/project/llm-summary/venv"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <binary> <plan.json> [seed_file]"
    exit 1
fi

BINARY="$(realpath "$1")"
PLAN="$(realpath "$2")"
SEED="${3:+$(realpath "$3")}"

if [ ! -f "$BINARY" ]; then
    echo "Error: binary not found: $BINARY"
    exit 1
fi
if [ ! -f "$PLAN" ]; then
    echo "Error: plan not found: $PLAN"
    exit 1
fi

# Output dir next to the plan file
OUT_DIR="$(dirname "$PLAN")"
LOG_FILE="$OUT_DIR/thoroupy_$(basename "$BINARY" .ucsan).log"

# Activate venv
source "$VENV_DIR/bin/activate"

# Run from thoroupy dir (needed for manager imports)
cd "$THOROUPY_DIR"

echo "Binary:  $BINARY"
echo "Plan:    $PLAN"
echo "Log:     $LOG_FILE"
[ -n "$SEED" ] && echo "Seed:    $SEED"
echo "---"

python run_policy.py "$BINARY" "$PLAN" ${SEED:+"$SEED"} --output-dir "$OUT_DIR" 2>&1 \
    | tee "$LOG_FILE"

# Print summary from log
echo "---"
grep -E "\[Plan\].*(Done|infeasible|All target|Uncovered)" "$LOG_FILE" 2>/dev/null || true
