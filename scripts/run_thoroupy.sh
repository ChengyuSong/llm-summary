#!/usr/bin/env bash
# Run thoroupy policy scheduler on a harness binary with a trace plan.
#
# Usage:
#   ./scripts/run_thoroupy.sh [--output-dir DIR] <binary> <plan.json> [seed_file]
#
# Example:
#   ./scripts/run_thoroupy.sh harnesses/zlibstatic/gzputc.ucsan harnesses/zlibstatic/plan_gzputc.json
#   ./scripts/run_thoroupy.sh --output-dir /tmp/out harnesses/zlibstatic/gzputc.ucsan harnesses/zlibstatic/plan_gzputc.json

set -euo pipefail

THOROUPY_DIR="$HOME/fuzzing/ucsan/thoroupy"
VENV_DIR="$HOME/project/llm-summary/venv"

# Parse options
OUT_DIR=""
CONFIG_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --output-dir) OUT_DIR="$(realpath "$2")"; shift 2 ;;
        --config) CONFIG_ARG="--config $(realpath "$2")"; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    echo "Usage: $0 [--output-dir DIR] [--config config.json] <binary> <plan.json> [seed_file]"
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

# Default output dir: next to the plan file
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$(dirname "$PLAN")"
fi
LOG_FILE="$OUT_DIR/thoroupy_$(basename "$BINARY" .ucsan).log"

# Auto-detect runtime config next to the plan if not specified
if [ -z "$CONFIG_ARG" ]; then
    PLAN_BASE="$(basename "$PLAN" .json)"
    # plan_foo_validation.json → runtime_foo.json
    RT_NAME="${PLAN_BASE#plan_}"
    RT_NAME="${RT_NAME%_validation}"
    RT_PATH="$(dirname "$PLAN")/runtime_${RT_NAME}.json"
    if [ -f "$RT_PATH" ]; then
        CONFIG_ARG="--config $RT_PATH"
    fi
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Run from thoroupy dir (needed for manager imports)
cd "$THOROUPY_DIR"

echo "Binary:  $BINARY"
echo "Plan:    $PLAN"
echo "Log:     $LOG_FILE"
[ -n "$SEED" ] && echo "Seed:    $SEED"
[ -n "$CONFIG_ARG" ] && echo "Config:  ${CONFIG_ARG#--config }"
echo "---"

python run_policy.py "$BINARY" "$PLAN" ${SEED:+"$SEED"} --output-dir "$OUT_DIR" $CONFIG_ARG 2>&1 \
    | tee "$LOG_FILE"

# Print summary from log
echo "---"
grep -E "\[Plan\].*(Done|infeasible|All target|Uncovered)" "$LOG_FILE" 2>/dev/null || true
