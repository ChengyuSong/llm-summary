#!/bin/bash
# A/B test: compare --cache-mode none vs instructions vs source
# Uses a subset of zstdlib (320 functions, 4 passes: allocation, free, init, memsafe)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source venv/bin/activate

SRC_DB="func-scans/zstdlib-test/functions.db"
BACKEND="claude"
MODEL="${1:-claude-sonnet-4-6@default}"
PASSES="--type allocation --type free --type init --type memsafe"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="cache_ab_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=== A/B Test: Prompt Caching ==="
echo "Backend: $BACKEND"
echo "Model: $MODEL"
echo "DB: $SRC_DB"
echo "Results: $RESULTS_DIR/"
echo ""

run_mode() {
    local mode="$1"
    local db_copy="${RESULTS_DIR}/functions_${mode}.db"
    local log_file="${RESULTS_DIR}/log_${mode}.txt"
    local llm_log="${RESULTS_DIR}/llm_${mode}.log"

    echo "--- Mode: $mode ---"
    cp "$SRC_DB" "$db_copy"

    echo "Running summarize with --cache-mode $mode ..."
    /usr/bin/time -v llm-summary summarize \
        --db "$db_copy" \
        --backend "$BACKEND" \
        --model "$MODEL" \
        --cache-mode "$mode" \
        --init-stdlib \
        $PASSES \
        --log-llm "$llm_log" \
        -v 2>&1 | tee -a "$log_file"

    echo ""
    echo "Completed mode=$mode"
    echo "Log: $log_file"
    echo ""
}

# Run all 3 modes sequentially
for mode in none instructions source; do
    run_mode "$mode"
done

echo ""
echo "=== All modes complete ==="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "To compare:"
echo "  grep -E 'LLM calls|Cache hits|Cache read|Cache creation|Functions processed|Errors' $RESULTS_DIR/log_*.txt"
