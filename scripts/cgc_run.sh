#!/bin/bash
# CGC benchmark pipeline: extract ground truth, scan, summarize, verify, evaluate.
#
# Make sure you activate the virtual environment first!
# source ~/project/llm-summary/venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND=""
MODEL=""
LLM_HOST=""
LLM_PORT=""
FROM_PHASE=0
FILTER=""
LIMIT=""
FORCE=""
INCREMENTAL=""
VERBOSE=""
CGC_DIR="/data/csong/cgc/cb-multios"
KAMAIN_BIN="kanalyzer"
FUNC_SCANS_DIR="func-scans/cgc"
GT_FILE="cgc_ground_truth.json"

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 --backend <backend> [options]

Run the CGC benchmark pipeline: extract ground truth, scan, summarize, verify, evaluate.

Required:
  --backend NAME       LLM backend (claude, gemini, ollama, llamacpp)

Optional:
  --model NAME         Model override
  --llm-host HOST      Host for local backends
  --llm-port PORT      Port for local backends
  --from-phase N       Start from phase N (0-6), skip earlier phases (default: 0)
  --filter NAME        Only process challenges matching this substring
  --limit N            Limit to at most N challenges
  --cgc-dir PATH       Path to cb-multios directory (default: $CGC_DIR)
  --kamain-bin PATH    Path to KAMain/kanalyzer binary (default: $KAMAIN_BIN)
  --force              Force re-summarize/verify even if cached
  --incremental        Only re-summarize functions with stale callee summaries
  -v, --verbose        Verbose output

Phases:
  0  extract ground truth       (cgc_extract_ground_truth.py)
  1  prepare + scan + callgraph (cgc_prepare.py)
  3  summarize                  (batch_summarize.py)
  4  verify                     (batch_verify.py)
  5  patch re-scan              (cgc_prepare.py --patch)
  6  patched summarize+verify   (llm-summary summarize --incremental)
  7  evaluate                   (cgc_evaluate.py)
EOF
    exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)
            BACKEND="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --llm-host)
            LLM_HOST="$2"; shift 2 ;;
        --llm-port)
            LLM_PORT="$2"; shift 2 ;;
        --from-phase)
            FROM_PHASE="$2"; shift 2 ;;
        --filter)
            FILTER="$2"; shift 2 ;;
        --limit)
            LIMIT="$2"; shift 2 ;;
        --cgc-dir)
            CGC_DIR="$2"; shift 2 ;;
        --kamain-bin)
            KAMAIN_BIN="$2"; shift 2 ;;
        --force|-f)
            FORCE="--force"; shift ;;
        --incremental)
            INCREMENTAL="--incremental"; shift ;;
        --verbose|-v)
            VERBOSE="--verbose"; shift ;;
        *)
            echo "Unknown option: $1"
            usage ;;
    esac
done

if [[ -z "$BACKEND" ]]; then
    echo "ERROR: --backend is required"
    usage
fi

# ── Build arg strings ────────────────────────────────────────────────────────
FILTER_ARGS=""
[[ -n "$FILTER" ]] && FILTER_ARGS="--filter $FILTER"
[[ -n "$LIMIT" ]]  && FILTER_ARGS="$FILTER_ARGS --limit $LIMIT"

LLM_ARGS="--backend $BACKEND"
[[ -n "$MODEL" ]]    && LLM_ARGS="$LLM_ARGS --model $MODEL"
[[ -n "$LLM_HOST" ]] && LLM_ARGS="$LLM_ARGS --llm-host $LLM_HOST"
[[ -n "$LLM_PORT" ]] && LLM_ARGS="$LLM_ARGS --llm-port $LLM_PORT"

# ── Helpers ───────────────────────────────────────────────────────────────────
TOTAL_START=$(date +%s)

run_phase() {
    local phase_num="$1"
    local label="$2"
    shift 2

    if [[ $phase_num -lt $FROM_PHASE ]]; then
        echo "--- Phase $phase_num: $label [SKIPPED] ---"
        return 0
    fi

    echo ""
    echo "=== Phase $phase_num: $label ==="
    echo "  > $*"
    local start=$(date +%s)

    "$@"
    local rc=$?

    local elapsed=$(( $(date +%s) - start ))
    if [[ $rc -ne 0 ]]; then
        echo "FAILED phase $phase_num ($label) after ${elapsed}s (exit $rc)"
        exit $rc
    fi
    echo "--- Phase $phase_num done in ${elapsed}s ---"
}

# ── Print config ──────────────────────────────────────────────────────────────
echo "Pipeline: CGC benchmark"
echo "Backend:  $BACKEND"
[[ -n "$MODEL" ]]    && echo "Model:    $MODEL"
[[ -n "$FILTER" ]]   && echo "Filter:   $FILTER"
[[ -n "$LIMIT" ]]    && echo "Limit:    $LIMIT"
echo "CGC dir:  $CGC_DIR"
echo "From:     phase $FROM_PHASE"
[[ -n "$FORCE" ]]       && echo "Force:       yes"
[[ -n "$INCREMENTAL" ]] && echo "Incremental: yes"
[[ -n "$VERBOSE" ]]     && echo "Verbose:  yes"
echo ""

# ── Phase 0: extract ground truth ────────────────────────────────────────────
run_phase 0 "extract ground truth" \
    python3 scripts/cgc_extract_ground_truth.py \
        --cgc-dir "$CGC_DIR" -o "$GT_FILE" $FILTER_ARGS $VERBOSE

# ── Phase 1: prepare + scan + call graph ─────────────────────────────────────
run_phase 1 "prepare + scan + callgraph" \
    python3 scripts/cgc_prepare.py \
        --cgc-dir "$CGC_DIR" --func-scans-dir "$FUNC_SCANS_DIR" \
        --kamain-bin "$KAMAIN_BIN" \
        $FILTER_ARGS $VERBOSE

# ── Phase 3: summarize ───────────────────────────────────────────────────────
run_phase 3 "summarize" \
    python3 scripts/batch_summarize.py \
        --projects-json scripts/cgc_projects.json \
        --func-scans-dir "$FUNC_SCANS_DIR" \
        $FILTER_ARGS $LLM_ARGS --init-stdlib $FORCE $INCREMENTAL $VERBOSE

# ── Phase 4: verify ──────────────────────────────────────────────────────────
run_phase 4 "verify" \
    python3 scripts/batch_verify.py \
        --func-scans-dir "$FUNC_SCANS_DIR" \
        $FILTER_ARGS $LLM_ARGS $FORCE $INCREMENTAL $VERBOSE

# ── Phase 5: patch re-scan ───────────────────────────────────────────────────
run_phase 5 "patch re-scan" \
    python3 scripts/cgc_prepare.py \
        --cgc-dir "$CGC_DIR" --func-scans-dir "$FUNC_SCANS_DIR" \
        --patch $FILTER_ARGS $VERBOSE

# ── Phase 6: patched incremental summarize + verify ──────────────────────────
if [[ 6 -ge $FROM_PHASE ]]; then
    echo ""
    echo "=== Phase 6: patched incremental summarize + verify ==="

    # Find all patched DBs matching the filter
    for patched_db in "$FUNC_SCANS_DIR"/*/functions_patched.db; do
        [[ -f "$patched_db" ]] || continue
        challenge_name="$(basename "$(dirname "$patched_db")")"

        # Apply filter if set
        if [[ -n "$FILTER" ]]; then
            if ! echo "$challenge_name" | grep -qi "$FILTER"; then
                continue
            fi
        fi

        echo "  [$challenge_name] summarize (incremental)..."
        llm-summary summarize \
            --db "$patched_db" \
            --type allocation --type free --type init --type memsafe \
            $LLM_ARGS --incremental --init-stdlib $VERBOSE

        echo "  [$challenge_name] verify (incremental)..."
        llm-summary summarize \
            --db "$patched_db" \
            --type verify \
            $LLM_ARGS --incremental $VERBOSE
    done

    echo "--- Phase 6 done ---"
else
    echo ""
    echo "--- Phase 6: patched incremental summarize + verify [SKIPPED] ---"
fi

# ── Phase 7: evaluate ────────────────────────────────────────────────────────
run_phase 7 "evaluate" \
    python3 scripts/cgc_evaluate.py \
        --ground-truth "$GT_FILE" \
        --func-scans-dir "$FUNC_SCANS_DIR" \
        $FILTER_ARGS $VERBOSE

# ── Done ──────────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "=== CGC benchmark pipeline complete in ${TOTAL_ELAPSED}s ==="
