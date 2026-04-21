#!/bin/bash
# Single-project pipeline: run all phases for one project from gpr_projects.json.
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
PREPROCESS=1
WITH_CHECK=0
CHECK_ONLY=0
FORCE=""
INCREMENTAL=""
VERBOSE=""
PASSTHROUGH=()

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <project-name> --backend <backend> [options] [-- extra-args...]

Run the code-contract analysis pipeline for a single project.

Positional:
  PROJECT              Project name from gpr_projects.json

Required:
  --backend NAME       LLM backend (claude, gemini, ollama, llamacpp)

Optional:
  --model NAME         Model override
  --llm-host HOST      Host for local backends (ollama, llamacpp)
  --llm-port PORT      Port for local backends
  --from-phase N       Start from phase N (0-4), skip earlier phases (default: 0)
  --skip-build         Shorthand for --from-phase 2 (skip build-learn and discover-link-units)
  --with-check         After phase 4, run 'llm-summary check' per target and
                       write check_report.json next to each functions.db.
  --check-only         Skip phases 0-3; re-run verification against cached
                       contracts (no contract generation) and persist issues.
  --no-preprocess      Disable clang -E macro expansion during scan (on by default)
  --force              Force re-summarize even if cached
  --incremental        Only re-summarize functions with stale callee summaries
  -v, --verbose        Verbose output
  --                   Pass remaining args through to batch scripts

Phases:
  0  build-learn          (batch_build_learn.py)
  1  discover-link-units  (batch_discover_link_units.py)
  2  scan targets         (batch_scan_targets.py)
  3  call graph + sidecar (batch_call_graph_gen.py, KAMain --ir-sidecar-dir)
  4  code-contract        (batch_code_contract.py; per-function summarize +
                           verify interleaved; optional entry-point check)
EOF
    exit 1
}

# ── Parse args ────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    usage
fi

PROJECT="$1"
shift

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
        --skip-build)
            FROM_PHASE=2; shift ;;
        --with-check)
            WITH_CHECK=1; shift ;;
        --check-only)
            CHECK_ONLY=1; FROM_PHASE=4; shift ;;
        --preprocess)
            PREPROCESS=1; shift ;;
        --no-preprocess)
            PREPROCESS=0; shift ;;
        --force|-f)
            FORCE="--force"; shift ;;
        --incremental)
            INCREMENTAL="--incremental"; shift ;;
        --verbose|-v)
            VERBOSE="--verbose"; shift ;;
        --)
            shift; PASSTHROUGH=("$@"); break ;;
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
FILTER_ARGS="--filter $PROJECT --limit 1"

LLM_ARGS="--backend $BACKEND"
[[ -n "$MODEL" ]]    && LLM_ARGS="$LLM_ARGS --model $MODEL"
[[ -n "$LLM_HOST" ]] && LLM_ARGS="$LLM_ARGS --llm-host $LLM_HOST"
[[ -n "$LLM_PORT" ]] && LLM_ARGS="$LLM_ARGS --llm-port $LLM_PORT"

CHECK_ARG=""
[[ $WITH_CHECK -eq 1 ]] && CHECK_ARG="--check"

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
echo "Pipeline: code-contract (single-project)"
echo "Project:  $PROJECT"
echo "Backend:  $BACKEND"
[[ -n "$MODEL" ]]    && echo "Model:    $MODEL"
[[ -n "$LLM_HOST" ]] && echo "LLM host: $LLM_HOST"
[[ -n "$LLM_PORT" ]] && echo "LLM port: $LLM_PORT"
echo "From:     phase $FROM_PHASE"
[[ $WITH_CHECK -eq 1 ]]  && echo "Check:       yes (entry-point obligations)"
[[ $CHECK_ONLY -eq 1 ]]  && echo "Check-only:  yes (verify cached contracts)"
[[ $PREPROCESS -eq 0 ]]  && echo "Preproc:     no"
[[ -n "$FORCE" ]]        && echo "Force:       yes"
[[ -n "$INCREMENTAL" ]]  && echo "Incremental: yes"
[[ -n "$VERBOSE" ]]      && echo "Verbose:     yes"
echo ""

# ── Phase 0: build-learn ─────────────────────────────────────────────────────
run_phase 0 "build-learn" \
    python3 scripts/batch_build_learn.py \
        $FILTER_ARGS $LLM_ARGS $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 1: discover-link-units ─────────────────────────────────────────────
run_phase 1 "discover-link-units" \
    python3 scripts/batch_discover_link_units.py \
        $FILTER_ARGS $LLM_ARGS $FORCE $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 2: scan targets ────────────────────────────────────────────────────
PREPROCESS_ARG=""
[[ $PREPROCESS -eq 1 ]] && PREPROCESS_ARG="--preprocess"

run_phase 2 "scan targets" \
    python3 scripts/batch_scan_targets.py \
        $FILTER_ARGS $PREPROCESS_ARG $INCREMENTAL $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Phase 3: call graph + IR sidecar ─────────────────────────────────────────
run_phase 3 "call graph" \
    python3 scripts/batch_call_graph_gen.py \
        $FILTER_ARGS --compositional $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

CHECK_ONLY_ARG=""
[[ $CHECK_ONLY -eq 1 ]] && CHECK_ONLY_ARG="--verify-only"

# ── Phase 4: code-contract (summarize + per-function verify) ─────────────────
run_phase 4 "code-contract" \
    python3 scripts/batch_code_contract.py \
        $FILTER_ARGS $LLM_ARGS $CHECK_ARG $FORCE $INCREMENTAL $CHECK_ONLY_ARG $VERBOSE \
        "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

# ── Done ──────────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "=== Pipeline complete for '$PROJECT' in ${TOTAL_ELAPSED}s ==="
