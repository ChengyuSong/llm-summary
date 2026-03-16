# Bug Triage Integration

Integration plan for incorporating the pbfuzz bug triage agent
(`~/fuzzing/pbfuzz/`) into llm-summary, using our LLM backend abstraction
for model switching.

## Motivation

The verification pass (Pass 5) finds memory safety issues via Hoare-logic
reasoning, but cannot:

- **Confirm** whether an issue is real or a false positive
- **Reproduce** it with a concrete input
- **Rank** issues by reachability from entry points

The pbfuzz agent can do this through a structured workflow with debugger
integration, but is locked to Claude Agent SDK — can't use Gemini, local
models, etc.

## Key Insight from pbfuzz Evaluation

The phase-gated workflow state machine (PLAN -> IMPLEMENT -> EXECUTE ->REFLECT)
is essential. Without strict phase enforcement, the agent gets stuck — some
execution states are hard to reason about, and the agent either loops on
analysis without acting, or jumps to execution without a plan.

## What to Reuse from pbfuzz

### Keep (proven valuable)

| Component | Why |
|---|---|
| **Workflow state machine** | Phase gating prevents agent from getting stuck |
| **GDB via named pipes** (`gdb.sh`) | Lets LLM set breakpoints, inspect vars, confirm bugs |
| **Structured memory** (BugPredicates, Preconditions, RootCauses, TriggerPlans) | Persistent state across ReAct iterations |
| **Deviation detector** concept | Identify why a path wasn't reached |
| **Prompt structure** (project_config + workflow_state) | Separates static context from dynamic state |

### Replace

| pbfuzz Component | llm-summary Replacement |
|---|---|
| Claude Agent SDK | `complete_with_tools()` ReAct loop (any backend) |
| MCP call_graph server | `functions.db` queries (already have callers/callees) |
| MCP search server | Direct source reads from DB |
| MCP corpus server | Call graph + summaries in DB |
| MCP workflow server | Python-side phase controller (no MCP overhead) |
| MCP fuzzer server | Thoroupy / standalone harness runner |

### Skip

| Component | Why |
|---|---|
| MCP protocol layer | Unnecessary when tools are Python functions in-process |
| DAP/LLDB debugger (`debugger.py`) | Complex; GDB named-pipe approach is simpler and portable |
| AFLGo static analysis inputs | We have our own call graph and function summaries |
| cursor-cli dependency | Replaced by our backend abstraction |

## Architecture

```
Verification Pass (existing)
  | SafetyIssue[] (in functions.db)
  v
Triage Controller (new)
  | Phase-gated workflow: ANALYZE -> PLAN -> CONFIRM -> REFLECT
  | Uses complete_with_tools() with any backend
  |
  | Tools available to LLM:
  |   read_function(name)      -- source from DB
  |   get_callers(name)        -- call graph from DB
  |   get_callees(name)        -- call graph from DB
  |   get_contracts(name)      -- pre/post conditions from DB
  |   read_source_file(path)   -- raw file read
  |   run_gdb(binary, cmds)    -- GDB session via named pipe
  |   run_harness(binary, input) -- execute harness, capture output
  v
TriageReport (new)
  | Per-issue: verdict, reasoning, evidence (GDB trace if available)
  v
Harness Generator (existing, enhanced)
  | Receives triage context: which path to target, preconditions
```

## Workflow Phases

Adapted from pbfuzz's PLAN -> IMPLEMENT -> EXECUTE -> REFLECT, tuned for
triage rather than fuzzing:

### ANALYZE (read-only)

- Read the `SafetyIssue` details: location, kind, severity, description
- Read function source, callee contracts, caller context from DB
- Identify the bug predicate: what condition triggers this issue
- Identify reaching preconditions: what caller state reaches this path
- **Allowed tools**: `read_function`, `get_callers`, `get_callees`,
  `get_contracts`, `read_source_file`
- **Output**: BugPredicates, Preconditions, RootCauses

### PLAN

- Design a confirmation strategy: what input/state would trigger the issue
- Create TriggerPlans with complexity scores
- If issue looks unreachable (all paths guarded), can verdict as `unlikely`
  and skip remaining phases
- **Allowed tools**: same as ANALYZE + write to workflow state
- **Output**: TriggerPlans, preliminary verdict

### CONFIRM (requires harness binary)

- Execute harness under GDB with breakpoints at issue location
- Check if bug path is reachable with crafted input
- Capture evidence: call stack, variable values at breakpoint
- **Allowed tools**: `run_gdb`, `run_harness`
- **Output**: Evidence, updated verdict

### REFLECT

- Analyze why confirmation succeeded or failed
- For unreached paths: identify which precondition was violated
- Update verdict with evidence
- If inconclusive and attempts remain, loop back to PLAN
- **Allowed tools**: read-only, analysis tools
- **Output**: Final `TriageVerdict`

## Data Models

```python
class TriageVerdict:
    issue: SafetyIssue           # from verification pass
    verdict: str                 # confirmed | likely | unlikely | false_positive
    confidence: float            # 0.0 - 1.0
    reasoning: str               # LLM explanation
    evidence: list[str]          # GDB traces, path analysis
    bug_predicates: list[dict]   # extracted triggering conditions
    preconditions: list[dict]    # reaching conditions
    root_cause: str | None       # vulnerability category

class TriageWorkflowState:
    phase: str                   # ANALYZE | PLAN | CONFIRM | REFLECT
    issue: SafetyIssue
    bug_predicates: list[dict]
    preconditions: list[dict]
    root_causes: list[dict]
    trigger_plans: list[dict]
    evidence: list[str]
    metrics: dict                # attempts, phase transitions
    log: list[str]               # action history (capped)
```

## Phase Gating

The controller enforces phase transitions in Python (no MCP needed):

```python
ALLOWED_TRANSITIONS = {
    "ANALYZE": ["PLAN"],
    "PLAN": ["CONFIRM", "REFLECT"],  # REFLECT if early verdict
    "CONFIRM": ["REFLECT"],
    "REFLECT": ["PLAN", "DONE"],     # loop back or finish
}

PHASE_TOOLS = {
    "ANALYZE": {"read_function", "get_callers", "get_callees",
                "get_contracts", "read_source_file"},
    "PLAN":    {"read_function", "get_callers", "get_callees",
                "get_contracts", "read_source_file"},
    "CONFIRM": {"run_gdb", "run_harness", "read_source_file"},
    "REFLECT": {"read_function", "get_callers", "get_callees",
                "get_contracts", "read_source_file"},
}
```

When the LLM calls a tool not allowed in the current phase, the controller
returns an error message with guidance on which phase to transition to.

## Implementation Phases

### Phase 1: Static triage (no execution)

- `src/llm_summary/triage.py` — controller + tool definitions
- CLI: `llm-summary triage --db <db> --severity high,medium`
- ANALYZE and PLAN phases only (no CONFIRM)
- Uses `complete_with_tools()` ReAct loop
- Output: JSON report with per-issue verdicts
- Works with any backend

### Phase 2: GDB integration

- Port `gdb.sh` named-pipe wrapper
- Add `run_gdb` tool to the ReAct loop
- CONFIRM phase enabled
- Requires harness binary (from `gen-harness`)
- Pipeline: triage -> gen-harness -> confirm

### Phase 3: Batch pipeline + harness chaining

- `scripts/batch_triage.py` — run triage across all high/medium issues
- Auto-invoke `gen-harness` for confirmed issues
- Integration with thoroupy for deeper confirmation
- Aggregate report generation

### Phase 4: Feedback loop

- Triage results improve future verification (fewer false positives)
- Confirmed issues feed into targeted harness generation
- Track triage accuracy over time (ground truth from manual review)

## Key Files

| File | Purpose |
|---|---|
| `src/llm_summary/triage.py` | Triage controller, workflow, tool defs |
| `src/llm_summary/models.py` | TriageVerdict, TriageWorkflowState models |
| `src/llm_summary/cli.py` | `triage` subcommand |
| `scripts/gdb_pipe.sh` | GDB named-pipe wrapper (ported from pbfuzz) |
| `scripts/batch_triage.py` | Batch triage runner |

## Example Usage

```bash
# Static triage (Phase 1)
llm-summary triage --db func-scans/libpng/functions.db \
  --severity high,medium --backend claude

# With GDB confirmation (Phase 2)
llm-summary triage --db func-scans/libpng/functions.db \
  --severity high --backend gemini \
  --harness-dir harnesses/libpng/ --confirm

# Batch (Phase 3)
python scripts/batch_triage.py --project libpng --backend claude \
  --severity high,medium --confirm
```
