# Bug Triage Agent

Phase-gated ReAct agent that proves memory safety issues (from verification
Pass 5) are either **safe** (cannot manifest) or **feasible** (reachable with
concrete inputs). Uses `complete_with_tools()` with any LLM backend.

## Origin: pbfuzz Lessons

The agent descends from the pbfuzz bug triage agent (`~/fuzzing/pbfuzz/`).
Key takeaway: without strict phase enforcement, the agent loops on analysis
or jumps to conclusions. The phase-gated state machine solved this.

### Reused from pbfuzz

- **Workflow state machine** — phase gating prevents agent from getting stuck
- **Structured memory** — persistent state across ReAct iterations
- **Prompt structure** — static context (project config) separated from
  dynamic state (workflow phase)

### Replaced

| pbfuzz | llm-summary |
|---|---|
| Claude Agent SDK | `complete_with_tools()` ReAct loop (any backend) |
| MCP call_graph/search/corpus servers | `functions.db` queries (in-process) |
| MCP workflow server | Python-side phase controller |
| MCP fuzzer server | ucsan / standalone harness runner |
| MCP protocol layer | Direct Python tool functions |
| cursor-cli | LLM backend abstraction |

## Architecture

```
Verification Pass (existing)
  | SafetyIssue[] (in functions.db)
  v
TriageAgent (triage.py)
  | Phase-gated workflow: ANALYZE -> HYPOTHESIZE -> VERDICT
  | Uses complete_with_tools() with any backend
  |
  | Tools (phase-filtered):
  |   read_function_source(name)   -- source from DB
  |   get_callers(name)            -- call graph + caller source
  |   get_callees(name)            -- direct + resolved indirect calls
  |   get_summaries(name)          -- contracts, allocs, frees, inits
  |   get_verification_summary(name) -- full issue list
  |   git_show(path)               -- read tracked files (optional)
  |   git_grep(pattern)            -- search repo (optional)
  |   git_ls_tree(path)            -- list repo tree (optional)
  |   transition_phase(next)       -- advance workflow
  |   submit_verdict(...)          -- final proof
  v
TriageResult[]
  | Per-issue: hypothesis, reasoning, contracts/path, relevant_functions
  v
gen-harness --validate (existing, enhanced)
  | Reads verdict, generates ucsan harness per entry function
```

## Workflow Phases

Three phases, strictly enforced by `TriageToolExecutor`:

### ANALYZE (read-only)

- Read `SafetyIssue` details: location, kind, severity, description
- Read function source, callee contracts, caller context from DB
- Identify the bug predicate and reaching preconditions
- Optionally inspect project source via git tools
- **Tools**: `read_function_source`, `get_callers`, `get_callees`,
  `get_summaries`, `get_verification_summary`, git tools, `transition_phase`

### HYPOTHESIZE

- Formulate a proof: either safety (updated contracts) or feasibility
  (call chain + assumptions)
- Same tools as ANALYZE for continued investigation
- **Tools**: same as ANALYZE

### VERDICT

- Submit the final proof via `submit_verdict`
- **Tools**: `submit_verdict` only

### Two Outcomes

**Safety proof** — the issue cannot manifest:
- Updated/new contracts that prove the property holds
- Example: all callers pass non-null, so null deref is impossible

**Feasibility proof** — the issue is reachable:
- Concrete call chain from entry to bug site
- Input assumptions that trigger it

Both include `relevant_functions` (scope for validation) and
`assumptions`/`assertions` (for ucsan harness).

## Phase Gating

Enforced in Python by `TriageToolExecutor`:

```python
ALLOWED_TRANSITIONS = {
    "ANALYZE": ["HYPOTHESIZE"],
    "HYPOTHESIZE": ["VERDICT"],
}

PHASE_TOOLS = {
    "ANALYZE":     _DB_TOOLS | GIT_TOOL_NAMES | {"transition_phase"},
    "HYPOTHESIZE": _DB_TOOLS | GIT_TOOL_NAMES | {"transition_phase"},
    "VERDICT":     {"submit_verdict"},
}
```

When the LLM calls a tool not allowed in the current phase, the executor
returns an error with guidance on which tools are available.

## Data Models

```python
@dataclass
class TriageResult:
    function_name: str
    issue_index: int
    issue: SafetyIssue
    hypothesis: str              # "safe" or "feasible"
    reasoning: str               # natural language proof

    # Safety proof
    updated_contracts: list[dict]  # new/strengthened contracts

    # Feasibility proof
    feasible_path: list[str]     # call chain from entry to bug site

    # For symbolic validation (ucsan)
    assumptions: list[str]       # input constraints
    assertions: list[str]        # violation conditions to check
    relevant_functions: list[str]  # functions to keep as real code
    validation_plan: list[dict]  # test case harnesses
```

### submit_verdict Schema

```python
submit_verdict(
    hypothesis: "safe" | "feasible",
    reasoning: str,

    # Safety proof:
    updated_contracts: [{
        "target": str,        # parameter/field name
        "contract_kind": "not_null" | "nullable" | "not_freed" |
                         "initialized" | "buffer_size",
        "description": str,
        "size_expr": str,     # for buffer_size (optional)
    }],

    # Feasibility proof:
    feasible_path: [str],     # call chain

    # Both:
    assumptions: [str],
    assertions: [str],
    relevant_functions: [str],
    validation_plan: [{"entries": [str]}],  # optional
)
```

## Validation Pipeline

Triage results feed into `gen-harness --validate` for symbolic confirmation:

```
llm-summary triage --db <db> -f func -o verdict.json
  ↓
llm-summary gen-harness --db <db> --validate verdict.json
  ↓
For each verdict:
  1. Extract relevant_functions, validation_plan
  2. Find entry functions (no callers within relevant set)
  3. Generate C harness via TRIAGE_VALIDATE_PROMPT
  4. Compile with ko-clang → .ucsan binary
  5. Run ucsan for symbolic validation
```

### TRIAGE_VALIDATE_PROMPT

Generates a C shim for ucsan concolic execution with contract-to-assertion
mapping:

| Contract kind | Assertion |
|---|---|
| `not_null` | `__assert_cond(ptr != NULL, id)` |
| `nullable` | (no assertion) |
| `not_freed` | `__assert_allocated(ptr, 0, id)` |
| `buffer_size(N)` | `__assert_allocated(ptr, N, id)` |
| `initialized(N)` | `__assert_init(ptr, N, id)` |
| `freed` | `__assert_freed(ptr, id)` |

Post-condition qualifiers: `[may_be_null]` suppresses non-NULL assert;
`[when COND]` wraps assertion in `if (COND) { ... }`.

## Git Tools Integration

When `--project-path` is provided, the agent gains access to the project
repository via `GitTools`:

- `git_show` — read tracked files at any ref
- `git_grep` — search file contents
- `git_ls_tree` — list directory structure

These use git plumbing commands with `--` separators and input validation
for injection prevention. Shared with other agents via `git_tools.py`.

When `--project-path` is omitted, git tools are hidden from the tool list
and the agent works purely from DB-provided source.

## CLI Usage

```bash
# Triage all high-severity issues
llm-summary triage --db func-scans/libpng/functions.db \
  --severity high --backend claude -v

# Triage specific function
llm-summary triage --db func-scans/zlib/functions.db \
  -f deflate -v

# Triage specific issue by index
llm-summary triage --db func-scans/zlib/functions.db \
  -f gz_write --issue-index 0 -v

# With git tools for source inspection
llm-summary triage --db func-scans/libpng/functions.db \
  -f png_read_row --project-path /data/csong/opensource/libpng -v

# Save results for validation
llm-summary triage --db func-scans/zlib/functions.db \
  -f deflate -o verdict.json --backend gemini

# Validate triage results with ucsan harness
llm-summary gen-harness --db func-scans/zlib/functions.db \
  --validate verdict.json --ko-clang-path ~/fuzzing/ucsan/ko-clang \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json
```

### CLI Options

```
llm-summary triage --db <path> [options]

Required:
  --db PATH                      Database file

Optional:
  -f, --function NAME...         Function(s) to triage (default: all with issues)
  --severity {high|medium|low}   Filter by issue severity
  --issue-index N                Triage specific issue (requires single -f)
  --backend {claude|openai|ollama|llamacpp|gemini}  (default: claude)
  --model STR                    Model override
  --project-path PATH            Enable git tools
  -o, --output PATH              JSON output file (default: summary to stdout)
  -v, --verbose                  Print detailed logs
  --disable-thinking             Disable extended thinking
  --llm-host, --llm-port         For local backends
```

## Key Files

| File | Purpose |
|---|---|
| `src/llm_summary/triage.py` | TriageAgent, TriageToolExecutor, system prompt, phase gating |
| `src/llm_summary/agent_tools.py` | Tool definitions (DB read tools, triage-only tools), ToolExecutor |
| `src/llm_summary/git_tools.py` | GitTools class, git_show/git_grep/git_ls_tree |
| `src/llm_summary/harness_generator.py` | validate_triage(), TRIAGE_VALIDATE_PROMPT |
| `src/llm_summary/models.py` | SafetyIssue, TriageResult, VerificationSummary |
| `src/llm_summary/cli.py` | `triage` subcommand, `gen-harness --validate` |

## Future Work

- **GDB integration**: Named-pipe based debugger for runtime confirmation
  (port `gdb.sh` from pbfuzz, add `run_gdb`/`run_harness` tools)
- **Batch pipeline**: `scripts/batch_triage.py` for bulk triage across projects
- **Feedback loop**: Triage accuracy tracking, confirmed issues feeding back
  into verification to reduce false positives
