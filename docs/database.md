# Database Schema

The tool uses SQLite to store all analysis data. This document describes the database schema.

## Tables

### `functions`

Stores extracted function metadata and source code.

```sql
CREATE TABLE functions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,      -- e.g., "char*(size_t)"
    canonical_signature TEXT,     -- Typedef-resolved signature
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,                  -- Complete function source code
    pp_source TEXT,               -- Preprocessed source (used for hashing)
    source_hash TEXT,             -- SHA-256 hash for change detection
    params_json TEXT,             -- Parameter list with types
    callsites_json TEXT,          -- Direct callsites (file/line/callee_expr)
    attributes TEXT DEFAULT '',   -- e.g., 'noreturn', 'pure'
    decl_header TEXT,             -- For externs: header path from `clang -E`
    UNIQUE(name, signature, file_path)
);
```

**Indexes:**
- `idx_functions_name`: Fast lookup by function name
- `idx_functions_file`: Fast lookup by file path

### `code_contract_summaries` (primary)

Per-function Hoare-style contracts produced by the `summarize` (code-contract) pass — the primary summary table consumed by `check`, `triage`, `gen-harness`, `lookup`, and `export`.

```sql
CREATE TABLE code_contract_summaries (
    function_id INTEGER PRIMARY KEY REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    noreturn INTEGER NOT NULL DEFAULT 0,
    body_annotated TEXT,                -- Function source with `// @requires/@ensures` pinned
    model TEXT NOT NULL,
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    tokens_cache_read INTEGER NOT NULL DEFAULT 0,
    tokens_cache_write INTEGER NOT NULL DEFAULT 0,
    struggle_max REAL NOT NULL DEFAULT 0.0,
    struggle_scores TEXT NOT NULL DEFAULT '{}',  -- per-property struggle scores
    retried INTEGER NOT NULL DEFAULT 0,
    retry_model TEXT,                   -- Model used on retry, if any
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Summary JSON format:** keyed by property (`memsafe`, `memleak`, `overflow`).

```json
{
  "function": "process_buffer",
  "properties": ["memsafe", "overflow"],
  "requires": {
    "memsafe":  ["buf != NULL", "buffer_size(buf) >= len"],
    "overflow": ["len <= INT_MAX"]
  },
  "ensures": {
    "memsafe":  ["true"],
    "overflow": ["return value in [0, len]"]
  },
  "modifies": {"memsafe": ["*buf"], "overflow": []},
  "notes":    {"memsafe": "..."},
  "origin":   {"memsafe": [{"kind": "local"}, {"kind": "callee", "callee": "memcpy", "idx": 1}]},
  "analysis": {"memsafe": "..."},
  "confidence": {"memsafe": "high", "overflow": "medium"},
  "noreturn": false,
  "inline_body": ""
}
```

**Design notes:**
- **No verdict field** — bug-finding is the separate `check` pass that walks `requires` chains from entry functions back to their `origin`.
- **`origin[P][i]`** records whether the i-th `requires[P]` clause is local (a leaf operation) or propagated from a callee (with the callee name + clause index). Powers witness chains in `check`.
- **`inline_body`** is non-empty only when the function was inlined at callsites instead of being summarized (small bodies); when set, `properties` are empty and the body is pasted at every caller.

### Legacy summary tables

The five tables below were populated by the legacy multi-pass pipeline (allocation/free/init/memsafe/verify). The CLI no longer writes to them — `summarize` runs only the code-contract pass — but the schemas remain so that older databases keep loading and the underlying pass classes (still importable) can be driven from Python.

### `allocation_summaries` (legacy)

Stores LLM-generated allocation summaries.

```sql
CREATE TABLE allocation_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,   -- Full JSON summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,              -- Which LLM generated this
    UNIQUE(function_id)
);
```

**Summary JSON format:**
```json
{
  "function": "create_buffer",
  "allocations": [
    {
      "type": "heap",
      "source": "malloc",
      "size_expr": "n + 1",
      "size_params": ["n"],
      "returned": true,
      "stored_to": null,
      "may_be_null": true
    }
  ],
  "parameters": {
    "n": {
      "role": "size_indicator",
      "used_in_allocation": true
    }
  },
  "description": "Allocates n+1 bytes for null-terminated buffer."
}
```

### `free_summaries` (legacy)

Stores LLM-generated free/deallocation summaries (Pass 2).

```sql
CREATE TABLE free_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,   -- Full JSON summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,              -- Which LLM generated this
    UNIQUE(function_id)
);
```

**Summary JSON format:**
```json
{
  "function": "png_destroy_info_struct",
  "frees": [
    {
      "target": "info_ptr->palette",
      "target_kind": "field",
      "deallocator": "png_free",
      "conditional": true,
      "nulled_after": true
    },
    {
      "target": "info_ptr",
      "target_kind": "parameter",
      "deallocator": "png_free",
      "conditional": false,
      "nulled_after": true
    }
  ],
  "description": "Frees all dynamically allocated fields of the info struct, then frees the struct itself."
}
```

**`target_kind` values:** `parameter`, `field`, `local`, `return_value`

### `init_summaries` (legacy)

Stores LLM-generated initialization summaries (Pass 3).

```sql
CREATE TABLE init_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,   -- Full JSON summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,              -- Which LLM generated this
    UNIQUE(function_id)
);
```

**Summary JSON format:**
```json
{
  "function": "png_set_IHDR",
  "inits": [
    {
      "target": "info_ptr->width",
      "target_kind": "field",
      "initializer": "assignment",
      "byte_count": "sizeof(png_uint_32)"
    },
    {
      "target": "*out",
      "target_kind": "parameter",
      "initializer": "memset",
      "byte_count": "n"
    }
  ],
  "description": "Always initializes all IHDR-related fields of the info structure."
}
```

**`target_kind` values:** `parameter`, `field`, `return_value` (no `local` — locals are not caller-visible post-conditions)

### `memsafe_summaries` (legacy)

Stores LLM-generated safety contract summaries (Pass 4 — pre-conditions).

```sql
CREATE TABLE memsafe_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,   -- Full JSON summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,              -- Which LLM generated this
    UNIQUE(function_id)
);
```

**Summary JSON format:**
```json
{
  "function": "process_buffer",
  "contracts": [
    {
      "target": "buf",
      "contract_kind": "not_null",
      "description": "buf must not be NULL"
    },
    {
      "target": "buf",
      "contract_kind": "buffer_size",
      "description": "buf must point to at least len bytes",
      "size_expr": "len",
      "relationship": "byte_count"
    },
    {
      "target": "ctx",
      "contract_kind": "initialized",
      "description": "ctx must be initialized before use"
    }
  ],
  "description": "Requires buf to be non-NULL with at least len bytes, and ctx to be initialized."
}
```

**`contract_kind` values:** `not_null`, `not_freed`, `buffer_size`, `initialized`

**`size_expr` and `relationship`:** Only present for `buffer_size` contracts.

### `verification_summaries` (legacy)

Stores verification results and simplified contracts (Pass 5).

```sql
CREATE TABLE verification_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,   -- Full JSON summary
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,              -- Which LLM generated this
    UNIQUE(function_id)
);
```

**Summary JSON format:**
```json
{
  "function": "process_data",
  "simplified_contracts": [
    {
      "target": "buf",
      "contract_kind": "buffer_size",
      "description": "buf must point to at least len bytes",
      "size_expr": "len",
      "relationship": "byte_count"
    }
  ],
  "issues": [
    {
      "location": "call to memcpy at line 42",
      "issue_kind": "buffer_overflow",
      "description": "memcpy destination may be too small",
      "severity": "high",
      "callee": "memcpy",
      "contract_kind": "buffer_size"
    }
  ],
  "description": "buf size contract propagated; memcpy may overflow if caller passes insufficient buffer."
}
```

**`issue_kind` values:** `null_deref`, `buffer_overflow`, `use_after_free`, `double_free`, `uninitialized_use`

**`severity` values:** `high` (definite violation), `medium` (depends on caller), `low` (unlikely/defensive)

**`simplified_contracts`:** Reuses `MemsafeContract` format — the subset of Pass 4 contracts NOT satisfied internally by the function.

### `leak_summaries`, `integer_overflow_summaries` (legacy)

Optional legacy passes; same `(function_id, summary_json, model_used, created_at, updated_at)` shape as the other legacy summary tables. Not driven by the current CLI.

### `call_edges`

Stores the call graph with callsite information.

```sql
CREATE TABLE call_edges (
    id INTEGER PRIMARY KEY,
    caller_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    callee_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    is_indirect INTEGER DEFAULT 0,
    file_path TEXT,               -- Callsite location
    line INTEGER,
    column INTEGER
);
```

**Indexes:**
- `idx_call_edges_caller`: Fast lookup of callees
- `idx_call_edges_callee`: Fast lookup of callers

### `address_taken_functions`

Functions whose addresses are taken somewhere in the codebase.

```sql
CREATE TABLE address_taken_functions (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    signature TEXT NOT NULL,      -- For type matching
    target_type TEXT NOT NULL DEFAULT 'address_taken',
    UNIQUE(function_id, target_type)
);
```

### `address_flows`

Tracks where function addresses flow to.

```sql
CREATE TABLE address_flows (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    flow_target TEXT NOT NULL,    -- e.g., "struct task.callback"
    file_path TEXT,
    line_number INTEGER,
    context_snippet TEXT,         -- Surrounding code for LLM context
    UNIQUE(function_id, flow_target, file_path, line_number)
);
```

**Indexes:**
- `idx_address_flows_function`: Fast lookup by function

**Flow target examples:**
- `var:handler` - Assigned to local variable
- `field:callback` - Assigned to struct field
- `param:register_callback[0]` - Passed as function argument
- `array:handlers` - Stored in array

### `indirect_callsites`

Indirect call expressions found in the codebase.

```sql
CREATE TABLE indirect_callsites (
    id INTEGER PRIMARY KEY,
    caller_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    callee_expr TEXT NOT NULL,    -- e.g., "ctx->handler"
    signature TEXT NOT NULL,      -- Expected signature
    context_snippet TEXT,         -- Surrounding code for LLM
    UNIQUE(caller_function_id, file_path, line_number, callee_expr)
);
```

**Indexes:**
- `idx_indirect_callsites_caller`: Fast lookup by caller function

### `indirect_call_targets`

LLM-resolved targets for indirect calls.

```sql
CREATE TABLE indirect_call_targets (
    callsite_id INTEGER REFERENCES indirect_callsites(id) ON DELETE CASCADE,
    target_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    confidence TEXT,              -- "high", "medium", "low"
    llm_reasoning TEXT,           -- LLM's explanation
    PRIMARY KEY(callsite_id, target_function_id)
);
```

### `address_flow_summaries`

LLM-generated flow summaries for address-taken functions.

```sql
CREATE TABLE address_flow_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    flow_destinations_json TEXT NOT NULL,
    semantic_role TEXT,
    likely_callers_json TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id)
);
```

**Indexes:**
- `idx_flow_summaries_function`: Fast lookup by function

### `build_configs`

Stores build configurations discovered by the build agent.

```sql
CREATE TABLE build_configs (
    project_path TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    build_system TEXT NOT NULL,
    configuration_json TEXT,
    script_path TEXT,
    artifacts_dir TEXT,
    compile_commands_path TEXT,
    llm_backend TEXT,
    llm_model TEXT,
    build_attempts INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_built_at TIMESTAMP
);
```

**Indexes:**
- `idx_build_configs_name`: Fast lookup by project name

### `container_summaries`

Stores container/wrapper function analysis results.

```sql
CREATE TABLE container_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    container_arg INTEGER NOT NULL,
    store_args_json TEXT NOT NULL,
    load_return INTEGER NOT NULL DEFAULT 0,
    container_type TEXT NOT NULL,
    confidence TEXT NOT NULL,
    heuristic_score INTEGER,
    heuristic_signals_json TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id)
);
```

**Indexes:**
- `idx_container_summaries_function`: Fast lookup by function

### `typedefs`

Type declarations extracted from source (typedefs, C++ using aliases, struct/class/union).

```sql
CREATE TABLE typedefs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'typedef',
    underlying_type TEXT NOT NULL,
    canonical_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    UNIQUE(name, kind, file_path)
);
```

**Kind values:** `typedef`, `using`, `struct`, `class`, `union`

**Indexes:**
- `idx_typedefs_name`: Fast lookup by typedef name

### `issue_reviews`

Records human/agent triage verdicts for issues flagged by the code-contract pass. Issues are matched by a stable fingerprint so re-running `summarize` doesn't lose the verdict when the issue list reorders.

```sql
CREATE TABLE issue_reviews (
    id INTEGER PRIMARY KEY,
    function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    issue_index INTEGER NOT NULL,        -- Index in latest summary's issues[]
    issue_fingerprint TEXT NOT NULL,     -- Stable hash across re-runs
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id, issue_fingerprint)
);
```

**Status values:** `pending`, `confirmed`, `false_positive`, `wontfix`

**Indexes:**
- `idx_issue_reviews_function`
- `idx_issue_reviews_status`

Driven by the `review-issue` and `show-issues` commands; consumed by `gen-harness --assess-issue`.

### `function_blocks`

Switch-case (and similar) chunks split out from large functions so the LLM can summarize them piecewise. Each block is summarized like a synthetic function and the parent's summary references the per-block results.

```sql
CREATE TABLE function_blocks (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,           -- e.g., 'switch_case'
    label TEXT NOT NULL,          -- Case label or block identifier
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,
    suggested_name TEXT,          -- LLM-proposed extracted-function name
    suggested_signature TEXT,
    summary_json TEXT             -- Per-block summary
);
```

**Indexes:**
- `idx_blocks_function`: Fast lookup by parent function

### `function_ir_facts`

Per-function IR facts imported from KAMain's `--ir-sidecar-dir` output. One row per function in this DB whose name matches a function in any imported sidecar. `facts_json` holds the raw per-function blob (effects, branches, ranges, int_ops, features, ir_hash, cg_hash) used by Phase 3 of the code-contract pipeline.

```sql
CREATE TABLE function_ir_facts (
    function_id INTEGER PRIMARY KEY REFERENCES functions(id) ON DELETE CASCADE,
    ir_hash TEXT,
    cg_hash TEXT,
    facts_json TEXT NOT NULL,
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `function_scan_issues`

Frontend-time issues from clang's diagnostic output during the `scan` compile (e.g. `-Winteger-overflow` on constant-expression UB). One row per `(function_id, line, kind)`. `kind` mirrors the clang warning flag (e.g. `integer-overflow`) minus the leading `-W`. The code-contract pass reads these before any LLM round trip.

```sql
CREATE TABLE function_scan_issues (
    function_id INTEGER NOT NULL REFERENCES functions(id) ON DELETE CASCADE,
    line INTEGER,
    column INTEGER,
    kind TEXT NOT NULL,
    message TEXT,
    PRIMARY KEY(function_id, line, kind)
);
```

**Indexes:**
- `idx_scan_issues_function`: Fast lookup by function

### `scan_metadata`

Free-form key/value store used to track project repo state for incremental scanning (e.g. last-scanned commit hash, source-tree root, scan tool version).

```sql
CREATE TABLE scan_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

## Entity Relationships

```
functions
    │
    ├──1:1──▶ code_contract_summaries          (primary, written by `summarize`)
    ├──1:1──▶ function_ir_facts                (KAMain IR sidecar import)
    ├──1:N──▶ function_scan_issues             (clang frontend warnings)
    ├──1:N──▶ function_blocks                  (split chunks of large functions)
    ├──1:N──▶ issue_reviews                    (triage verdicts)
    │
    ├──1:1──▶ container_summaries
    │
    ├──1:N──▶ call_edges (as caller)
    ├──1:N──▶ call_edges (as callee)
    │
    ├──1:1──▶ address_taken_functions
    ├──1:N──▶ address_flows
    ├──1:1──▶ address_flow_summaries
    │
    ├──1:N──▶ indirect_callsites (as caller)
    ├──1:N──▶ indirect_call_targets (as target)
    │
    └── legacy 1:1 ──▶ allocation_summaries, free_summaries, init_summaries,
                       memsafe_summaries, verification_summaries,
                       leak_summaries, integer_overflow_summaries

build_configs   (standalone, keyed by project_path)
typedefs        (standalone, keyed by name+kind+file)
scan_metadata   (standalone, key/value)
```

## Common Queries

### Get function with its code-contract summary
```sql
SELECT f.*, s.summary_json, s.noreturn, s.struggle_max
FROM functions f
LEFT JOIN code_contract_summaries s ON f.id = s.function_id
WHERE f.name = 'process_buffer';
```

### Get call graph for a function
```sql
SELECT
    caller.name as caller_name,
    callee.name as callee_name,
    e.file_path,
    e.line,
    e.column
FROM call_edges e
JOIN functions caller ON e.caller_id = caller.id
JOIN functions callee ON e.callee_id = callee.id
WHERE caller.name = 'main';
```

### Find all callers of a function
```sql
SELECT DISTINCT f.name, f.file_path
FROM functions f
JOIN call_edges e ON f.id = e.caller_id
WHERE e.callee_id = (SELECT id FROM functions WHERE name = 'malloc');
```

### Get functions needing re-analysis
```sql
SELECT f.*
FROM functions f
LEFT JOIN code_contract_summaries s ON f.id = s.function_id
WHERE s.function_id IS NULL;
```

### List pending high-severity issues
```sql
SELECT f.name, r.issue_index, r.reason
FROM issue_reviews r
JOIN functions f ON r.function_id = f.id
WHERE r.status = 'pending';
```

### Get indirect call candidates
```sql
SELECT f.name, f.signature, atf.signature as expected_sig
FROM address_taken_functions atf
JOIN functions f ON atf.function_id = f.id
WHERE atf.signature = 'void (*)(void*, int)';
```

## Database Operations

### Change Detection

The `source_hash` field enables efficient change detection:

```python
def needs_update(func: Function) -> bool:
    current_hash = compute_source_hash(func.source)
    stored = db.get_function(func.id)
    return stored.source_hash != current_hash
```

### Cascade Invalidation

When a function changes, invalidate it and all callers:

```python
def invalidate_and_cascade(func_id: int) -> list[int]:
    invalidated = []
    to_process = [func_id]

    while to_process:
        current = to_process.pop()
        if current in invalidated:
            continue

        db.delete_summary(current)
        invalidated.append(current)

        callers = db.get_callers(current)
        to_process.extend(callers)

    return invalidated
```

## File Location

Default database file: `summaries.db` in current directory. The standard layout used by the batch scripts is `func-scans/<project>/<target>/functions.db` (one DB per link unit).

Override with `--db` option:
```bash
llm-summary scan --compile-commands build/compile_commands.json --db /path/to/custom.db
llm-summary summarize --db /path/to/custom.db
```
