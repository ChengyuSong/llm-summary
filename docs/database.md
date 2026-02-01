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
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,                  -- Complete function source code
    source_hash TEXT,             -- SHA-256 hash for change detection
    UNIQUE(name, signature, file_path)
);
```

**Indexes:**
- `idx_functions_name`: Fast lookup by function name
- `idx_functions_file`: Fast lookup by file path

### `allocation_summaries`

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
    UNIQUE(function_id)
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
    context_snippet TEXT          -- Surrounding code for LLM context
);
```

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
    context_snippet TEXT          -- Surrounding code for LLM
);
```

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

## Entity Relationships

```
functions
    │
    ├──1:1──▶ allocation_summaries
    │
    ├──1:N──▶ call_edges (as caller)
    ├──1:N──▶ call_edges (as callee)
    │
    ├──1:1──▶ address_taken_functions
    ├──1:N──▶ address_flows
    │
    ├──1:N──▶ indirect_callsites (as caller)
    └──1:N──▶ indirect_call_targets (as target)
```

## Common Queries

### Get function with summary
```sql
SELECT f.*, s.summary_json
FROM functions f
LEFT JOIN allocation_summaries s ON f.id = s.function_id
WHERE f.name = 'create_buffer';
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
LEFT JOIN allocation_summaries s ON f.id = s.function_id
WHERE s.id IS NULL;
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

Default database file: `summaries.db` in current directory.

Override with `--db` option:
```bash
llm-summary analyze /path/to/project --db /path/to/custom.db
```
