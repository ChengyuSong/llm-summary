# Indirect Call Analysis

Indirect calls are resolved using KAMain, an LLVM IR-based static analyzer that performs CFL-reachability-based points-to analysis.

## Overview

Indirect calls (function pointer calls, virtual method calls) are challenging for static analysis because the call target is not known at compile time. KAMain resolves them soundly by analyzing `.bc` files produced during the build.

```
┌─────────────────────┐
│ build-learn          │
│ (-g -flto=full       │
│  -save-temps=obj)    │
└──────────┬──────────┘
           │ .bc files
           ▼
┌─────────────────────┐
│ KAMain              │
│ (CFL-reachability   │
│  points-to analysis)│
└──────────┬──────────┘
           │ --callgraph-json
           ▼
┌─────────────────────┐
│ import-callgraph    │
│ (JSON -> call_edges)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ call_edges table    │
│ (direct + indirect) │
└─────────────────────┘
```

## Prerequisites

- `.bc` files compiled with `-g` (debug info required for source line numbers)
- Build with `-flto=full -save-temps=obj` to produce `.bc` alongside object files
- The `build-learn` command automatically includes these flags

## Workflow

```bash
# 1. Generate a script to run KAMain
llm-summary generate-kanalyzer-script \
  --project libpng \
  --artifacts-dir /data/build-artifacts/libpng \
  --output-json /tmp/libpng_cg.json \
  -o run_kamain.sh

# 2. Run KAMain (produces JSON call graph)
bash run_kamain.sh

# 3. Import into database
llm-summary import-callgraph \
  --json /tmp/libpng_cg.json \
  --db analysis.db \
  --clear-edges \
  --verbose
```

### CLI Options

`generate-kanalyzer-script`:
- `--project` (required): project name under `build-scripts/`
- `--artifacts-dir`: path to `.bc` files (default: `build-scripts/<project>/artifacts`)
- `--output-json` (required): output path for KAMain JSON
- `--kamain-bin`: path to KAMain binary
- `--allocator-file`, `--container-file`: optional JSON inputs for KAMain

`import-callgraph`:
- `--json` (required): path to KAMain JSON
- `--db`: database file
- `--clear-edges`: remove existing `call_edges` before import
- `--verbose`: print matching stats and stub creation

## Function Matching

The importer matches KAMain function entries to existing DB functions using:

1. **Name match**: External functions matched by name (most common)
2. **File+name match**: Internal functions matched by `(name, file_path)`
3. **Suffix match**: Falls back to file suffix matching when build directories differ
4. **Demangling**: C++ mangled names (`_Z...`) demangled via `c++filt`
5. **Stub creation**: Unmatched functions (e.g., libc, zlib) get minimal DB entries

## Coverage

KAMain resolves indirect calls where both the function pointer assignment and the call site are within the analyzed `.bc` files. It cannot resolve:

- **User-supplied callbacks**: Pointers set by application code outside the library (e.g., `png_set_read_fn` callbacks)
- **Cross-library calls**: Pointers flowing through external dependencies not included in the `.bc` set

## Pre-scan

The `scan` command can be run independently to find address-taken functions and indirect callsites from source code using libclang (no LLVM IR needed):

```bash
llm-summary scan \
  --compile-commands build/compile_commands.json \
  --db out.db \
  --verbose
```

This populates the `address_taken_functions`, `address_flows`, and `indirect_callsites` tables. These are useful for understanding which callsites KAMain was able to resolve.

### Target Types (`TargetType` enum)

| Type | Detection | Example |
|------|-----------|---------|
| `address_taken` | `&func`, implicit conversion, passed as arg | `callback = handler; register(handler)` |
| `virtual_method` | `cursor.is_virtual_method()` | `virtual void on_event()` |
| `constructor_attr` | `__attribute__((constructor))` | Init functions called before `main()` |
| `destructor_attr` | `__attribute__((destructor))` | Cleanup functions called after `main()` |
| `section_placed` | `__attribute__((section(".init*"/".fini*"/".ctors"/".dtors")))` | Linker-called functions |
| `ifunc` | `__attribute__((ifunc("resolver")))` | GNU indirect function resolvers |
| `weak_symbol` | `__attribute__((weak))` | Override-able default implementations |

## Validation Results (libpng)

| Metric | Value |
|--------|-------|
| Functions in KAMain JSON | 529 |
| Matched to existing DB | 528 (99.8%) |
| Stubs created | 32 (external libc/zlib) |
| Total call edges | 1789 |
| Direct edges | 1736 |
| Indirect edges (resolved) | 53 |
| Scan callsites resolved by KAMain | 9/19 (47%) |
| Unresolved (user callbacks) | 10/19 |

### compile_commands.json Support

When `compile_commands.json` is provided to `scan`:
- Per-file compile flags are extracted and passed to libclang
- Proper macro expansion ensures conditionally-compiled functions are found
- Only source files (`.c`, `.cpp`) are processed (headers included via `#include`)

## Database Tables

```sql
-- Call graph edges (direct + indirect from KAMain)
call_edges (id, caller_id, callee_id, is_indirect, file_path, line, column)

-- Functions whose addresses are taken (from scan)
address_taken_functions (id, function_id, signature, target_type)

-- Where function addresses flow to (from scan)
address_flows (id, function_id, flow_target, file_path, line_number, context_snippet)

-- Indirect call sites (from scan)
indirect_callsites (id, caller_function_id, file_path, line_number, callee_expr, signature, context_snippet)
```

## File Reference

| File | Purpose |
|------|---------|
| `callgraph_import.py` | Import KAMain JSON call graph into DB |
| `compile_commands.py` | Parse compile_commands.json |
| `indirect/scanner.py` | Find address-taken functions and other indirect call targets |
| `indirect/callsites.py` | Find indirect callsites |
| `models.py` | TargetType, CallEdge, AddressTakenFunction |
| `db.py` | Database schema and methods |
| `cli.py` | scan, generate-kanalyzer-script, import-callgraph commands |
| `scripts/batch_scan_targets.py` | Batch scan all built projects |
