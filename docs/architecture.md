# Architecture Overview

This document describes the architecture of the LLM-based memory allocation summary analysis tool.

## System Overview

The tool performs compositional, bottom-up analysis of C/C++ code to generate memory allocation summaries. It processes functions in dependency order (callees before callers) so that callee summaries are available when analyzing callers.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Source Files   │────▶│ Function        │────▶│ Call Graph      │
│  (.c/.cpp/.h)   │     │ Extractor       │     │ Builder         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Summary         │◀────│ LLM Summary     │◀────│ Topological     │
│ Database        │     │ Generator       │     │ Ordering (SCCs) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Components

### 1. Function Extractor (`extractor.py`)

Uses libclang to parse C/C++ source files and extract function definitions.

**Key classes:**
- `FunctionExtractor`: Basic extraction with function signatures and source
- `FunctionExtractorWithBodies`: Full parsing including function bodies for call analysis

**Output:** List of `Function` objects containing:
- Function name and signature
- File path and line numbers
- Complete source code
- Source hash for change detection

### 2. Call Graph Builder (`callgraph.py`)

Builds the call graph by analyzing function bodies for call expressions.

**Features:**
- Extracts direct function calls
- Records callsite locations (file, line, column)
- Integrates with indirect call analysis

**Key class:** `CallGraphBuilder`

### 3. Indirect Call Analysis (`indirect/`)

Handles function pointer calls and virtual methods.

#### Scanner (`indirect/scanner.py`)
Finds functions whose addresses are taken:
- `&function` expressions
- Function passed as callback argument
- Function assigned to pointer variable

#### Callsite Finder (`indirect/callsites.py`)
Identifies indirect call expressions:
- `ptr->callback(args)`
- `handlers[i](args)`
- `(*fptr)(args)`

#### Resolver (`indirect/resolver.py`)
Uses LLM to determine likely targets for indirect calls based on:
- Matching function signatures
- Code context
- Address flow information

### 4. Topological Ordering (`ordering.py`)

Computes processing order using Tarjan's SCC algorithm.

**Features:**
- Identifies strongly connected components (recursive function groups)
- Orders SCCs topologically (callees before callers)
- Handles mutual recursion

**Key class:** `ProcessingOrderer`

### 5. LLM Backends (`llm/`)

Abstraction layer for different LLM providers.

**Base class:** `LLMBackend`
- `complete(prompt, system)`: Generate completion
- `complete_with_metadata(prompt, system)`: Get completion with token counts

**Implementations:**
- `ClaudeBackend`: Anthropic Claude API
- `OpenAIBackend`: OpenAI API (also works with compatible APIs)
- `OllamaBackend`: Local models via Ollama

### 6. Summary Generator (`summarizer.py`)

Core analysis engine that generates allocation summaries.

**Process:**
1. Get functions in topological order
2. For each function:
   - Gather callee summaries
   - Build LLM prompt with function source and callee info
   - Parse LLM response into structured summary
   - Store in database

**Key classes:**
- `AllocationSummarizer`: Main summarization logic
- `IncrementalSummarizer`: Handles updates when source changes

### 7. Database (`db.py`)

SQLite storage for all analysis data.

**Tables:**
- `functions`: Function metadata and source
- `allocation_summaries`: Generated summaries as JSON
- `call_edges`: Call graph with callsite locations
- `address_taken_functions`: Functions whose addresses are taken
- `address_flows`: Where function addresses flow to
- `indirect_callsites`: Indirect call expressions
- `indirect_call_targets`: Resolved indirect call targets

### 8. Standard Library (`stdlib.py`)

Pre-defined summaries for common C standard library functions.

**Covered functions:**
- Memory: `malloc`, `calloc`, `realloc`, `free`, `aligned_alloc`
- Strings: `strdup`, `strndup`, `asprintf`
- Files: `fopen`, `fdopen`, `tmpfile`, `opendir`
- Memory mapping: `mmap`, `munmap`

### 9. CLI (`cli.py`)

Command-line interface using Click.

**Commands:**
- `analyze`: Full analysis with LLM
- `extract`: Function and call graph extraction only
- `callgraph`: Export call graph
- `show`: Display summaries
- `lookup`: Look up specific function
- `stats`: Database statistics
- `export`: Export to JSON
- `init-stdlib`: Add stdlib summaries
- `clear`: Clear database

## Data Flow

### Analysis Pipeline

```
1. Source Files
   │
   ▼
2. Function Extraction (libclang)
   │
   ├──▶ Functions stored in DB
   │
   ▼
3. Call Graph Construction
   │
   ├──▶ Direct calls extracted from AST
   ├──▶ Indirect callsites identified
   ├──▶ Address-taken functions found
   │
   ▼
4. Indirect Call Resolution (LLM)
   │
   ├──▶ Candidates filtered by signature
   ├──▶ LLM determines likely targets
   │
   ▼
5. Topological Ordering
   │
   ├──▶ SCCs computed
   ├──▶ Processing order determined
   │
   ▼
6. Summary Generation (LLM)
   │
   ├──▶ Process in order (callees first)
   ├──▶ Include callee summaries in prompt
   ├──▶ Parse and store results
   │
   ▼
7. Summary Database
```

### Incremental Updates

When source files change:

1. Compute new source hash
2. Compare with stored hash
3. If changed:
   - Invalidate function's summary
   - Cascade invalidation to all callers
4. Re-analyze invalidated functions

## Memory Safety Analysis Framework

The system uses a multi-pass, Hoare-logic-inspired approach to check memory safety. Post-condition passes (1-3) summarize what each function *produces*. The pre-condition pass (4) summarizes what each function *requires*. The verification pass (5) checks that post-conditions satisfy pre-conditions at each call site.

All passes are bottom-up (callees before callers) and independent of each other except where noted.

### Pass 1: Allocation Summary (post-condition) — existing

Captures memory allocations and buffer-size pairs produced by each function.

- What gets allocated (heap/stack/static), via which allocator, size expression
- Which parameters affect allocation size
- Buffer-size pairs established: `(buffer, size)` relationships produced by the function
- Supports project-specific allocators via `--allocator-file`

**Summarizer:** `AllocationSummarizer`

### Pass 2: Free Summary (post-condition) — planned

Captures which buffers get freed by each function.

- Which parameter, field, or return value gets freed
- Via which deallocator (`free`, `png_free`, etc.)
- Conditional or unconditional
- Whether the pointer is nulled after free

Feeds temporal safety checks (use-after-free, double-free).

### Pass 3: Initialization Summary (post-condition) — planned

Captures which memory gets initialized by each function.

- Which output parameters, fields, or allocated buffers are initialized
- Partial vs. full initialization
- Conditional initialization (only on success path, etc.)

Feeds uninitialized-use checks.

### Pass 4: Access Precondition (pre-condition) — planned

Captures what contracts must hold for safe execution of each function. This is the *requirement* side — what callers must guarantee.

- **Buffer-size contracts**: parameter X must point to at least Y bytes/elements
- **Not-null contracts**: parameter X must not be NULL
- **Not-freed contracts**: parameter X must point to live (not yet freed) memory
- **Must-be-initialized contracts**: parameter X must be initialized before use (as pointer dereference, branch condition, arithmetic operand, etc.)

Note: uninitialized *read* into a variable is benign; uninitialized *use* (dereference, branch, index) is the safety issue.

### Pass 5: Verification (LLM-based) — planned

Checks post-conditions against pre-conditions at each call site.

Since summaries are in natural language (e.g., "allocates n+1 bytes" vs. "requires buffer of at least strlen(s)+1 bytes"), mechanical matching is insufficient. The LLM evaluates whether the caller's established post-conditions satisfy the callee's pre-conditions.

| Safety class | Post-condition passes | Pre-condition (pass 4) |
|---|---|---|
| Buffer overflow | 1 (allocation size) | buffer-size contracts |
| Null dereference | 1 (may_be_null) | not-null contracts |
| Use-after-free | 2 (what's freed) | not-freed contracts |
| Double free | 2 (what's freed) | not-freed contracts |
| Uninitialized use | 3 (what's initialized) | must-be-initialized contracts |

**Dependencies:** Passes 1-3 are independent and can run in parallel. Pass 4 should run after 1-3 so the LLM knows what contract categories to look for. Pass 5 requires all four preceding passes.

## Design Decisions

### Why libclang?

- Accurate parsing of C/C++ including macros and templates
- Provides full AST access for call extraction
- Handles complex preprocessor directives
- Industry-standard tool

### Why SQLite?

- Single-file database, easy to manage
- Supports complex queries for lookups
- ACID transactions for data integrity
- No external server required

### Why bottom-up analysis?

- Callee summaries provide context for caller analysis
- Enables compositional reasoning
- Reduces redundant LLM calls
- Matches how humans understand code

### Why JSON for summaries?

- Flexible schema evolution
- Easy to parse and generate
- Human-readable for debugging
- LLMs handle JSON well
