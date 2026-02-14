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

### 6. Graph Traversal Driver (`driver.py`)

Unified bottom-up traversal engine. Builds the call graph once, computes SCCs, and runs one or more summary passes over functions in topological order (callees first). When `--type allocation --type free` is used, both passes execute in a single traversal instead of duplicating graph construction. All four passes (`allocation`, `free`, `init`, `memsafe`) can run together.

**Key classes:**
- `BottomUpDriver`: Owns graph building (cached) and SCC traversal. `run(passes, force, dirty_ids)` executes all passes per function.
- `SummaryPass` (Protocol): Interface each pass implements — `get_cached()`, `summarize()`, `store()`
- `AllocationPass`: Adapter wrapping `AllocationSummarizer`
- `FreePass`: Adapter wrapping `FreeSummarizer`
- `InitPass`: Adapter wrapping `InitSummarizer`
- `MemsafePass`: Adapter wrapping `MemsafeSummarizer`
- `VerificationPass`: Adapter wrapping `VerificationSummarizer`

**Incremental support:** When `dirty_ids` is provided, the driver computes the affected set (dirty functions + transitive callers via reverse edges) and only re-summarizes those; all others load from cache.

### 7. Summary Generators (`summarizer.py`, `free_summarizer.py`, `init_summarizer.py`, `memsafe_summarizer.py`, `verification_summarizer.py`)

Per-function LLM summarization logic. Each summarizer builds a prompt from the function source and callee summaries, queries the LLM, and parses the structured response.

**Key classes:**
- `AllocationSummarizer`: Allocation/buffer-size-pair analysis
- `FreeSummarizer`: Free/deallocation analysis
- `InitSummarizer`: Initialization post-condition analysis
- `MemsafeSummarizer`: Safety contract (pre-condition) analysis
- `VerificationSummarizer`: Cross-pass verification and contract simplification
- `IncrementalSummarizer`: Handles source-change invalidation, delegates re-summarization to `BottomUpDriver`

### 8. Database (`db.py`)

SQLite storage for all analysis data.

**Tables:**
- `functions`: Function metadata and source
- `allocation_summaries`: Generated allocation summaries as JSON
- `free_summaries`: Generated free/deallocation summaries as JSON
- `init_summaries`: Generated initialization summaries as JSON
- `memsafe_summaries`: Generated safety contract summaries as JSON
- `verification_summaries`: Generated verification results as JSON
- `call_edges`: Call graph with callsite locations
- `address_taken_functions`: Functions whose addresses are taken
- `address_flows`: Where function addresses flow to
- `indirect_callsites`: Indirect call expressions
- `indirect_call_targets`: Resolved indirect call targets

### 9. Standard Library (`stdlib.py`)

Pre-defined allocation, free, and initialization summaries for common C standard library functions.

**Allocation summaries:**
- Memory: `malloc`, `calloc`, `realloc`, `free`, `aligned_alloc`
- Strings: `strdup`, `strndup`, `asprintf`
- Files: `fopen`, `fdopen`, `tmpfile`, `opendir`
- Memory mapping: `mmap`, `munmap`

**Free summaries:**
- `free`, `realloc`, `fclose`, `closedir`, `munmap`, `freeaddrinfo`

**Init summaries:**
- `calloc`, `memset`, `memcpy`, `memmove`, `strncpy`, `snprintf`, `strdup`, `strndup`

**Memsafe summaries:**
- `memcpy`, `memmove`, `memset`, `free`, `strlen`, `strcpy`, `strncpy`, `strcmp`, `snprintf`, `printf`, `fprintf`, `fwrite`, `fread`, `malloc`

### 10. CLI (`cli.py`)

Command-line interface using Click.

**Commands:**
- `summarize`: Generate allocation, free, init, memsafe, and/or verify summaries (`--type allocation`, `--type free`, `--type init`, `--type memsafe`, `--type verify`)
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
5. BottomUpDriver (driver.py)
   │
   ├──▶ Build call graph + compute SCCs (once)
   ├──▶ Traverse in topological order (callees first)
   ├──▶ Run all registered passes per function:
   │      AllocationPass, FreePass, InitPass, MemsafePass, etc.
   │
   ▼
6. Summary Generation (LLM, per pass)
   │
   ├──▶ Gather callee summaries from prior results
   ├──▶ Build prompt, query LLM, parse response
   ├──▶ Store result in DB
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

### Pass 2: Free Summary (post-condition) — implemented

Captures which buffers get freed by each function.

- **Target**: what gets freed (`ptr`, `info_ptr->palette`, `row_buf`)
- **Target kind**: `parameter`, `field`, `local`, or `return_value`
- **Deallocator**: `free`, `png_free`, `g_free`, or project-specific
- **Conditional**: whether the free is inside an if/error path
- **Nulled after**: whether the pointer is set to NULL after free
- Supports project-specific deallocators via `--deallocator-file`

Feeds temporal safety checks (use-after-free, double-free).

**Summarizer:** `FreeSummarizer` (`free_summarizer.py`)
**DB table:** `free_summaries`
**CLI:** `llm-summary summarize --type free`

### Pass 3: Initialization Summary (post-condition) — implemented

Captures what each function **always** initializes on all non-error exit paths (caller-visible only).

- **Target**: what gets initialized (`*out`, `ctx->data`, `return value`)
- **Target kind**: `parameter` (output param), `field` (struct field via param), or `return_value`
- **Initializer**: how it's initialized (`memset`, `assignment`, `calloc`, `callee:func_name`)
- **Byte count**: how many bytes (`n`, `sizeof(T)`, `full`, or null)

Only unconditional, guaranteed initializations visible to the caller. Local variables are excluded (not a post-condition). Feeds uninitialized-use checks (Pass 5).

**Summarizer:** `InitSummarizer` (`init_summarizer.py`)
**DB table:** `init_summaries`
**CLI:** `llm-summary summarize --type init`

### Pass 4: Safety Contracts (pre-condition) — implemented

Captures what contracts must hold for safe execution of each function. This is the *requirement* side — what callers must guarantee.

- **Not-null contracts** (`not_null`): pointer parameters that are dereferenced must not be NULL
- **Not-freed contracts** (`not_freed`): pointers passed to free/dealloc must point to live memory
- **Buffer-size contracts** (`buffer_size`): pointers used in memcpy/indexing must have sufficient capacity (includes `size_expr` and `relationship`)
- **Initialized contracts** (`initialized`): variables/fields used in deref, branch, or index must be initialized

Callee contracts that a function does NOT satisfy internally are propagated as the function's own contracts.

Note: uninitialized *read* into a variable is benign; uninitialized *use* (dereference, branch, index) is the safety issue.

**Summarizer:** `MemsafeSummarizer` (`memsafe_summarizer.py`)
**DB table:** `memsafe_summaries`
**CLI:** `llm-summary summarize --type memsafe`

### Pass 5: Verification & Contract Simplification — implemented

Cross-pass verification that checks post-conditions against pre-conditions at each call site. For each function, the verifier:

1. **Internal safety check** — does the function itself perform unsafe operations?
2. **Callee pre-condition satisfaction** — at each call site, are the callee's memsafe contracts satisfied?
3. **Contract simplification** — removes Pass 4 contracts that the function satisfies internally, keeping only contracts that must propagate to callers.
4. **Issue reporting** — unsatisfied pre-conditions become `SafetyIssue` findings with severity levels.

The verifier queries the DB directly for Passes 1-3 callee post-conditions and Pass 4 raw contracts (cross-pass data), while receiving `VerificationSummary` callee summaries from the driver for already-verified callees (simplified contracts).

| Safety class | Post-condition passes | Pre-condition (pass 4) |
|---|---|---|
| Buffer overflow | 1 (allocation size) | buffer-size contracts |
| Null dereference | 1 (may_be_null) | not-null contracts |
| Use-after-free | 2 (what's freed) | not-freed contracts |
| Double free | 2 (what's freed) | not-freed contracts |
| Uninitialized use | 3 (what's initialized) | must-be-initialized contracts |

Issue severity: **high** (definite violation), **medium** (depends on caller), **low** (unlikely/defensive).

**Summarizer:** `VerificationSummarizer` (`verification_summarizer.py`)
**DB table:** `verification_summaries`
**CLI:** `llm-summary summarize --type verify`

**Dependencies:** Passes 1-4 are independent and run together in a single `BottomUpDriver` traversal when multiple `--type` flags are given. Pass 5 requires all four prior passes to exist (prerequisite check in CLI).

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
