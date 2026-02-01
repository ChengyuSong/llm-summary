# Python API Reference

This document describes the Python API for programmatic use.

## Models (`llm_summary.models`)

### Function

Represents an extracted C/C++ function.

```python
from llm_summary.models import Function

func = Function(
    name="create_buffer",
    file_path="/path/to/file.c",
    line_start=10,
    line_end=25,
    source="char* create_buffer(size_t n) { ... }",
    signature="char*(size_t)",
    id=None,  # Set after database insertion
    source_hash=None,  # Computed automatically
)
```

### Allocation

Represents a memory allocation within a function.

```python
from llm_summary.models import Allocation, AllocationType

alloc = Allocation(
    alloc_type=AllocationType.HEAP,
    source="malloc",
    size_expr="n + 1",
    size_params=["n"],
    returned=True,
    stored_to=None,
    may_be_null=True,
)
```

**AllocationType values:**
- `HEAP`: Dynamic allocation (malloc, new, etc.)
- `STACK`: Stack allocation (VLAs, alloca)
- `STATIC`: Static/global allocation
- `UNKNOWN`: Cannot determine

### AllocationSummary

Complete summary for a function.

```python
from llm_summary.models import AllocationSummary, ParameterInfo

summary = AllocationSummary(
    function_name="create_buffer",
    allocations=[alloc],
    parameters={
        "n": ParameterInfo(role="size_indicator", used_in_allocation=True)
    },
    description="Allocates n+1 bytes for null-terminated buffer.",
)

# Convert to dictionary
data = summary.to_dict()
```

### CallEdge

Represents a call graph edge with callsite information.

```python
from llm_summary.models import CallEdge

edge = CallEdge(
    caller_id=1,
    callee_id=2,
    is_indirect=False,
    file_path="/path/to/file.c",
    line=42,
    column=5,
)

# Get callsite as string
print(edge.callsite_str())  # "/path/to/file.c:42:5"
```

## Database (`llm_summary.db`)

### SummaryDB

Main database interface.

```python
from llm_summary.db import SummaryDB

# Create/open database
db = SummaryDB("summaries.db")  # File-based
db = SummaryDB(":memory:")       # In-memory

# Always close when done
db.close()

# Or use context manager pattern
try:
    db = SummaryDB("summaries.db")
    # ... operations ...
finally:
    db.close()
```

#### Function Operations

```python
# Insert function
func_id = db.insert_function(func)

# Batch insert
func_ids = db.insert_functions_batch([func1, func2, func3])

# Lookup
func = db.get_function(func_id)
funcs = db.get_function_by_name("malloc")
funcs = db.get_function_by_name("foo", signature="int(int)")
funcs = db.get_functions_by_file("/path/to/file.c")
all_funcs = db.get_all_functions()
```

#### Summary Operations

```python
# Store summary
db.upsert_summary(func, summary, model_used="claude-sonnet-4-20250514")

# Retrieve
summary = db.get_summary("create_buffer")
summary = db.get_summary("overloaded", signature="int(int)")
summary = db.get_summary_by_function_id(func_id)
summaries = db.get_summaries_by_file("/path/to/file.c")

# Check if update needed
if db.needs_update(func):
    # Re-analyze function
    pass
```

#### Call Graph Operations

```python
# Add edges
db.add_call_edge(edge)
db.add_call_edges_batch([edge1, edge2, edge3])

# Query
callee_ids = db.get_callees(caller_id)
caller_ids = db.get_callers(callee_id)
all_edges = db.get_all_call_edges()
edges = db.get_call_edges_by_caller(caller_id)

# Invalidation
invalidated_ids = db.invalidate_and_cascade(func_id)
```

#### Utility

```python
# Statistics
stats = db.get_stats()
# Returns: {"functions": 100, "allocation_summaries": 50, ...}

# Clear all data
db.clear_all()
```

## Extractor (`llm_summary.extractor`)

### FunctionExtractor

Extract functions from C/C++ files.

```python
from llm_summary.extractor import FunctionExtractor

extractor = FunctionExtractor(
    compile_args=["-I/path/to/includes"],  # Optional
    libclang_path=None,  # Auto-detect
)

# Single file
functions = extractor.extract_from_file("/path/to/file.c")

# Multiple files
functions = extractor.extract_from_files(["/path/a.c", "/path/b.c"])

# Directory
functions = extractor.extract_from_directory(
    "/path/to/project",
    extensions=(".c", ".cpp", ".h"),
    recursive=True,
)
```

## Call Graph (`llm_summary.callgraph`)

### CallGraphBuilder

Build call graph from source files.

```python
from llm_summary.callgraph import CallGraphBuilder
from llm_summary.db import SummaryDB

db = SummaryDB("project.db")
builder = CallGraphBuilder(db)

# Build from files (also extracts functions)
edges = builder.build_from_files(["/path/a.c", "/path/b.c"])

# Build from directory
edges = builder.build_from_directory("/path/to/project")

# Get call graph as adjacency list
graph = builder.get_call_graph()  # {caller_id: [callee_ids]}
reverse = builder.get_reverse_call_graph()  # {callee_id: [caller_ids]}
```

## Ordering (`llm_summary.ordering`)

### ProcessingOrderer

Compute processing order for bottom-up analysis.

```python
from llm_summary.ordering import ProcessingOrderer

# graph: {node_id: [callee_ids]}
orderer = ProcessingOrderer(graph)

# Get SCCs in processing order (callees first)
for scc in orderer.get_processing_order():
    print(f"Processing SCC: {scc}")

# Check if function is recursive
is_recursive = orderer.is_recursive(func_id)

# Get SCC members
members = orderer.get_scc_members(func_id)

# Statistics
stats = orderer.get_stats()
# {"nodes": 100, "edges": 150, "sccs": 80, "recursive_sccs": 5, "largest_scc": 3}
```

## LLM Backends (`llm_summary.llm`)

### Creating Backends

```python
from llm_summary.llm import create_backend

# Factory function
llm = create_backend("claude", model="claude-sonnet-4-20250514")
llm = create_backend("openai", model="gpt-4o")
llm = create_backend("ollama", model="llama3.1")

# Direct instantiation
from llm_summary.llm import ClaudeBackend, OpenAIBackend, OllamaBackend

llm = ClaudeBackend(
    model="claude-sonnet-4-20250514",
    api_key="...",  # Or use ANTHROPIC_API_KEY env var
    max_tokens=4096,
)

llm = OpenAIBackend(
    model="gpt-4o",
    api_key="...",  # Or use OPENAI_API_KEY env var
    base_url=None,  # For compatible APIs
)

llm = OllamaBackend(
    model="llama3.1",
    base_url="http://localhost:11434",
)
```

### Using Backends

```python
# Simple completion
response = llm.complete("Analyze this code...")

# With system message
response = llm.complete(
    "Analyze this function",
    system="You are a code analysis expert."
)

# With metadata (token counts)
result = llm.complete_with_metadata("Analyze this...")
print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
print(f"Content: {result.content}")
```

## Summarizer (`llm_summary.summarizer`)

### AllocationSummarizer

Generate allocation summaries.

```python
from llm_summary.summarizer import AllocationSummarizer
from llm_summary.llm import create_backend
from llm_summary.db import SummaryDB

db = SummaryDB("project.db")
llm = create_backend("claude")

summarizer = AllocationSummarizer(db, llm, verbose=True)

# Summarize single function
summary = summarizer.summarize_function(func, callee_summaries={})

# Summarize all functions in order
summaries = summarizer.summarize_all(force=False)

# Statistics
print(summarizer.stats)
# {"functions_processed": 50, "llm_calls": 45, "cache_hits": 5, "errors": 0}
```

### IncrementalSummarizer

Handle updates when source changes.

```python
from llm_summary.summarizer import IncrementalSummarizer

inc = IncrementalSummarizer(db, llm, verbose=True)

# Update single function
invalidated = inc.update_function(func)

# Re-sync all invalidated
summaries = inc.resync_invalidated()

# Update entire file
summaries = inc.update_file("/path/to/file.c", new_functions)
```

## Standard Library (`llm_summary.stdlib`)

### Pre-defined Summaries

```python
from llm_summary.stdlib import (
    get_stdlib_summary,
    get_all_stdlib_summaries,
    is_stdlib_allocator,
)

# Get specific summary
summary = get_stdlib_summary("malloc")

# Get all
all_summaries = get_all_stdlib_summaries()

# Check if allocator
if is_stdlib_allocator("strdup"):
    print("strdup allocates memory")
```

## Complete Example

```python
from llm_summary.db import SummaryDB
from llm_summary.extractor import FunctionExtractor
from llm_summary.callgraph import CallGraphBuilder
from llm_summary.summarizer import AllocationSummarizer
from llm_summary.llm import create_backend

# Initialize
db = SummaryDB("analysis.db")
llm = create_backend("claude")

try:
    # Extract functions
    extractor = FunctionExtractor()
    functions = extractor.extract_from_directory("/path/to/project")
    db.insert_functions_batch(functions)

    # Build call graph
    builder = CallGraphBuilder(db)
    builder.build_from_directory("/path/to/project")

    # Generate summaries
    summarizer = AllocationSummarizer(db, llm, verbose=True)
    summaries = summarizer.summarize_all()

    # Query results
    for func in db.get_all_functions():
        summary = db.get_summary_by_function_id(func.id)
        if summary and summary.allocations:
            print(f"{func.name}: {[a.source for a in summary.allocations]}")

finally:
    db.close()
```
