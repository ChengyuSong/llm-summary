# Extending the Tool

This document describes how to extend, customize, and consume results from the tool.

## Consuming Analysis Results

The analysis pipeline produces a SQLite database with per-function summaries across five passes. This section explains how to interpret and query the results.

### Pass Structure

Each function has up to five summary records, built bottom-up (callees before callers):

| Pass | Table | What it captures | Key fields |
|---|---|---|---|
| 1 — Allocation | `allocation_summaries` | What gets allocated, sizes, may-be-null | `allocations[].source`, `size_expr`, `may_be_null`, `buffer_size_pairs` |
| 2 — Free | `free_summaries` | What gets freed, conditionally, nulled-after | `frees[].target`, `deallocator`, `conditional`, `nulled_after` |
| 3 — Init | `init_summaries` | What gets initialized (caller-visible) | `inits[].target`, `initializer`, `byte_count` |
| 4 — Memsafe | `memsafe_summaries` | Pre-conditions required for safe execution | `contracts[].target`, `contract_kind`, `size_expr` |
| 5 — Verify | `verification_summaries` | Verified issues + simplified contracts | `issues[]`, `simplified_contracts[]` |

Passes 1-3 are **post-conditions** (what functions produce). Pass 4 is **pre-conditions** (what functions require). Pass 5 checks post-conditions against pre-conditions and simplifies.

For full table schemas and JSON formats, see [database.md](database.md). For the conceptual framework (Hoare-logic approach, safety classes), see [architecture.md](architecture.md).

### Pass 5 Supersedes Pass 4

After verification, use `verification_summaries.simplified_contracts` instead of raw `memsafe_summaries.contracts`. Simplified contracts are the subset that the function does NOT satisfy internally — only these propagate to callers. If a function has no verification summary yet, fall back to raw Pass 4 contracts.

### Querying Issues

```sql
-- All high-severity issues with function context
SELECT f.name, f.file_path, v.summary_json
FROM verification_summaries v
JOIN functions f ON v.function_id = f.id
WHERE json_extract(v.summary_json, '$.issues') != '[]';
```

```python
from llm_summary.db import SummaryDB

db = SummaryDB("functions.db")
for func in db.get_all_functions():
    vs = db.get_verification_summary_by_function_id(func.id)
    if vs is None:
        continue
    for issue in vs.issues:
        if issue.severity == "high":
            print(f"{func.name}: {issue.issue_kind} at {issue.location}")
            if issue.callee:
                # Cross-reference: look up callee's contracts
                callee_funcs = db.get_function_by_name(issue.callee)
                for cf in callee_funcs:
                    ms = db.get_memsafe_summary_by_function_id(cf.id)
                    # ms.contracts has the violated pre-condition
```

### Cross-Pass Correlation

An issue in Pass 5 references data from earlier passes. To get full context:

- **`buffer_overflow`** with `callee` set: look up the callee's `buffer_size` contract in `memsafe_summaries`, and the caller's allocation size in `allocation_summaries`
- **`use_after_free`** / **`double_free`**: look up what the callee frees in `free_summaries`
- **`uninitialized_use`**: check what the callee initializes in `init_summaries`
- **`null_deref`**: check `allocation_summaries` for `may_be_null` on the relevant allocation

### Issue Severity

- **high**: definite violation (unconditional unsafe op, no guard) — likely real bugs
- **medium**: potential violation depending on caller behavior — need caller context to confirm
- **low**: unlikely (error path only, defensive concern) — informational

## Adding a New LLM Backend

### 1. Create Backend Class

Create a new file in `src/llm_summary/llm/`:

```python
# src/llm_summary/llm/my_backend.py

from .base import LLMBackend, LLMResponse


class MyBackend(LLMBackend):
    """Custom LLM backend."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(model)
        self.api_key = api_key
        # Initialize your client

    @property
    def default_model(self) -> str:
        return "my-default-model"

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Generate completion."""
        # Call your API
        response = self._call_api(prompt, system)
        return response

    def complete_with_metadata(
        self, prompt: str, system: str | None = None
    ) -> LLMResponse:
        """Generate completion with metadata."""
        # Call your API and get token counts
        content, input_tokens, output_tokens = self._call_api_with_counts(
            prompt, system
        )
        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
```

### 2. Register in Factory

Update `src/llm_summary/llm/__init__.py`:

```python
from .my_backend import MyBackend

__all__ = [..., "MyBackend"]

def create_backend(backend_type: str, model: str | None = None, **kwargs):
    if backend_type == "my_backend":
        return MyBackend(model=model, **kwargs)
    # ... existing backends
```

### 3. Add CLI Option

Update `src/llm_summary/cli.py`:

```python
@click.option(
    "--backend",
    type=click.Choice(["claude", "openai", "ollama", "my_backend"]),
    default="claude"
)
```

## Adding Standard Library Functions

### 1. Add to STDLIB_SUMMARIES

Edit `src/llm_summary/stdlib.py`:

```python
STDLIB_SUMMARIES["my_function"] = AllocationSummary(
    function_name="my_function",
    allocations=[
        Allocation(
            alloc_type=AllocationType.HEAP,
            source="my_function",
            size_expr="size parameter",
            size_params=["size"],
            returned=True,
            may_be_null=True,
        )
    ],
    parameters={
        "size": ParameterInfo(role="size_indicator", used_in_allocation=True),
    },
    description="Allocates memory of specified size.",
)
```

### 2. Generate from Documentation

Use `StdlibSummaryGenerator` to create summaries from man pages:

```python
from llm_summary.stdlib import StdlibSummaryGenerator
from llm_summary.llm import create_backend

llm = create_backend("claude")
generator = StdlibSummaryGenerator(llm)

# Get man page content
manpage = subprocess.run(
    ["man", "-P", "cat", "3", "getline"],
    capture_output=True, text=True
).stdout

# Generate summaries
summaries = generator.generate_from_manpage(manpage)
for summary in summaries:
    print(summary.to_dict())
```

## Customizing Prompts

### 1. Modify Prompt Templates

Edit files in `prompts/` directory:

- `allocation.txt`: Main summary generation prompt
- `indirect_call.txt`: Indirect call resolution prompt
- `stdlib.txt`: Stdlib documentation parsing prompt

### 2. Use Custom Prompts in Code

```python
from llm_summary.summarizer import AllocationSummarizer

class CustomSummarizer(AllocationSummarizer):
    def summarize_function(self, func, callee_summaries=None):
        # Build custom prompt
        prompt = self._build_custom_prompt(func, callee_summaries)

        # Query LLM
        response = self.llm.complete(prompt)

        # Parse response
        return self._parse_response(response, func.name)

    def _build_custom_prompt(self, func, callee_summaries):
        return f"""
        Custom prompt for {func.name}...
        """
```

## Adding New Analysis Types

### Example: Memory Leak Detection

```python
# src/llm_summary/leak_detector.py

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import Function

LEAK_PROMPT = """
Analyze this function for potential memory leaks.

Function:
```c
{source}
```

Callee summaries:
{callee_summaries}

Identify:
1. Allocations that may not be freed
2. Error paths that skip cleanup
3. Missing null checks after allocation

Respond in JSON format:
{{
  "leaks": [
    {{
      "allocation_line": 10,
      "reason": "Not freed on error path",
      "severity": "high"
    }}
  ]
}}
"""

class LeakDetector:
    def __init__(self, db: SummaryDB, llm: LLMBackend):
        self.db = db
        self.llm = llm

    def detect_leaks(self, func: Function) -> list[dict]:
        # Get callee summaries
        callee_info = self._get_callee_info(func)

        # Build prompt
        prompt = LEAK_PROMPT.format(
            source=func.source,
            callee_summaries=callee_info,
        )

        # Query LLM
        response = self.llm.complete(prompt)

        # Parse and return
        return self._parse_response(response)
```

## Adding Database Tables

### 1. Update Schema

Edit `src/llm_summary/db.py`:

```python
SCHEMA = """
... existing tables ...

-- New table for custom data
CREATE TABLE IF NOT EXISTS my_custom_table (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    custom_field TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_custom_function ON my_custom_table(function_id);
"""
```

### 2. Add Methods

```python
class SummaryDB:
    # ... existing methods ...

    def add_custom_data(self, func_id: int, data: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO my_custom_table (function_id, custom_field) VALUES (?, ?)",
            (func_id, data),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_custom_data(self, func_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT custom_field FROM my_custom_table WHERE function_id = ?",
            (func_id,),
        ).fetchall()
        return [row["custom_field"] for row in rows]
```

## Adding CLI Commands

```python
# In src/llm_summary/cli.py

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--db", "db_path", default="summaries.db")
@click.option("--output", "-o", type=click.Path())
def my_command(path, db_path, output):
    """Description of my command."""
    db = SummaryDB(db_path)

    try:
        # Implement command logic
        result = do_analysis(db, path)

        if output:
            with open(output, "w") as f:
                f.write(result)
        else:
            console.print(result)

    finally:
        db.close()
```

## Testing Extensions

### Add Unit Tests

```python
# tests/test_my_extension.py

import pytest
from llm_summary.my_module import MyClass

class TestMyExtension:
    def test_basic_functionality(self):
        obj = MyClass()
        result = obj.my_method()
        assert result == expected

    def test_edge_case(self):
        # Test edge cases
        pass
```

### Run Tests

```bash
pytest tests/test_my_extension.py -v
```

## Best Practices

1. **Follow existing patterns**: Look at similar code for guidance
2. **Add type hints**: Use Python type annotations
3. **Write tests**: Cover new functionality with tests
4. **Update documentation**: Add to relevant docs
5. **Handle errors gracefully**: Don't crash on edge cases
6. **Use the database**: Store results for caching/querying
