"""LLM-based allocation summary generator."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    Function,
    ParameterInfo,
)
from .ordering import ProcessingOrderer


ALLOCATION_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate memory allocation summaries.

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Summaries

{callee_summaries}

## Task

Generate a memory allocation summary for this function. Identify:

1. **Allocations**: Any memory allocations (malloc, calloc, realloc, new, etc.)
   - Type: heap, stack, or static
   - Source: The allocating function/operator
   - Size expression: How size is computed
   - Size parameters: Which function parameters affect size
   - Returned: Is the allocation returned?
   - Stored to: Is it stored to a field/global?
   - May be null: Can allocation fail?

2. **Parameters**: Role of each parameter
   - Role: size_indicator, buffer, count, pointer_out, etc.
   - Used in allocation: Does it affect allocation size?

3. **Description**: One-sentence summary of what this function allocates

Consider:
- Wrapper functions that call allocators
- Size calculations (n + 1, n * sizeof(T), etc.)
- Conditional allocations
- Allocations stored to struct fields or output parameters

Respond in JSON format:
```json
{{
  "function": "{name}",
  "allocations": [
    {{
      "type": "heap|stack|static",
      "source": "allocator function name",
      "size_expr": "size expression or null",
      "size_params": ["parameter names affecting size"],
      "returned": true|false,
      "stored_to": "field/variable name or null",
      "may_be_null": true|false
    }}
  ],
  "parameters": {{
    "param_name": {{
      "role": "role description",
      "used_in_allocation": true|false
    }}
  }},
  "description": "One-sentence description"
}}
```

If the function does not allocate memory (directly or via callees), return:
```json
{{
  "function": "{name}",
  "allocations": [],
  "parameters": {{}},
  "description": "Does not allocate memory"
}}
```
"""


class AllocationSummarizer:
    """Generates allocation summaries for functions using LLM."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, AllocationSummary] | None = None,
    ) -> AllocationSummary:
        """
        Generate allocation summary for a single function.

        Args:
            func: The function to summarize
            callee_summaries: Pre-computed summaries for callees

        Returns:
            AllocationSummary for the function
        """
        if callee_summaries is None:
            callee_summaries = {}

        # Build callee summaries section
        callee_section = self._build_callee_section(func, callee_summaries)

        # Build prompt
        prompt = ALLOCATION_SUMMARY_PROMPT.format(
            source=func.source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            callee_summaries=callee_section,
        )

        # Query LLM
        try:
            if self.verbose:
                print(f"  Summarizing: {func.name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            summary = self._parse_response(response, func.name)
            self._stats["functions_processed"] += 1

            return summary

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error summarizing {func.name}: {e}")

            # Return empty summary on error
            return AllocationSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def summarize_all(
        self,
        force: bool = False,
    ) -> dict[int, AllocationSummary]:
        """
        Summarize all functions in the database in dependency order.

        Args:
            force: If True, re-summarize even if summary exists

        Returns:
            Mapping of function ID to summary
        """
        # Build call graph
        edges = self.db.get_all_call_edges()
        graph: dict[int, list[int]] = {}

        for edge in edges:
            if edge.caller_id not in graph:
                graph[edge.caller_id] = []
            graph[edge.caller_id].append(edge.callee_id)

        # Add all functions to graph (some may have no callees)
        for func in self.db.get_all_functions():
            if func.id is not None and func.id not in graph:
                graph[func.id] = []

        # Get processing order
        orderer = ProcessingOrderer(graph)

        if self.verbose:
            stats = orderer.get_stats()
            print(f"Processing {stats['nodes']} functions in {stats['sccs']} SCCs")
            if stats["recursive_sccs"] > 0:
                print(f"  ({stats['recursive_sccs']} recursive SCCs)")

        # Process in order
        summaries: dict[int, AllocationSummary] = {}

        for scc in orderer.get_processing_order():
            # For recursive SCCs, we may need multiple passes
            # For now, just process each function once
            for func_id in scc:
                func = self.db.get_function(func_id)
                if func is None:
                    continue

                # Check if summary exists and is current
                if not force:
                    existing = self.db.get_summary_by_function_id(func_id)
                    if existing and not self.db.needs_update(func):
                        summaries[func_id] = existing
                        self._stats["cache_hits"] += 1
                        continue

                # Build callee summaries
                callee_ids = graph.get(func_id, [])
                callee_summaries = {}

                for callee_id in callee_ids:
                    if callee_id in summaries:
                        callee_func = self.db.get_function(callee_id)
                        if callee_func:
                            callee_summaries[callee_func.name] = summaries[callee_id]

                # Generate summary
                summary = self.summarize_function(func, callee_summaries)
                summaries[func_id] = summary

                # Store in database
                self.db.upsert_summary(func, summary, model_used=self.llm.model)

        return summaries

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, AllocationSummary],
    ) -> str:
        """Build the callee summaries section for the prompt."""
        if not callee_summaries:
            return "No callee summaries available (leaf function or external calls only)."

        lines = []
        for name, summary in callee_summaries.items():
            if summary.allocations:
                alloc_desc = ", ".join(
                    f"{a.source}({a.size_expr or 'unknown size'})"
                    for a in summary.allocations
                )
                lines.append(f"- `{name}`: Allocates via {alloc_desc}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'No allocations'}")

        return "\n".join(lines)

    def _parse_response(self, response: str, func_name: str) -> AllocationSummary:
        """Parse LLM response into AllocationSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return AllocationSummary(
                    function_name=func_name,
                    description="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return AllocationSummary(
                function_name=func_name,
                description=f"JSON parse error: {e}",
            )

        # Parse allocations
        allocations = []
        for a in data.get("allocations", []):
            alloc_type_str = a.get("type", "unknown").lower()
            try:
                alloc_type = AllocationType(alloc_type_str)
            except ValueError:
                alloc_type = AllocationType.UNKNOWN

            allocations.append(
                Allocation(
                    alloc_type=alloc_type,
                    source=a.get("source", ""),
                    size_expr=a.get("size_expr"),
                    size_params=a.get("size_params", []),
                    returned=a.get("returned", False),
                    stored_to=a.get("stored_to"),
                    may_be_null=a.get("may_be_null", True),
                )
            )

        # Parse parameters
        parameters = {}
        for name, info in data.get("parameters", {}).items():
            parameters[name] = ParameterInfo(
                role=info.get("role", ""),
                used_in_allocation=info.get("used_in_allocation", False),
            )

        return AllocationSummary(
            function_name=data.get("function", func_name),
            allocations=allocations,
            parameters=parameters,
            description=data.get("description", ""),
        )


class IncrementalSummarizer:
    """Handles incremental updates when source files change."""

    def __init__(self, db: SummaryDB, llm: LLMBackend, verbose: bool = False):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.summarizer = AllocationSummarizer(db, llm, verbose)

    def update_function(self, func: Function) -> list[int]:
        """
        Update summary for a function and invalidate dependents.

        Returns list of invalidated function IDs.
        """
        if func.id is None:
            func.id = self.db.insert_function(func)

        # Invalidate this function and all its callers
        invalidated = self.db.invalidate_and_cascade(func.id)

        if self.verbose:
            print(f"Invalidated {len(invalidated)} functions")

        return invalidated

    def resync_invalidated(self) -> dict[int, AllocationSummary]:
        """
        Re-summarize all functions that need updating.

        Returns mapping of function ID to new summary.
        """
        return self.summarizer.summarize_all(force=False)

    def update_file(self, file_path: str, new_functions: list[Function]) -> dict[int, AllocationSummary]:
        """
        Update summaries for all functions in a file.

        Args:
            file_path: Path to the updated file
            new_functions: Newly extracted functions from the file

        Returns:
            Updated summaries
        """
        # Get existing functions in this file
        existing = self.db.get_functions_by_file(file_path)
        existing_by_sig = {(f.name, f.signature): f for f in existing}

        # Find changed and new functions
        for func in new_functions:
            key = (func.name, func.signature)
            if key in existing_by_sig:
                old_func = existing_by_sig[key]
                if self.db.needs_update(func):
                    func.id = old_func.id
                    self.update_function(func)
            else:
                # New function
                self.db.insert_function(func)

        # Re-sync all invalidated
        return self.resync_invalidated()
