"""LLM-based free/deallocation summary generator."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    FreeOp,
    FreeSummary,
    Function,
)

FREE_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate deallocation (free) summaries.

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Free Summaries

{callee_summaries}

## Task

Generate a deallocation summary for this function. Identify every buffer or resource
that this function frees (directly or via callees).

For each free operation, identify:

1. **target**: What gets freed — the expression (e.g., "ptr", "info_ptr->palette", "row_buf")
2. **target_kind**: One of:
   - "parameter" — a function parameter is freed
   - "field" — a struct field (accessed via parameter or global) is freed
   - "local" — a local variable is freed
   - "return_value" — the freed pointer is also returned (rare)
3. **deallocator**: The function that performs the free (e.g., "free", "png_free", "g_free")
4. **conditional**: true if the free is inside an if-block, error path, or conditional
5. **nulled_after**: true if the pointer is set to NULL after the free

Consider:
- Direct calls to free/deallocator functions
- Wrapper functions that free (use callee summaries)
- Conditional frees (inside if-blocks, error paths)
- Whether the pointer is NULLed after free (defensive pattern)

Respond in JSON format:
```json
{{
  "function": "{name}",
  "frees": [
    {{
      "target": "expression being freed",
      "target_kind": "parameter|field|local|return_value",
      "deallocator": "free function name",
      "conditional": true|false,
      "nulled_after": true|false
    }}
  ],
  "description": "One-sentence description of what this function frees"
}}
```

If the function does not free any memory (directly or via callees), return:
```json
{{
  "function": "{name}",
  "frees": [],
  "description": "Does not free memory"
}}
```
"""


class FreeSummarizer:
    """Generates free/deallocation summaries for functions using LLM."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
        deallocators: list[str] | None = None,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.deallocators = deallocators or []
        self._stats = {
            "functions_processed": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        }
        self._progress_current = 0
        self._progress_total = 0

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, FreeSummary] | None = None,
    ) -> FreeSummary:
        """Generate free summary for a single function."""
        if callee_summaries is None:
            callee_summaries = {}

        callee_section = self._build_callee_section(func, callee_summaries)

        prompt = FREE_SUMMARY_PROMPT.format(
            source=func.source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            callee_summaries=callee_section,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(f"  ({self._progress_current}/{self._progress_total}) Summarizing (free): {func.name}")
                else:
                    print(f"  Summarizing (free): {func.name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            summary = self._parse_response(response, func.name)
            self._stats["functions_processed"] += 1

            return summary

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error summarizing (free) {func.name}: {e}")

            return FreeSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, FreeSummary],
    ) -> str:
        """Build the callee free summaries section for the prompt."""
        if not callee_summaries and not self.deallocators:
            return "No callee free summaries available (leaf function or external calls only)."

        lines = []
        for name, summary in callee_summaries.items():
            if summary.frees:
                free_desc = ", ".join(
                    f"{f.deallocator}({f.target})"
                    for f in summary.frees
                )
                lines.append(f"- `{name}`: Frees {free_desc}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'Does not free memory'}")

        # Append project-specific deallocators not already covered
        covered_names = set(callee_summaries.keys())
        for dealloc_name in self.deallocators:
            if dealloc_name not in covered_names:
                lines.append(f"- `{dealloc_name}`: Known deallocator (project-specific)")

        if not lines:
            return "No callee free summaries available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [free pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> FreeSummary:
        """Parse LLM response into FreeSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return FreeSummary(
                    function_name=func_name,
                    description="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return FreeSummary(
                function_name=func_name,
                description=f"JSON parse error: {e}",
            )

        # Parse frees
        _VALID_KINDS = {"parameter", "field", "local", "return_value"}
        frees = []
        for f in data.get("frees", []):
            target_kind = f.get("target_kind", "local")
            if target_kind not in _VALID_KINDS:
                target_kind = "local"
            frees.append(
                FreeOp(
                    target=f.get("target", ""),
                    target_kind=target_kind,
                    deallocator=f.get("deallocator", "free"),
                    conditional=f.get("conditional", False),
                    nulled_after=f.get("nulled_after", False),
                )
            )

        return FreeSummary(
            function_name=data.get("function", func_name),
            frees=frees,
            description=data.get("description", ""),
        )
