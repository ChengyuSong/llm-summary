"""LLM-based initialization summary generator."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Function,
    InitOp,
    InitSummary,
)

INIT_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate initialization summaries (post-conditions).

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Initialization Summaries

{callee_summaries}

## Task

Generate an initialization summary for this function. Identify what this function
**always** initializes on ALL non-error exit paths — only guaranteed, unconditional
initializations. This is a post-condition: only things visible to the CALLER matter.

Only include caller-visible initializations:
- **Output parameters**: memory written via pointer parameters (e.g., `*out = value`)
- **Struct fields**: fields written via a parameter (e.g., `ctx->data = ...`)
- **Return values**: the function's return value itself

Do NOT include local variables — they are not visible to the caller after return.

For each initialization, identify:

1. **target**: What gets initialized — the expression (e.g., "*out", "ctx->data", "return value")
2. **target_kind**: One of:
   - "parameter" — an output parameter is written via pointer dereference
   - "field" — a struct field is written via a parameter
   - "return_value" — the return value is always set
3. **initializer**: How it's initialized (e.g., "memset", "assignment", "calloc", "callee:func_name")
4. **byte_count**: How many bytes are initialized — "n", "sizeof(T)", "full", or null if unknown

Consider:
- Direct assignments to output parameters and struct fields
- Calls to memset, memcpy, calloc, etc. (use callee summaries)
- Only include initializations that happen on ALL non-error exit paths
- If a field is only initialized on some paths, do NOT include it

Respond in JSON format:
```json
{{
  "function": "{name}",
  "inits": [
    {{
      "target": "expression being initialized",
      "target_kind": "parameter|field|return_value",
      "initializer": "how it is initialized",
      "byte_count": "n|sizeof(T)|full|null"
    }}
  ],
  "description": "One-sentence description of what this function always initializes"
}}
```

If the function does not unconditionally initialize any caller-visible state, return:
```json
{{
  "function": "{name}",
  "inits": [],
  "description": "Does not unconditionally initialize caller-visible state"
}}
```
"""


class InitSummarizer:
    """Generates initialization summaries for functions using LLM."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        log_file: str | None = None,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
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
        callee_summaries: dict[str, InitSummary] | None = None,
    ) -> InitSummary:
        """Generate init summary for a single function."""
        if callee_summaries is None:
            callee_summaries = {}

        callee_section = self._build_callee_section(func, callee_summaries)

        prompt = INIT_SUMMARY_PROMPT.format(
            source=func.source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            callee_summaries=callee_section,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(f"  ({self._progress_current}/{self._progress_total}) Summarizing (init): {func.name}")
                else:
                    print(f"  Summarizing (init): {func.name}")

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
                print(f"  Error summarizing (init) {func.name}: {e}")

            return InitSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, InitSummary],
    ) -> str:
        """Build the callee init summaries section for the prompt."""
        if not callee_summaries:
            return "No callee initialization summaries available (leaf function or external calls only)."

        lines = []
        for name, summary in callee_summaries.items():
            if summary.inits:
                init_desc = ", ".join(
                    f"{i.initializer}({i.target})"
                    for i in summary.inits
                )
                lines.append(f"- `{name}`: Initializes {init_desc}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'Does not initialize caller-visible state'}")

        if not lines:
            return "No callee initialization summaries available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [init pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> InitSummary:
        """Parse LLM response into InitSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return InitSummary(
                    function_name=func_name,
                    description="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return InitSummary(
                function_name=func_name,
                description=f"JSON parse error: {e}",
            )

        # Parse inits
        _VALID_KINDS = {"parameter", "field", "return_value"}
        inits = []
        for i in data.get("inits", []):
            target_kind = i.get("target_kind", "parameter")
            if target_kind not in _VALID_KINDS:
                target_kind = "parameter"
            inits.append(
                InitOp(
                    target=i.get("target", ""),
                    target_kind=target_kind,
                    initializer=i.get("initializer", "assignment"),
                    byte_count=i.get("byte_count"),
                )
            )

        return InitSummary(
            function_name=data.get("function", func_name),
            inits=inits,
            description=data.get("description", ""),
        )
