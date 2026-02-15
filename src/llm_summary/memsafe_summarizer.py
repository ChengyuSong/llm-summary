"""LLM-based safety contract (pre-condition) summary generator."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Function,
    MemsafeContract,
    MemsafeSummary,
)

MEMSAFE_SUMMARY_PROMPT = """You are analyzing C/C++ code to generate safety pre-condition contracts.

## Function to Analyze

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## Callee Safety Contracts

{callee_summaries}

## Task

Generate safety contracts (pre-conditions) for this function. Identify what the
**caller must guarantee** for this function to execute in a memory-safe manner.

For each contract, identify:

1. **target**: The parameter or expression the contract applies to (e.g., "ptr", "buf", "ctx->data")
2. **contract_kind**: One of:
   - "not_null" — pointer parameter that is dereferenced and must not be NULL
   - "not_freed" — pointer passed to free/dealloc that must point to live memory
   - "buffer_size" — pointer used with memcpy/memset/indexing that must have sufficient capacity
   - "initialized" — variable/field used in dereference, branch, or index that must be initialized
3. **description**: Brief description of the requirement
4. **size_expr**: (buffer_size only) The size expression required, e.g., "n", "sizeof(T)", "strlen(src)+1"
5. **relationship**: (buffer_size only) One of "byte_count" or "element_count"

Rules:
- Pointer params that are **dereferenced** (read/write through `*p`, `p->field`, `p[i]`) → `not_null`
- Params passed to `free()` or deallocators → `not_freed`
- Params used in memcpy/memset/array indexing with a size → `buffer_size` (include size_expr + relationship)
- Params/fields used in dereference, branch, or index before being set → `initialized`
- If a callee has contracts that this function does NOT satisfy internally, propagate them as this function's own contracts
- Do NOT include contracts for parameters that are checked (e.g., `if (ptr == NULL) return`) before use
- Only include size_expr and relationship for buffer_size contracts

Respond in JSON format:
```json
{{
  "function": "{name}",
  "contracts": [
    {{
      "target": "parameter or expression",
      "contract_kind": "not_null|not_freed|initialized|buffer_size",
      "description": "brief description of the requirement",
      "size_expr": "n (buffer_size only, omit otherwise)",
      "relationship": "byte_count (buffer_size only, omit otherwise)"
    }}
  ],
  "description": "One-sentence summary of this function's safety requirements"
}}
```

If the function has no safety pre-conditions (e.g., all pointers are checked before use), return:
```json
{{
  "function": "{name}",
  "contracts": [],
  "description": "No safety pre-conditions required"
}}
```
"""


class MemsafeSummarizer:
    """Generates safety contract summaries for functions using LLM."""

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
        callee_summaries: dict[str, MemsafeSummary] | None = None,
    ) -> MemsafeSummary:
        """Generate safety contract summary for a single function."""
        if callee_summaries is None:
            callee_summaries = {}

        callee_section = self._build_callee_section(func, callee_summaries)

        prompt = MEMSAFE_SUMMARY_PROMPT.format(
            source=func.source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            callee_summaries=callee_section,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(f"  ({self._progress_current}/{self._progress_total}) Summarizing (memsafe): {func.name}")
                else:
                    print(f"  Summarizing (memsafe): {func.name}")

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
                print(f"  Error summarizing (memsafe) {func.name}: {e}")

            return MemsafeSummary(
                function_name=func.name,
                description=f"Error generating summary: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, MemsafeSummary],
    ) -> str:
        """Build the callee safety summaries section for the prompt."""
        if not callee_summaries:
            return "No callee safety contracts available (leaf function or external calls only)."

        lines = []
        for name, summary in callee_summaries.items():
            if summary.contracts:
                contract_descs = []
                for c in summary.contracts:
                    if c.contract_kind == "buffer_size" and c.size_expr:
                        contract_descs.append(f"{c.target}: {c.contract_kind}({c.size_expr})")
                    else:
                        contract_descs.append(f"{c.target}: {c.contract_kind}")
                lines.append(f"- `{name}`: Requires {', '.join(contract_descs)}")
            else:
                lines.append(f"- `{name}`: {summary.description or 'No safety pre-conditions'}")

        if not lines:
            return "No callee safety contracts available (leaf function or external calls only)."

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [memsafe pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> MemsafeSummary:
        """Parse LLM response into MemsafeSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return MemsafeSummary(
                    function_name=func_name,
                    description="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return MemsafeSummary(
                function_name=func_name,
                description=f"JSON parse error: {e}",
            )

        # Parse contracts
        _VALID_KINDS = {"not_null", "not_freed", "initialized", "buffer_size"}
        contracts = []
        for c in data.get("contracts", []):
            contract_kind = c.get("contract_kind", "not_null")
            if contract_kind not in _VALID_KINDS:
                contract_kind = "not_null"

            size_expr = None
            relationship = None
            if contract_kind == "buffer_size":
                size_expr = c.get("size_expr")
                relationship = c.get("relationship")

            contracts.append(
                MemsafeContract(
                    target=c.get("target", ""),
                    contract_kind=contract_kind,
                    description=c.get("description", ""),
                    size_expr=size_expr,
                    relationship=relationship,
                )
            )

        return MemsafeSummary(
            function_name=data.get("function", func_name),
            contracts=contracts,
            description=data.get("description", ""),
        )
