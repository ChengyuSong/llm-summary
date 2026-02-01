"""LLM-based indirect call resolution."""

from ..db import SummaryDB
from ..llm.base import LLMBackend
from ..models import IndirectCallsite, IndirectCallTarget


INDIRECT_CALL_PROMPT = """You are analyzing indirect function calls in C/C++ code to determine which functions could be called.

## Indirect Call Site

The following indirect call appears in function `{caller_name}`:

```c
{context_snippet}
```

The indirect call expression is: `{callee_expr}`
Expected function signature: `{signature}`
Location: {file_path}:{line_number}

## Candidate Functions

These functions have matching signatures and their addresses are taken somewhere in the codebase:

{candidates_section}

## Address Flow Information

{address_flows_section}

## Task

Based on the code context and how function pointers are typically used, determine which candidate functions could realistically be called at this indirect call site.

For each candidate, assess:
1. Does the naming/purpose match the call context?
2. Is there evidence the function pointer could reach this value?
3. What's your confidence (high/medium/low)?

Respond in JSON format:
```json
{{
  "targets": [
    {{
      "function": "function_name",
      "confidence": "high|medium|low",
      "reasoning": "Brief explanation"
    }}
  ]
}}
```

Only include functions that could realistically be called. If no candidates seem likely, return an empty targets array.
"""


class IndirectCallResolver:
    """
    Uses LLM to resolve indirect call targets.

    Given an indirect call site and candidate functions (address-taken with
    matching signatures), asks the LLM to determine likely targets.
    """

    def __init__(self, db: SummaryDB, llm: LLMBackend):
        self.db = db
        self.llm = llm

    def resolve_callsite(self, callsite: IndirectCallsite) -> list[IndirectCallTarget]:
        """Resolve a single indirect call site."""
        # Get caller function name
        caller = self.db.get_function(callsite.caller_function_id)
        caller_name = caller.name if caller else "unknown"

        # Get candidate functions (address-taken with matching signature)
        candidates = self.db.get_address_taken_functions(callsite.signature)

        if not candidates:
            # Try without exact signature match - let LLM filter
            candidates = self.db.get_address_taken_functions()

        if not candidates:
            return []

        # Build candidate section
        candidates_section = self._build_candidates_section(candidates)

        # Build address flows section
        address_flows_section = self._build_address_flows_section(candidates)

        # Build prompt
        prompt = INDIRECT_CALL_PROMPT.format(
            caller_name=caller_name,
            context_snippet=callsite.context_snippet or "",
            callee_expr=callsite.callee_expr,
            signature=callsite.signature,
            file_path=callsite.file_path,
            line_number=callsite.line_number,
            candidates_section=candidates_section,
            address_flows_section=address_flows_section,
        )

        # Query LLM
        try:
            response = self.llm.complete(prompt)
            targets = self._parse_response(response, callsite, candidates)

            # Store resolved targets
            for target in targets:
                self.db.add_indirect_call_target(target)

            return targets

        except Exception as e:
            print(f"Warning: Failed to resolve callsite: {e}")
            return []

    def resolve_all_callsites(self) -> dict[int, list[IndirectCallTarget]]:
        """Resolve all unresolved indirect call sites."""
        callsites = self.db.get_indirect_callsites()
        results = {}

        for callsite in callsites:
            if callsite.id is None:
                continue

            # Check if already resolved
            existing = self.db.get_indirect_call_targets(callsite.id)
            if existing:
                results[callsite.id] = existing
                continue

            targets = self.resolve_callsite(callsite)
            results[callsite.id] = targets

        return results

    def _build_candidates_section(self, candidates: list) -> str:
        """Build the candidates section for the prompt."""
        lines = []
        for i, atf in enumerate(candidates, 1):
            func = self.db.get_function(atf.function_id)
            if func:
                # Get function description from its source (first few lines or signature)
                first_lines = func.source.split("\n")[:3] if func.source else []
                snippet = "\n".join(first_lines)
                lines.append(
                    f"{i}. `{func.name}` (signature: {atf.signature})\n"
                    f"   File: {func.file_path}\n"
                    f"   ```c\n   {snippet}\n   ```"
                )
        return "\n\n".join(lines) if lines else "No candidates found."

    def _build_address_flows_section(self, candidates: list) -> str:
        """Build the address flows section for the prompt."""
        lines = []
        for atf in candidates:
            func = self.db.get_function(atf.function_id)
            if not func:
                continue

            flows = self.db.get_address_flows(atf.function_id)
            if flows:
                lines.append(f"Function `{func.name}` address flows to:")
                for flow in flows[:5]:  # Limit to first 5 flows
                    lines.append(f"  - {flow.flow_target} at {flow.file_path}:{flow.line_number}")

        return "\n".join(lines) if lines else "No address flow information available."

    def _parse_response(
        self,
        response: str,
        callsite: IndirectCallsite,
        candidates: list,
    ) -> list[IndirectCallTarget]:
        """Parse LLM response into IndirectCallTarget objects."""
        import json
        import re

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
                return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        targets = []
        target_list = data.get("targets", [])

        # Build function name to ID mapping
        name_to_id = {}
        for atf in candidates:
            func = self.db.get_function(atf.function_id)
            if func:
                name_to_id[func.name] = atf.function_id

        for item in target_list:
            func_name = item.get("function", "")
            confidence = item.get("confidence", "low")
            reasoning = item.get("reasoning", "")

            if func_name in name_to_id and callsite.id is not None:
                targets.append(
                    IndirectCallTarget(
                        callsite_id=callsite.id,
                        target_function_id=name_to_id[func_name],
                        confidence=confidence,
                        llm_reasoning=reasoning,
                    )
                )

        return targets
