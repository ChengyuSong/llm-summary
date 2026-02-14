"""LLM-based verification and contract simplification (Pass 5)."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Function,
    MemsafeContract,
    SafetyIssue,
    VerificationSummary,
)

VERIFICATION_PROMPT = """You are verifying memory safety of a C/C++ function using Hoare-logic-style reasoning.

## Function Under Verification

```c
{source}
```

Function: `{name}`
Signature: `{signature}`
File: {file_path}

## This Function's Raw Safety Contracts (from Pass 4)

{own_contracts}

## Callee Information

{callee_section}

## Verification Tasks

### Check 1: Internal Safety
Does the function itself perform any unsafe operations (null dereference, buffer overflow,
use-after-free, double-free, use of uninitialized memory)?

### Check 2: Callee Pre-condition Satisfaction
For EACH call to a callee that has pre-conditions, determine whether this function
establishes those pre-conditions before the call.

A pre-condition is SATISFIED if:
- The function checks the condition before the call (e.g., null check guard)
- The value comes from an allocation that guarantees the property (e.g., calloc -> initialized)
- The value flows from a parameter that this function requires via its OWN contracts (propagation)

A pre-condition is UNSATISFIED if the function neither checks it nor propagates it.

### Check 3: Contract Simplification
For each of the function's own raw contracts: is it satisfied internally? If yes, remove it.
Only keep contracts that MUST be propagated to callers.

## Severity Guidelines

- **high**: Definite violation (unconditional unsafe op, no guard)
- **medium**: Potential violation depending on caller behavior
- **low**: Unlikely (error path only, defensive concern)

## Output

Respond with JSON:
```json
{{
  "function": "{name}",
  "simplified_contracts": [
    {{
      "target": "parameter or expression",
      "contract_kind": "not_null|not_freed|initialized|buffer_size",
      "description": "brief description",
      "size_expr": "n (buffer_size only, omit otherwise)",
      "relationship": "byte_count (buffer_size only, omit otherwise)"
    }}
  ],
  "issues": [
    {{
      "location": "line 42 or call to foo at line 42",
      "issue_kind": "null_deref|buffer_overflow|use_after_free|double_free|uninitialized_use",
      "description": "what the problem is",
      "severity": "high|medium|low",
      "callee": "callee_name (if callee contract violation, omit if internal)",
      "contract_kind": "which contract kind was violated (omit if internal)"
    }}
  ],
  "description": "One-sentence verification summary"
}}
```

If no issues and no contracts remain:
```json
{{
  "function": "{name}",
  "simplified_contracts": [],
  "issues": [],
  "description": "Function is internally safe with no propagated contracts"
}}
```
"""

_VALID_ISSUE_KINDS = {
    "null_deref",
    "buffer_overflow",
    "use_after_free",
    "double_free",
    "uninitialized_use",
}
_VALID_SEVERITIES = {"high", "medium", "low"}
_VALID_CONTRACT_KINDS = {"not_null", "not_freed", "initialized", "buffer_size"}


class VerificationSummarizer:
    """Verifies memory safety and simplifies contracts using cross-pass data."""

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
            "issues_found": 0,
            "contracts_simplified": 0,
        }
        self._progress_current = 0
        self._progress_total = 0

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def summarize_function(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary] | None = None,
    ) -> VerificationSummary:
        """Verify a function and simplify its contracts."""
        if callee_summaries is None:
            callee_summaries = {}

        callee_section = self._build_callee_section(func, callee_summaries)
        own_contracts = self._build_own_contracts_section(func)

        prompt = VERIFICATION_PROMPT.format(
            source=func.source,
            name=func.name,
            signature=func.signature,
            file_path=func.file_path,
            own_contracts=own_contracts,
            callee_section=callee_section,
        )

        try:
            if self.verbose:
                if self._progress_total > 0:
                    print(
                        f"  ({self._progress_current}/{self._progress_total}) "
                        f"Verifying: {func.name}"
                    )
                else:
                    print(f"  Verifying: {func.name}")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            summary = self._parse_response(response, func.name)
            self._stats["functions_processed"] += 1
            self._stats["issues_found"] += len(summary.issues)

            # Count simplified contracts: raw - remaining
            raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
            if raw_memsafe:
                raw_count = len(raw_memsafe.contracts)
                remaining_count = len(summary.simplified_contracts)
                self._stats["contracts_simplified"] += max(
                    0, raw_count - remaining_count
                )

            return summary

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error verifying {func.name}: {e}")

            return VerificationSummary(
                function_name=func.name,
                description=f"Error during verification: {e}",
            )

    def _build_callee_section(
        self,
        func: Function,
        callee_summaries: dict[str, VerificationSummary],
    ) -> str:
        """Build cross-pass callee context for the prompt."""
        if func.id is None:
            return "No callee information available."

        callee_ids = self.db.get_callees(func.id)
        if not callee_ids:
            return "No callees (leaf function)."

        lines = []
        for callee_id in callee_ids:
            callee_func = self.db.get_function(callee_id)
            if callee_func is None:
                continue

            callee_name = callee_func.name
            section_lines = [f"### `{callee_name}`"]

            # Pre-conditions: prefer simplified (verified) over raw Pass 4
            if callee_name in callee_summaries:
                verified = callee_summaries[callee_name]
                if verified.simplified_contracts:
                    section_lines.append("**Pre-conditions (simplified):**")
                    for c in verified.simplified_contracts:
                        if c.contract_kind == "buffer_size" and c.size_expr:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind}({c.size_expr}) "
                                f"-- {c.description}"
                            )
                        else:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind} -- {c.description}"
                            )
                else:
                    section_lines.append(
                        "**Pre-conditions:** None (all satisfied internally)"
                    )
            else:
                # Fall back to raw Pass 4
                raw_memsafe = self.db.get_memsafe_summary_by_function_id(callee_id)
                if raw_memsafe and raw_memsafe.contracts:
                    section_lines.append("**Pre-conditions (raw):**")
                    for c in raw_memsafe.contracts:
                        if c.contract_kind == "buffer_size" and c.size_expr:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind}({c.size_expr}) "
                                f"-- {c.description}"
                            )
                        else:
                            section_lines.append(
                                f"  - {c.target}: {c.contract_kind} -- {c.description}"
                            )
                else:
                    section_lines.append("**Pre-conditions:** None")

            # Post-conditions from Passes 1-3
            post_parts = []

            # Pass 1: Allocations
            alloc_summary = self.db.get_summary_by_function_id(callee_id)
            if alloc_summary and alloc_summary.allocations:
                alloc_descs = []
                for a in alloc_summary.allocations:
                    desc = f"{a.source}"
                    if a.size_expr:
                        desc += f"({a.size_expr})"
                    extras = []
                    if a.may_be_null:
                        extras.append("may_be_null")
                    if a.returned:
                        extras.append("returned")
                    if extras:
                        desc += f" [{', '.join(extras)}]"
                    alloc_descs.append(desc)
                post_parts.append(f"  Allocations: {'; '.join(alloc_descs)}")
                if alloc_summary.buffer_size_pairs:
                    for bsp in alloc_summary.buffer_size_pairs:
                        post_parts.append(
                            f"  Buffer-size pair: ({bsp.buffer}, {bsp.size}) "
                            f"{bsp.relationship}"
                        )

            # Pass 2: Frees
            free_summary = self.db.get_free_summary_by_function_id(callee_id)
            if free_summary and free_summary.frees:
                free_descs = []
                for f in free_summary.frees:
                    desc = f"{f.deallocator}({f.target})"
                    extras = []
                    if f.conditional:
                        extras.append("conditional")
                    if f.nulled_after:
                        extras.append("nulled_after")
                    if extras:
                        desc += f" [{', '.join(extras)}]"
                    free_descs.append(desc)
                post_parts.append(f"  Frees: {'; '.join(free_descs)}")

            # Pass 3: Initializations
            init_summary = self.db.get_init_summary_by_function_id(callee_id)
            if init_summary and init_summary.inits:
                init_descs = []
                for i in init_summary.inits:
                    desc = f"{i.initializer}({i.target})"
                    if i.byte_count:
                        desc += f" [{i.byte_count} bytes]"
                    init_descs.append(desc)
                post_parts.append(f"  Initializations: {'; '.join(init_descs)}")

            if post_parts:
                section_lines.append("**Post-conditions:**")
                section_lines.extend(post_parts)
            else:
                section_lines.append("**Post-conditions:** None available")

            lines.append("\n".join(section_lines))

        if not lines:
            return "No callee information available."

        return "\n\n".join(lines)

    def _build_own_contracts_section(self, func: Function) -> str:
        """Format this function's raw Pass 4 contracts."""
        if func.id is None:
            return "No raw contracts available."

        raw_memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if not raw_memsafe or not raw_memsafe.contracts:
            return "No raw safety contracts (Pass 4 found no pre-conditions)."

        lines = []
        for c in raw_memsafe.contracts:
            if c.contract_kind == "buffer_size" and c.size_expr:
                lines.append(
                    f"- {c.target}: {c.contract_kind}({c.size_expr}) -- {c.description}"
                )
            else:
                lines.append(f"- {c.target}: {c.contract_kind} -- {c.description}")

        return "\n".join(lines)

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"Function: {func_name} [verification pass]\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")

    def _parse_response(self, response: str, func_name: str) -> VerificationSummary:
        """Parse LLM response into VerificationSummary."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return VerificationSummary(
                    function_name=func_name,
                    description="Failed to parse LLM response",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return VerificationSummary(
                function_name=func_name,
                description=f"JSON parse error: {e}",
            )

        # Parse simplified contracts
        contracts = []
        for c in data.get("simplified_contracts", []):
            contract_kind = c.get("contract_kind", "not_null")
            if contract_kind not in _VALID_CONTRACT_KINDS:
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

        # Parse issues
        issues = []
        for i in data.get("issues", []):
            issue_kind = i.get("issue_kind", "null_deref")
            if issue_kind not in _VALID_ISSUE_KINDS:
                issue_kind = "null_deref"

            severity = i.get("severity", "medium")
            if severity not in _VALID_SEVERITIES:
                severity = "medium"

            issues.append(
                SafetyIssue(
                    location=i.get("location", ""),
                    issue_kind=issue_kind,
                    description=i.get("description", ""),
                    severity=severity,
                    callee=i.get("callee"),
                    contract_kind=i.get("contract_kind"),
                )
            )

        return VerificationSummary(
            function_name=data.get("function", func_name),
            simplified_contracts=contracts,
            issues=issues,
            description=data.get("description", ""),
        )
