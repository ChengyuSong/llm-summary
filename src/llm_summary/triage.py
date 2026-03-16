"""Bug triage agent: prove safety or feasibility of verification issues.

The agent analyzes each SafetyIssue from the verification pass by tracing
caller/callee context through the DB. It produces one of two proofs:

1. **Safety proof**: Updated pre/post-conditions showing the issue cannot
   manifest. Key output: updated contracts for the DB so the verifier
   won't flag the same false positive again.

2. **Feasibility proof**: A concrete execution path showing the violation
   is reachable. Includes assumptions and assertions for future symbolic
   validation via ucsan.

Uses a phase-gated ReAct loop (ANALYZE → HYPOTHESIZE → VERDICT) to prevent
the agent from getting stuck.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import (
    Function,
    SafetyIssue,
    VerificationSummary,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TriageResult:
    """Result of triaging a single SafetyIssue."""

    function_name: str
    issue_index: int
    issue: SafetyIssue
    hypothesis: str  # "safe" or "feasible"
    reasoning: str  # natural language proof

    # Safety proof: updated contracts that eliminate the issue
    updated_contracts: list[dict[str, Any]] = field(default_factory=list)

    # Feasibility proof: caller chain that can trigger the issue
    feasible_path: list[str] = field(default_factory=list)

    # For future ucsan validation
    assumptions: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "issue_index": self.issue_index,
            "issue": self.issue.to_dict(),
            "hypothesis": self.hypothesis,
            "reasoning": self.reasoning,
            "updated_contracts": self.updated_contracts,
            "feasible_path": self.feasible_path,
            "assumptions": self.assumptions,
            "assertions": self.assertions,
        }


# ---------------------------------------------------------------------------
# Workflow phases & gating
# ---------------------------------------------------------------------------

ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "ANALYZE": ["HYPOTHESIZE"],
    "HYPOTHESIZE": ["VERDICT"],
}

PHASE_TOOLS: dict[str, set[str]] = {
    "ANALYZE": {
        "read_function_source",
        "get_callers",
        "get_callees",
        "get_summaries",
        "get_verification_summary",
        "transition_phase",
    },
    "HYPOTHESIZE": {
        "read_function_source",
        "get_callers",
        "get_callees",
        "get_summaries",
        "get_verification_summary",
        "transition_phase",
    },
    "VERDICT": {
        "submit_verdict",
    },
}


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool-use schema)
# ---------------------------------------------------------------------------

TRIAGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_function_source",
        "description": (
            "Read the full source code of a function, similar to a Read/cat "
            "tool but looked up by function name from the project database. "
            "Returns macro-annotated source (original lines shown as "
            "'// (macro)' comments above their expanded form), file path, "
            "line range, and signature."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function to read.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_callers",
        "description": (
            "Search for all functions that call the given function, similar "
            "to Grep but searching the call graph instead of text. Returns "
            "caller names with signatures, file paths, and full source code "
            "so you can see how arguments are passed to the callee."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the callee function to find callers of.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_callees",
        "description": (
            "Get all functions called by the given function, including "
            "resolved indirect call targets (function pointers, vtable "
            "calls). Returns fully-qualified callee function names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function whose callees to list.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_summaries",
        "description": (
            "Get all analysis summaries for any function: memory safety "
            "pre-conditions, simplified contracts from verification, "
            "post-conditions (allocations, frees, initializations), and "
            "verification issues. Works for the target function, its "
            "callers, or its callees."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "get_verification_summary",
        "description": (
            "Get the full verification summary for a function, including "
            "all issues found and simplified contracts. Use this to see "
            "what the verifier reported for any function (callers, callees, "
            "or the target itself)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the function.",
                },
            },
            "required": ["function_name"],
        },
    },
    {
        "name": "transition_phase",
        "description": (
            "Transition to the next workflow phase. "
            "ANALYZE->HYPOTHESIZE, HYPOTHESIZE->VERDICT."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "next_phase": {
                    "type": "string",
                    "enum": ["HYPOTHESIZE", "VERDICT"],
                    "description": "The phase to transition to.",
                },
            },
            "required": ["next_phase"],
        },
    },
    {
        "name": "submit_verdict",
        "description": (
            "Submit the final triage verdict. Only callable in VERDICT phase. "
            "You must provide either a safety proof (with updated_contracts) "
            "or a feasibility proof (with feasible_path)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hypothesis": {
                    "type": "string",
                    "enum": ["safe", "feasible"],
                    "description": (
                        "safe: the issue cannot manifest given caller constraints. "
                        "feasible: the violation condition is reachable."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Detailed natural language proof. For safety proofs, "
                        "explain which caller constraints prevent the issue. "
                        "For feasibility proofs, describe the execution path."
                    ),
                },
                "updated_contracts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "contract_kind": {
                                "type": "string",
                                "enum": [
                                    "not_null", "nullable", "not_freed",
                                    "initialized", "buffer_size",
                                ],
                            },
                            "description": {"type": "string"},
                            "size_expr": {"type": "string"},
                        },
                        "required": ["target", "contract_kind", "description"],
                    },
                    "description": (
                        "For 'safe' hypothesis: updated/additional contracts "
                        "that prove the issue away. These will be written back "
                        "to the DB so the verifier won't flag this again."
                    ),
                },
                "feasible_path": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "For 'feasible' hypothesis: the call chain that can "
                        "trigger the issue. E.g. ['main', 'process_input', "
                        "'parse_header', 'target_func (overflow at line 42)']."
                    ),
                },
                "assumptions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Constraints on inputs for this proof. For safety: "
                        "'width <= MAX_WIDTH because caller validates'. "
                        "For feasibility: 'width is user-controlled via "
                        "parse_header()'."
                    ),
                },
                "assertions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The violation condition to check. E.g. "
                        "'width * height overflows uint32_t', "
                        "'buf[offset] is out-of-bounds when offset >= len'."
                    ),
                },
            },
            "required": ["hypothesis", "reasoning"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM_PROMPT = """\
You are a memory safety bug triage agent for C/C++ code.

## Task

You are given a potential memory safety issue found by static verification.
Your job is to produce a PROOF — either proving the issue is safe (cannot
manifest) or proving it is feasible (can be triggered).

## Two Outcomes

### 1. Safety Proof (hypothesis: "safe")

Prove the issue cannot manifest by showing that caller constraints prevent
the triggering condition. You MUST provide:
- **updated_contracts**: New or strengthened pre/post-conditions that
  eliminate the issue. These get written back to the DB so the verifier
  won't report this false positive again.
- **assumptions**: What caller constraints make this safe.
  E.g., "all callers validate `size <= BUF_MAX` before calling this function"
- **assertions**: The violation condition that cannot be reached.
  E.g., "size * elem_size cannot overflow because size <= 1024 and elem_size <= 8"

### 2. Feasibility Proof (hypothesis: "feasible")

Prove the violation condition IS reachable by identifying a concrete
execution path. You MUST provide:
- **feasible_path**: The call chain from entry point to the bug site.
- **assumptions**: What input conditions trigger the path.
  E.g., "attacker controls width and height via PNG header fields"
- **assertions**: The violation condition that gets triggered.
  E.g., "width * height overflows uint32_t, causing undersized allocation"

## Key Principle

The verifier checks each function in isolation, assuming its pre-conditions
hold. But callers may constrain arguments more tightly than the declared
contracts. Your job is to check the CALLER CONTEXT:

- An integer overflow in `alloc(width * height)` is real only if callers
  can pass values large enough to overflow
- A null deref is real only if callers can actually pass NULL
- A buffer overflow is real only if callers can pass a size exceeding bounds

## Workflow (strictly enforced)

### ANALYZE phase
1. Read the flagged function's source code
2. Read caller functions — understand how the function is invoked
3. Check callee contracts — understand what guarantees callees provide
4. Trace the data flow: where do the problematic parameters come from?

### HYPOTHESIZE phase
1. Formulate your hypothesis: safe or feasible
2. Gather remaining evidence (you can still read functions/contracts)
3. Build your proof: identify specific constraints or paths

### VERDICT phase
1. Call submit_verdict with your proof

## Rules
- Always read the function source and at least its direct callers
- For safety proofs: you MUST show which callers constrain the parameters
- For feasibility proofs: you MUST show a concrete call chain
- Include assumptions and assertions — these will be used for symbolic validation
- Do not guess — if caller constraints are ambiguous, lean toward "feasible"
"""


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


def _resolve_function(db: SummaryDB, name: str) -> Function | None:
    """Find a function by name, returning the first match or None."""
    funcs = db.get_function_by_name(name)
    return funcs[0] if funcs else None


class TriageToolExecutor:
    """Executes triage tools against the function database."""

    def __init__(self, db: SummaryDB, verbose: bool = False):
        self.db = db
        self.verbose = verbose
        self._current_phase = "ANALYZE"
        self._func_cache: dict[str, Function | None] = {}

    @property
    def phase(self) -> str:
        return self._current_phase

    def _get_func(self, name: str) -> Function | None:
        if name not in self._func_cache:
            self._func_cache[name] = _resolve_function(self.db, name)
        return self._func_cache[name]

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call, enforcing phase gating."""
        allowed = PHASE_TOOLS.get(self._current_phase, set())
        if tool_name not in allowed:
            return {
                "error": (
                    f"Tool '{tool_name}' not allowed in {self._current_phase} "
                    f"phase. Allowed: {sorted(allowed)}. "
                    f"Use transition_phase to advance."
                ),
            }
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        result: dict[str, Any] = handler(tool_input)
        return result

    # -- read_function_source --

    def _tool_read_function_source(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        return {
            "function": func.name,
            "signature": func.signature or "",
            "file_path": func.file_path,
            "line_start": func.line_start,
            "line_end": func.line_end,
            "source": func.llm_source[:20000],
        }

    # -- get_callers --

    def _tool_get_callers(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        caller_ids = self.db.get_callers(func.id)
        callers = []
        for cid in caller_ids:
            caller = self.db.get_function(cid)
            if caller is None:
                continue
            info: dict[str, Any] = {
                "name": caller.name,
                "signature": caller.signature or "",
                "file_path": caller.file_path,
                "source": caller.llm_source[:8000],
            }
            callers.append(info)

        return {
            "function": name,
            "caller_count": len(callers),
            "callers": callers,
        }

    # -- get_callees --

    def _tool_get_callees(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        callee_ids = self.db.get_callees(func.id)
        edges = self.db.get_call_edges_by_caller(func.id)
        # Build a set of indirect callee IDs for annotation
        indirect_ids = {e.callee_id for e in edges if e.is_indirect}

        callees = []
        for cid in callee_ids:
            callee = self.db.get_function(cid)
            if callee is None:
                continue
            # Fully-qualified: file_path::name(signature)
            fq = f"{callee.file_path}::{callee.name}"
            if callee.signature:
                fq += f" {callee.signature}"
            if cid in indirect_ids:
                fq += " [indirect]"
            callees.append(fq)

        return {
            "function": name,
            "callee_count": len(callees),
            "callees": callees,
        }

    # -- get_summaries --

    def _tool_get_summaries(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        result: dict[str, Any] = {
            "function": func.name,
            "signature": func.signature,
        }

        memsafe = self.db.get_memsafe_summary_by_function_id(func.id)
        if memsafe and memsafe.contracts:
            result["pre_conditions"] = [
                {
                    "target": c.target,
                    "kind": c.contract_kind,
                    "description": c.description,
                    **({"size_expr": c.size_expr} if c.size_expr else {}),
                }
                for c in memsafe.contracts
            ]
        else:
            result["pre_conditions"] = []

        vsummary = self.db.get_verification_summary_by_function_id(func.id)
        if vsummary:
            if vsummary.simplified_contracts is not None:
                result["simplified_contracts"] = [
                    {
                        "target": c.target,
                        "kind": c.contract_kind,
                        "description": c.description,
                        **({"size_expr": c.size_expr} if c.size_expr else {}),
                    }
                    for c in vsummary.simplified_contracts
                ]
            result["issues_count"] = len(vsummary.issues)

        alloc = self.db.get_summary_by_function_id(func.id)
        if alloc and alloc.allocations:
            result["allocations"] = [
                {
                    "source": a.source,
                    **({"size_expr": a.size_expr} if a.size_expr else {}),
                    "may_be_null": a.may_be_null,
                    "returned": a.returned,
                }
                for a in alloc.allocations
            ]

        free_summary = self.db.get_free_summary_by_function_id(func.id)
        if free_summary and free_summary.frees:
            result["frees"] = [
                {
                    "deallocator": f.deallocator,
                    "target": f.target,
                    "conditional": f.conditional,
                    **({"condition": f.condition} if f.condition else {}),
                    "nulled_after": f.nulled_after,
                }
                for f in free_summary.frees
            ]

        init_summary = self.db.get_init_summary_by_function_id(func.id)
        if init_summary and init_summary.inits:
            result["initializations"] = [
                {
                    "initializer": i.initializer,
                    "target": i.target,
                    **({"byte_count": i.byte_count} if i.byte_count else {}),
                }
                for i in init_summary.inits
            ]

        return result

    # -- get_verification_summary --

    def _tool_get_verification_summary(self, inp: dict[str, Any]) -> dict[str, Any]:
        name = inp["function_name"]
        func = self._get_func(name)
        if func is None:
            return {"error": f"Function '{name}' not found in database."}
        if func.id is None:
            return {"error": f"Function '{name}' has no ID."}

        vsummary = self.db.get_verification_summary_by_function_id(func.id)
        if vsummary is None:
            return {"function": name, "status": "not_verified"}

        return {
            "function": name,
            "description": vsummary.description,
            "simplified_contracts": (
                [
                    {
                        "target": c.target,
                        "kind": c.contract_kind,
                        "description": c.description,
                        **({"size_expr": c.size_expr} if c.size_expr else {}),
                    }
                    for c in vsummary.simplified_contracts
                ]
                if vsummary.simplified_contracts is not None
                else None
            ),
            "issues": [i.to_dict() for i in vsummary.issues],
        }

    # -- transition_phase --

    def _tool_transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        next_phase = inp["next_phase"]
        allowed = ALLOWED_TRANSITIONS.get(self._current_phase, [])
        if next_phase not in allowed:
            return {
                "error": (
                    f"Cannot transition from {self._current_phase} to "
                    f"{next_phase}. Allowed: {allowed}"
                ),
            }
        prev = self._current_phase
        self._current_phase = next_phase
        return {"previous_phase": prev, "current_phase": next_phase}

    # -- submit_verdict --

    def _tool_submit_verdict(self, inp: dict[str, Any]) -> dict[str, Any]:
        return {"accepted": True, **inp}


# ---------------------------------------------------------------------------
# Triage controller (ReAct loop)
# ---------------------------------------------------------------------------

MAX_TURNS = 25


class TriageAgent:
    """Triage verification issues using an LLM ReAct loop."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose

    def triage_issue(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        vsummary: VerificationSummary,
    ) -> TriageResult:
        """Triage a single issue via LLM ReAct loop."""
        executor = TriageToolExecutor(self.db, verbose=self.verbose)
        user_prompt = self._build_issue_prompt(func, issue, issue_index, vsummary)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        if self.verbose:
            print(f"\n[Triage] {func.name} issue #{issue_index}: "
                  f"{issue.issue_kind} at {issue.location}")
            print(f"  {issue.description}")

        verdict_result: dict[str, Any] | None = None

        for turn in range(MAX_TURNS):
            response = self.llm.complete_with_tools(
                messages=messages,
                tools=TRIAGE_TOOL_DEFINITIONS,
                system=TRIAGE_SYSTEM_PROMPT,
            )

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(f"[Triage] LLM stopped at turn {turn + 1} ({stop})")
                break

            if stop != "tool_use":
                if self.verbose:
                    print(f"[Triage] Unexpected stop_reason: {stop}")
                break

            assistant_content: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []

            for block in response.content:
                if hasattr(block, "text") and block.type == "text":
                    entry: dict[str, Any] = {"type": "text", "text": block.text}
                    if getattr(block, "thought", False):
                        entry["thought"] = True
                    sig = getattr(block, "thought_signature", None)
                    if sig:
                        entry["thought_signature"] = sig
                    assistant_content.append(entry)
                    if self.verbose and not getattr(block, "thought", False):
                        print(f"  [LLM] {block.text[:300]}")

                elif block.type == "tool_use":
                    tool_entry: dict[str, Any] = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                    sig = getattr(block, "thought_signature", None)
                    if sig:
                        tool_entry["thought_signature"] = sig
                    assistant_content.append(tool_entry)

                    result = executor.execute(block.name, block.input)

                    if self.verbose:
                        err = result.get("error")
                        if err:
                            print(f"  [Tool] {block.name} -> ERROR: {err[:150]}")
                        elif block.name == "submit_verdict":
                            print(f"  [Tool] submit_verdict -> "
                                  f"{result.get('hypothesis')}")
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"  [Tool] {block.name}({arg})")

                    if block.name == "submit_verdict" and result.get("accepted"):
                        verdict_result = result

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            if verdict_result is not None:
                break

        return self._build_result(func, issue, issue_index, verdict_result)

    def triage_function(
        self,
        func: Function,
        *,
        severity_filter: set[str] | None = None,
    ) -> list[TriageResult]:
        """Triage all issues for a function."""
        if func.id is None:
            return []

        vsummary = self.db.get_verification_summary_by_function_id(func.id)
        if vsummary is None or not vsummary.issues:
            return []

        results = []
        for i, issue in enumerate(vsummary.issues):
            if severity_filter and issue.severity not in severity_filter:
                continue
            result = self.triage_issue(func, issue, i, vsummary)
            results.append(result)
        return results

    def _build_issue_prompt(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        vsummary: VerificationSummary,
    ) -> str:
        lines = [
            "## Issue to Triage",
            "",
            f"Function: `{func.name}`",
            f"Signature: `{func.signature}`",
            f"File: {func.file_path}:{func.line_start}",
            "",
            f"### Issue #{issue_index}",
            f"- **Kind**: {issue.issue_kind}",
            f"- **Location**: {issue.location}",
            f"- **Severity**: {issue.severity}",
            f"- **Description**: {issue.description}",
        ]
        if issue.callee:
            lines.append(f"- **Callee involved**: {issue.callee}")
        if issue.contract_kind:
            lines.append(f"- **Contract violated**: {issue.contract_kind}")

        if vsummary.description:
            lines.extend(["", "### Verifier Assessment", vsummary.description])

        if len(vsummary.issues) > 1:
            lines.extend(["", "### Other Issues in This Function"])
            for j, other in enumerate(vsummary.issues):
                if j == issue_index:
                    continue
                lines.append(
                    f"- #{j}: [{other.severity}] {other.issue_kind} "
                    f"at {other.location} — {other.description}"
                )

        lines.extend([
            "",
            "### Instructions",
            "Follow the workflow: ANALYZE -> HYPOTHESIZE -> VERDICT.",
            "Start by reading the function source, then check its callers.",
            "Produce either a safety proof or a feasibility proof.",
        ])
        return "\n".join(lines)

    def _build_result(
        self,
        func: Function,
        issue: SafetyIssue,
        issue_index: int,
        verdict: dict[str, Any] | None,
    ) -> TriageResult:
        if verdict is not None:
            hyp = verdict.get("hypothesis", "feasible")
            if hyp not in ("safe", "feasible"):
                hyp = "feasible"
            return TriageResult(
                function_name=func.name,
                issue_index=issue_index,
                issue=issue,
                hypothesis=hyp,
                reasoning=verdict.get("reasoning", ""),
                updated_contracts=verdict.get("updated_contracts", []),
                feasible_path=verdict.get("feasible_path", []),
                assumptions=verdict.get("assumptions", []),
                assertions=verdict.get("assertions", []),
            )

        return TriageResult(
            function_name=func.name,
            issue_index=issue_index,
            issue=issue,
            hypothesis="feasible",
            reasoning="Agent did not submit a verdict within turn limit.",
        )
