"""Contract-coverage agent: audit a library's docs against its contracts.

The complement to `triage`: triage proves/refutes individual issues; this
agent enumerates the public API of one library and looks for caveats that
the documentation/headers/examples document but the code-contract pipeline
missed. Output is a JSON catalog of gaps, each with an exact quote and a
suggested contract clause, usable as input to a re-run of the contract
pass with hand-seeded clauses.

Phases (linear):

1. **SEARCH** — read the manual / public header / example program. Build
   candidate caveats with exact quotes.
2. **CHECK** — fetch each candidate's existing contract; drop those
   already captured; keep real gaps. Also flag APIs with no contract row
   at all (`missing_contract`).
3. **REPORT** — emit one `submit_gaps(...)` call with the full catalog.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_tools import (
    CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS,
    GIT_TOOL_NAMES,
    READ_TOOL_DEFINITIONS,
    ToolExecutor,
)
from .db import SummaryDB
from .git_tools import GIT_TOOL_DEFINITIONS as _GIT_TOOL_DEFS
from .git_tools import GitTools
from .llm.base import LLMBackend

# ---------------------------------------------------------------------------
# Header parsing — extract public API names from a C header.
# ---------------------------------------------------------------------------

# libpng / many libs that follow PNG_EXPORT(ordinal, type, name, args)
_PNG_EXPORT_RE = re.compile(
    r"^\s*PNG_EXPORTA?\s*\(\s*\d+\s*,\s*[^,]+,\s*(\w+)\s*,",
    re.MULTILINE,
)

# Naive C function declaration: `<type> name(...)`. Best-effort only —
# falls back when no PNG_EXPORT pattern is present. Allows leading
# whitespace; rejects macro/typedef noise via the `;` requirement and
# the C-keyword filter on the captured name.
_C_FUNC_DECL_RE = re.compile(
    r"^[ \t]*[\w\s\*\(\),]*?\b([A-Za-z_]\w+)\s*\([^;{]*\)\s*;",
    re.MULTILINE,
)

# Identifiers that look like function names but are actually keywords or
# common storage-class noise we want to skip in the naive fallback.
_C_KEYWORDS: set[str] = {
    "if", "for", "while", "switch", "return", "sizeof", "static",
    "extern", "inline", "const", "volatile", "struct", "union",
    "enum", "typedef", "void", "int", "char", "short", "long",
    "float", "double", "signed", "unsigned",
}


def parse_public_apis(header_text: str) -> list[str]:
    """Return ordered, de-duplicated list of public API names in *header_text*.

    Tries the libpng PNG_EXPORT macro first (catches every export the
    library considers public). If none match, falls back to a naive scan
    for `<type> name(...);` declarations.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _PNG_EXPORT_RE.findall(header_text):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if ordered:
        return ordered

    for name in _C_FUNC_DECL_RE.findall(header_text):
        if name in _C_KEYWORDS or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ContractGap:
    """A single missing or wrong contract clause."""

    function: str
    category: str  # see PHASE_TOOLS schema enum
    evidence_source: str
    evidence_quote: str
    suggested_clause: str
    property: str = ""
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "category": self.category,
            "property": self.property,
            "evidence_source": self.evidence_source,
            "evidence_quote": self.evidence_quote,
            "suggested_clause": self.suggested_clause,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContractGap:
        return cls(
            function=str(d.get("function", "")),
            category=str(d.get("category", "")),
            evidence_source=str(d.get("evidence_source", "")),
            evidence_quote=str(d.get("evidence_quote", "")),
            suggested_clause=str(d.get("suggested_clause", "")),
            property=str(d.get("property", "")),
            explanation=str(d.get("explanation", "")),
        )


@dataclass
class ContractCheckResult:
    """The output of one library audit."""

    library: str
    target: str
    summary: str = ""
    gaps: list[ContractGap] = field(default_factory=list)
    completed: bool = False  # True iff agent submitted a gap catalog

    def to_dict(self) -> dict[str, Any]:
        return {
            "library": self.library,
            "target": self.target,
            "summary": self.summary,
            "gap_count": len(self.gaps),
            "gaps": [g.to_dict() for g in self.gaps],
            "completed": self.completed,
        }


# ---------------------------------------------------------------------------
# Workflow phases & gating
# ---------------------------------------------------------------------------

ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "SEARCH": ["CHECK"],
    "CHECK": ["REPORT"],
}

_DB_TOOLS = {
    "read_function_source",
    "get_callers",
    "get_callees",
    "get_contracts",
}

PHASE_TOOLS: dict[str, set[str]] = {
    # SEARCH: explore docs / headers / examples; enumerate APIs.
    "SEARCH": (
        GIT_TOOL_NAMES
        | {"list_public_apis", "list_apis_without_contracts"}
        | {"transition_phase"}
    ),
    # CHECK: verify candidates against the contract DB and source.
    "CHECK": (
        _DB_TOOLS
        | GIT_TOOL_NAMES
        | {"list_public_apis", "list_apis_without_contracts"}
        | {"transition_phase"}
    ),
    # REPORT: terminal — only the gap catalog submission is allowed.
    "REPORT": {"submit_gaps"},
}


# ---------------------------------------------------------------------------
# Tool definitions — assembled from shared + contract-check-specific
# ---------------------------------------------------------------------------

# Reuse only the read tools that make sense for an API-coverage audit.
_READ_TOOLS_FOR_CHECK = [
    t for t in READ_TOOL_DEFINITIONS if t["name"] in _DB_TOOLS
]


# transition_phase tool definition (specialized for SEARCH→CHECK→REPORT).
_TRANSITION_PHASE_TOOL: dict[str, Any] = {
    "name": "transition_phase",
    "description": (
        "Transition to the next workflow phase. "
        "SEARCH->CHECK, CHECK->REPORT. "
        "REPORT is terminal — call submit_gaps to finish."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "next_phase": {
                "type": "string",
                "enum": ["CHECK", "REPORT"],
                "description": "The phase to transition to.",
            },
        },
        "required": ["next_phase"],
    },
}


CONTRACT_CHECK_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    *_READ_TOOLS_FOR_CHECK,
    *CONTRACT_CHECK_ONLY_TOOL_DEFINITIONS,
    _TRANSITION_PHASE_TOOL,
    *_GIT_TOOL_DEFS,
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CONTRACT_CHECK_SYSTEM_PROMPT = """\
You audit a C library's public-API code contracts against its documented
behavior and its source code, finding caveats the contracts miss.

## Inputs

You are given:
- The library name and build target.
- The contract database via tools (one row per function).
- The library's git repository via git_show / git_grep / git_ls_tree.

You must DISCOVER the docs yourself — there are no hints. Use
git_ls_tree at the repo root and likely subdirs (`include/`, `docs/`,
`examples/`, `contrib/`) to find: the public header (usually
`<libname>.h`), a manual (`*manual*`, `*.md`, `README*`, anything
under `docs/`), and a canonical example program (`example*.c`,
`examples/`, `contrib/`). Use git_grep when names are non-obvious.

## Phases (strictly enforced)

### SEARCH phase
Build candidate caveats with EXACT quotes.

1. git_ls_tree the repo root; identify the public header, manual, and
   example program.
2. Use list_public_apis on the public header to enumerate the API surface.
3. Use list_apis_without_contracts on that list to capture the
   "missing_contract" gap class up front (those go into the catalog as-is).
4. Read sections of the manual (git_show with start_line/max_lines) and
   look for caveat language: "you must", "do not", "undefined", "before
   calling", "the caller is responsible for", "after this", "must not".
5. Read the public header for Doxygen / `/* ... */` comments next to
   declarations — they often document caveats the manual omits.
6. Read the canonical example program (if present) — patterns it sets up
   before calling an API are usually preconditions; cleanup after the
   call is usually a postcondition / lifecycle obligation.
7. Build an internal list of {function, candidate caveat, exact quote,
   source location}.

### CHECK phase
For each candidate caveat:

1. Fetch the function's existing contract (get_contracts).
2. Compare the candidate to the contract's requires/ensures/modifies.
3. Drop candidates the contract already captures (any reasonable
   paraphrase counts — we are looking for GAPS, not vocabulary mismatches).
4. For survivors, classify into one of these categories:

   - missing_contract  — the function has NO code-contract row at all
   - missing_requires  — caller obligation not in any requires[prop]
   - missing_ensures   — guarantee/effect not in any ensures[prop]
   - missing_modifies  — state mutation undocumented in modifies[prop]
   - ordering          — call-order rule between two APIs not encoded
   - lifecycle         — alloc/free/ownership rule not encoded
   - error_path        — error-return / NULL handling documented but
                         not encoded
   - inconsistency     — source/contract conflict (contract says X,
                         source clearly does Y)

5. If you suspect inconsistency, read the source (read_function_source)
   to confirm before reporting.
6. You may also call list_apis_without_contracts again with newly seen
   names if you discover APIs in the manual that you did not get from
   list_public_apis.

### REPORT phase
Call submit_gaps EXACTLY ONCE with the full catalog. The call is terminal.

## Quality bar

- Every gap MUST include a verbatim quote from a doc / header / source
  location AND a suggested contract clause. If you cannot quote it, do
  NOT report it.
- The suggested_clause should be a directly addable predicate, e.g.
  `"png_ptr->io_ptr != NULL"` or
  `"after this call, png_ptr->row_buf may be allocated; caller must call
  png_destroy_read_struct to free"`. Either C-expression or short
  English is fine — clarity over syntax.
- For ordering / lifecycle / error_path categories, name the OTHER
  function involved in the explanation.
- Do not invent caveats. If the manual is silent and the source plainly
  shows safe behavior, there is no gap.

## Pacing

You have a turn budget; use it on whole-library coverage, not deep
dives. Skim manual sections rather than reading every line. Focus on:

1. APIs without contracts (free wins, captured by step 3 of SEARCH).
2. Lifecycle / ordering rules between APIs (these matter most for
   downstream symbolic validation).
3. APIs whose manual entry contains caveat language.

Skip routine getters/setters whose contracts look correct.
"""


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------


class ContractCheckToolExecutor:
    """Phase-gated wrapper around ToolExecutor for contract-check."""

    def __init__(
        self,
        db: SummaryDB,
        verbose: bool = False,
        git_tools: GitTools | None = None,
        project_path: Path | None = None,
    ) -> None:
        self._executor = ToolExecutor(
            db,
            verbose=verbose,
            git_tools=git_tools,
            project_path=project_path,
        )
        self._current_phase = "SEARCH"

    @property
    def phase(self) -> str:
        return self._current_phase

    def execute(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        allowed = PHASE_TOOLS.get(self._current_phase, set())
        if tool_name not in allowed:
            return {
                "error": (
                    f"Tool '{tool_name}' not allowed in {self._current_phase} "
                    f"phase. Allowed: {sorted(allowed)}. "
                    f"Use transition_phase to advance."
                ),
            }
        if tool_name == "transition_phase":
            return self._transition_phase(tool_input)
        return self._executor.execute(tool_name, tool_input)

    def _transition_phase(self, inp: dict[str, Any]) -> dict[str, Any]:
        next_phase = inp["next_phase"]
        allowed_next = ALLOWED_TRANSITIONS.get(self._current_phase, [])
        if next_phase not in allowed_next:
            return {
                "error": (
                    f"Cannot transition from {self._current_phase} to "
                    f"{next_phase}. Allowed: {allowed_next}"
                ),
            }
        prev = self._current_phase
        self._current_phase = next_phase
        return {"previous_phase": prev, "current_phase": next_phase}


# ---------------------------------------------------------------------------
# Contract-check controller (ReAct loop)
# ---------------------------------------------------------------------------

DEFAULT_MAX_TURNS = 80


class ContractCheckAgent:
    """Audit one library's public-API contracts via an LLM ReAct loop."""

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend,
        verbose: bool = False,
        project_path: Path | None = None,
    ) -> None:
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.project_path = project_path

    def check_library(
        self,
        library: str,
        target: str,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> ContractCheckResult:
        """Run the audit. Returns the gap catalog."""
        if self.project_path is None:
            raise ValueError(
                "ContractCheckAgent requires project_path (the git repo).",
            )

        git = GitTools(self.project_path)
        executor = ContractCheckToolExecutor(
            self.db,
            verbose=self.verbose,
            git_tools=git,
            project_path=self.project_path,
        )
        user_prompt = self._build_prompt(library, target)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        if self.verbose:
            print(f"\n[ContractCheck] {library}/{target} "
                  f"(max_turns={max_turns})")

        gaps_result: dict[str, Any] | None = None

        for turn in range(max_turns):
            allowed = PHASE_TOOLS.get(executor.phase, set())
            tools = [
                t for t in CONTRACT_CHECK_TOOL_DEFINITIONS
                if t["name"] in allowed
            ]

            response = self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                system=CONTRACT_CHECK_SYSTEM_PROMPT,
            )

            stop = getattr(response, "stop_reason", None)
            if stop in ("end_turn", "stop"):
                if self.verbose:
                    print(
                        f"[ContractCheck] LLM stopped at turn {turn + 1} "
                        f"({stop})",
                    )
                break
            if stop != "tool_use":
                if self.verbose:
                    print(f"[ContractCheck] Unexpected stop_reason: {stop}")
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
                            print(
                                f"  [Tool] {block.name} -> ERROR: "
                                f"{err[:150]}",
                            )
                        elif block.name == "submit_gaps":
                            print(
                                f"  [Tool] submit_gaps -> "
                                f"{len(result.get('gaps', []))} gaps",
                            )
                        else:
                            arg = json.dumps(block.input)
                            if len(arg) > 80:
                                arg = arg[:80] + "..."
                            print(f"  [Tool] {block.name}({arg})")

                    if (
                        block.name == "submit_gaps"
                        and result.get("accepted")
                    ):
                        gaps_result = result

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            if gaps_result is not None:
                break

        return self._build_result(library, target, gaps_result)

    def _build_prompt(self, library: str, target: str) -> str:
        return "\n".join([
            "## Library to Audit",
            "",
            f"- **Library**: {library}",
            f"- **Target**: {target}",
            "",
            "### Instructions",
            "Follow the workflow: SEARCH -> CHECK -> REPORT.",
            "Start by using git_ls_tree at the repo root to discover the "
            "public header (typically `<libname>.h` or under `include/`), "
            "the manual (look for `*manual*`, `docs/`, or `*.txt` at the "
            "root), and a canonical example program (look for "
            "`example*.c`, `examples/`, or `contrib/`). If a layout is "
            "non-obvious, use git_grep to locate it.",
            "Then call list_public_apis on the header and "
            "list_apis_without_contracts on the result to seed the "
            "missing-contract gaps. Skim the manual / example for caveat "
            "language and verify against the contract DB.",
            "End with one submit_gaps call containing the full catalog.",
        ])

    def _build_result(
        self,
        library: str,
        target: str,
        verdict: dict[str, Any] | None,
    ) -> ContractCheckResult:
        if verdict is None:
            return ContractCheckResult(
                library=library,
                target=target,
                summary="Agent did not submit a gap catalog within turn limit.",
                completed=False,
            )

        gaps = [
            ContractGap.from_dict(g)
            for g in verdict.get("gaps", [])
            if isinstance(g, dict)
        ]
        return ContractCheckResult(
            library=str(verdict.get("library", library)),
            target=str(verdict.get("target", target)),
            summary=str(verdict.get("summary", "")),
            gaps=gaps,
            completed=True,
        )
