"""LLM-guided seed generation for concolic validation.

Generates unittest-style C test files that initialize objects with concrete
plausible values, then call ``__ucsan_symbolize_input()`` to register them
for symbolic execution.  A ReAct agent loop explores the project source via
git tools, generates seed code, test-compiles it, and submits working seeds.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .git_tools import GIT_TOOL_DEFINITIONS, GitTools
from .llm.base import LLMBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEED_TURNS = 30
MAX_COMPILE_ATTEMPTS = 5

_GIT_TOOL_NAMES = {"git_show", "git_ls_tree", "git_grep"}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SEED_SYSTEM_PROMPT = """\
You are a concolic-execution seed generator. Your job is to craft C test \
files that provide **smart initial inputs** for the ucsan symbolic executor.

Each seed file defines a ``void test()`` function that:
1. Allocates and initializes local objects with **concrete plausible values**.
2. Calls ``__ucsan_symbolize_input()`` on each object.
3. Runs the **precondition + target-call block** copied from the shim.

## CRITICAL: concrete values MUST satisfy preconditions (requires)

The precondition block (assume_* calls from the shim) will **abort** if \
any precondition is violated. Your concrete values are the starting state \
for the solver — if they already violate a precondition, the seed is \
useless because execution terminates immediately.

Read the requires section carefully and ensure every concrete value \
satisfies every applicable precondition. For example, if the requires say \
``s->pending + 4 + stored_len <= s->pending_buf_size``, your concrete \
values for pending, stored_len, and pending_buf_size MUST satisfy that \
inequality.

## Git tools

You have read-only access to the project source via git tools. Use them to \
discover:
- Struct field meanings (valid ranges, enum values, magic constants)
- How the target function validates its inputs (what checks must pass)
- Related test code or usage examples in the project

## Workflow

1. Read the verdict and shim code to understand what inputs are needed.
2. Read the preconditions (requires) and plan concrete values that satisfy them.
3. Use git tools to look up struct definitions, constants, and usage patterns.
4. Generate a seed C file with init + symbolize + precondition/call block.
5. Call ``compile_seed`` to verify it compiles.
6. If it compiles, call ``submit_seed`` to accept it.
7. Generate up to 5 seeds total. Each should target a different scenario \
or path through the code.

## Loop threshold

The concolic executor has a **loop_threshold** that limits how many \
iterations a loop can execute before termination. If reaching a target \
basic block requires passing through a loop N times, the threshold must \
be >= N or the executor will abort before reaching it.

**Keep loop iteration counts minimal.** Use the smallest values that \
still reach the target basic blocks. For example, if the target BB is \
after a loop over ``num_palette`` entries and the interesting path \
requires ``num_palette > 256``, use ``257`` instead of ``300``.

When you submit a seed, note the expected max loop iterations in the \
description so the harness runner can set the threshold accordingly.

## Rules

- Each seed must be a **complete, self-contained C file** that compiles \
with ko-clang.
- Copy the headers and extern declarations from the shim code. \
Do NOT copy ``__shim_*`` stub functions — they are linked separately.
- The test() body has three parts in order:
  1. **Init + symbolize**: your concrete values + __ucsan_symbolize_input calls
  2. **Precondition block**: the assume_* calls from the shim (copy verbatim)
  3. **Target call + asserts**: the function call and assert_* from the shim
- Do NOT remove or modify the assume_*/assert_* calls — they enforce contracts.
- Variable names in your init block MUST match the shim's test() parameter \
names so the precondition block can reference them.
- Always verify with ``compile_seed`` before ``submit_seed``.
- If compilation fails, fix the errors and retry.
"""

SEED_CONVENTIONS = """\
## __ucsan_symbolize_input API

```c
extern void __ucsan_symbolize_input(void *ptr, unsigned long size, int id);
```

- ``ptr``: pointer to an initialized local object
- ``size``: ``sizeof(object)`` in bytes
- ``id``: 1-based unique object identifier (0 is reserved)

### Semantics

On first run (no solver input), the runtime snapshots the concrete bytes \
at ``ptr`` as the initial seed value. The solver then explores mutations \
from this starting point.

On re-runs, the solver overwrites ``ptr`` with generated bytes before \
the function continues — the concrete init code still executes but gets \
overwritten.

### Rules

- Entry function is always ``void test()`` — **no parameters**.
- Initialize objects as local variables with designated initializers or \
memset + field assignment.
- Call ``__ucsan_symbolize_input()`` **after** all initialization, \
**before** the precondition block.
- Register **leaves first, parents after** — if obj1.next points to obj2, \
register obj2 before obj1.
- ``id`` values: sequential integers starting from 1.
- For buffers/arrays: allocate with malloc, fill with plausible data, \
then symbolize.
- Include the same headers as the shim (copy from shim code).
- Forward-declare ``__ucsan_symbolize_input``.
- Variable names MUST match the shim's test() parameter names exactly \
(e.g. if the shim has ``deflate_state * s``, your local must be named ``s``).

### Seed structure

```
void test() {
    // Part 1: YOUR CODE — allocate + init with concrete plausible values
    // Part 2: YOUR CODE — __ucsan_symbolize_input() calls
    // Part 3: COPIED FROM SHIM — assume_* preconditions + target call + assert_*
}
```

### Example

Given a shim with:
```c
void test(struct node *head) {
    assume_cond(head != NULL, 0);
    head = assume_allocated(head, sizeof(*head), 0);
    int r = target(head);
    assert_cond(r >= 0, 1);
}
```

The seed becomes:
```c
void test() {
    struct node obj2 = { .v = 42, .next = NULL };
    struct node head_obj = { .v = 10, .next = &obj2 };
    struct node *head = &head_obj;

    __ucsan_symbolize_input(&obj2, sizeof(obj2), 2);
    __ucsan_symbolize_input(&head_obj, sizeof(head_obj), 1);

    // preconditions + call + postconditions (from shim)
    assume_cond(head != NULL, 0);
    head = assume_allocated(head, sizeof(*head), 0);
    int r = target(head);
    assert_cond(r >= 0, 1);
}
```
"""

SEED_REFINE_CONVENTIONS = """\
## Coverage-Guided Refinement

The previous seeds were run through the concolic executor. Below are the \
coverage results showing which basic blocks and paths were explored.

Your task: generate **new** seeds that target the **missed** paths. \
Focus on:
- BBs that were never reached — what concrete values would steer \
execution there?
- Traces that were attempted but failed — what structural invariants \
did the solver miss?
- Edge cases: boundary values, empty inputs, maximum sizes.

Do NOT regenerate seeds similar to the previous ones — they already \
covered those paths.
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

SEED_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "compile_seed",
        "description": (
            "Compile a seed C file with ko-clang. "
            "Returns {success: bool, errors: string}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete C source code for the seed file.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "submit_seed",
        "description": (
            "Accept a seed C file. Call this after compile_seed succeeds. "
            "Can be called multiple times to submit multiple seed variants."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete C source code for the seed file.",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Brief description of what this seed tests "
                        "(e.g. 'num_palette exceeds PNG_MAX_PALETTE_LENGTH')."
                    ),
                },
            },
            "required": ["code"],
        },
    },
] + GIT_TOOL_DEFINITIONS


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class SeedExecutor:
    """Executes seed-generation tool calls."""

    def __init__(
        self,
        compile_fn: Callable[[str, str, str | None], tuple[bool, str]],
        ucsan_config: str,
        file_path: str | None,
        git_tools: GitTools | None,
        include_flags: list[str] | None = None,
    ) -> None:
        self.compile_fn = compile_fn
        self.ucsan_config = ucsan_config
        self.file_path = file_path
        self.git = git_tools
        self.include_flags = include_flags or []
        self.seeds: list[tuple[str, str]] = []  # (code, description)
        self.compile_attempts = 0
        self._last_compiled: str | None = None

    def execute(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name in _GIT_TOOL_NAMES:
            if self.git is None:
                return {"error": f"Tool '{tool_name}' unavailable: no project path"}
            return self.git.dispatch(tool_name, tool_input)
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        result: dict[str, Any] = handler(tool_input)
        return result

    def _tool_compile_seed(self, inp: dict[str, Any]) -> dict[str, Any]:
        code = inp.get("code", "")
        if not code:
            return {"success": False, "errors": "No code provided"}
        self.compile_attempts += 1
        if self.compile_attempts > MAX_COMPILE_ATTEMPTS:
            return {
                "success": False,
                "errors": (
                    f"Compile attempt limit ({MAX_COMPILE_ATTEMPTS}) reached. "
                    "Call submit_seed to accept current code or move on."
                ),
            }
        ok, errors = self.compile_fn(code, self.ucsan_config, self.file_path)
        if ok:
            self._last_compiled = code
            return {"success": True, "errors": ""}
        return {"success": False, "errors": errors}

    def _tool_submit_seed(self, inp: dict[str, Any]) -> dict[str, Any]:
        code = inp.get("code", "")
        if not code and not self._last_compiled:
            return {"error": "No code provided"}
        if self._last_compiled:
            code = self._last_compiled
        desc = inp.get("description", f"seed_{len(self.seeds)}")
        self.seeds.append((code, desc))
        # Reset compile state for next seed
        self.compile_attempts = 0
        self._last_compiled = None
        return {
            "accepted": True,
            "seed_index": len(self.seeds) - 1,
            "total_seeds": len(self.seeds),
        }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _run_seed_agent(
    user_prompt: str,
    llm: LLMBackend,
    executor: SeedExecutor,
    verbose: bool = False,
) -> list[tuple[str, str]]:
    """Run the ReAct agent loop. Returns list of (code, description) tuples."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    # Filter git tools if unavailable
    tools = list(SEED_TOOL_DEFINITIONS)
    if executor.git is None:
        tools = [t for t in tools if t["name"] not in _GIT_TOOL_NAMES]

    for turn in range(MAX_SEED_TURNS):
        response = llm.complete_with_tools(
            messages=messages,
            tools=tools,
            system=SEED_SYSTEM_PROMPT,
        )

        stop = getattr(response, "stop_reason", None)
        if stop in ("end_turn", "stop"):
            if verbose:
                print(f"    [seed-agent] Stopped at turn {turn + 1}")
            break

        if stop != "tool_use":
            if verbose:
                print(f"    [seed-agent] Unexpected stop: {stop}")
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
                if verbose and not getattr(block, "thought", False):
                    print(f"    [seed-agent] {block.text[:200]}")

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

                if verbose:
                    err = result.get("error")
                    if err:
                        print(f"    [seed-agent] {block.name} -> ERROR: {err[:150]}")
                    elif block.name == "compile_seed":
                        ok = result.get("success")
                        print(f"    [seed-agent] compile_seed -> "
                              f"{'OK' if ok else 'FAIL'}")
                    elif block.name == "submit_seed":
                        idx = result.get("seed_index", "?")
                        print(f"    [seed-agent] submit_seed #{idx}")
                    else:
                        arg = json.dumps(block.input)
                        if len(arg) > 80:
                            arg = arg[:80] + "..."
                        print(f"    [seed-agent] {block.name}({arg})")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    return executor.seeds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_shim_test_body(shim_code: str) -> str | None:
    """Extract the body of test() from the shim code.

    Returns the lines between the opening { and closing } of the test()
    function, which contain assume_* preconditions, the target call,
    and assert_* postconditions.
    """
    lines = shim_code.split("\n")
    # Find "void test(" definition
    start = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("void test(")
                and not stripped.startswith("extern ")
                and not stripped.startswith("/*")):
            start = i
            break
    if start < 0:
        return None

    # Find opening brace
    brace_line = -1
    if "{" in lines[start]:
        brace_line = start
    elif start + 1 < len(lines) and "{" in lines[start + 1]:
        brace_line = start + 1
    if brace_line < 0:
        return None

    # Find matching closing brace
    depth = 0
    end = -1
    for j in range(brace_line, len(lines)):
        depth += lines[j].count("{") - lines[j].count("}")
        if depth == 0:
            end = j
            break
    if end < 0:
        return None

    # Extract body lines (between { and })
    body_lines = lines[brace_line + 1:end]
    return "\n".join(body_lines)


def _format_verdict_context(triage_context: dict[str, Any]) -> str:
    """Format triage verdict for the seed prompt."""
    lines = []
    lines.append(f"- Hypothesis: **{triage_context.get('hypothesis', 'unknown')}**")
    lines.append(f"- Issue: [{triage_context.get('severity', '')}] "
                 f"{triage_context.get('issue_kind', '')} — "
                 f"{triage_context.get('issue_description', '')}")
    lines.append(f"- Reasoning: {triage_context.get('reasoning', 'N/A')}")

    assumptions = triage_context.get("assumptions", [])
    if assumptions:
        lines.append("\nAssumptions:")
        for i, a in enumerate(assumptions, 1):
            lines.append(f"  {i}. {a}")

    assertions = triage_context.get("assertions", [])
    if assertions:
        lines.append("\nAssertions:")
        for i, a in enumerate(assertions, 1):
            lines.append(f"  {i}. {a}")

    return "\n".join(lines)


def generate_seed_tests(
    func_name: str,
    shim_code: str,
    triage_context: dict[str, Any],
    output_dir: Path,
    llm: LLMBackend,
    compile_fn: Callable[[str, str, str | None], tuple[bool, str]],
    ucsan_config: str,
    file_path: str | None = None,
    git_tools: GitTools | None = None,
    include_flags: list[str] | None = None,
    verbose: bool = False,
) -> list[Path]:
    """Generate seed tests via ReAct agent.

    Args:
        func_name: Target function name.
        shim_code: Generated shim C code (used as reference for headers/stubs).
        triage_context: Verdict context dict (hypothesis, reasoning, etc.).
        output_dir: Directory to write seed files.
        llm: LLM backend with tool-use support.
        compile_fn: Callable(code, ucsan_config, file_path) -> (ok, errors).
        ucsan_config: YAML config string (entry/scope/shims).
        file_path: Source file path (for compile_commands include resolution).
        git_tools: Optional GitTools for project exploration.
        include_flags: Extra include flags for compilation.
        verbose: Print progress.

    Returns:
        List of paths to generated seed .c files.
    """
    executor = SeedExecutor(
        compile_fn=compile_fn,
        ucsan_config=ucsan_config,
        file_path=file_path,
        git_tools=git_tools,
        include_flags=include_flags,
    )

    verdict_text = _format_verdict_context(triage_context)

    # Extract precondition + call + postcondition block from the shim
    shim_test_body = _extract_shim_test_body(shim_code)
    precond_section = ""
    if shim_test_body:
        precond_section = (
            "## Precondition + Call Block (copy into every seed's test())\n\n"
            "After your init + symbolize code, include this block verbatim. "
            "It enforces preconditions (assume_*), calls the target, and "
            "checks postconditions (assert_*). Do NOT modify it.\n\n"
            f"```c\n{shim_test_body}\n```\n\n"
        )

    user_prompt = (
        "Generate seed test cases for concolic validation of a triage verdict.\n\n"
        "## Triage Verdict\n\n"
        f"{verdict_text}\n\n"
        "## Shim Code (reference — copy headers, NOT stubs)\n\n"
        f"```c\n{shim_code}\n```\n\n"
        f"{precond_section}"
        "## Your Task\n\n"
        f"Generate 1-5 seed test files for `{func_name}`. Each seed should:\n"
        "1. Copy the headers and extern declarations from the shim code. "
        "Do NOT copy `__shim_*` stub functions (linked separately).\n"
        "2. Write a `void test()` with:\n"
        "   a. Concrete init — allocate and initialize objects with "
        "plausible values that **satisfy all preconditions**.\n"
        "   b. Symbolize — call `__ucsan_symbolize_input()` on each object.\n"
        "   c. Precondition + call block — copy the assume_*/target call/"
        "assert_* block from the shim's test() body verbatim.\n\n"
        "**Your variable names MUST match the shim's test() parameter names** "
        "so the precondition block references work.\n\n"
        "Each seed should target a **different scenario** described in the "
        "verdict (e.g., different input sizes, boundary values, different "
        "code paths).\n\n"
        f"{SEED_CONVENTIONS}"
    )

    if verbose:
        print(f"  Generating seeds for: {func_name}")

    seeds = _run_seed_agent(user_prompt, llm, executor, verbose=verbose)

    # Write seed files
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (code, desc) in enumerate(seeds):
        seed_path = output_dir / f"seed_{func_name}_{i}.c"
        seed_path.write_text(code)
        if verbose:
            print(f"    Wrote: {seed_path} ({desc})")
        paths.append(seed_path)

    return paths


def refine_seed_tests(
    func_name: str,
    shim_code: str,
    previous_seeds: list[str],
    coverage_results: dict[str, Any],
    output_dir: Path,
    llm: LLMBackend,
    compile_fn: Callable[[str, str, str | None], tuple[bool, str]],
    ucsan_config: str,
    file_path: str | None = None,
    git_tools: GitTools | None = None,
    include_flags: list[str] | None = None,
    verbose: bool = False,
) -> list[Path]:
    """Refine seeds based on coverage gaps.

    Args:
        func_name: Target function name.
        shim_code: Generated shim C code.
        previous_seeds: List of previous seed C code strings.
        coverage_results: Coverage data (covered/missed BBs, traces).
        output_dir: Directory to write new seed files.
        llm: LLM backend with tool-use support.
        compile_fn: Callable(code, ucsan_config, file_path) -> (ok, errors).
        ucsan_config: YAML config string.
        file_path: Source file path.
        git_tools: Optional GitTools.
        include_flags: Extra include flags.
        verbose: Print progress.

    Returns:
        List of paths to new seed .c files.
    """
    executor = SeedExecutor(
        compile_fn=compile_fn,
        ucsan_config=ucsan_config,
        file_path=file_path,
        git_tools=git_tools,
        include_flags=include_flags,
    )

    # Format previous seeds
    prev_text = ""
    for i, code in enumerate(previous_seeds):
        prev_text += f"### Seed {i}\n\n```c\n{code}\n```\n\n"

    # Format coverage
    covered = coverage_results.get("covered_bbs", [])
    missed = coverage_results.get("missed_bbs", [])
    traces = coverage_results.get("traces", [])

    cov_text = f"- Covered BBs: {len(covered)}\n"
    cov_text += f"- Missed BBs: {len(missed)}\n"
    if missed:
        cov_text += f"- Missed BB IDs: {missed[:50]}\n"
    if traces:
        cov_text += f"- Explored traces: {len(traces)}\n"
        for t in traces[:10]:
            goal = t.get("goal", "")
            status = t.get("status", "")
            cov_text += f"  - {goal}: {status}\n"

    # Extract precondition + call block from the shim
    shim_test_body = _extract_shim_test_body(shim_code)
    precond_section = ""
    if shim_test_body:
        precond_section = (
            "## Precondition + Call Block (copy into every seed's test())\n\n"
            "After your init + symbolize code, include this block verbatim.\n\n"
            f"```c\n{shim_test_body}\n```\n\n"
        )

    user_prompt = (
        "Refine seed test cases based on coverage results.\n\n"
        f"{SEED_REFINE_CONVENTIONS}\n\n"
        "## Coverage Results\n\n"
        f"{cov_text}\n\n"
        "## Previous Seeds\n\n"
        f"{prev_text}\n"
        "## Shim Code (reference)\n\n"
        f"```c\n{shim_code}\n```\n\n"
        f"{precond_section}"
        "## Your Task\n\n"
        f"Generate 1-3 new seed test files for `{func_name}` targeting the "
        "missed paths. Follow the same format as previous seeds — include "
        "the precondition + call block from the shim — but with different "
        "concrete values.\n\n"
        f"{SEED_CONVENTIONS}"
    )

    if verbose:
        print(f"  Refining seeds for: {func_name}")

    seeds = _run_seed_agent(user_prompt, llm, executor, verbose=verbose)

    # Write seed files (continue numbering from existing seeds)
    existing = list(output_dir.glob(f"seed_{func_name}_*.c"))
    start_idx = len(existing)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, (code, desc) in enumerate(seeds):
        seed_path = output_dir / f"seed_{func_name}_{start_idx + i}.c"
        seed_path.write_text(code)
        if verbose:
            print(f"    Wrote: {seed_path} ({desc})")
        paths.append(seed_path)

    return paths
