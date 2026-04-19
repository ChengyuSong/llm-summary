#!/usr/bin/env python3
"""Hoare-style compositional summary pipeline (per docs/design-llm-first.md).

Per function (bottom-up over call graph):
  1. Compute property set: subset of {memsafe, memleak} the function touches,
     based on simple source features + callee summaries.
  2. For each in-scope property, one LLM call producing requires/ensures lines.
  3. Inline callee summaries appear as a header block at the top of the source.
  4. NO verdict in stored summaries.
  5. Callee `requires` are propagated VERBATIM, never invented or strengthened.

Entry-point check (Hoare-only):
  Scan entry's requires lines; non-trivial requires = potential bug.

This is v0: one LLM call per (function, property). KV-cached multi-turn Q&A
is the next iteration once the per-property prompt is stable.

Usage:
    source ~/project/llm-summary/venv/bin/activate
    python scripts/contract_pipeline.py \\
        --yml /path/to/task.yml \\
        --backend claude --model claude-haiku-4-5@20251001 \\
        --output results.json --summary-out summaries.json -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import juliet_eval as je

from llm_summary.builder.json_utils import extract_json
from llm_summary.cli import _load_compile_commands
from llm_summary.db import SummaryDB
from llm_summary.extractor import FunctionExtractor
from llm_summary.llm import LLMBackend, build_backend_kwargs, create_backend
from llm_summary.llm.base import make_json_response_format
from llm_summary.models import Function, _annotate_macro_diff

log = logging.getLogger("contract_pipeline")

# ── Property scope (design doc principle 6) ─────────────────────────────────
#
# Property families this pipeline summarizes — must mirror what the OLD
# system supports (juliet_baseline._MEMSAFE_KINDS / _LEAK_KINDS / _OVERFLOW_KINDS):
#   memsafe  — null_deref, buffer_overflow, use_after_free, double_free,
#              invalid_free, uninitialized_use
#   memleak  — memory_leak
#   overflow — integer_overflow, division_by_zero, shift_ub
#
# Per-FUNCTION adaptive scoping (in `_property_set`) skips a property only
# when the function's body and its callees show no behavior that could
# touch it. We do NOT gate on the benchmark's declared subproperty — the
# benchmark just provides a ground-truth `expected_verdict`; what we
# report is the union of what each in-scope property says about `main`.

PROPERTIES = ["memsafe", "memleak", "overflow"]


# ── Per-function summary (NO verdict) ───────────────────────────────────────

PROPERTY_SCHEMA = {
    "type": "object",
    "properties": {
        "requires": {"type": "array", "items": {"type": "string"}},
        "ensures":  {"type": "array", "items": {"type": "string"}},
        "modifies": {"type": "array", "items": {"type": "string"}},
        "notes":    {"type": "string"},
        # Property-independent: set true iff this function has NO returning
        # path (e.g. body unconditionally aborts/exits/longjmps). The same
        # answer is expected from every per-property call; we OR them.
        "noreturn": {"type": "boolean"},
    },
    "required": ["requires", "ensures"],
}


@dataclass
class FunctionSummary:
    """Hoare-style per-function summary. NO verdict field by design."""
    function: str
    properties: list[str] = field(default_factory=list)
    requires: dict[str, list[str]] = field(default_factory=dict)
    ensures: dict[str, list[str]] = field(default_factory=dict)
    modifies: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    # Property-independent: callsites to a noreturn callee cut the path.
    # Sources: explicit `__attribute__((noreturn))` / `_Noreturn` on the
    # declaration, stdlib seed (abort/exit/...), or LLM-detected
    # body-always-aborts. Used by callers for path narrowing.
    noreturn: bool = False

    def has_requires(self, prop: str) -> bool:
        return any(_is_nontrivial(r) for r in self.requires.get(prop, []))

    def has_ensures(self, prop: str) -> bool:
        return any(_is_nontrivial(e) for e in self.ensures.get(prop, []))


def _is_nontrivial(predicate: str) -> bool:
    s = predicate.strip().lower()
    return s not in ("", "true", "1", "tt", "(no observable effect)",
                     "no resource acquired", "nothing acquired")


# ── Stdlib / harness contracts (A/B-test scaffold; will move to stdlib.py) ──
#
# Pre-built FunctionSummary entries for opaque callees (libc, sv-comp helpers,
# compiler builtins). Seeded into `summaries` before the topo walk so the
# existing `_inline_callee_contracts` flow surfaces them at every callsite —
# no code path needs to special-case "external function".

def _summary(
    name: str,
    *,
    noreturn: bool = False,
    **per_prop: dict[str, list[str] | str],
) -> "FunctionSummary":
    """Compact constructor: _summary("malloc", memsafe={"ensures": [...]})."""
    s = FunctionSummary(function=name, properties=list(per_prop.keys()))
    for prop, slots in per_prop.items():
        s.requires[prop] = list(slots.get("requires", []))   # type: ignore[arg-type]
        s.ensures[prop]  = list(slots.get("ensures",  []))   # type: ignore[arg-type]
        s.modifies[prop] = list(slots.get("modifies", []))   # type: ignore[arg-type]
        n = slots.get("notes", "")
        if isinstance(n, str) and n:
            s.notes[prop] = n
    s.noreturn = noreturn
    return s


def _build_stdlib_contracts() -> dict[str, "FunctionSummary"]:
    out: dict[str, FunctionSummary] = {}

    # ── Allocators (SV-COMP: never NULL) ──
    out["malloc"] = _summary(
        "malloc",
        memsafe={"ensures": ["result != NULL",
                             "result allocated for size bytes (uninitialized)"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result"]},
    )
    out["calloc"] = _summary(
        "calloc",
        memsafe={"ensures": ["result != NULL",
                             "result allocated for nmemb*size bytes",
                             "result[0..nmemb*size-1] initialized to zero"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result"]},
    )
    out["realloc"] = _summary(
        "realloc",
        memsafe={"requires": ["ptr is NULL or previously allocated and not yet freed"],
                 "ensures":  ["result != NULL",
                              "result allocated for size bytes",
                              "ptr ownership transferred to result (do not use ptr after)"]},
        memleak={"ensures": ["acquires heap allocation tied to result; releases ptr"]},
    )
    out["reallocarray"] = _summary(
        "reallocarray",
        memsafe={"requires": ["ptr is NULL or previously allocated and not yet freed"],
                 "ensures":  ["result != NULL",
                              "result allocated for nmemb*size bytes",
                              "ptr ownership transferred to result"]},
        memleak={"ensures": ["acquires heap allocation tied to result; releases ptr"]},
    )

    # ── Stack alloc (no leak obligation) ──
    for name in ("alloca", "__builtin_alloca"):
        out[name] = _summary(
            name,
            memsafe={"ensures": ["result != NULL",
                                 "result allocated for size bytes on stack (uninitialized)",
                                 "result valid only until enclosing function returns"]},
        )

    # ── free ──
    out["free"] = _summary(
        "free",
        memsafe={"requires": ["ptr is NULL or from malloc/calloc/realloc and not yet freed"]},
        memleak={"ensures": ["releases ptr"]},
        # ptr's referent becomes invalid; modeled as modifying the heap region.
    )

    # ── SV-COMP nondet (full type range; empty ensures = unconstrained) ──
    nondet_signed = (
        "__VERIFIER_nondet_int", "__VERIFIER_nondet_short",
        "__VERIFIER_nondet_long", "__VERIFIER_nondet_longlong",
        "__VERIFIER_nondet_char", "__VERIFIER_nondet_schar",
        "__VERIFIER_nondet_pchar",
    )
    nondet_unsigned = (
        "__VERIFIER_nondet_uint", "__VERIFIER_nondet_ushort",
        "__VERIFIER_nondet_ulong", "__VERIFIER_nondet_ulonglong",
        "__VERIFIER_nondet_uchar", "__VERIFIER_nondet_size_t",
        "__VERIFIER_nondet_u32", "__VERIFIER_nondet_u16", "__VERIFIER_nondet_u8",
    )
    nondet_other = (
        "__VERIFIER_nondet_bool", "__VERIFIER_nondet_float",
        "__VERIFIER_nondet_double", "__VERIFIER_nondet_pointer",
        "__VERIFIER_nondet_loff_t",
    )
    for n in nondet_signed + nondet_unsigned + nondet_other:
        out[n] = _summary(
            n,
            overflow={"notes": "returns the full type range; treat result as unconstrained"},
        )

    # ── SV-COMP control flow (path-pruning; aborts on false) ──
    for n in ("assume_abort_if_not", "__VERIFIER_assume"):
        out[n] = _summary(
            n,
            overflow={"ensures": ["cond holds on the (only) returning path"],
                      "notes":   "aborts unless cond is true; assume cond after the call"},
            memsafe={"ensures": ["cond holds on the (only) returning path"]},
        )

    out["__VERIFIER_error"] = _summary(
        "__VERIFIER_error",
        noreturn=True,
        overflow={"ensures": ["does not return (program asserts failure)"],
                  "notes":   "reaching this is the bug under sv-comp reach-error"},
    )
    out["abort"] = _summary(
        "abort",
        noreturn=True,
        overflow={"ensures": ["does not return"]},
    )
    out["exit"]  = _summary(
        "exit",
        noreturn=True,
        overflow={"ensures": ["does not return"]},
        memleak={"ensures":  ["all program-lifetime allocations released by OS"]},
    )
    for n in ("_exit", "_Exit", "quick_exit", "__assert_fail", "__assert",
              "__assert_perror_fail", "longjmp", "siglongjmp",
              "pthread_exit", "thrd_exit"):
        out[n] = _summary(n, noreturn=True)

    return out


_STDLIB_CONTRACTS: dict[str, "FunctionSummary"] = _build_stdlib_contracts()


# ── SA features (regex; KAMain sidecar replaces this in v1) ─────────────────

@dataclass
class Features:
    has_deref: bool
    has_alloc: bool
    has_free: bool
    has_index: bool
    has_arith: bool   # +, -, *, ++, --  (signed-overflow risk)
    has_div: bool     # /, %             (division_by_zero / INT_MIN/-1)
    has_shift: bool   # <<, >>           (shift_ub)

    @property
    def memsafe_relevant(self) -> bool:
        return self.has_deref or self.has_index or self.has_alloc or self.has_free

    @property
    def memleak_relevant(self) -> bool:
        return self.has_alloc or self.has_free

    @property
    def overflow_relevant(self) -> bool:
        return self.has_arith or self.has_div or self.has_shift


_DEREF_RE = re.compile(r"(\*\s*\w)|(\w\s*->\s*\w)")
_INDEX_RE = re.compile(r"\w\s*\[")
_ALLOC_RE = re.compile(
    r"\b(malloc|calloc|realloc|alloca|kmalloc|kzalloc|kcalloc|"
    r"vmalloc|vzalloc|xmalloc|xzalloc|ldv_malloc|ldv_calloc|ldv_realloc|"
    r"ldv_zalloc|ldv_xmalloc|ldv_xzalloc)\s*\("
)
_FREE_RE = re.compile(r"\b(free|kfree|vfree|kvfree|ldv_free)\s*\(")
# Arithmetic: any +, -, *, ++, -- not part of a comment/pointer-deref/etc.
# Crude over-approximation is fine — adaptive scoping favors recall.
_ARITH_RE = re.compile(r"(\+\+|--|[^/*+\-]\s*[+\-*]\s*\w)")
_DIV_RE = re.compile(r"\w\s*[/%]\s*\w")
_SHIFT_RE = re.compile(r"<<|>>")


def _extract_features(source: str) -> Features:
    return Features(
        has_deref=bool(_DEREF_RE.search(source)),
        has_alloc=bool(_ALLOC_RE.search(source)),
        has_free=bool(_FREE_RE.search(source)),
        has_index=bool(_INDEX_RE.search(source)),
        has_arith=bool(_ARITH_RE.search(source)),
        has_div=bool(_DIV_RE.search(source)),
        has_shift=bool(_SHIFT_RE.search(source)),
    )


def _property_set(
    features: Features,
    callee_summaries: list[FunctionSummary],
) -> list[str]:
    """Adaptive scoping: which subset of PROPERTIES applies here.

    A property is in scope if either (a) the function body shows behavior
    that could touch it, or (b) any callee published a non-trivial
    requires/ensures for it (so the caller may need to discharge or
    propagate). Crude regex on body — KAMain JSON sidecar replaces this
    in v1.
    """
    def callee_touches(prop: str) -> bool:
        return any(
            s.has_requires(prop) or s.has_ensures(prop)
            for s in callee_summaries
        )

    props: list[str] = []
    if features.memsafe_relevant or callee_touches("memsafe"):
        props.append("memsafe")
    if features.memleak_relevant or callee_touches("memleak"):
        props.append("memleak")
    if features.overflow_relevant or callee_touches("overflow"):
        props.append("overflow")
    return props


# ── Prompts (per-property; verbatim callee discharge enforced) ──────────────

SYSTEM_PROMPT = """\
You produce Hoare-style pre/post summaries for C functions, ONE PROPERTY AT A TIME.

## Hard rules

1. **Specific predicates; code-form preferred, prose OK when natural.** \
Each item must pin a CONCRETE fact about a named variable, field, position, \
or range. Code expressions in the source language (C or C++) are preferred \
when they fit (`p != NULL`, `0 <= i && i < n`, `len <= sizeof(buf)`, \
`setEventCalled == 1`). Compact value-range forms are also fine for \
integer ranges (`n: [0, INT_MAX]`, `i: [1, len-1]`). Prose is acceptable \
when it compresses the fact more cleanly than a code expression would \
(`s null-terminated; '\\0' at index length-1`, `p[0..length-1] initialized`, \
`result allocated for n elements`). Avoid BOTH formal-logic notation \
(`\\valid(p)`, `\\forall x`) AND over-compressed prose that hides specifics \
(`p is a valid string` — what does "valid" mean? what's the length? is it \
null-terminated and where?). The downstream consumer is another LLM call: \
clarity-for-that-reader is the bar, not formal-tool checkability.

2. **Caller-accessible names only — STRICT.** Every `requires` / `ensures` \
clause is read by the CALLER. The caller cannot see anything you declared \
inside the function body. Reference ONLY:
   - this function's parameter names (read them off the signature, not \
     the body);
   - the literal `result` for the return value;
   - out-parameter cells (`*out_p`, `out_p->field` — only if `out_p` is a \
     parameter);
   - globals or `static` storage visible at the call site;
   - facts about memory the parameters point into (`s[0..n-1]`, \
     `*p initialized`, `s null-terminated`).
   FORBIDDEN: any identifier introduced by `int foo`, `char *p = s`, \
`for (int i = ...)`, etc. inside the body. Loop counters, walking pointers, \
clamp-locals — none of these may appear in `requires` or `ensures`.
   Concrete examples of common wrong outputs:
   - WRONG: function `f(const char *s)` whose body walks `p = s; p++` \
     publishing `requires: (long)(p - s) <= INT_MAX` — `p` is a body-local \
     walking pointer; the caller has no `p`.
     RIGHT: restate in caller terms (e.g. `strlen-of-s-as-cstring <= \
     INT_MAX`), or drop the clause and let the verify pass handle the \
     body cast.
   - WRONG: a no-argument function whose body declares `int length = ...; \
     if (length < 1) length = 1;` publishing `requires: length >= 1` — \
     `length` is body-local and the function takes no parameters.
     RIGHT: omit the clause. A no-arg function CANNOT have a meaningful \
     `requires` referencing body state; that invariant belongs in `notes` \
     (or in `ensures` if it escapes via `result`).
   If you cannot restate a callee's published clause in YOUR caller-visible \
terms, drop it and rely on the verify pass instead of forwarding a \
malformed clause upward.

3. **Callee discharge is VERBATIM.** When a callee K's contract says \
`requires[P]: phi`, you may either:
   (a) DISCHARGE phi at the callsite — cite a fact already on the path \
       (e.g., "p was just assigned `malloc(n)` whose `ensures` includes \
       `result != NULL`");
   (b) PROPAGATE phi as your own `requires[P]` — verbatim (after the \
       caller-name substitution required by rule 2), optionally with the \
       callsite's path-condition prepended (e.g., `cond ==> phi`).
   You MAY NOT INVENT preconditions a callee did not declare. If callee K's \
contract is `requires[P]: true`, you must NOT add a precondition on K's \
arguments "because K probably needs them valid". If K's contract has no \
preconditions, K's arguments are unconstrained from K's perspective.

4. **NO VERDICT.** You output only `requires` / `ensures` / `modifies`. \
Whether the body satisfies its own contract is checked separately and is \
not your concern here. Never write "this function is safe/unsafe".

5. **Conservative on uncertainty**:
   - `requires`: include the obligation when in doubt (FP > FN for soundness).
   - `ensures`: weaken to a safe over-approximation (often empty / `true`).
   - `modifies`: include the location if it might be written to.

6. **Approximate when predicates grow unwieldy.** Precision is a goal, but a
   ten-clause disjunction or a deeply nested conditional defeats downstream
   reasoning. When a predicate becomes too complex, approximate
   CONSERVATIVELY:
   - `ensures`: WEAKEN (over-approximate the post-state; e.g., replace
     `result == c1 || ... || result == c20` with `result >= -2147483648 &&
     result <= 259`). Never strengthen.
   - `requires`: STRENGTHEN (under-approximate the pre-state; demand more of
     the caller to keep sound). Never weaken.
   You decide when to approximate; flag in `notes` when you do.

7. **`modifies` scope — stack + heap, not globals.** C zero-initializes
   statics and globals at program start (C11 §6.7.9¶10), so reading them
   before any explicit write is well-defined. We track `modifies` to support
   use-before-initialization reasoning at the CALLER, which only matters for:
     - **Stack locals** (uninitialized at entry; UB on read-before-write);
     - **Heap memory from `malloc` / `realloc`** (uninitialized; UB on
       read-before-write). `calloc` returns zeroed memory — exempt;
     - **Out-parameters** the caller passes in (caller treats them as
       freshly-initialized after the call returns).
   Do NOT list globals or `static` storage in `modifies` — those don't
   create use-before-init obligations.

8. **External / harness functions are summarised for you.** When a callsite \
calls a function whose body you can't see (stdlib, `__VERIFIER_*`, etc.), \
its `requires` / `ensures` / `modifies` are inlined as `// >>>` comments \
above the call. Trust those exactly; do not re-derive their behaviour from \
naming conventions or harness assumptions.

9. **Noreturn callees cut the path.** A callsite annotated `// >>> noreturn: \
true` (or whose callee block shows `noreturn: true`) does not return. If it \
sits in the THEN-arm of `if (G) noreturnCallee();`, code after the `if` \
runs only when the call did not happen — so subsequent code may assume \
`!G`. If the function's body unconditionally aborts/exits/longjmps on every \
path, set `noreturn: true` in your output (it is a property-independent \
signal — emit the same value in every per-property call).
"""


MEMSAFE_PROMPT = """\
Analyze function `{name}` for MEMORY SAFETY only.

In-scope: null deref, out-of-bounds access, use-after-free, double-free, \
init-before-read.

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about pointer validity, \
buffer bounds, init state, or alloc/free state — predicates like \
`p != NULL`, `0 <= i && i < n`, `*p initialized`, `not freed(p)`. \
Do NOT emit integer-overflow predicates (`x != INT_MIN`, `n <= INT_MAX/2`) \
or memory-leak predicates here — those belong to other passes.

## Method — buffer / string fact tracking (publish what the body establishes)

Memsafe verification across function boundaries depends on the producer
publishing the size↔buffer pairings the consumer needs. Per system rule 1,
prose is acceptable as long as the fact is specific. Walk the body and
publish in `ensures` every concrete fact the body establishes about
returned, out-parameter, or modified buffers:

- **Allocation size pairing**: when you `malloc(N)` (or any allocator) and
  return / store the pointer, pair the pointer with N by name in `ensures`:
  `result allocated for N bytes`, or `result points to N elements of T`.
  Always cite N. Bare "result is heap-allocated" is too weak — the
  consumer can't bound any index.
- **Element count vs byte count**: `malloc(N * sizeof(T))` cast to `T*` →
  `N elements`; `malloc(N)` cast to `T*` → `N / sizeof(T)` elements (often
  a bug). Publish the unit you actually have.
- **C-string null terminator**: when you write `s[k] = '\\0'`, publish the
  terminator position: `s[k] == '\\0'`, or `s null-terminated at index k`.
  This makes the C-string LENGTH derivable as `k`. Without it, a consumer
  like `cstrlen(s)` cannot show that its loop terminates within the
  allocation. If the terminator position depends on a parameter (e.g.
  `s[length-1] = '\\0'`), name that parameter.
- **Initialization range**: when a loop writes `s[0..k]`, publish
  `s[0..k] initialized` (prose is fine here — the C-expression form would
  be a quantifier). If you only initialize a SUBSET of the buffer, publish
  the subset, not the whole buffer.
- **Pre-existing facts**: anything you LEARNED from a callsite (e.g.
  `t = malloc(...); if (t == 0) myexit(1);` → `t != NULL` after the
  branch) and that is still true at the function exit is also publishable
  if a caller would care.

If a producer fails to publish one of these facts, the consumer either
(a) can't write a meaningful `requires` (FN — bug missed because the
relevant invariant has no name), or (b) writes a `requires` the caller
cannot discharge (FP — false alarm propagated up the chain). Both are
worse than a slightly verbose `ensures`.

The header block lists each callee's published pre/post for memsafe. Use them \
verbatim; do NOT invent preconditions a callee did not declare.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<expr or short prose>", ...],
  "ensures":  ["<expr or short prose>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}

Guidance:
- `requires` examples: `Context != NULL` (only if a callee declared it),
  `len <= sizeof(buf)` (if function indexes `buf[i]` with `i < len`),
  `initialized(p, n)` (if function reads `*p`).
- `ensures` examples: `*out initialized for n bytes`, `return value != NULL`,
  `<inherits callee.ensures[memsafe]>`.
- `modifies`: list stack locals and heap regions written here (out-params,
  *p where p was malloc'd in this function, etc.). SKIP globals/statics
  (zero-init at startup → no use-before-init obligation).
- If function has no memsafe obligations and no memsafe-relevant ensures,
  output empty arrays. Empty is the right answer for many simple functions.
"""


MEMLEAK_PROMPT = """\
Analyze function `{name}` for MEMORY LEAKS only.

In-scope: every heap allocation either (a) released within the function on \
every return path, (b) returned to the caller, or (c) stored in a \
caller-visible location. Anything else leaks.

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about resource ownership and \
release — clauses like `caller releases <sym>`, `acquires fd: caller \
must close`, `all acquisitions released`, `<callee>.requires[memleak] \
holds`. Do NOT emit integer-overflow or pointer-validity predicates here.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<expr or short clause>", ...],
  "ensures":  ["<expr or short clause>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}

Guidance:
- `requires` examples: `caller releases <sym>` (function acquires resource and
  hands it off), `<callee>.requires[memleak] holds` (propagated verbatim).
- `ensures` examples: `no resource acquired`, `acquires fd: caller must close`,
  `all acquisitions released`.
- If the function does not allocate, free, or call anything that does, output
  empty arrays.
"""


OVERFLOW_PROMPT = """\
Analyze function `{name}` for INTEGER UB. SUMMARY ONLY — produce \
preconditions/postconditions that downstream verification can use. \
DO NOT decide whether a bug exists (that's the verify pass).

## Hard scope for `requires` and `ensures` in this pass

Every `requires` and `ensures` item MUST be about integer values:
signed-overflow ranges (`x != INT_MIN`, `n <= INT_MAX/2`), divisor
nonzero (`d != 0`), shift bounds (`0 <= s && s < 32`), or value-range
results (`result >= 0 && result <= INT_MAX`).

Do NOT emit pointer-validity predicates (`p != NULL`, `p != NP`),
initialization predicates (`*p initialized`, `field set`), or other
non-integer concerns here. Those belong to the memsafe pass and will
appear in this function's other property entries — leave them out.

## In-scope operations

- Signed arithmetic: `+`, `-`, `*`, `++`, `--`, unary `-` (overflow risk).
- Division / modulo: `/`, `%` (zero-divisor, plus `INT_MIN/-1`, `INT_MIN%-1`).
- Shifts: `<<`, `>>` (negative amount, amount >= bit-width of promoted left
  operand, left-shifting a negative signed value).

## Method — value-range analysis

Walk the function statement by statement, tracking a value range per integer
variable. Emit each range either as a code predicate (`x >= 1 && x <= 100`)
or as a compact value-range form (`x: [1, 100]`, `n: [INT_MIN+1, INT_MAX]`).
Pick whichever is shorter for the case at hand. Source language is whatever
the input file uses (C or C++); use that language's expression syntax.

1. **Initialise**: from C constants, callee `ensures[overflow]` ranges, and
   any callee `requires[overflow]` already discharged on this path.
2. **Narrow on branches**: after `if (x > 0)`, treat x as `x >= 1` on the true
   side and `x <= 0` on the false side. After a callsite whose summary
   says it aborts unless `c` (a callee `ensures` like `c on returning path`),
   treat `c` as a path fact for subsequent code.
3. **Arithmetic**: combine the operand ranges. If the result range can fall
   outside the operand type's range under the inputs you've tracked, emit a
   `requires` predicate that excludes the bad inputs.
4. **Callee discharge** (verbatim): for each callee K with
   `requires[overflow]: phi`, EITHER discharge phi at the callsite by citing
   the path facts that imply it, OR propagate phi as YOUR `requires`
   (verbatim, optionally with the path-condition prepended:
   `cond ==> phi`). NEVER invent a precondition a callee did not declare.

## Output form — code or value-range, not English

- `requires` items are boolean expressions or value-range forms that must
  hold on entry. Examples:
    `n != INT_MIN`                         (function computes `-n` on int)
    `n: [INT_MIN+1, INT_MAX]`              (same fact, range form)
    `n <= INT_MAX / 2`                     (function computes `2*n` on int)
    `divisor != 0`                         (function computes `x / divisor`)
    `shift_amt: [0, 31]`                   (function does `x << shift_amt`)
    `cond ==> n != INT_MIN`                (precondition only on a path)

- `ensures` items are boolean expressions or value-range forms about the
  return value and out-parameters on exit. Use the literal name `result`
  for the return value, and `*out_p` for an out-parameter. Examples:
    `result: [0, INT_MAX]`                 (return value range)
    `result == a + b`                      (exact value, holds under requires)
    `*out_p >= 0`                          (out-parameter postcondition)
    `result != 0 ==> *err == 0`            (conditional postcondition)

  If the function does not return an integer or has no integer effect,
  `ensures` may be empty. Empty is a valid answer.

- `modifies` items are stack locals and heap locations whose value the
  function changes (so the caller can reason about use-before-init for
  out-params and freshly-allocated memory). SKIP globals and `static`
  storage — C zero-inits them, so they create no use-before-init
  obligation.

- `notes` is one line of free-form context shown to YOUR caller alongside
  your contract (and to the verifier). Useful for facts that don't fit
  cleanly in `requires` / `ensures` (e.g. "this function aborts unless cond
  holds", "loop initialises s[0..length-1]"). Keep it short.

## Reminders for compositional discipline

- A function with no signed arithmetic, no division/modulo, no shifts, AND
  whose callees all publish empty `requires[overflow]` and `ensures[overflow]`
  → output empty arrays.
- DO NOT add a `requires` "just because the function has nondet inputs and
  performs arithmetic" — emit the predicate that ACTUALLY excludes the UB
  case (`x != INT_MIN`, `denom != 0`, `0 <= s < 32`). If you can write the
  predicate, write it; if you can't pin it down, propagate the callee
  obligation verbatim.

## C-semantics reminders (apply BEFORE flagging an op as UB)

- **Integer promotion.** Operands narrower than `int` (`char`, `short`,
  `_Bool`) are promoted to `int` before arithmetic; check overflow at the
  PROMOTED type, not the original.
- **Unsigned wrap is well-defined.** `unsigned` arithmetic wraps modulo
  2^N — not UB. Do not flag wrap on unsigned types.
- **Literal type follows C rules.** Unsuffixed decimal constants take the
  smallest of `int` / `long` / `long long` that fits; `LL` / `ULL` force
  long-long. The leading `-` on a literal is the unary-minus operator
  applied AFTER the literal's type is fixed.
- **Compare against the result type's range.** A signed op overflows only
  when its mathematical result falls outside the result type's
  representable range. Values that fit exactly (including the type's
  min/max) are not overflow.

## Data model

{data_model_note}

The header block lists each callee's published pre/post for overflow. Use
them verbatim; do NOT invent preconditions a callee did not declare.

{callee_block}

=== SOURCE ===
{source}

Output JSON exactly matching:
{{
  "requires": ["<expr or x: [lo, hi]>", ...],
  "ensures":  ["<expr or x: [lo, hi]>", ...],
  "modifies": ["<sym>", ...],
  "notes":    "<one-line context propagated to YOUR caller alongside contract>"
}}
"""


PROPERTY_PROMPT = {
    "memsafe": MEMSAFE_PROMPT,
    "memleak": MEMLEAK_PROMPT,
    "overflow": OVERFLOW_PROMPT,
}


# ── Callee summary block (inline, per-property) ─────────────────────────────

def _format_callee_for_property(s: FunctionSummary, prop: str) -> list[str]:
    reqs = s.requires.get(prop, [])
    ens = s.ensures.get(prop, [])
    mods = s.modifies.get(prop, [])
    note = s.notes.get(prop, "").strip()
    lines = [f"  {s.function}:"]
    if s.noreturn:
        lines.append("    noreturn: true")
    lines.append(
        f"    requires[{prop}]: " + ("; ".join(reqs) if reqs else "true")
    )
    lines.append(
        f"    ensures[{prop}]:  "
        + ("; ".join(ens) if ens else "(no observable effect)")
    )
    if mods:
        lines.append(f"    modifies: " + ", ".join(mods))
    if note:
        lines.append(f"    notes:    {note}")
    return lines


def _build_callee_block(
    func: Function,
    summaries: dict[str, FunctionSummary],
    prop: str,
    callee_names: list[str],
) -> str:
    in_scope = [n for n in callee_names if n in summaries]

    if not in_scope:
        return "=== CALLEE SUMMARIES ===\n(no in-scope callees)"

    parts = ["=== CALLEE SUMMARIES ==="]
    for name in in_scope:
        parts.extend(_format_callee_for_property(summaries[name], prop))
    return "\n".join(parts)


def _ordered_callee_names(
    func: Function,
    edges: dict[str, set[str]],
    summaries: dict[str, "FunctionSummary"] | None = None,
) -> list[str]:
    """Stable order: first by recorded callsite order, then any DB-only
    callees (e.g., resolved indirect targets) appended in name order.

    Includes external callees that have a seeded summary (stdlib / harness
    contracts) so they appear in the prompt's callee block alongside
    project-internal callees.
    """
    proj = edges.get(func.name, set())
    extern = set()
    if summaries is not None:
        for cs in func.callsites:
            name = cs.get("callee")
            if name and name not in proj and name in summaries:
                extern.add(name)
    in_scope = proj | extern

    seen: set[str] = set()
    out: list[str] = []
    for cs in func.callsites:
        name = cs.get("callee")
        if name and name in in_scope and name not in seen:
            seen.add(name)
            out.append(name)
    for name in sorted(in_scope):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _inline_callee_contracts(
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    prop: str,
) -> str:
    """Build the source the LLM sees for `prop`: raw lines with each
    callsite preceded by `// >>> callee.requires/ensures/modifies` hint
    lines, then layered through macro/sizeof annotation.

    Hints inserted at `cs['line_in_body']` (0-based offset into raw
    `func.source`). Skipped for callsites whose callee has no recorded
    summary or has nothing to say about this property (keeps output
    terse for the boilerplate cases).
    """
    raw_lines = func.source.splitlines()
    insertions: dict[int, list[str]] = {}
    # In-scope = anything we have a summary for. That covers both
    # project-internal callees (added during the topo walk) and stdlib /
    # harness contracts (seeded before the walk). The `edges` map only
    # tracks intra-project edges, so we cannot use it as the gate.
    for cs in func.callsites:
        callee_name = cs.get("callee")
        if not callee_name or callee_name not in summaries:
            continue
        line_in_body = cs.get("line_in_body")
        if line_in_body is None or line_in_body < 0 or line_in_body >= len(raw_lines):
            continue
        s = summaries[callee_name]
        reqs = [r for r in s.requires.get(prop, []) if _is_nontrivial(r)]
        ens = [e for e in s.ensures.get(prop, []) if _is_nontrivial(e)]
        mods = s.modifies.get(prop, [])
        note = s.notes.get(prop, "").strip()
        if not reqs and not ens and not mods and not note and not s.noreturn:
            continue
        body_line = raw_lines[line_in_body]
        indent = " " * (len(body_line) - len(body_line.lstrip()))
        hint: list[str] = [
            f"{indent}// >>> {callee_name} contract for {prop}:"
        ]
        if s.noreturn:
            hint.append(f"{indent}// >>>   noreturn: true")
        if reqs:
            hint.append(f"{indent}// >>>   requires: " + "; ".join(reqs))
        if ens:
            hint.append(f"{indent}// >>>   ensures:  " + "; ".join(ens))
        if mods:
            hint.append(f"{indent}// >>>   modifies: " + ", ".join(mods))
        if note:
            hint.append(f"{indent}// >>>   notes:    " + note)
        insertions.setdefault(line_in_body, []).extend(hint)

    if not insertions:
        return func.llm_source

    out: list[str] = []
    for i, line in enumerate(raw_lines):
        if i in insertions:
            out.extend(insertions[i])
        out.append(line)
    annotated = "\n".join(out)
    if func.pp_source and func.pp_source != func.source:
        return _annotate_macro_diff(annotated, func.pp_source)
    return annotated


# ── Bottom-up driver ────────────────────────────────────────────────────────

def _build_callsite_edges(functions: list[Function]) -> dict[str, set[str]]:
    """Fallback edge map (name → set of callee names) built from the
    per-function `callsites` lists populated by the FunctionExtractor.
    Captures only direct callees; misses indirect-call resolution."""
    by_name = {f.name: f for f in functions}
    edges: dict[str, set[str]] = {f.name: set() for f in functions}
    for f in functions:
        for cs in f.callsites:
            callee = cs.get("callee")
            if callee and callee in by_name:
                edges[f.name].add(callee)
    return edges


def _build_db_edges(
    db: SummaryDB, functions: list[Function],
) -> dict[str, set[str]]:
    """Build edge map (name → callee names) from the DB's `call_edges`
    table plus resolved indirect targets from
    `indirect_callsites` ⨝ `indirect_call_targets`.

    Restricted to the in-scope `functions` set so boilerplate (filtered
    in `_load_functions_from_db`) and any extra DB rows don't leak in.
    """
    in_scope_names = {f.name: f.id for f in functions if f.id is not None}
    in_scope_ids = {fid for fid in in_scope_names.values()}
    id_to_name: dict[int, str] = {f.id: f.name for f in functions if f.id is not None}

    edges: dict[str, set[str]] = {f.name: set() for f in functions}

    # Direct call edges (and is_indirect=1 rows that the scan resolved
    # in-place, e.g., when the indirect-call resolver wrote them back).
    for row in db.conn.execute(
        "SELECT caller_id, callee_id FROM call_edges"
    ).fetchall():
        caller_id = row["caller_id"]
        callee_id = row["callee_id"]
        if caller_id in in_scope_ids and callee_id in in_scope_ids:
            edges[id_to_name[caller_id]].add(id_to_name[callee_id])

    # Resolved indirect-call targets (when the indirect-call resolver
    # left the resolution in indirect_call_targets instead of inserting
    # back into call_edges).
    for row in db.conn.execute(
        "SELECT ic.caller_function_id AS caller_id, "
        "       ict.target_function_id AS callee_id "
        "FROM indirect_callsites ic "
        "JOIN indirect_call_targets ict ON ic.id = ict.callsite_id"
    ).fetchall():
        caller_id = row["caller_id"]
        callee_id = row["callee_id"]
        if caller_id in in_scope_ids and callee_id in in_scope_ids:
            edges[id_to_name[caller_id]].add(id_to_name[callee_id])

    return edges


def _reachable_from(
    functions: list[Function],
    entry_name: str,
    edges: dict[str, set[str]],
) -> list[Function]:
    """BFS from entry over `edges`; return functions reachable from entry."""
    if entry_name not in edges:
        return []
    seen: set[str] = {entry_name}
    queue: list[str] = [entry_name]
    while queue:
        name = queue.pop()
        for callee in edges.get(name, ()):
            if callee not in seen:
                seen.add(callee)
                queue.append(callee)
    return [f for f in functions if f.name in seen]


def _topo_order(
    functions: list[Function],
    edges: dict[str, set[str]],
) -> list[Function]:
    """Reverse-topological order (callees first). Recursion-safe."""
    by_name = {f.name: f for f in functions}
    order: list[Function] = []
    visited: set[str] = set()
    on_stack: set[str] = set()

    def visit(f: Function) -> None:
        if f.name in visited or f.name in on_stack:
            return
        on_stack.add(f.name)
        for callee in edges.get(f.name, ()):
            if callee in by_name and callee != f.name:
                visit(by_name[callee])
        on_stack.discard(f.name)
        visited.add(f.name)
        order.append(f)

    for f in functions:
        visit(f)
    return order


def _load_functions_from_db(db_path: Path) -> tuple[list[Function], SummaryDB]:
    """Pull pre-scanned functions (with pp_source / llm_source annotations)
    from the SummaryDB built by `llm-summary scan`. Returns (functions, db)
    so the caller can also query call_edges/indirect-call tables."""
    db = SummaryDB(str(db_path))
    funcs = db.get_all_functions()
    funcs = [
        f for f in funcs
        if f.name not in je.BOILERPLATE_FUNCTIONS and f.source
    ]
    return funcs, db


def _extract_functions(i_path: Path, work_dir: Path) -> list[Function]:
    """Fallback path when no pre-scanned DB exists — re-extract from source."""
    cc_json = [{
        "directory": str(i_path.parent),
        "file": str(i_path),
        "command": f"clang -c {i_path.name}",
    }]
    cc_path = work_dir / "compile_commands.json"
    cc_path.write_text(json.dumps(cc_json))
    compile_commands, _ = _load_compile_commands(str(cc_path), None)
    extractor = FunctionExtractor(
        compile_commands=compile_commands,
        enable_preprocessing=(i_path.suffix != ".i"),
    )
    funcs = extractor.extract_from_file(i_path)
    return [
        f for f in funcs
        if f.name not in je.BOILERPLATE_FUNCTIONS and f.source
    ]


def _entry_function_name(functions: list[Function]) -> str | None:
    for f in functions:
        if f.name == "main":
            return f.name
    return None


def _check_entry(
    summary: FunctionSummary,
) -> tuple[bool, dict[str, list[str]]]:
    """Hoare-only entry check across ALL properties.

    Returns (predicted_safe, obligations_by_property).
    Any non-trivial `requires[P]` at the entry = potential bug for P, since
    no caller exists to discharge it. Conservative — IL feasibility analysis
    would tighten FPs in a later layer.
    """
    obligations: dict[str, list[str]] = {}
    for prop in PROPERTIES:
        non_trivial = [r for r in summary.requires.get(prop, [])
                       if _is_nontrivial(r)]
        if non_trivial:
            obligations[prop] = non_trivial
    return len(obligations) == 0, obligations


# ── Verification pass (per function, per in-scope property) ─────────────────
#
# Summarization produced per-function pre/post for each property. Verify
# decides whether the body actually satisfies its own contract: for each
# in-scope property, look for places in the body where the property could be
# violated despite the published `requires` being assumed true. This is the
# missing half — without it, intra-entry UB (e.g. `int x = INT_MAX + 1` in
# `main`) never surfaces because there's no caller to receive a precondition.

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "description": {"type": "string"},
                },
                "required": ["kind", "description"],
            },
        },
    },
    "required": ["issues"],
}


VERIFY_SYSTEM_PROMPT = """\
You verify whether a C function body satisfies its own published Hoare-style
contract for a single safety property. You are NOT producing the contract
(that was a prior pass) — you are looking for property violations *inside*
the body, given the body, the contract, and each callee's contract.

## Hard rules

1. **Assume `requires` hold.** The function's published `requires[P]` are
   given to you as facts on entry. Do NOT report a "missing precondition"
   or recommend tightening the contract. If a precondition you wish existed
   is not present, the body must establish it before the relevant operation
   or that operation is a violation.

2. **Discharge callee `requires[P]` at every callsite.** Each callsite is
   annotated with `// >>> callee contract for P` showing requires/ensures/
   modifies. If a callee's `requires[P]` may NOT hold at the callsite given
   the path facts, that is a violation in THIS function (the caller failed
   to discharge it).

3. **Stay inside the property.** Only flag issues belonging to property P —
   memsafe, memleak, or overflow as instructed. Do NOT mix.

4. **Be specific.** Each issue cites a concrete operation, not a category
   ("the function does arithmetic so it might overflow" is not an issue —
   "`x = 2147483647 + 1` overflows on the unconditional path" is).

5. **External / harness functions are summarised for you.** Each callsite is
   annotated with the callee's `requires` / `ensures` / `modifies`. Trust
   those exactly — including for stdlib (`malloc`, `free`, etc.) and harness
   helpers (`__VERIFIER_*`, `assume_abort_if_not`-style aborts). Don't
   re-derive their behaviour from the name.

6. **Noreturn callees cut the path.** A callsite annotated
   `// >>> noreturn: true` does not return. Code after such a call (on the
   same straight-line path) is unreachable. If the call sits in the
   THEN-arm of `if (G) noreturnCallee();`, code after the `if` runs only
   when the call did not happen — so subsequent code may assume `!G`.

7. **Empty list is the right answer when the body is safe under its
   published contract.**
"""


VERIFY_MEMSAFE_PROMPT = """\
VERIFY function `{name}` for MEMORY SAFETY violations under its published
memsafe contract.

In-scope kinds (use these exact `kind` strings):
  - `null_deref`         — `*p`, `p->f`, `p[i]` with p potentially NULL
  - `buffer_overflow`    — `a[i]` with i potentially outside [0, len(a))
  - `use_after_free`     — deref of a pointer freed earlier on the path
  - `double_free`        — free of a pointer already freed on the path
  - `invalid_free`       — free of a pointer not from malloc / not at base
  - `uninitialized_use`  — read of a stack/heap byte not yet written
  - `callee_requires`    — a callsite where callee.requires[memsafe] may not
                           hold given the path facts (cite the callee + which
                           clause); this is the caller's bug to flag

## Function's published memsafe contract (assume `requires` hold on entry)
{own_contract}

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for memsafe`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "<one of the kinds above>",
     "line": <int|null>,
     "description": "<one-line: cite the operation/callsite and why the published requires (plus path facts) don't cover it>"}}
  ]
}}
"""


VERIFY_MEMLEAK_PROMPT = """\
VERIFY function `{name}` for MEMORY LEAKS under its published memleak contract.

In-scope kinds:
  - `memory_leak`     — heap allocation on a path that doesn't reach `free`,
                        isn't returned to the caller, and isn't stored in
                        caller-visible memory before the function returns
  - `callee_requires` — a callsite where callee.requires[memleak] may not hold
                        (e.g. the caller fails to release a resource the
                        callee declares the caller must release)

## Function's published memleak contract (assume `requires` hold on entry)
{own_contract}

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for memleak`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "memory_leak|callee_requires",
     "line": <int|null>,
     "description": "<one-line: cite the allocation and the path on which it leaks>"}}
  ]
}}
"""


VERIFY_OVERFLOW_PROMPT = """\
VERIFY function `{name}` for INTEGER UB under its published overflow contract.

In-scope kinds:
  - `integer_overflow` — signed `+` / `-` / `*` / `++` / `--` / unary `-`
                         whose result can fall outside the operand type
  - `division_by_zero` — `/` or `%` with a divisor that may be zero
  - `shift_ub`         — `<<` / `>>` with negative amount, amount >= bit-width
                         of the promoted left operand, or left-shift of a
                         negative signed value
  - `callee_requires`  — a callsite where callee.requires[overflow] may not
                         hold (cite which clause)

## Method (value-range; use the source language's expression syntax)

Walk the body statement by statement, tracking each integer's range from:
  (a) the function's published `requires[overflow]`,
  (b) C constants assigned along the path,
  (c) callee `ensures[overflow]` after each callsite,
  (d) branch narrowing (`if (x > 0)` ⇒ `x >= 1` on the true side),
  (e) `assume_abort_if_not(c)` adds `c` to the path.
For each in-scope op, ask: can the operand range admit UB? If yes, that op
is an `integer_overflow` / `division_by_zero` / `shift_ub` issue.

For each callsite: do the path facts imply every clause of
callee.requires[overflow]? If not, emit `callee_requires`.

## C-semantics reminders (apply BEFORE flagging an op as UB)

- **Integer promotion.** Operands narrower than `int` (`char`, `short`,
  `_Bool`) are promoted to `int` before arithmetic; check overflow at the
  PROMOTED type, not the original.
- **Unsigned wrap is well-defined.** `unsigned` arithmetic wraps modulo
  2^N — not UB. Do not flag wrap on unsigned types.
- **Literal type follows C rules.** Unsuffixed decimal constants take the
  smallest of `int` / `long` / `long long` that fits; `LL` / `ULL` force
  long-long. The leading `-` on a literal is the unary-minus operator
  applied AFTER the literal's type is fixed.
- **Compare against the result type's range.** A signed op overflows only
  when its mathematical result falls outside the result type's
  representable range. Values that fit exactly (including the type's
  min/max) are not overflow.
- **Noreturn callees in guards.** A callsite annotated
  `// >>> noreturn: true` (e.g. `abort`, `exit`, `__VERIFIER_error`) does
  not return; if it sits in `if (G) noreturnCallee();`, code after the
  `if` may assume `!G`.

## Data model

{data_model_note}

## Function's published overflow contract (assume `requires` hold on entry)
{own_contract}

{callee_block}

=== SOURCE (callsites annotated with `// >>> callee contract for overflow`) ===
{source}

Output JSON exactly matching:
{{
  "issues": [
    {{"kind": "integer_overflow|division_by_zero|shift_ub|callee_requires",
     "line": <int|null>,
     "description": "<one-line: cite the operation and the operand range that admits UB>"}}
  ]
}}

Empty list = body is overflow-safe under its published requires.
"""


VERIFY_PROMPT = {
    "memsafe": VERIFY_MEMSAFE_PROMPT,
    "memleak": VERIFY_MEMLEAK_PROMPT,
    "overflow": VERIFY_OVERFLOW_PROMPT,
}


def _format_own_contract(summary: FunctionSummary, prop: str) -> str:
    reqs = [r for r in summary.requires.get(prop, []) if _is_nontrivial(r)]
    ens = [e for e in summary.ensures.get(prop, []) if _is_nontrivial(e)]
    mods = summary.modifies.get(prop, [])
    lines = [
        f"  requires[{prop}]: " + ("; ".join(reqs) if reqs else "true"),
        f"  ensures[{prop}]:  " + ("; ".join(ens) if ens else "(no observable effect)"),
    ]
    if mods:
        lines.append(f"  modifies: " + ", ".join(mods))
    return "\n".join(lines)


def _verify_one(
    func: Function,
    summary: FunctionSummary,
    summaries: dict[str, FunctionSummary],
    llm: LLMBackend,
    log_fp: Any,
    edges: dict[str, set[str]],
    data_model_note: str = "",
) -> tuple[dict[str, list[dict[str, Any]]], int, int, int]:
    """Per-property verification for one function. Returns
    (issues_by_prop, calls, in_tok, out_tok)."""
    issues_by_prop: dict[str, list[dict[str, Any]]] = {}
    if not summary.properties:
        return issues_by_prop, 0, 0, 0

    callee_names = _ordered_callee_names(func, edges, summaries)
    response_format = make_json_response_format(VERIFY_SCHEMA, name="verify")
    in_tok = out_tok = calls = 0

    if log_fp:
        log_fp.write(f"\n--- VERIFY ({func.name}) ---\n")

    for prop in summary.properties:
        callee_block = _build_callee_block(
            func, summaries, prop, callee_names
        )
        source_inlined = _inline_callee_contracts(
            func, summaries, edges, prop
        )
        own_contract = _format_own_contract(summary, prop)
        fmt_kwargs: dict[str, Any] = dict(
            name=func.name,
            own_contract=own_contract,
            callee_block=callee_block,
            source=source_inlined,
        )
        if prop == "overflow":
            fmt_kwargs["data_model_note"] = data_model_note
        prompt = VERIFY_PROMPT[prop].format(**fmt_kwargs)

        if log_fp:
            log_fp.write(f"\n--- VERIFY USER ({prop}) ---\n{prompt}\n")

        _dump_target = os.environ.get("DUMP_VERIFY")
        if _dump_target and _dump_target == func.name:
            _out = Path(f"/tmp/verify_{func.name}_{prop}.json")
            _out.write_text(json.dumps({
                "system": VERIFY_SYSTEM_PROMPT,
                "prompt": prompt,
                "response_format": response_format,
                "schema": VERIFY_SCHEMA,
            }, indent=2))
            log.info("Dumped verify prompt to %s", _out)

        resp = llm.complete_with_metadata(
            prompt, system=VERIFY_SYSTEM_PROMPT,
            response_format=response_format,
        )
        calls += 1
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens

        if log_fp:
            log_fp.write(
                f"\n--- VERIFY RESPONSE ({prop}) ---\n{resp.content}\n"
            )

        try:
            data = extract_json(resp.content)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"{func.name}/{prop} verify: invalid JSON: {e}\n"
                f"{resp.content[:200]}"
            ) from e

        raw_issues = data.get("issues") or []
        kept: list[dict[str, Any]] = []
        for it in raw_issues:
            if not isinstance(it, dict):
                continue
            kind = str(it.get("kind") or "").strip()
            desc = str(it.get("description") or "").strip()
            if not kind or not desc:
                continue
            kept.append({
                "kind": kind,
                "line": it.get("line"),
                "description": desc,
            })
        if kept:
            issues_by_prop[prop] = kept

    return issues_by_prop, calls, in_tok, out_tok


def _summarize_one(
    func: Function,
    summaries: dict[str, FunctionSummary],
    llm: LLMBackend,
    log_fp: Any,
    edges: dict[str, set[str]],
    data_model_note: str = "",
) -> tuple[FunctionSummary, int, int, int]:
    """Per-property summarization for one function. v0: one LLM call per
    in-scope property."""
    raw_for_features = func.llm_source  # macro-annotated + sizeof-annotated
    features = _extract_features(raw_for_features)
    callee_names = _ordered_callee_names(func, edges, summaries)
    callee_summaries = [summaries[n] for n in callee_names if n in summaries]
    props = _property_set(features, callee_summaries)

    summary = FunctionSummary(function=func.name, properties=props)
    # Seed noreturn from explicit attribute (e.g. extern decl with
    # `__attribute__((noreturn))` or `_Noreturn`); LLM may also emit
    # `noreturn: true` per-property below — OR-merge.
    if func.attributes and "noreturn" in func.attributes.lower():
        summary.noreturn = True
    in_tok = out_tok = calls = 0

    if log_fp:
        log_fp.write(f"\n\n===== FUNCTION: {func.name} =====\n")
        log_fp.write(f"--- FEATURES ---\n")
        log_fp.write(
            f"deref={features.has_deref} index={features.has_index} "
            f"alloc={features.has_alloc} free={features.has_free}\n"
        )
        log_fp.write(f"--- PROPERTIES ---\n{props}\n")

    if not props:
        if log_fp:
            log_fp.write("(no properties in scope; emitting empty summary)\n")
        return summary, 0, 0, 0

    response_format = make_json_response_format(
        PROPERTY_SCHEMA, name="property_summary"
    )

    for prop in props:
        callee_block = _build_callee_block(func, summaries, prop, callee_names)
        source_inlined = _inline_callee_contracts(
            func, summaries, edges, prop
        )
        fmt_kwargs: dict[str, Any] = dict(
            name=func.name,
            callee_block=callee_block,
            source=source_inlined,
        )
        if prop == "overflow":
            fmt_kwargs["data_model_note"] = data_model_note
        prompt = PROPERTY_PROMPT[prop].format(**fmt_kwargs)

        if log_fp:
            log_fp.write(f"\n--- USER ({prop}) ---\n{prompt}\n")

        _dump_target = os.environ.get("DUMP_SUMMARIZE")
        if _dump_target and _dump_target == func.name:
            _out = Path(f"/tmp/summarize_{func.name}_{prop}.json")
            _out.write_text(json.dumps({
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "response_format": response_format,
                "schema": PROPERTY_SCHEMA,
            }, indent=2))
            log.info("Dumped summarize prompt to %s", _out)

        resp = llm.complete_with_metadata(
            prompt, system=SYSTEM_PROMPT, response_format=response_format,
        )
        calls += 1
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens

        if log_fp:
            log_fp.write(f"\n--- RESPONSE ({prop}) ---\n{resp.content}\n")

        try:
            data = extract_json(resp.content)
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(
                f"{func.name}/{prop}: invalid JSON: {e}\n"
                f"{resp.content[:200]}"
            ) from e

        summary.requires[prop] = list(data.get("requires") or [])
        summary.ensures[prop] = list(data.get("ensures") or [])
        summary.modifies[prop] = list(data.get("modifies") or [])
        summary.notes[prop] = str(data.get("notes") or "")
        if bool(data.get("noreturn", False)):
            summary.noreturn = True

    return summary, calls, in_tok, out_tok


@dataclass
class TaskRun:
    predicted_safe: bool
    obligations: dict[str, list[str]]
    issues: dict[str, dict[str, list[dict[str, Any]]]]  # function → prop → [issue]
    calls: int
    in_tok: int
    out_tok: int
    summaries: dict[str, FunctionSummary]


_DATA_MODEL_NOTES = {
    "LP64": ("LP64 (sizeof: int=4, long=8, long long=8, void*=8). "
             "Type ranges: int [-2^31, 2^31-1]; long/long long [-2^63, 2^63-1]."),
    "ILP32": ("ILP32 (sizeof: int=4, long=4, long long=8, void*=4). "
              "Type ranges: int / long [-2^31, 2^31-1]; long long [-2^63, 2^63-1]."),
}


def _data_model_note(model: str | None) -> str:
    """Render a one-paragraph data-model note for the prompt. Defaults to
    LP64 when the task didn't declare one."""
    key = (model or "LP64").upper()
    return _DATA_MODEL_NOTES.get(key, _DATA_MODEL_NOTES["LP64"])


def run_one_task(
    i_path: Path,
    work_dir: Path,
    llm: LLMBackend,
    log_llm: str | None = None,
    db_path: Path | None = None,
    verify: bool = True,
    data_model: str | None = None,
) -> TaskRun:
    """Run summarize (always) and verify (when `verify=True`) over the
    call graph reachable from `main`.

    The combined safety verdict is:
        predicted_safe = (no non-trivial requires at the entry) AND
                         (verify pass found no issue in any function).

    Without verify, the second clause is vacuously true (legacy v0
    behaviour); with verify, intra-entry UB and any latent body
    violations also surface.

    If `db_path` is provided and exists, loads pre-scanned functions
    (with macro-annotated `pp_source`) from the DB; otherwise re-extracts
    from the source file as a fallback.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    db: SummaryDB | None = None
    if db_path is not None and db_path.exists():
        log.debug("  Loading functions from DB: %s", db_path)
        functions, db = _load_functions_from_db(db_path)
    else:
        if db_path is not None:
            log.debug("  No DB at %s — re-extracting from source", db_path)
        functions = _extract_functions(i_path, work_dir)
    if not functions:
        raise RuntimeError("No functions extracted")

    entry = _entry_function_name(functions)
    if entry is None:
        raise RuntimeError("No entry function (main) found")

    if db is not None:
        edges = _build_db_edges(db, functions)
        n_edges = sum(len(v) for v in edges.values())
        log.debug("  Call graph: %d edges from DB (call_edges + indirect)",
                  n_edges)
    else:
        edges = _build_callsite_edges(functions)
        log.debug("  Call graph: %d edges from func.callsites (direct only)",
                  sum(len(v) for v in edges.values()))

    functions = _reachable_from(functions, entry, edges)
    log.debug("  Reachable from %s: %d functions", entry, len(functions))

    # Seed with stdlib / harness contracts so opaque callees (malloc,
    # __VERIFIER_nondet_*, assume_abort_if_not, etc.) flow through the
    # callee-inlining path the same way as user functions.
    summaries: dict[str, FunctionSummary] = dict(_STDLIB_CONTRACTS)
    issues: dict[str, dict[str, list[dict[str, Any]]]] = {}
    in_tok = out_tok = calls = 0
    log_fp = open(log_llm, "w") if log_llm else None

    dm_note = _data_model_note(data_model)

    try:
        ordered = list(_topo_order(functions, edges))
        for func in ordered:
            s, c, i, o = _summarize_one(
                func, summaries, llm, log_fp, edges=edges,
                data_model_note=dm_note,
            )
            calls += c
            in_tok += i
            out_tok += o
            summaries[func.name] = s

        if verify:
            for func in ordered:
                summary = summaries[func.name]
                if not summary.properties:
                    continue
                func_issues, c, i, o = _verify_one(
                    func, summary, summaries, llm, log_fp, edges=edges,
                    data_model_note=dm_note,
                )
                calls += c
                in_tok += i
                out_tok += o
                if func_issues:
                    issues[func.name] = func_issues
    finally:
        if log_fp:
            log_fp.close()

    entry_summary = summaries.get(entry)
    if entry_summary is None:
        raise RuntimeError(f"No summary produced for entry `{entry}`")

    if verify:
        # Verify is the source of truth: body issues + un-discharged callee
        # requires are the only real bugs. Entry's own `requires` is just
        # propagated metadata — the verifier already discharged each clause
        # at the callsite (or flagged it as `callee_requires`).
        obligations = {}
        predicted_safe = len(issues) == 0
    else:
        # No-verify ablation: fall back to the entry-requires heuristic.
        _, obligations = _check_entry(entry_summary)
        predicted_safe = len(obligations) == 0

    return TaskRun(
        predicted_safe=predicted_safe,
        obligations=obligations,
        issues=issues,
        calls=calls,
        in_tok=in_tok,
        out_tok=out_tok,
        summaries=summaries,
    )


# ── Per-task result ─────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    yml_file: str
    i_file: str
    cwe: str
    variant: str
    expected_safe: bool
    subproperty: str
    obligations: dict[str, list[str]] = field(default_factory=dict)
    issues: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    predicted_safe: bool | None = None
    correct: bool | None = None
    classification: str = ""
    error: str | None = None
    elapsed_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hoare-style summary pipeline (memsafe + memleak). "
            "Per docs/design-llm-first.md."
        ),
    )
    parser.add_argument("--benchmarks", default=None,
                        help="Path to benchmark directory")
    parser.add_argument("--cwe", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--backend", default="claude")
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-host", default="localhost")
    parser.add_argument("--llm-port", type=int, default=None)
    parser.add_argument("--output", "-o",
                        default="contract_pipeline_results.json")
    parser.add_argument("--summary-out", default=None,
                        help="Per-function summaries (JSON) for inspection")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--set-file", default=None)
    parser.add_argument("--yml", default=None,
                        help="Single .yml task file (smoke test)")
    parser.add_argument("--filter", default=None)
    parser.add_argument("--func-scans-dir", default=je.DEFAULT_FUNC_SCANS,
                        help="Directory containing pre-scanned functions.db "
                             "per task (mirrors juliet_eval). When present, "
                             "we load Function objects (with pp_source / "
                             "macro+sizeof annotations) from the DB instead "
                             "of re-extracting.")
    parser.add_argument("--work-dir",
                        default="func-scans/sv-benchmarks-contract",
                        help="Fallback working directory used only when no "
                             "pre-scanned DB is found for a task.")
    parser.add_argument("--llm-log-dir", default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the per-function verify pass; only the "
                             "entry-requires check decides safety. Useful "
                             "for ablation against the summarize-only v0.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.yml:
        yml_path = Path(args.yml)
        if not yml_path.exists():
            log.error("YAML task file not found: %s", yml_path)
            sys.exit(1)
        info = je.parse_yml(yml_path)
        if info is None:
            log.error("No relevant property in %s", yml_path)
            sys.exit(1)
        tasks = [(yml_path, info)]
    elif args.set_file:
        set_path = Path(args.set_file)
        if not set_path.exists():
            log.error("Set file not found: %s", set_path)
            sys.exit(1)
        tasks = je.collect_tasks_from_set_file(set_path, args.cwe)
    elif args.benchmarks:
        bdir = Path(args.benchmarks)
        if not bdir.exists():
            log.error("Benchmarks directory not found: %s", bdir)
            sys.exit(1)
        tasks = je.collect_tasks(bdir, args.cwe, args.variant)
    else:
        log.error("One of --yml, --set-file, or --benchmarks is required")
        sys.exit(1)

    log.info("Collected %d tasks", len(tasks))

    if args.filter:
        tasks = [(p, i) for p, i in tasks if args.filter in p.stem]
        log.info("Filtered to %d tasks", len(tasks))
    if args.limit > 0:
        tasks = tasks[: args.limit]

    backend_kwargs = build_backend_kwargs(
        args.backend, llm_host=args.llm_host, llm_port=args.llm_port,
        disable_thinking=args.disable_thinking,
    )
    llm = create_backend(args.backend, model=args.model, **backend_kwargs)
    log.info("Backend: %s  Model: %s", args.backend, llm.model)

    work_base = Path(args.work_dir)
    func_scans_base = Path(args.func_scans_dir)
    results: list[TaskResult] = []
    all_summaries: dict[str, dict[str, Any]] = {}
    tp = tn = fp = fn = errors = 0

    for idx, (yml_path, info) in enumerate(tasks):
        i_path = Path(info["source_path"])
        expected_safe: bool = info["expected_verdict"]
        subprop: str = info["subproperty"]

        variant = ("bad" if "_bad" in yml_path.stem
                   else "good" if "_good" in yml_path.stem
                   else "")
        stem = yml_path.stem
        work_dir = work_base / yml_path.parent.name / stem
        db_path = func_scans_base / yml_path.parent.name / stem / "functions.db"

        result = TaskResult(
            yml_file=yml_path.name,
            i_file=info["i_file"],
            cwe=stem.split("---")[0] if "---" in stem else stem.split("_")[0],
            variant=variant,
            expected_safe=expected_safe,
            subproperty=subprop,
        )

        log.info("[%d/%d] %s (expected: %s, subprop: %s)",
                 idx + 1, len(tasks), stem,
                 "safe" if expected_safe else "UNSAFE", subprop)

        log_llm: str | None = None
        if args.llm_log_dir:
            ld = Path(args.llm_log_dir)
            ld.mkdir(parents=True, exist_ok=True)
            log_llm = str(ld / f"{stem}.log")

        t0 = time.time()
        try:
            run = run_one_task(
                i_path, work_dir, llm,
                log_llm=log_llm, db_path=db_path,
                verify=not args.no_verify,
                data_model=info.get("data_model"),
            )
            result.elapsed_s = time.time() - t0
            result.llm_calls = run.calls
            result.input_tokens = run.in_tok
            result.output_tokens = run.out_tok
            result.obligations = run.obligations
            result.issues = run.issues
            result.predicted_safe = run.predicted_safe
            result.correct = run.predicted_safe == expected_safe

            if expected_safe and run.predicted_safe:
                result.classification = "TN"; tn += 1
            elif expected_safe and not run.predicted_safe:
                result.classification = "FP"; fp += 1
            elif not expected_safe and not run.predicted_safe:
                result.classification = "TP"; tp += 1
            else:
                result.classification = "FN"; fn += 1

            n_oblig = sum(len(v) for v in run.obligations.values())
            n_issues = sum(
                len(per_prop)
                for per_prop_dict in run.issues.values()
                for per_prop in per_prop_dict.values()
            )
            log.info(
                "  -> %s (%d oblig / %d issues, %d calls, %.1fs, "
                "%dk in + %dk out)",
                result.classification, n_oblig, n_issues, run.calls,
                result.elapsed_s, run.in_tok // 1000, run.out_tok // 1000,
            )

            if args.summary_out:
                all_summaries[stem] = {
                    name: asdict(s) for name, s in run.summaries.items()
                }

        except Exception as e:
            result.elapsed_s = time.time() - t0
            result.error = str(e)
            errors += 1
            log.error("  -> ERROR: %s", e)

        results.append(result)

    total = len(results)
    scored = total - errors
    correct = tp + tn
    accuracy = correct / scored if scored else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0)

    log.info("")
    log.info("=== Results ===")
    log.info("Total: %d  Scored: %d  Correct: %d  Accuracy: %.1f%%",
             total, scored, correct, accuracy * 100)
    log.info("TP: %d  TN: %d  FP: %d  FN: %d  Errors: %d",
             tp, tn, fp, fn, errors)
    log.info("Precision: %.3f  Recall: %.3f  F1: %.3f",
             precision, recall, f1)
    total_in = sum(r.input_tokens for r in results)
    total_out = sum(r.output_tokens for r in results)
    log.info("Tokens: %dk input, %dk output",
             total_in // 1000, total_out // 1000)

    output = {
        "summary": {
            "total": total, "scored": scored, "correct": correct,
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "errors": errors,
        },
        "config": {
            "backend": args.backend,
            "model": llm.model,
            "cwe": args.cwe,
            "variant": args.variant,
            "approach": "hoare-pre-post-memsafe-memleak",
        },
        "results": [asdict(r) for r in results],
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    log.info("Results saved to %s", args.output)

    if args.summary_out and all_summaries:
        Path(args.summary_out).write_text(json.dumps(all_summaries, indent=2))
        log.info("Summaries saved to %s", args.summary_out)


if __name__ == "__main__":
    main()
