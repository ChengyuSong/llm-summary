# LLM Prompts

This document describes the prompts the analyzer sends to the LLM. The current
primary path is the **code-contract pipeline** (one Hoare-style summary +
verifier per safety property). The earlier multi-pass pipeline
(allocation / free / init / memsafe / verification, plus optional leak and
integer-overflow passes) is still importable from Python but is no longer
driven by the CLI.

Prompt source files:

| File | Pipeline |
|------|----------|
| `src/llm_summary/code_contract/prompts.py` | Code-contract (current) |
| `src/llm_summary/summarizer.py` | Allocation pass (legacy) |
| `src/llm_summary/free_summarizer.py` | Free pass (legacy) |
| `src/llm_summary/init_summarizer.py` | Init pass (legacy) |
| `src/llm_summary/memsafe_summarizer.py` | Memsafe contracts (legacy) |
| `src/llm_summary/verification_summarizer.py` | Verification (legacy) |
| `src/llm_summary/leak_summarizer.py` | Leak detection (legacy) |
| `src/llm_summary/integer_overflow_summarizer.py` | Integer overflow (legacy) |
| `src/llm_summary/allocator.py` | Allocator / deallocator candidate detection |
| `prompts/indirect_call.txt` | Indirect-call resolution (legacy LLM mode) |
| `prompts/stdlib.txt` | Standard library seeding from man pages |

---

## Code-Contract Pipeline (current)

The code-contract pass produces a per-function, per-property Hoare-style
contract — `requires` / `ensures` / `modifies` plus an `analysis` chain-of-
thought, `notes`, and a self-rated `confidence`. A separate **verify** prompt
then checks the body against its own contract for property violations. Both
sides run **once per property** (`memsafe`, `memleak`, `overflow`); each
property gets its own scope rules and JSON, so the model is not asked to mix
concerns in a single response.

### System rules (shared by all summary properties)

A single `SYSTEM_PROMPT` precedes every per-property summary call. The hard
rules it enforces:

1. **Specific predicates; code-form preferred, prose OK when natural.** Pin
   concrete facts about named symbols. C expressions when they fit
   (`p != NULL`, `0 <= i && i < n`); compact value-range form for integers
   (`n: [0, INT_MAX]`); prose only when it compresses better than C
   (`s null-terminated; '\0' at index length-1`). Forbidden: formal-logic
   notation (`\valid(p)`) AND vague prose (`p is a valid string`).
2. **Caller-accessible names only — STRICT.** `requires` / `ensures` are
   read by the caller. Reference only this function's parameters, the
   literal `result`, out-parameter cells, globals/statics, and facts about
   memory the parameters point into. Body-locals (loop counters, walking
   pointers, clamp temporaries) are FORBIDDEN. If you can't restate a
   callee's requires in caller-visible terms, drop it.
3. **Callee discharge is VERBATIM and MANDATORY.** Every callee
   `requires[P]` clause must be either DISCHARGED at the callsite (cite a
   path fact) or PROPAGATED upward verbatim (optionally guarded by the
   path condition: `cond ==> phi`). Silent drops are not allowed. **You
   may not invent preconditions a callee did not declare.**
4. **No verdict.** Output only `requires` / `ensures` / `modifies` / notes /
   confidence. Whether the body satisfies its own contract is the verify
   pass's job.
5. **Conservative on uncertainty.** `requires` strengthen, `ensures` weaken,
   `modifies` over-include.
6. **Approximate when predicates grow unwieldy.** `ensures` may weaken;
   `requires` may strengthen. Flag the approximation in `notes`.
7. **`modifies` scope — stack + heap, not globals.** C zero-inits globals at
   startup, so reads-before-write on globals are well-defined. Track only
   stack locals, malloc/realloc heap, and out-parameters.
8. **External / harness functions are summarised for you.** Stdlib +
   harness callsites are inlined as `// >>> callee contract for P:` comment
   blocks. Trust them verbatim; do not re-derive from the name.
9. **Noreturn callees cut the path.** A callsite annotated
   `// >>> noreturn: true` does not return; downstream code may assume the
   negation of any guard that gated the noreturn call.

### Per-property summary prompts

Each property gets its own user prompt with a hard scope clause and method
guidance — the model sees only the rules relevant to that property.

#### `memsafe`

In-scope: null deref, out-of-bounds access, use-after-free, double-free,
init-before-read. Predicate kinds allowed: pointer validity, buffer bounds,
init state, alloc/free state. Integer-overflow and leak predicates are
explicitly excluded.

Method walks the body twice — once for what the caller must guarantee
(emit `requires`), once for what the body establishes (emit `ensures`):

- Unguarded deref → `p != NULL` in `requires`.
- Buffer access at `buf[expr]` → decompose `expr`, emit
  `malloc_size(buf) >= max_index` (or `... >= offset + len` for memcpy/
  memset). For struct-field access patterns (`ctx->buf[ctx->pos+ctx->len]`),
  emit a relational invariant (`ctx->pos + ctx->len < ctx->capacity`).
- Read of `*p` → `*p initialized`. **Write-only deref like `p->f = v` does
  NOT obligate `*p initialized` — only `p != NULL`.**
- `malloc(N)` returned/stored → `malloc_size(result) == N` (always cite N;
  bare "result is heap-allocated" is too weak).
- `p = malloc(); if (!p) abort();` → `result != NULL` in `ensures`.
- `free(p)` (or deallocator) → `freed(p)` in `ensures`.
- `s[k] = '\0'` → publish the terminator position so downstream `cstrlen`
  loops can be shown to terminate.

Output JSON: `{analysis, requires[], ensures[], modifies[], notes,
confidence}`. `analysis` comes FIRST (2–4 sentences walking the key
operations); `confidence` (`high|medium|low`) comes LAST.

#### `memleak`

In-scope: every heap allocation either (a) released within the function on
every return path, (b) returned to the caller, or (c) stored in a
caller-visible location. Predicates speak only of resource ownership:
`caller releases <sym>`, `acquires fd: caller must close`, `all
acquisitions released`, propagated callee `requires[memleak]`. Pointer
validity and integer predicates are out of scope here.

Same `{analysis, requires[], ensures[], modifies[], notes, confidence}`
JSON shape.

#### `overflow`

In-scope: signed `+ - * ++ -- unary-`, `/` `%` (zero-divisor + `INT_MIN/-1`),
shifts (`<< >>` with negative or out-of-range amount, left-shift of
negative). Predicate kinds: signed-overflow ranges (`x != INT_MIN`,
`n <= INT_MAX/2`), divisor-nonzero, shift bounds, value-range results.
Pointer-validity and init-state predicates are explicitly excluded.

The prompt embeds the **C-semantics reminders** the model must apply
*before* flagging:

- **Integer promotion** to `int` for narrower operands.
- **Unsigned wrap is well-defined** — never flag wrap on unsigned types.
- **Literal types** follow C rules; `-2147483648` is `unary-minus` applied
  to a literal whose type is fixed first.
- **Compare against the result-type range.** Worked example:
  `65536 * -32768 == INT_MIN` lands on the boundary, IS representable,
  NOT overflow.
- **Unsigned-to-signed within-range cast** is well-defined; don't flag
  `(int) strlen(s)` when the value obviously fits.

The prompt also includes a **data-model preamble** (`LP64` or `ILP32`)
rendered by `data_model_note()`; the verify pass uses the same preamble so
both sides agree on type widths. Two output forms accepted: code
predicate (`x >= 1 && x <= 100`) or value-range (`x: [1, 100]`).

**Published output bounds**: when the function writes a derivable narrow
range to a caller-visible location (struct field, out-param), publish it
in `ensures` so callers reading that location for shifts/comparisons can
discharge their own `requires`.

### Verify prompts

After the summary lands, a per-property verify call reads the body again,
this time with the function's own contract pinned and each callsite
annotated with the callee's `requires` / `ensures` / `modifies` (built by
`build_callee_block` in `code_contract/inliner.py`). The verifier emits an
`issues[]` list with `{kind, line, analysis, is_ub}`; only `is_ub: true`
entries become real findings.

Verify rules:

1. **Assume `requires` hold on entry** — never report a "missing
   precondition"; if it's missing, the body must establish it before the
   relevant op, otherwise that op is a violation.
2. **Discharge callee `requires[P]` at every callsite** — if it may not
   hold given the path facts, emit a `callee_requires` issue.
3. **Stay inside the property** — memsafe issues only in the memsafe
   verifier, etc.
4. **Be specific** — cite a concrete operation, not a category.
5. **Trust inlined external/harness contracts** verbatim, including
   `__attribute__((noreturn))` annotations.
6. **Noreturn callees cut the path.**
7. **Empty issue list is correct when the body is safe.**
8. **Globals / statics are never NULL** when annotated `// global` in the
   type-defs section.

In-scope kinds per property:

| Property | Kinds |
|----------|-------|
| `memsafe` | `null_deref`, `buffer_overflow`, `use_after_free`, `double_free`, `invalid_free`, `uninitialized_use`, `callee_requires` |
| `memleak` | `memory_leak`, `callee_requires` |
| `overflow` | `integer_overflow`, `division_by_zero`, `shift_ub`, `callee_requires` |

The memsafe verifier additionally watches for **unsigned wrap in bounds
guards** (e.g. `if (pos + len >= space)` on `size_t` — wrap can falsify the
guard and skip a realloc, especially on ILP32 where `pos + len` can exceed
`SIZE_MAX`).

### Front-end warning channel

When the scan compile produced clang `-Wall` diagnostics for the function,
the source is prefixed with a `// FRONTEND WARNINGS (clang -Wall) — assess
feasibility:` block. Each verifier treats those warnings as **candidate
issues** and decides feasibility from the path facts; the overflow verifier
in particular keeps constant-folded UB (`INT_MAX + 1`) flagged as
`integer_overflow` even when no `add` instruction survives in IR.

### Inlining shortcut and SCC handling

Two non-prompt features influence what the model sees but are worth
mentioning since they show up in the prompt context:

- **`inline_body`**: when a small function has a single project caller and
  isn't part of an SCC, the body is pasted verbatim at every callsite
  (`// >>> body of <name>:`) instead of being summarized. The model sees
  the inline source rather than a contract block.
- **SCC convergence**: recursive groups iterate the per-property prompt
  until a change-detector LLM judges the contracts have converged.

### Struggle scoring + auto-retry

After parsing each per-property response, a struggle scorer
(`code_contract/struggle.py`) measures hedge density and length anomaly.
When the score crosses a threshold, the property is auto-retried once on a
stronger backend (Claude). The struggle scores and `retried` / `retry_model`
columns land in `code_contract_summaries` for downstream triage.

### Malformed-JSON retry

The pass keeps the conversation context and retries once when JSON parsing
fails on the first response — much cheaper than re-priming the cache.

---

## Allocator / Deallocator Candidate Detection

The build phase asks the model to confirm whether a given function is an
allocator or deallocator before it gets a stdlib-style contract.

**Source:** `src/llm_summary/allocator.py` (`ALLOCATOR_PROMPT`,
`DEALLOCATOR_PROMPT`).

This is heuristic-pre-filtered — only candidates flagged by static signals
(`malloc`/`calloc`/`realloc` reachable, return-pointer pattern, etc.) are
sent. The prompt asks for a single JSON answer with the allocator role
(direct, wrapper, etc.), the size argument index when applicable, and a
confidence rating. Confirmed candidates feed downstream V-Snapshot
construction and the legacy allocation pass.

---

## Allocation Pass (legacy)

**Source:** `src/llm_summary/summarizer.py` — `ALLOCATION_SUMMARY_PROMPT`,
`ALLOCATION_SYSTEM_PROMPT` + `ALLOCATION_USER_PROMPT`,
`ALLOC_TASK_PROMPT`, `BLOCK_ALLOCATION_PROMPT`.

Three template families exist for the same prompt:

- **Single-message** (`ALLOCATION_SUMMARY_PROMPT`) — no caching, source +
  task in one user turn.
- **Cache mode "instructions"** — task instructions in the system message
  (cached across all functions in a pass), source in the user message.
- **Cache mode "source"** — source in the system message (cached across
  passes for the same function), task in the user message.

The instructions block (`_ALLOC_INSTRUCTIONS`) is the single source of
truth shared by all three families.

What the prompt asks for:

1. **Allocations and returned pointers** — heap allocations AND non-heap
   returned-pointer provenance (`static`, `parameter_derived`,
   `escaped_stack`). Each entry carries `type`, `source`, `size_expr`,
   `size_params`, `returned`, `stored_to`, `may_be_null`. The
   `may_be_null` field is set false when the function null-checks the
   allocation and calls a noreturn function on the failure path, OR when a
   callee summary marked the allocation `[never null]`.
2. **Parameters** — role (`size_indicator`, `buffer`, `count`,
   `pointer_out`, …) and whether each affects allocation size.
3. **Buffer-size pairs** (post-condition only) — `param_pair`,
   `struct_field`, or `flexible_array` relationships this function
   *establishes* between a buffer and its size. Pairs the function merely
   reads are out of scope.
4. **Description** — one-sentence summary.

Hard rules added since the doc was first written:

- **Enumerate every distinct allocation site individually** — no collapsing.
- **Callee propagation** — propagate callee allocations stored to
  caller-visible locations as your own. Don't drop callee allocs just
  because this function doesn't allocate directly.
- **No false stack reports** — local arrays / compound literals / static
  const tables are NOT allocations unless they escape (returned/stored).

Sub-block prompt (`BLOCK_ALLOCATION_PROMPT`) is used for switch-case
chunking of large functions: the model summarizes one case at a time and
proposes a synthetic name + signature for the chunk; the parent's summary
references the per-block results via `function_blocks`.

JSON schema (`_alloc_json_schema`) is rendered with `{{` / `{{{{` brace
escaping depending on whether the template is single-format or double-
format (the cache-source variant is double-format because the schema is
already inside an outer `.format()` substitution).

Example response is unchanged from the early version of this doc:

```json
{
  "function": "create_buffer",
  "description": "Allocates n+1 bytes of heap memory for a null-terminated buffer.",
  "parameters": {
    "n": {"role": "size_indicator", "used_in_allocation": true}
  },
  "allocations": [
    {
      "type": "heap",
      "source": "malloc",
      "size_expr": "n + 1",
      "size_params": ["n"],
      "returned": true,
      "stored_to": null,
      "may_be_null": true
    }
  ],
  "buffer_size_pairs": []
}
```

---

## Free Pass (legacy)

**Source:** `src/llm_summary/free_summarizer.py` — `FREE_SUMMARY_PROMPT`,
`FREE_SYSTEM_PROMPT` + `FREE_USER_PROMPT`, `FREE_TASK_PROMPT`,
`BLOCK_FREE_PROMPT`. Same three-template structure as the allocation pass.

Key changes since the original doc:

- **`frees` and `resource_releases` are split.** Heap deallocations
  (`free`, `realloc(p, 0)`, `munmap`, `xfree`/`g_free` wrappers, `fclose`,
  `closedir`) go in `frees`; non-heap cleanup (`close(fd)`, `sem_destroy`,
  `pthread_join`, …) goes in `resource_releases`. Internal stdio buffer
  management (`fprintf` may realloc its buffer) is intentionally NOT
  reported — implementation detail, not a free.
- **C++ `this`** is treated as an implicit pointer parameter; `delete this`
  and `delete m_data` (which is `this->m_data`) get the matching
  `target_kind`.
- **Per-op `description`** for loop-based or transitive frees ("frees all
  elements in a linked list", "frees all entries in a hash table"). Omit
  for simple single-pointer frees.
- **`condition`** field on conditional frees is now caller-observable
  only — only function parameters and/or `result`. Over-approximation is
  fine; soundness > precision. Internal locals and callee results are
  forbidden.
- **`fclose` / `closedir`** are explicitly listed in the deallocator
  examples (they free the FILE\*/DIR\* heap object).
- **Caller-visible abstraction**: if `cleanup(ctx)` frees three fields,
  report ONE entry (`target=ctx`, `deallocator=cleanup`); the callee's own
  summary captures the per-field detail. Direct frees alongside the
  cleanup call are reported separately.
- **Conditional collapse**: when a callee's free is gated by an arg the
  caller sets to a constant, the prompt instructs the model to *omit* the
  free if the constant evaluates the guard false, or mark it
  unconditional if always true.
- **Enumerate every distinct free site individually** — no collapsing.

---

## Init Pass (legacy)

**Source:** `src/llm_summary/init_summarizer.py` — `INIT_SUMMARY_PROMPT`
plus the `_INIT_INSTRUCTIONS` block shared across cache modes, and
`BLOCK_INIT_PROMPT` for chunked large functions.

Asks for caller-visible initializations only (output parameters, struct
fields written via a parameter or `this`, return value). Per-init fields:
`target`, `target_kind`, `initializer`, `byte_count`, `conditional`,
`condition`.

Changes since the original doc:

- **Return-value rule**: if every exit returns a value (including NULL,
  0, error codes), the return value IS unconditionally initialized — a
  function that returns NULL on error and a pointer on success still
  always initializes its return.
- **`byte_count` normalization**: must be a concrete expression from the
  code (`n`, `len`, `sizeof(int)`, `count * sizeof(T)`) or `null`. Vague
  values like `"full"` are rejected and normalized to `null`.
- **Output value ranges** added to the response — narrower-than-type
  ranges for return values and out-params (e.g. a loop counter bounded
  by `i < MAX` published as `[0, MAX)`). Skip tautological full-type
  ranges.
- **Noreturn detection** added to the response — `noreturn: true` if every
  path ends in a known-noreturn call (`abort`, `exit`,
  `__builtin_unreachable`, callee with `__attribute__((noreturn))`) or an
  infinite loop. `noreturn_condition` for path-conditional cases.
- **Callee propagation**: if a callee initializes a caller-visible
  location, propagate the init.
- **Caller-observable conditions only** — same rule as the free pass.
- **C++ `this`** member writes recognized as field inits.

---

## Memsafe Contracts Pass (legacy)

**Source:** `src/llm_summary/memsafe_summarizer.py` —
`MEMSAFE_SUMMARY_PROMPT`, `BLOCK_MEMSAFE_PROMPT`. Generates safety
**pre-conditions** (Pass 4 of the legacy 5-pass pipeline).

Contract kinds:

- `disallow_null` — pointer dereferenced (`*p`, `p->f`, `p[i]`) without a
  prior NULL check on the same path. C++ `this` is implicitly dereferenced
  whenever a member field is accessed.
- `allow_null` — pointer NULL-checked before any deref. Carries an
  optional `condition` for cases where the null tolerance only holds in a
  specific branch (e.g. `n == 0`).
- `not_freed` — pointer passed to `free` / a deallocator; must point to
  live heap.
- `buffer_size` — pointer used with `memcpy` / `memset` / array indexing;
  must have sufficient capacity. Carries `size_expr` (use **numeric
  values** for compile-time constants, NOT macro names) and `relationship`
  (`byte_count` / `element_count`). Optional `condition`.
- `initialized` — variable/field whose value is **read** before being
  written. **Write-only deref like `p->field = val` is NOT
  uninitialized use** — it only needs `disallow_null`.

Notable refinements since the original doc:

- The pass renames `not_null` → `disallow_null` and `nullable` →
  `allow_null` to make the producer/consumer direction explicit.
- Callee pre-conditions are inlined as `/* PRE[callee(actual_args)]: */`
  comments above each callsite; the prompt instructs the model to apply
  formal→actual substitution and propagate any unsatisfied callee
  obligation as the function's own contract.
- An optional `{alias_context}` slot is rendered when V-Snapshot has
  produced may-alias info; aliasing locations / parameters are listed so
  the model can reason about them. (TODO: same hook is planned for the
  code-contract pass — see `docs/architecture.md` §10.)

---

## Verification Pass (legacy)

**Source:** `src/llm_summary/verification_summarizer.py` —
`VERIFICATION_PROMPT`, `BLOCK_VERIFICATION_PROMPT`.

Hoare-logic-style checker: **assume the function's own pre-conditions hold**,
walk the body statement by statement, and only flag something as an issue
if it can occur even with all pre-conditions satisfied. Tracks
pointer/buffer state (null, non-null, freed, initialized, allocated size)
**and** integer value ranges (bounds from checks, assignments, return
values).

Output JSON: `simplified_contracts` (subset of Pass-4 contracts NOT
satisfied internally — these propagate upward to callers) plus `issues`.

Issue kinds (`_VALID_ISSUE_KINDS`): `null_deref`, `buffer_overflow`,
`use_after_free`, `double_free`, `uninitialized_use`, `invalid_free`,
`memory_leak`, `integer_overflow`. The pass's leak / overflow detection
has since been split into dedicated passes (see below); the verification
prompt still enumerates the kinds for backward compat.

Changes since the original doc:

- **Hoare-logic state tracking** is the new method (replaced the older
  pattern-checklist style).
- **Drops pre-conditions for entry functions** — entry-point obligations
  belong to the program's caller-of-main, not the function.
- **Trust callee contracts**: an explicit "do NOT override this with
  general knowledge" rule. If `PRE[callee(...)]: no pre-conditions`,
  passing NULL is safe — the callee's summary said so.
- **Unchecked `may_be_null` return dereferenced → `null_deref`** —
  explicit rule because the early version often missed this.
- **Indirect call via NULL function pointer → `null_deref`** explicitly
  named.
- **Type-defs section** prepended to the source (typedefs, structs, file-
  scope statics) so the model can resolve narrow-type bounds without
  guessing.
- **Macro-expanded source** (`pp_source` from `clang -E`) used by all
  passes; PRE/POST annotations are placed at the original-source
  callsites via `llm_source` + expression matching.
- **`{own_alloc_free_section}`** lifted from the leak pass's simplified
  alloc/free list, so the verifier sees the same compositional facts the
  leak pass uses.
- **Frontend warnings** (clang `-Wall`) are passed through as candidate
  issues; the verifier judges feasibility.
- **Severity** rubric: `high` = unconditional, `medium` = specific path,
  `low` = error path.

The "Not a bug if" reminders include: guarded by runtime check, covered
by the function's own pre-conditions, guaranteed by callee post-condition,
or static/global (C zero-init at startup).

---

## Memory Leak Pass (legacy)

**Source:** `src/llm_summary/leak_summarizer.py` — `LEAK_SYSTEM_PROMPT`,
`LEAK_USER_PROMPT`. Added as a dedicated pass after the original doc
(commit `6c52add`) and now also seeds the verifier's
`{own_alloc_free_section}`.

Compares allocation summaries against free summaries to find unmatched
allocations. A heap allocation leaks when it is **not** freed before
return AND **not** returned to the caller AND **not** stored to a
caller-visible location. For entry-point `main()`, storing to a global
does NOT prevent a leak — there is no caller to observe it.

Output JSON has three lists:

1. **`leaks`** — bug entries with `allocation`, `stored_to`, `reason`,
   `severity`.
2. **`simplified_allocations`** — allocations NOT freed internally that
   ARE returned/stored. These are not leaks here, but callers must release
   them. Annotated with nullability so downstream callers know whether to
   null-check.
3. **`simplified_frees`** — frees of caller-provided pointers (parameters,
   struct fields passed in) — tells callers what this function frees.

Callee propagation: the prompt explicitly instructs the model to take
"unfreed allocations (caller must handle)" from each callee summary as
THIS function's responsibility, match against this function's frees, and
either resolve, propagate, or report.

Indirect-call targets are annotated in the callee section so the model
can match through function pointers.

---

## Integer Overflow Pass (legacy)

**Source:** `src/llm_summary/integer_overflow_summarizer.py` —
`OVERFLOW_SYSTEM_PROMPT`, `OVERFLOW_USER_PROMPT`. Split out of the
verification pass (commit `d4eef8f`). The current code-contract pipeline
folds overflow back in via its own `overflow` property — the legacy
overflow pass is preserved for the multi-pass driver.

In-scope: signed overflow (`+ - * ++ -- unary-`), division/modulo by zero
(plus `INT_MIN/-1` and `INT_MIN%-1`), shift UB. **Out of scope and
explicitly NOT reported**: unsigned wrapping, integer promotion artifacts,
unsigned-to-signed within-range casts, literal type widening, well-defined
edge values like `INT_MIN`, guarded arithmetic, dead/infeasible paths,
lines with a `// safe` IR-prover hint.

Method: value-range tracking through branches, arithmetic, callee output
ranges, and callee constraints.

Output triple:

1. **`constraints`** — pre-conditions on this function's parameters
   (`target`, `range`, `description`). Only those NOT internally guarded.
2. **`output_ranges`** — return value + out-param ranges. Always include,
   even when equal to the full type range — callers need this to detect
   downstream overflow.
3. **`issues`** — concrete UB findings with operand ranges cited.

Data model: defaults to LP64, with explicit ILP32 fallback when the
source declares it.

---

## Indirect Call Resolution

**Source:** `prompts/indirect_call.txt` (legacy LLM-based mode).

Used to determine likely targets for a function-pointer call. The current
batch flow uses **KAMain**'s CFL-reachability points-to analysis instead
(see `docs/indirect-call-analysis.md`); this prompt is retained for cases
where an LLM-only fallback is desired.

Inputs:

```
1. Indirect call site context (caller, expr, signature, location)
2. Candidate functions (address-taken with matching signature, with source snippet)
3. Address flow information (where the function's address has been observed
   flowing to: fields, params, arrays)
```

Output: `{"targets": [{"function", "confidence", "reasoning"}]}`. The
prompt asks the model to assess naming/purpose match, evidence the
pointer can reach this value, and a confidence rating. Empty targets is
the right answer when no candidate fits.

---

## Standard Library Prompt

**Source:** `prompts/stdlib.txt`.

Used to expand stdlib coverage by parsing man pages or header comments.
The model identifies functions that allocate memory (or release it) and
emits a JSON summary in the same shape as the in-project allocation /
free summaries, which is then loaded into the global `stdlib_cache.db`
(see `docs/architecture.md` §11). Hardcoded contracts in
`code_contract/stdlib.py` always win over cache entries.

---

## Prompt Engineering Patterns Used

Patterns the codebase has converged on after iteration on Qwen / Haiku /
Llama.cpp:

1. **Per-property prompts beat omnibus prompts.** Splitting `memsafe` /
   `memleak` / `overflow` into separate calls was a measurable win on
   small/local models.
2. **Verbose per-property prompts beat short-tail prompts.** A/B testing
   (`/tmp/cache_ab_findings.md`) showed Qwen drops "no resource
   acquired" trivial ensures with the short tail. The repo standardizes
   on the verbose form because small/fast models are the primary target.
3. **Analysis-first, confidence-last.** Putting the chain-of-thought
   `analysis` field at the top of the JSON (before the contract clauses)
   reliably improves quality vs. asking for it after. `confidence`
   self-rating goes at the end and is honestly under-rated — low-conf
   summaries get auto-retried on a stronger model.
4. **JSON response schemas for every pass.** Each call uses
   `make_json_response_format(...)` so the backend constrains the
   output. Prompts also include enum constraints inline as fallback for
   backends without schema support.
5. **Examples in prompts use the source language's syntax.** No
   `\valid(p)` / `\forall x` formal-logic notation; no over-compressed
   prose like `p is a valid string`.
6. **Negative anchoring.** Each prompt lists what is OUT of scope so the
   model doesn't drift into another property's predicates.
7. **Inline callee context as comment lines.** Both pipelines inline
   callee `requires` / `ensures` / `modifies` (and noreturn) at each
   callsite as `// >>> ...` comments, so the model never has to
   cross-reference a separate "callee table" mid-prompt.
8. **Frontend warnings as candidate issues.** Clang `-Wall` diagnostics
   from the scan compile are forwarded to the verifier as a feasibility
   question, not a verdict.
9. **Robust JSON parsing.** Responses are scanned for `\`\`\`json` fenced
   blocks first (preferring the LAST block — many models echo a partial
   draft before the final answer), with a raw-JSON fallback and a repair
   step for common malformations.
10. **Cache-mode templates.** Each legacy pass exposes three template
    families — single-message, "instructions" (system cached), and
    "source" (per-function source cached) — so the runner can pick the
    right cache strategy for the backend.
11. **Auto-retry on bad JSON.** The code-contract pass retries once with
    the conversation context preserved; cheaper than re-priming the
    cache.

### Response Parsing

```python
def _parse_response(self, response: str) -> Summary:
    # Prefer the LAST ```json ... ``` block (many models echo drafts).
    matches = list(re.finditer(r"```json\s*(.*?)\s*```", response, re.DOTALL))
    if matches:
        json_str = matches[-1].group(1)
    else:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        json_str = json_match.group(0) if json_match else ""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return _try_repair(json_str)  # quote/comma fixups
```
