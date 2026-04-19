# LLM-First Verification: Redesign

## Mission

Secure code in the era of LLMs and agents. The compositional verification
pipeline is the load-bearing piece. This doc captures the redesign that
replaces the original 7-pass per-property pipeline with **Hoare-style
forward summaries (`requires` / `ensures`) per function**, expressed as
**ACSL/JML-shaped annotations on C source** ("code-as-summary"). Each
function is summarized by **per-property single-turn calls over a cached
source-prep block**. Bug-finding at entry points reads only the summary
chain; cross-function feasibility analysis (incorrectness logic) is a
deferred layer.

Scope for the first prototype: **memsafe + memleak (+ overflow as
secondary)**. The 10-property taxonomy was a designer's wish list, not
a thing we know how to support soundly across the model tiers we care
about. Two-to-three well-defined property families are enough to test
the architecture and easier to keep honest.

This is a research project, not a production system. **No backward
compatibility constraints.** We rewrite anything that's wasteful and
delete anything that's no longer needed.

## Principles

1. **Use the LLM where it's uniquely valuable** — semantic intent,
   predicate generation, judgment under ambiguity, recognizing
   project-specific patterns. Don't pay LLM tokens for fact extraction
   the IR can do precisely.

2. **Don't pay LLM tokens for what static analysis already does well** —
   call graph, def-use, dominance, basic ranges, side-effect summaries,
   alias sets. We have libclang and KAMain. Use them.

3. **Code is the summary.** Empirically validated: ACSL/JML-shaped
   annotations on C source took small-model (Haiku, Qwen) false-negative
   rate to 0 in the format A/B test. Predicates are C-expressions over
   program variables, parameters, and globals — a notation LLMs have
   seen extensively in pretraining. Reserve NL for *intent* and
   *justification* fields, not facts. If a summary becomes longer than
   the body it summarizes, inline the body.

4. **Compositional discipline.** A summary is the function's pre/post,
   full stop. **No verdict.** No internal mechanism. The summary is
   what the caller sees and reasons over; everything else is hidden.
   When `f` calls `g`, `f`'s summary is computed against `g`'s pre/post
   *verbatim* — `f` may not strengthen, weaken, or invent `g`'s
   preconditions. (Note: empirically the canary `Context != NULL`
   hallucination is fixable by prompt cleanup alone — see "What the
   experiments showed". The verbatim-discharge rule is still useful
   discipline for deeper call graphs and is cheap to enforce.)

5. **Hoare-style forward summaries, not bug search.** Each pass walks
   forward from the entry of the function and answers narrow questions:
   "what must hold on entry for property P?" / "what is established on
   exit?". No backward chaining, no "find me all the bugs", no global
   reasoning. Bug-finding (incorrectness-logic-style reachability +
   triggering condition propagation) is a separate, later layer; it
   needs feasibility analysis the Hoare summaries alone cannot
   provide.

6. **Scope to memsafe + memleak (+ overflow).** Each family has
   grounded, IR-checkable operations (load/store, malloc/free, arith)
   to anchor the model. Properties whose IR signal is weaker (uninit,
   shift_ub, etc.) are deferred until we have evidence the pipeline
   handles them reliably.

7. **Local models must work — and they do, with code-as-summary.**
   Qwen-coder 30B / Llama 3.1 / Haiku-class are first-class targets and
   the *primary* deployment target. The format change to code-as-
   summary unlocks them (0 FN on the test set). Per-property single-
   turn calls with a cached source-prep block is the call shape; multi-
   round Q&A was tested and rejected.

8. **SA-driven property gating.** Don't ask property questions a
   function can't answer. SA features (KAMain JSON sidecar) tell us
   which operations the function performs; callee summaries tell us
   which preconditions need discharging. Together they produce the
   *applicable property set* per function — most simple functions skip
   all three property calls.

9. **Patch-assist is a first-class output.** Every reported violation
   carries enough information for a downstream patcher (LLM or human)
   to write the fix without re-querying source.

10. **Detection-assist is the same artifact.** A small model given a
    callee's `requires` / `ensures` and the caller's source should be
    able to flag callsites that violate the contract. The summary
    format must support this without invented JSON taxonomies. Code-
    form predicates (`p != NULL`) pattern-match more reliably than
    enum slots (`contract_kind: disallow_null`).

## What the experiments showed

Two rounds of A/B testing on `floppy_simpl3.cil-1` and
`floppy_simpl3.cil-2` settled the load-bearing design questions.

### Round 1: format A/B (yesterday)

Compared the legacy JSON contract format against ACSL/JML-shaped
annotations on C source ("code-as-summary"), with the per-property
single-pass `scripts/contract_pipeline.py` prototype.

- **Code-as-summary takes small-model FN to 0.** Haiku and Qwen both
  reached 0 false negatives on the property test set with the new
  format. The format change is the single largest quality lever.
- The previous prototype's residual issues (verdict cascade, accidental
  TPs from int-as-pointer hallucinations) sit on top of the format
  question, not under it.

### Round 2: pipeline shape A/B (today)

Compared three call shapes per (function, property) on Sonnet 4.5
(Vertex `rawPredict`) and llama.cpp / Qwen3.5-27B:

- **Mode 1**: uncached single-turn, full per-property prompt.
- **Mode 2**: cached system + source-prep, short property tail.
- **Mode 3**: cached, multi-turn (enumerate → derive → emit).

Findings:

- **Multi-turn (Mode 3) is dead.** Strictly dominated on both backends:
  +266% wall time on Sonnet, +119% on llama.cpp, no quality win. Do
  not build a multi-round Q&A driver.
- **Per-property single-turn over cached source is the right call
  shape.** Caching the SYSTEM_PROMPT + source-prep block once per
  function, then issuing one call per applicable property, amortizes
  the source cost across 2–3 property calls and keeps each individual
  call small enough for local models.
- **The `Context != NULL` canary is fixable by prompt cleanup alone.**
  Mode 2 on Sonnet drops the fabrication with no structural pipeline
  change — just a shorter property-tail with the rules in cached
  SYSTEM. The verdict-cascade ban from Principle 4 remains useful
  discipline, but it is no longer the load-bearing fix for the canary.
- **Verbose per-property prompts are the small-model-friendly default.**
  Sonnet improves with short tails (less over-fabrication). Qwen drops
  trivial ensures (`no resource acquired`) without explicit cuing.
  Since small/fast models are the primary target, **default to verbose
  per-property prompts**. Backend-aware prompt minimization (short
  tails for Sonnet) is deferred (see Open Questions).

### Why this beats the OLD pipeline

The OLD 7-pass design got per-property scoping right — that part is
preserved. What was wasteful: each pass re-derived calls / writes /
branches / ranges from source (~50–70% of token cost was redundancy);
preconditions were fragmented across three tables; each pass implicitly
did "find the bugs of kind P", which doesn't decompose well for local
models. Caching the source-prep block once per function eliminates the
redundancy in one move; forward-Hoare-only framing eliminates the bug-
search-per-pass overhead.

## Diagnosis of the original 7-pass design

The OLD design's right kernel was **per-property scoping** — each
prompt small enough for local models to handle. That kernel is
preserved.

What was wasteful or wrong:

- **Fact redundancy across passes.** Each pass re-read source to
  derive the same calls/writes/branches/ranges. Caching a single
  source-prep block once per function eliminates this.
- **Fact disagreement across passes.** No longer an issue with one
  shared cache feeding all property calls.
- **Contract fragmentation.** Preconditions in three tables. Replaced
  by one summary per function in code-as-summary form.
- **Bug search per pass.** Each pass implicitly searched ("find the
  null-deref bugs"). Replaced by forward-Hoare questions on the cached
  source.

## Division of labor

### Static analysis (no LLM tokens)

| Fact | Source | Notes |
|---|---|---|
| Function definitions, signatures, callsites | libclang | existing |
| Direct call graph | libclang/IR | existing |
| Indirect call targets | KAMain + LLM resolver | existing |
| Pointer aliases per function point | KAMain V-snapshot | existing |
| SSA def-use, dominance | LLVM IR | KAMain extension |
| Memory ops with operands | LLVM IR | KAMain extension — `load/store/GEP/memcpy/free/alloca/call` |
| Control flow guards | LLVM IR | KAMain extension — branch + dominance |
| Allocator/free callsite enumeration | IR + project allocator list | KAMain extension |
| Property-gating features | per `plan-pass-gating.md` | drives which properties to ask about |

### LLM (kept where uniquely valuable)

- **Predicate writing** — turning SA-extracted operations into
  `requires` / `ensures` lines as C-expressions.
- **Discharge judgment** — deciding whether a callee's `requires` is
  satisfied at this callsite given the path facts.
- **Indirect call resolution** — as today.
- **Custom allocator/container detection** — as today.
- **Build-learn agent** — as today.

The summary pass is the only LLM-heavy work; everything else is
data-driven aggregation.

## Architecture

```
Phase 0: build-learn          (LLM ReAct agent — unchanged)
Phase 1: scan                 (libclang — unchanged)
         functions, callsites, AST-level facts → functions.db
Phase 2: KAMain analysis      (lives in KAMain repo — extended)
         ├─ call graph + V-snapshot (existing)
         └─ NEW per-function IR facts emitted as JSON sidecar:
            memory ops, guards, SSA facts, alias annotations,
            property-gating features
Phase 3: summary              (NEW — per-property single-turn over
         pass                  cached source, per function in
                               llm-summary; bottom-up over the call
                               graph)
         input:  source + Phase 2 SA facts + callees' summaries
                  (annotated inline at each callsite)
         output: per-property `requires` / `ensures` lines
                  attached to the function's source
Phase 4: entry-point check    (NEW — Hoare-only at the entry)
         ├─ aggregate summaries across the call graph
         ├─ for each entry function, scan its `requires`:
         │    every non-trivial entry pre = potential bug
         └─ emit per-entry verdict + patch-ready obligation reports
Phase 5: surface              (existing eval / triage / harness;
                               rewired to read summaries)
         issue triage UI, ucsan harness gen, patch handoff,
         small-model violation detector consuming summaries
```

IL-style feasibility analysis for tightening Phase 4 FPs is a future
phase, not in this design.

**Repo split**: Phase 2 IR fact extraction lives in KAMain (small
extension to the existing CFL/V-snap pass — emits JSON sidecar next to
`.bc`, no DB writes; per `docs/todo-kamain-ir-sidecar.md`). The
llm-summary repo consumes the sidecar.

## Summary format

A function's summary is the function's source, with annotation lines
attached. Inspired by ACSL/JML — a format LLMs have seen in pretraining.

```c
// @requires[memsafe]: <inherits KeSetEvent.requires[memsafe]>
// @requires[memleak]: true
// @ensures[memsafe]: setEventCalled == 1
// @ensures[memleak]: nothing acquired
// @modifies: setEventCalled
// @returns: -1073741802
int FloppyPnpComplete(int DeviceObject, int Irp, int Context) {
  KeSetEvent(Context, 1, 0);
  return -1073741802;
}
```

For an unconditionally non-returning function:

```c
// @requires[memsafe]: true
// @requires[memleak]: true
// @ensures: ⊥                  // does not return; control reaches reach_error
void errorFn(void);
```

Three rules:

1. **Predicates are C-expressions** over program variables, parameters,
   and globals. `p != NULL`, `len <= sizeof(buf)`, `setEventCalled == 1`.
   Not `\valid(p)`, not `∀x . P(x)` — small models can't reliably parse
   formal logic notation.

2. **`requires` lines come from the function's own operations PLUS its
   callees' `requires`**. A callee's `requires[P]: φ` either is
   discharged at the callsite (proof by some path fact) or becomes
   this function's `requires[P]: φ` (verbatim, possibly with the
   callsite's path-condition prepended). Callee `requires` cannot be
   invented or strengthened by the caller. When the function inherits
   purely, prefer reference-by-name (`<inherits callee.requires[P]>`)
   over verbatim restatement to keep summaries sub-linear in subtree
   size.

3. **No verdict in the summary.** There is no "this function is safe /
   unsafe" tag. Whether the body satisfies its own contract is checked
   locally for diagnostics only and never cascaded to callers.
   `errorFn`'s `@ensures: ⊥` is a *fact* about its behavior — not an
   "unsafe" tag — and callers reason about it the way they reason
   about any other `ensures`.

For entry functions (`main` in sv-comp, exported APIs in libpng), the
same format applies. There is no caller to discharge `requires`, so
non-trivial entry `requires` lines are exactly the obligation reports
Phase 4 emits.

## Phase 3: per-property single-turn over cached source

Per function, in bottom-up order over the call graph:

1. **Compute the property set.** From the function's KAMain sidecar
   (which operations does it perform?) and its callees' summaries
   (which preconditions need discharging here?), determine which
   subset of {memsafe, memleak, overflow} the function actually
   touches. If empty, emit the empty summary and move on. Most simple
   functions land in {} or {memsafe}.

2. **Build the cached prompt.** Two cached blocks:
   - **System** (cached, session-stable): generic instructions —
     summary format rules, code-as-summary conventions, verbatim-
     discharge discipline.
   - **Source-prep user block** (cached, per-function): the function's
     source annotated with inline callee `requires` / `ensures` at
     each callsite, plus relevant SA facts from the KAMain sidecar.

3. **Issue one verbose property call per applicable property.** The
   property tail is the full `PROPERTY_PROMPT` for that family
   (memsafe / memleak / overflow), explicitly cuing trivial ensures
   (`no resource acquired`, range bounds, etc.) — small models need
   these cues. Each call asks for `requires` / `ensures` / `modifies`
   in code-as-summary form for that property only. Two or three
   property calls share both the system cache and the source-prep
   cache.

4. **Aggregate.** The per-property answers are appended to the
   function's annotated source as the summary. Stored as one row in
   `summaries`.

5. **Local-only verdict for diagnostics.** Optionally re-prompt:
   "given this function's `requires` and `ensures`, does the body
   satisfy them?". The answer is logged for debugging and never
   written into the summary store. This is the only place "verdict"
   exists, and it is invisible to callers.

### Backend integration

- **Anthropic backends**: `cache_control: ephemeral` markers on the
  system block (session-stable) and source-prep block (function-
  scoped). Property calls reuse both.
- **llama.cpp**: rely on the default slot KV cache (`cache_prompt:
  true`). Hold one persistent HTTP session per function so the slot
  stays warm across the property calls.

A small KV-cache abstraction in `llm/` hides the per-backend mechanics.

### Why this satisfies the four design goals

- **Concise.** Each function ships only the `requires` / `ensures`
  lines it actually has. Most functions have 0–2 of each. The summary
  is strictly smaller than the body it abstracts.
- **Sufficient.** Every callee precondition is either discharged at
  the callsite (by a cited path fact) or propagated verbatim.
  Conservative defaults on uncertainty (`requires: <verbatim callee
  req>`, `ensures: top` / `modifies: *`) preserve soundness.
- **Small-model-friendly.** Per-property scoping + verbose property
  prompts + cached source. Each call is small and explicitly cued.
- **LLM-friendly.** ACSL/JML-shaped annotations on C source, no
  invented taxonomy, predicates are C-expressions the model can read
  and check against the body.

## Phase 4: entry-point check

Hoare-only, mechanical, no LLM call required for the common path.

For each entry function (e.g., `main`, exported API):

1. Read its summary's `requires` lines.
2. Each `requires[P]: φ` where `φ ≢ true` is an undischarged
   obligation → potential violation of P at this entry.
3. Emit a report per non-trivial `requires`: which property, which
   predicate, the witness chain (which callsite in the entry's body,
   propagated from which callee, ultimately rooted at which leaf
   operation).

This is sound for bug-finding (every reported obligation is real) and
conservative for false-positive rate (an obligation might be
discharged by some run-time fact the analysis doesn't see — argv
structure, sv-comp `__VERIFIER_nondet_*` ranges).

Tightening the FP rate requires **incorrectness logic**: under-
approximating reachability + triggering condition propagation to
verify "is there actually a feasible state where this obligation is
unmet?". That is a planned future phase, not in this design.

## Predicate language

Compromise between LLM-friendliness and discharge-mechanizability:

- **Numeric**: `x op c`, `x op y` for `op ∈ {<, ≤, =, ≥, >, ≠}`,
  arithmetic over int constants. Discharge: ConstantRange / interval
  intersection.
- **Pointer typestate**: `not_null(p)`, `allocated(p)`, `freed(p)`,
  `initialized(p, n)`. Discharge: typestate machine + alias info.
- **Aliasing**: `disjoint(p, q)`, `aliases(p, q)`. Discharge: KAMain
  alias set.
- **Free-form** (escape hatch): `<C-style expression>`. Discharge:
  LLM judgment, recorded with confidence.

## What gets deleted

- 7 per-property summarizer/pass/table triples → one `summaries`
  table.
- `simplified_contracts` becomes redundant — entry-point check
  produces the remainder directly.
- Per-property prompts in `llm-prompts.md` — replaced by the verbose
  `PROPERTY_PROMPT` per family used in the per-property single-turn
  driver (memsafe + memleak + overflow at v0).
- The single-pass ContractRecord prototype (`scripts/
  contract_pipeline.py`) once the per-property cached driver is the
  default. Kept around until then for A/B comparison.

## What gets kept

- `build-learn` and Docker pipeline (Phase 0).
- libclang scan (Phase 1).
- KAMain CG + V-snapshot (Phase 2) — extended to emit per-function
  IR fact sidecars.
- Indirect call resolution (LLM-assisted).
- Container/allocator detection (LLM).
- Link-unit pipeline.
- `gen-harness`, triage, ucsan integration — they read summaries
  instead of the 7 tables.

## What gets added

- **KAMain JSON sidecar** (KAMain repo). Per-function IR walk
  emitting effects, guards, ranges, alias annotations, property-
  gating features. No DB writes; sidecar JSON only (per
  `docs/todo-kamain-ir-sidecar.md`).
- **Sidecar loader** (llm-summary repo).
- **`summaries` table** with one row per function:
  `(function, requires_memsafe, requires_memleak, requires_overflow,
  ensures_memsafe, ensures_memleak, ensures_overflow, modifies,
  body_annotated)`.
- **Per-property cached driver** — for each function: compute property
  set; issue one cached property call per applicable property; aggregate
  into the summary.
- **Entry-point checker** (Phase 4) — pure Python; scans entry
  summaries for non-trivial `requires`.
- **KV-cache abstraction** in `llm/` — Anthropic `cache_control`
  markers for big-model backends; persistent llama.cpp session for
  local.

## Migration / what to build first

Branch: `code-as-summary`. Incremental, A/B-validated.

1. **Property prompts v0** — verbose per-property prompts (memsafe,
   memleak, overflow) lifted from the cache A/B test. Already known to
   work on Qwen and Haiku.

2. **KAMain sidecar v0** (KAMain repo).
   - Emit features needed for property gating: load/store presence,
     malloc/free callsites, arith sites, branch conditions, parameter
     aliases.
   - JSON sidecar only; no DB writes.

3. **Per-property cached driver v0** (llm-summary repo, behind a flag).
   - Read KAMain sidecar + existing callee summaries.
   - Compute property set; issue one cached property call per
     applicable property; aggregate.
   - Run on `floppy_simpl3.cil-1` and `cil-2` first; compare to OLD
     pipeline (tokens, calls, elapsed) and to single-pass
     ContractRecord.
   - **Pass criteria**:
     - Local model (llama.cpp) **completes** the run (single-pass
       ContractRecord did not on FloppyPnp).
     - Summaries for `KeSetEvent` and `FloppyPnpComplete` carry
       *no fabricated `Context` precondition*.
     - No "verdict" field anywhere in stored summaries.
     - Total tokens lower than OLD 7-pass on the same set.

4. **Entry-point check v0.**
   - Pure-Python scan of `main`'s `requires` lines.
   - On `floppy_simpl3.cil-2` (expected UNSAFE), the check should
     surface a non-trivial entry `requires` for the actual bug
     (whatever it turns out to be once the summaries are clean).

5. **A/B sweep — both model tiers, three pipelines, on the same
   sv-comp set.**

   | | OLD 7-pass | single-pass ContractRecord | per-property cached |
   |---|---|---|---|
   | Sonnet | baseline | done | target |
   | Haiku | baseline | done (0 FN with code-as-summary) | target |
   | llama.cpp | baseline | timed out (FloppyPnp) | target |

   Headline: per-property cached on llama.cpp matches or beats OLD
   7-pass on F1 with fewer total tokens.

6. **Decision point.**
   - If per-property cached wins on local-model F1 with lower tokens,
     commit.
   - If not, most likely culprits in order: callee discharge rule too
     strict (FP) or too loose (still hallucinating); SA features
     under-driving the property gating (FN); KV cache not being
     honored by backend.

7. **Delete OLD passes/tables once per-property cached is the default.**

8. **Future: IL feasibility layer for FP tightening at entry
   points.** Track reaching/triggering conditions explicitly; verify
   that non-trivial entry `requires` are reachable under the entry's
   actual input domain. Out of scope for this design.

## Open questions

- **SCCs (mutual recursion).** Per-property cached driver needs a
  fixpoint. Probably easiest: iterate per SCC until summaries
  stabilize, with a small max-iteration cap.

- **Backend-aware prompt minimization (deferred).** Mode 2 in the
  cache A/B test showed Sonnet does better with short property tails
  than with the verbose form (less over-fabrication on the
  `Context != NULL` canary). Switching big-model backends to short
  tails would also save input tokens. **Deferred:** primary deployment
  target is small/fast models, where verbose tails are empirically
  better. Revisit only if big-model pipelines become the load-bearing
  configuration.

- **Caching cost in isolation.** The cache A/B test conflated caching
  with prompt cleanup. If we ever care about caching's marginal cost
  separately, run a 4th mode (short prompt, no `cache_control`).
  Probably not worth the cycles — caching is essentially free on
  cache reads and has modest write overhead.

- **Ensures inheritance representation.** Reference-by-name
  (`<inherits callee.ensures[P]>`) keeps summaries small but requires
  the consumer to expand on demand. Verbatim is more local-model-
  friendly. Default to verbatim for v0; switch to reference if
  summary size becomes a problem on bigger projects.

- **Properties beyond memsafe + memleak + overflow.** Init, shift_ub,
  div_by_zero — defer until v0 is solid; add one at a time, only
  after KAMain can flag the relevant operations precisely.

- **Caching across projects.** Natural key: `(function_name,
  source_hash, callee_summaries_hash)`. Same as today, keyed on the
  new schema.

## Why this should win

1. **Small models work.** Code-as-summary takes Haiku and Qwen to 0
   FN on the format A/B test. Per-property single-turn keeps each
   call small enough that local models finish (single-pass
   ContractRecord blew the per-call HTTP budget on FloppyPnp).

2. **Hallucinations are reduced by both prompt and discipline.**
   Prompt structure (rules in cached SYSTEM, short or verbose tail
   per backend) is the load-bearing fix for the canary `Context !=
   NULL` fabrication. The verbatim-discharge rule is a structural
   backstop for deeper call graphs and is cheap to enforce.

3. **Verdict cascade is gone.** Functions don't carry "unsafe" tags
   that propagate. `errorFn`'s behavior is captured by `@ensures: ⊥`
   (no return); callers reason about it the way they reason about any
   other `ensures`.

4. **Token economics improve over OLD multi-pass.** Caching the
   source-prep block once per function eliminates the OLD pipeline's
   50–70% redundancy. Three property calls share one cached source.

5. **Detection-assist comes for free.** A small model with a callee's
   `requires` lines and a caller's source can pattern-match
   `requires: p != NULL` against caller code more reliably than
   `contract_kind: disallow_null`. Same artifact, two consumers.

6. **Verification is honest.** Phase 4 is pure mechanism; the only
   LLM judgment in the bug-report path is whether a callsite
   discharges a callee `requires`. Every other step is data-driven.

7. **The architecture admits the IL layer cleanly.** Forward Hoare
   summaries are the substrate; bug-finding feasibility analysis
   reads them and adds reachability + triggering. We don't have to
   redesign anything to get to bug-finding-with-FP-control later.
