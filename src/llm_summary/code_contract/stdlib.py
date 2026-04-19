"""Stdlib / harness contract seeds for opaque callees.

Lifted from `scripts/contract_pipeline.py:142-251`, then split:
  - This module: pure libc / POSIX / compiler-builtin contracts. Always
    loaded by the pipeline.
  - `svcomp_stdlib.py`: sv-comp helpers (every `__VERIFIER_*`,
    `assume_abort_if_not`). Loaded only when running on sv-comp benchmarks.

Pre-built `CodeContractSummary` entries are seeded into the `summaries`
dict before the topo walk so the existing callee-inlining flow surfaces
them at every callsite — no code path needs to special-case "external
function".
"""

from __future__ import annotations

from typing import Any

from .models import CodeContractSummary


def _summary(
    name: str,
    *,
    noreturn: bool = False,
    **per_prop: Any,
) -> CodeContractSummary:
    """Compact constructor: _summary("malloc", memsafe={"ensures": [...]})."""
    s = CodeContractSummary(function=name, properties=list(per_prop.keys()))
    for prop, slots in per_prop.items():
        s.requires[prop] = list(slots.get("requires", []))
        s.ensures[prop]  = list(slots.get("ensures",  []))
        s.modifies[prop] = list(slots.get("modifies", []))
        n = slots.get("notes", "")
        if isinstance(n, str) and n:
            s.notes[prop] = n
    s.noreturn = noreturn
    return s


def _build_libc_contracts() -> dict[str, CodeContractSummary]:
    out: dict[str, CodeContractSummary] = {}

    # ── Allocators ──
    out["malloc"] = _summary(
        "malloc",
        memsafe={"ensures": ["result is NULL or allocated for size bytes (uninitialized)"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result if non-NULL"]},
    )
    out["calloc"] = _summary(
        "calloc",
        memsafe={"ensures": ["result is NULL or allocated for nmemb*size bytes",
                             "result[0..nmemb*size-1] initialized to zero when result != NULL"]},
        memleak={"ensures": ["acquires heap allocation: caller must release result if non-NULL"]},
    )
    out["realloc"] = _summary(
        "realloc",
        memsafe={
            "requires": ["ptr is NULL or previously allocated and not yet freed"],
            "ensures":  [
                "result is NULL or allocated for size bytes",
                "ptr ownership transferred to result on success "
                "(do not use ptr after)",
            ],
        },
        memleak={"ensures": [
            "on success: acquires heap allocation tied to result; releases ptr"
        ]},
    )
    out["reallocarray"] = _summary(
        "reallocarray",
        memsafe={"requires": ["ptr is NULL or previously allocated and not yet freed"],
                 "ensures":  ["result is NULL or allocated for nmemb*size bytes",
                              "ptr ownership transferred to result on success"]},
        memleak={"ensures": ["on success: acquires heap allocation tied to result; releases ptr"]},
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

    # ── Process termination (libc; noreturn) ──
    out["abort"] = _summary(
        "abort",
        noreturn=True,
        overflow={"ensures": ["does not return"]},
    )
    out["exit"] = _summary(
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


STDLIB_CONTRACTS: dict[str, CodeContractSummary] = _build_libc_contracts()
