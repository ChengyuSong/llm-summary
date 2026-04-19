"""SV-COMP harness contract seeds.

Loaded only when running on sv-comp benchmarks (CLI flag or auto-detect from
`Project.benchmark_kind`). Lifted from `scripts/contract_pipeline.py:194-243`.

`__VERIFIER_*` and `assume_abort_if_not` are sv-comp specific. The plan
keeps them out of the always-loaded stdlib so non-sv-comp projects (libpng,
openssl, CGC, ...) don't see contracts they will never call.

Note: sv-comp models `malloc` / `calloc` as never returning NULL. We extend
the libc contracts in `stdlib.py` accordingly when sv-comp mode is active.
"""

from __future__ import annotations

from .models import CodeContractSummary
from .stdlib import _summary


def _build_svcomp_contracts() -> dict[str, CodeContractSummary]:
    out: dict[str, CodeContractSummary] = {}

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

    return out


SVCOMP_CONTRACTS: dict[str, CodeContractSummary] = _build_svcomp_contracts()


def svcomp_malloc_overrides() -> dict[str, CodeContractSummary]:
    """SV-COMP-specific stronger ensures: malloc / calloc never return NULL.

    Under sv-comp's memory model these allocators are total. Merge these
    over the libc seeds in `stdlib.STDLIB_CONTRACTS` when the project is
    sv-comp.
    """
    out: dict[str, CodeContractSummary] = {}
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
    return out
