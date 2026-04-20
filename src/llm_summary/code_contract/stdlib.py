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

    # ── errno / setjmp ──
    out["__errno_location"] = _summary(
        "__errno_location",
        memsafe={"ensures": ["result != NULL",
                             "result points to thread-local int (errno)"]},
    )
    out["_setjmp"] = _summary(
        "_setjmp",
        memsafe={"requires": ["env points to a writable jmp_buf"],
                 "ensures":  ["env may be used by a subsequent longjmp "
                              "while the enclosing function is live"]},
        # Modeled as a normal-return call; longjmp is modeled separately as noreturn.
    )

    # ── string / mem ──
    out["memcmp"] = _summary(
        "memcmp",
        memsafe={"requires": ["s1 readable for n bytes",
                              "s2 readable for n bytes"]},
    )
    out["strerror"] = _summary(
        "strerror",
        memsafe={"ensures": ["result != NULL",
                             "result points to a NUL-terminated static/thread-local "
                             "buffer; do not free; may be overwritten by next call"]},
    )

    # ── string -> number ──
    out["atof"] = _summary(
        "atof",
        memsafe={"requires": ["nptr is non-NULL and points to a NUL-terminated string"]},
    )

    # ── math (write through pointer) ──
    out["frexp"] = _summary(
        "frexp",
        memsafe={"requires": ["exp is non-NULL and writable for sizeof(int)"]},
    )
    out["modf"] = _summary(
        "modf",
        memsafe={"requires": ["iptr is non-NULL and writable for sizeof(double)"]},
    )
    out["pow"] = _summary(
        "pow",
        memsafe={"ensures": ["pure: no memory effects"]},
    )

    # ── time ──
    out["gmtime"] = _summary(
        "gmtime",
        memsafe={"requires": ["timer is non-NULL and readable for sizeof(time_t)"],
                 "ensures":  ["result is NULL on failure or points to a static struct tm",
                              "static buffer may be overwritten by subsequent "
                              "gmtime/localtime calls; do not free"]},
    )

    # ── stdio ──
    out["ferror"] = _summary(
        "ferror",
        memsafe={"requires": ["stream is non-NULL and refers to an open FILE"]},
    )
    out["fflush"] = _summary(
        "fflush",
        memsafe={"requires": ["stream is NULL or refers to an open FILE"]},
    )
    out["remove"] = _summary(
        "remove",
        memsafe={"requires": ["pathname is non-NULL and points to a NUL-terminated string"]},
    )
    out["vsnprintf"] = _summary(
        "vsnprintf",
        memsafe={"requires": ["str is NULL and size == 0, or str is writable for size bytes",
                              "format is non-NULL and points to a NUL-terminated string",
                              "ap matches the conversions in format and points to live arguments"],
                 "ensures":  ["if size > 0 and str != NULL: str[0..min(size-1, written)] "
                              "written and str NUL-terminated within size bytes"]},
    )

    # ── POSIX file I/O ──
    out["open"] = _summary(
        "open",
        memsafe={"requires": ["pathname is non-NULL and points to a NUL-terminated string"],
                 "ensures":  ["result is -1 on failure or a non-negative file descriptor"]},
        memleak={"ensures": ["on success: acquires a file descriptor; "
                             "caller must close result"]},
    )
    out["close"] = _summary(
        "close",
        memsafe={"requires": ["fd is -1 or an open file descriptor not already closed"]},
        memleak={"ensures": ["releases fd if it referred to an open descriptor"]},
    )
    out["read"] = _summary(
        "read",
        memsafe={"requires": ["fd is an open file descriptor opened for reading",
                              "buf is writable for count bytes"]},
    )
    out["write"] = _summary(
        "write",
        memsafe={"requires": ["fd is an open file descriptor opened for writing",
                              "buf is readable for count bytes"]},
    )
    out["lseek64"] = _summary(
        "lseek64",
        memsafe={"requires": ["fd is an open file descriptor that supports seeking"]},
    )
    out["fcntl"] = _summary(
        "fcntl",
        memsafe={"requires": ["fd is an open file descriptor",
                              "varargs match the cmd's expected argument type, if any"]},
    )
    out["memchr"] = _summary(
        "memchr",
        memsafe={"requires": ["s is readable for n bytes"],
                 "ensures":  ["result is NULL or points within s[0..n-1]"]},
    )

    # ── musl internal allocator aliases ──
    out["__libc_free"] = _summary(
        "__libc_free",
        memsafe={"requires": ["p is NULL or a previously allocated pointer not yet freed"]},
        memleak={"ensures": ["releases the allocation pointed to by p if non-NULL"]},
    )
    out["__libc_realloc"] = _summary(
        "__libc_realloc",
        memsafe={
            "requires": ["ptr is NULL or previously allocated and not yet freed"],
            "ensures":  ["result is NULL or allocated for size bytes",
                         "ptr ownership transferred to result on success"],
        },
        memleak={"ensures": ["on success: acquires heap allocation tied to result; releases ptr"]},
    )

    # ── compiler-rt complex-arithmetic builtins ──
    # Emitted by clang for complex multiplication; pure arithmetic, no memory ops.
    for name in ("__mulsc3", "__muldc3", "__mulxc3"):
        out[name] = _summary(name, memsafe={})

    return out


STDLIB_CONTRACTS: dict[str, CodeContractSummary] = _build_libc_contracts()
