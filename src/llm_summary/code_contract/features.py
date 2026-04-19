"""Static-analysis feature gating for per-function property scope.

Decides which subset of {memsafe, memleak, overflow} applies to each
function. A property is in scope iff (a) the function body shows behavior
that could touch it, or (b) any callee published a non-trivial
requires/ensures for it.

Adapter contract:
  - `features_for(func) -> Features`
  - `property_set(features, callee_summaries) -> list[str]`

Until the KAMain JSON sidecar (`docs/todo-kamain-ir-sidecar.md`) ships,
`features_for` falls back to the regex helpers lifted from
`scripts/contract_pipeline.py:262-303`. Each call logs `XXX(mock-until-sidecar)`
at debug level so future audits can find every site that depends on the
mock. When KAMain lands, replace the body of `features_for` with the
sidecar lookup; callers stay unchanged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..models import Function
from .models import PROPERTIES, CodeContractSummary

log = logging.getLogger("code_contract.features")


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


def _extract_features_regex(source: str) -> Features:
    return Features(
        has_deref=bool(_DEREF_RE.search(source)),
        has_alloc=bool(_ALLOC_RE.search(source)),
        has_free=bool(_FREE_RE.search(source)),
        has_index=bool(_INDEX_RE.search(source)),
        has_arith=bool(_ARITH_RE.search(source)),
        has_div=bool(_DIV_RE.search(source)),
        has_shift=bool(_SHIFT_RE.search(source)),
    )


def features_for(func: Function) -> Features:
    """Return the SA feature set for `func`.

    Mock fallback: regex over `func.llm_source` (macro-annotated). When
    KAMain JSON sidecar is wired up, replace the body to read from the
    sidecar — caller signature stays the same.
    """
    log.debug("XXX(mock-until-sidecar) features_for %s using regex", func.name)
    return _extract_features_regex(func.llm_source)


def property_set(
    features: Features,
    callee_summaries: list[CodeContractSummary],
) -> list[str]:
    """Adaptive scoping: which subset of PROPERTIES applies here.

    A property is in scope if either (a) the function body shows behavior
    that could touch it, or (b) any callee published a non-trivial
    requires/ensures for it (so the caller may need to discharge or
    propagate).
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
    # Defensive: PROPERTIES drives the iteration order downstream.
    return [p for p in PROPERTIES if p in props]
