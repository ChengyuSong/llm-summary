"""Code-contract pipeline: Hoare-style per-function summaries (memsafe / memleak / overflow).

See docs/design-llm-first.md and the plan
~/.claude/plans/alright-we-create-a-gleaming-sutherland.md for design context.

The pipeline produces, per function and per in-scope property:
  - requires (preconditions)
  - ensures (postconditions)
  - modifies (stack/heap locations written)
  - notes (one-line context)
  - noreturn (property-independent flag)

NO verdict field by design. Entry-point checking is a separate Phase 4 pass
that scans entry-function `requires` for non-trivial obligations.
"""

from .models import PROPERTIES, CodeContractSummary, is_nontrivial

__all__ = ["PROPERTIES", "CodeContractSummary", "is_nontrivial"]
