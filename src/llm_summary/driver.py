"""Unified bottom-up graph traversal driver for summary passes."""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol

from .db import SummaryDB
from .models import AllocationSummary, FreeSummary, InitSummary, MemsafeSummary, VerificationSummary, Function
from .ordering import ProcessingOrderer


class SummaryPass(Protocol):
    """Interface each pass must implement."""

    name: str

    def get_cached(self, func_id: int, func: Function) -> Any | None:
        """Return cached summary if available, else None."""
        ...

    def summarize(self, func: Function, callee_summaries: dict[str, Any]) -> Any:
        """Generate a summary for *func* given its callee summaries."""
        ...

    def store(self, func: Function, summary: Any) -> None:
        """Persist *summary* to the database."""
        ...


class AllocationPass:
    """Adapter that wraps AllocationSummarizer as a SummaryPass."""

    name = "allocation"

    def __init__(self, summarizer, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> AllocationSummary | None:
        existing = self.db.get_summary_by_function_id(func_id)
        if existing and not self.db.needs_update(func):
            return existing
        return None

    def summarize(self, func: Function, callee_summaries: dict[str, AllocationSummary]) -> AllocationSummary:
        return self.summarizer.summarize_function(func, callee_summaries)

    def store(self, func: Function, summary: AllocationSummary) -> None:
        self.db.upsert_summary(func, summary, model_used=self.model)


class FreePass:
    """Adapter that wraps FreeSummarizer as a SummaryPass."""

    name = "free"

    def __init__(self, summarizer, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> FreeSummary | None:
        return self.db.get_free_summary_by_function_id(func_id)

    def summarize(self, func: Function, callee_summaries: dict[str, FreeSummary]) -> FreeSummary:
        return self.summarizer.summarize_function(func, callee_summaries)

    def store(self, func: Function, summary: FreeSummary) -> None:
        self.db.upsert_free_summary(func, summary, model_used=self.model)


class InitPass:
    """Adapter that wraps InitSummarizer as a SummaryPass."""

    name = "init"

    def __init__(self, summarizer, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> InitSummary | None:
        return self.db.get_init_summary_by_function_id(func_id)

    def summarize(self, func: Function, callee_summaries: dict[str, InitSummary]) -> InitSummary:
        return self.summarizer.summarize_function(func, callee_summaries)

    def store(self, func: Function, summary: InitSummary) -> None:
        self.db.upsert_init_summary(func, summary, model_used=self.model)


class MemsafePass:
    """Adapter that wraps MemsafeSummarizer as a SummaryPass."""

    name = "memsafe"

    def __init__(self, summarizer, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> MemsafeSummary | None:
        return self.db.get_memsafe_summary_by_function_id(func_id)

    def summarize(self, func: Function, callee_summaries: dict[str, MemsafeSummary]) -> MemsafeSummary:
        return self.summarizer.summarize_function(func, callee_summaries)

    def store(self, func: Function, summary: MemsafeSummary) -> None:
        self.db.upsert_memsafe_summary(func, summary, model_used=self.model)


class VerificationPass:
    """Adapter that wraps VerificationSummarizer as a SummaryPass."""

    name = "verify"

    def __init__(self, summarizer, db: SummaryDB, model: str):
        self.summarizer = summarizer
        self.db = db
        self.model = model

    def get_cached(self, func_id: int, func: Function) -> VerificationSummary | None:
        return self.db.get_verification_summary_by_function_id(func_id)

    def summarize(self, func: Function, callee_summaries: dict[str, VerificationSummary]) -> VerificationSummary:
        return self.summarizer.summarize_function(func, callee_summaries)

    def store(self, func: Function, summary: VerificationSummary) -> None:
        self.db.upsert_verification_summary(func, summary, model_used=self.model)


class BottomUpDriver:
    """Builds the call graph once, then runs one or more summary passes
    over functions in bottom-up (callee-first) topological order."""

    def __init__(self, db: SummaryDB, verbose: bool = False):
        self.db = db
        self.verbose = verbose
        self._graph: dict[int, list[int]] | None = None
        self._orderer: ProcessingOrderer | None = None

    def build_graph(self) -> tuple[dict[int, list[int]], ProcessingOrderer]:
        """Build (and cache) the call graph + ProcessingOrderer."""
        if self._graph is not None and self._orderer is not None:
            return self._graph, self._orderer

        edges = self.db.get_all_call_edges()
        graph: dict[int, list[int]] = {}

        for edge in edges:
            if edge.caller_id not in graph:
                graph[edge.caller_id] = []
            graph[edge.caller_id].append(edge.callee_id)

        # Add all functions to graph (some may have no callees)
        for func in self.db.get_all_functions():
            if func.id is not None and func.id not in graph:
                graph[func.id] = []

        self._graph = graph
        self._orderer = ProcessingOrderer(graph)
        return graph, self._orderer

    def compute_affected(
        self, dirty_ids: set[int], graph: dict[int, list[int]]
    ) -> set[int]:
        """Compute the set of function IDs affected by changes to *dirty_ids*.

        This is *dirty_ids* plus all transitive callers (via reverse edges).
        """
        # Build reverse graph
        reverse: dict[int, list[int]] = {}
        for caller, callees in graph.items():
            for callee in callees:
                if callee not in reverse:
                    reverse[callee] = []
                reverse[callee].append(caller)

        affected = set(dirty_ids)
        queue: deque[int] = deque(dirty_ids)
        while queue:
            node = queue.popleft()
            for caller in reverse.get(node, []):
                if caller not in affected:
                    affected.add(caller)
                    queue.append(caller)

        return affected

    def run(
        self,
        passes: list[SummaryPass],
        force: bool = False,
        dirty_ids: set[int] | None = None,
    ) -> dict[str, dict[int, Any]]:
        """Run all *passes* over the call graph in a single traversal.

        Args:
            passes: List of SummaryPass adapters to run.
            force: Re-summarize even if a cached summary exists.
            dirty_ids: If provided, only re-summarize these function IDs
                and their transitive callers.  Others load from cache.

        Returns:
            ``{pass.name: {func_id: summary}}``
        """
        graph, orderer = self.build_graph()

        # Compute affected set if incremental
        affected: set[int] | None = None
        if dirty_ids is not None:
            affected = self.compute_affected(dirty_ids, graph)

        if self.verbose:
            stats = orderer.get_stats()
            msg = f"Processing {stats['nodes']} functions in {stats['sccs']} SCCs"
            if stats["recursive_sccs"] > 0:
                msg += f" ({stats['recursive_sccs']} recursive)"
            if affected is not None:
                msg += f", {len(affected)} affected"
            print(msg)

        # Prepare per-pass result dicts
        results: dict[str, dict[int, Any]] = {p.name: {} for p in passes}

        processing_order = list(orderer.get_processing_order())
        total = sum(len(scc) for scc in processing_order)
        current = 0

        for scc in processing_order:
            for func_id in scc:
                current += 1
                func = self.db.get_function(func_id)
                if func is None:
                    continue

                # If incremental and function is not affected, load from cache
                if affected is not None and func_id not in affected:
                    for p in passes:
                        cached = p.get_cached(func_id, func)
                        if cached is not None:
                            results[p.name][func_id] = cached
                            p.summarizer._stats["cache_hits"] += 1
                    continue

                for p in passes:
                    # Check cache
                    if not force:
                        cached = p.get_cached(func_id, func)
                        if cached is not None:
                            results[p.name][func_id] = cached
                            p.summarizer._stats["cache_hits"] += 1
                            continue

                    # Gather callee summaries from this pass's results
                    callee_ids = graph.get(func_id, [])
                    callee_summaries: dict[str, Any] = {}
                    for callee_id in callee_ids:
                        if callee_id in results[p.name]:
                            callee_func = self.db.get_function(callee_id)
                            if callee_func:
                                callee_summaries[callee_func.name] = results[p.name][callee_id]

                    # Set progress on the underlying summarizer
                    p.summarizer._progress_current = current
                    p.summarizer._progress_total = total

                    # Generate summary
                    summary = p.summarize(func, callee_summaries)
                    results[p.name][func_id] = summary

                    # Store in database
                    p.store(func, summary)

        return results
