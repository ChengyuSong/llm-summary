"""Topological ordering and SCC computation for the call graph."""

from collections import defaultdict
from collections.abc import Iterator


def compute_sccs(graph: dict[int, list[int]]) -> list[list[int]]:
    """
    Compute strongly connected components using Tarjan's algorithm.

    Args:
        graph: Adjacency list representation (node -> list of successors)

    Returns:
        List of SCCs, where each SCC is a list of node IDs.
        SCCs are returned in reverse topological order (leaves first).
    """
    index_counter = [0]
    stack: list[int] = []
    lowlinks: dict[int, int] = {}
    index: dict[int, int] = {}
    on_stack: set[int] = set()
    sccs: list[list[int]] = []

    # Get all nodes (including those with no outgoing edges)
    all_nodes = set(graph.keys())
    for successors in graph.values():
        all_nodes.update(successors)

    def strongconnect(node: int) -> None:
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        for successor in graph.get(node, []):
            if successor not in index:
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], index[successor])

        # If node is a root node, pop the stack and generate an SCC
        if lowlinks[node] == index[node]:
            scc = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            sccs.append(scc)

    for node in all_nodes:
        if node not in index:
            strongconnect(node)

    return sccs


def topological_order_sccs(
    graph: dict[int, list[int]]
) -> list[list[int]]:
    """
    Compute SCCs and return them in topological order.

    Returns SCCs ordered so that if SCC A depends on SCC B, then B comes before A.
    This means leaf SCCs (callees) come first.
    """
    sccs = compute_sccs(graph)

    # Tarjan's algorithm returns SCCs in reverse topological order
    # So we just return them as-is
    return sccs


def get_processing_order(graph: dict[int, list[int]]) -> Iterator[list[int]]:
    """
    Get the order in which functions should be processed.

    Yields groups of function IDs that can be processed together.
    Each group is an SCC. Groups are yielded in dependency order
    (callees before callers).
    """
    sccs = topological_order_sccs(graph)

    # SCCs are already in reverse topological order from Tarjan's
    for scc in sccs:
        yield scc


class ProcessingOrderer:
    """Determines the order to process functions for bottom-up analysis."""

    def __init__(self, graph: dict[int, list[int]]):
        """
        Initialize with call graph.

        Args:
            graph: Call graph as adjacency list (caller -> [callees])
        """
        self.graph = graph
        self._sccs: list[list[int]] | None = None
        self._scc_index: dict[int, int] | None = None

    @property
    def sccs(self) -> list[list[int]]:
        """Get all SCCs in processing order."""
        if self._sccs is None:
            self._sccs = topological_order_sccs(self.graph)
        return self._sccs

    @property
    def scc_index(self) -> dict[int, int]:
        """Get mapping from node ID to SCC index."""
        if self._scc_index is None:
            self._scc_index = {}
            for i, scc in enumerate(self.sccs):
                for node in scc:
                    self._scc_index[node] = i
        return self._scc_index

    def get_processing_order(self) -> Iterator[list[int]]:
        """Yield SCCs in processing order (callees first)."""
        for scc in self.sccs:
            yield scc

    def is_recursive(self, func_id: int) -> bool:
        """Check if a function is part of a recursive SCC."""
        scc_idx = self.scc_index.get(func_id)
        if scc_idx is None:
            return False
        return len(self.sccs[scc_idx]) > 1

    def get_scc_members(self, func_id: int) -> list[int]:
        """Get all members of the SCC containing this function."""
        scc_idx = self.scc_index.get(func_id)
        if scc_idx is None:
            return [func_id]
        return self.sccs[scc_idx]

    def get_external_callees(self, scc: list[int]) -> list[int]:
        """Get callees of an SCC that are outside the SCC."""
        scc_set = set(scc)
        external = set()

        for node in scc:
            for callee in self.graph.get(node, []):
                if callee not in scc_set:
                    external.add(callee)

        return list(external)

    def get_scc_graph(self) -> dict[int, list[int]]:
        """
        Get the DAG of SCCs.

        Returns a graph where nodes are SCC indices and edges represent
        calls between SCCs.
        """
        scc_graph: dict[int, list[int]] = defaultdict(list)

        for node, callees in self.graph.items():
            src_scc = self.scc_index.get(node)
            if src_scc is None:
                continue

            for callee in callees:
                dst_scc = self.scc_index.get(callee)
                if dst_scc is not None and dst_scc != src_scc:
                    if dst_scc not in scc_graph[src_scc]:
                        scc_graph[src_scc].append(dst_scc)

        return dict(scc_graph)

    def get_stats(self) -> dict[str, int]:
        """Get statistics about the call graph."""
        num_nodes = len(set(self.graph.keys()) | {n for ns in self.graph.values() for n in ns})
        num_edges = sum(len(callees) for callees in self.graph.values())
        num_sccs = len(self.sccs)
        recursive_sccs = sum(1 for scc in self.sccs if len(scc) > 1)
        largest_scc = max(len(scc) for scc in self.sccs) if self.sccs else 0

        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "sccs": num_sccs,
            "recursive_sccs": recursive_sccs,
            "largest_scc": largest_scc,
        }
