"""Tests for the ordering module."""

import pytest

from llm_summary.ordering import (
    compute_sccs,
    topological_order_sccs,
    ProcessingOrderer,
)


class TestComputeSCCs:
    """Tests for SCC computation."""

    def test_empty_graph(self):
        """Empty graph should return empty SCCs."""
        sccs = compute_sccs({})
        assert sccs == []

    def test_single_node(self):
        """Single node should be its own SCC."""
        sccs = compute_sccs({1: []})
        assert len(sccs) == 1
        assert 1 in sccs[0]

    def test_no_cycles(self):
        """Linear chain should have each node as its own SCC."""
        # 1 -> 2 -> 3
        graph = {1: [2], 2: [3], 3: []}
        sccs = compute_sccs(graph)

        assert len(sccs) == 3
        for scc in sccs:
            assert len(scc) == 1

    def test_simple_cycle(self):
        """Simple cycle should be one SCC."""
        # 1 -> 2 -> 3 -> 1
        graph = {1: [2], 2: [3], 3: [1]}
        sccs = compute_sccs(graph)

        assert len(sccs) == 1
        assert set(sccs[0]) == {1, 2, 3}

    def test_multiple_sccs(self):
        """Graph with multiple SCCs."""
        # SCC1: 1 <-> 2
        # SCC2: 3 -> 4 -> 5 -> 3
        # 1 -> 3 (cross-SCC edge)
        graph = {
            1: [2, 3],
            2: [1],
            3: [4],
            4: [5],
            5: [3],
        }
        sccs = compute_sccs(graph)

        assert len(sccs) == 2

        scc_sets = [set(scc) for scc in sccs]
        assert {1, 2} in scc_sets
        assert {3, 4, 5} in scc_sets

    def test_self_loop(self):
        """Self-loop should create SCC of size 1."""
        graph = {1: [1]}
        sccs = compute_sccs(graph)

        assert len(sccs) == 1
        assert 1 in sccs[0]


class TestTopologicalOrder:
    """Tests for topological ordering of SCCs."""

    def test_linear_chain(self):
        """Linear chain should be ordered leaf-first."""
        # 1 -> 2 -> 3
        graph = {1: [2], 2: [3], 3: []}
        sccs = topological_order_sccs(graph)

        # Should be [3], [2], [1] (leaves first)
        assert len(sccs) == 3
        order = [scc[0] for scc in sccs]

        # 3 should come before 2, which should come before 1
        assert order.index(3) < order.index(2)
        assert order.index(2) < order.index(1)

    def test_diamond(self):
        """Diamond pattern should have correct ordering."""
        #     1
        #    / \
        #   2   3
        #    \ /
        #     4
        graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
        sccs = topological_order_sccs(graph)

        order = [scc[0] for scc in sccs]

        # 4 should come first (leaf)
        assert order[0] == 4
        # 1 should come last (root)
        assert order[-1] == 1


class TestProcessingOrderer:
    """Tests for ProcessingOrderer class."""

    def test_get_processing_order(self):
        """Test getting processing order."""
        graph = {1: [2], 2: [3], 3: []}
        orderer = ProcessingOrderer(graph)

        order = list(orderer.get_processing_order())
        assert len(order) == 3

        # Flatten to get node order
        flat_order = [node for scc in order for node in scc]

        # Leaves first
        assert flat_order.index(3) < flat_order.index(2)
        assert flat_order.index(2) < flat_order.index(1)

    def test_is_recursive(self):
        """Test recursive detection."""
        graph = {
            1: [2],
            2: [1],  # Mutual recursion
            3: [],   # Not recursive
        }
        orderer = ProcessingOrderer(graph)

        assert orderer.is_recursive(1) is True
        assert orderer.is_recursive(2) is True
        assert orderer.is_recursive(3) is False

    def test_get_scc_members(self):
        """Test getting SCC members."""
        graph = {1: [2], 2: [1], 3: []}
        orderer = ProcessingOrderer(graph)

        members = orderer.get_scc_members(1)
        assert set(members) == {1, 2}

        members = orderer.get_scc_members(3)
        assert members == [3]

    def test_get_external_callees(self):
        """Test getting external callees."""
        # SCC {1, 2} calls into 3
        graph = {1: [2, 3], 2: [1], 3: []}
        orderer = ProcessingOrderer(graph)

        scc = orderer.get_scc_members(1)
        external = orderer.get_external_callees(scc)

        assert 3 in external
        assert 1 not in external
        assert 2 not in external

    def test_get_stats(self):
        """Test statistics computation."""
        graph = {
            1: [2, 3],
            2: [1],    # Creates SCC with 1
            3: [4],
            4: [5],
            5: [3],    # Creates SCC with 3, 4
        }
        orderer = ProcessingOrderer(graph)

        stats = orderer.get_stats()
        assert stats["nodes"] == 5
        assert stats["edges"] == 6
        assert stats["sccs"] == 2
        assert stats["recursive_sccs"] == 2
        assert stats["largest_scc"] == 3

    def test_get_scc_graph(self):
        """Test SCC DAG computation."""
        # SCC1: {1, 2}
        # SCC2: {3}
        # SCC1 -> SCC2
        graph = {1: [2, 3], 2: [1], 3: []}
        orderer = ProcessingOrderer(graph)

        scc_graph = orderer.get_scc_graph()

        # Should have edge from SCC containing 1 to SCC containing 3
        scc1_idx = orderer.scc_index[1]
        scc3_idx = orderer.scc_index[3]

        assert scc3_idx in scc_graph.get(scc1_idx, [])
