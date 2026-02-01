"""Tests for standard library summaries."""

import pytest

from llm_summary.stdlib import (
    get_stdlib_summary,
    get_all_stdlib_summaries,
    is_stdlib_allocator,
    STDLIB_SUMMARIES,
)
from llm_summary.models import AllocationType


class TestStdlibSummaries:
    """Tests for pre-defined stdlib summaries."""

    def test_malloc_summary(self):
        """Test malloc summary."""
        summary = get_stdlib_summary("malloc")

        assert summary is not None
        assert summary.function_name == "malloc"
        assert len(summary.allocations) == 1

        alloc = summary.allocations[0]
        assert alloc.alloc_type == AllocationType.HEAP
        assert alloc.source == "malloc"
        assert alloc.returned is True
        assert alloc.may_be_null is True
        assert "size" in alloc.size_params

    def test_calloc_summary(self):
        """Test calloc summary."""
        summary = get_stdlib_summary("calloc")

        assert summary is not None
        assert len(summary.allocations) == 1

        alloc = summary.allocations[0]
        assert alloc.size_expr == "nmemb * size"
        assert "nmemb" in alloc.size_params
        assert "size" in alloc.size_params

    def test_realloc_summary(self):
        """Test realloc summary."""
        summary = get_stdlib_summary("realloc")

        assert summary is not None
        assert len(summary.allocations) == 1

        assert "ptr" in summary.parameters
        assert "size" in summary.parameters
        assert summary.parameters["size"].used_in_allocation is True
        assert summary.parameters["ptr"].used_in_allocation is False

    def test_free_summary(self):
        """Test free summary (no allocation)."""
        summary = get_stdlib_summary("free")

        assert summary is not None
        assert len(summary.allocations) == 0
        assert "ptr" in summary.parameters

    def test_strdup_summary(self):
        """Test strdup summary."""
        summary = get_stdlib_summary("strdup")

        assert summary is not None
        assert len(summary.allocations) == 1

        alloc = summary.allocations[0]
        assert alloc.source == "strdup"
        assert "strlen" in alloc.size_expr

    def test_nonexistent_function(self):
        """Test looking up nonexistent function."""
        summary = get_stdlib_summary("nonexistent_function")
        assert summary is None

    def test_get_all_summaries(self):
        """Test getting all stdlib summaries."""
        summaries = get_all_stdlib_summaries()

        assert len(summaries) > 0
        assert "malloc" in summaries
        assert "free" in summaries

        # Should be a copy
        summaries["test"] = None
        assert "test" not in STDLIB_SUMMARIES


class TestIsStdlibAllocator:
    """Tests for is_stdlib_allocator function."""

    def test_allocators(self):
        """Test identifying allocating functions."""
        assert is_stdlib_allocator("malloc") is True
        assert is_stdlib_allocator("calloc") is True
        assert is_stdlib_allocator("realloc") is True
        assert is_stdlib_allocator("strdup") is True

    def test_non_allocators(self):
        """Test identifying non-allocating functions."""
        assert is_stdlib_allocator("free") is False
        assert is_stdlib_allocator("fclose") is False

    def test_unknown_functions(self):
        """Test unknown functions."""
        assert is_stdlib_allocator("unknown_function") is False
