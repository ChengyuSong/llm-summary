"""Tests for the data models."""

import pytest

from llm_summary.models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    Function,
    ParameterInfo,
)


class TestFunction:
    """Tests for Function model."""

    def test_create_function(self):
        """Test creating a function."""
        func = Function(
            name="test_func",
            file_path="/path/to/file.c",
            line_start=10,
            line_end=20,
            source="void test_func() {}",
            signature="void()",
        )

        assert func.name == "test_func"
        assert func.file_path == "/path/to/file.c"
        assert func.id is None

    def test_function_hash(self):
        """Test function hashing for use in dicts/sets."""
        func1 = Function(
            name="test", file_path="/f.c", line_start=1, line_end=5,
            source="void test() {}", signature="void()"
        )
        func2 = Function(
            name="test", file_path="/f.c", line_start=1, line_end=5,
            source="void test() {}", signature="void()"
        )
        func3 = Function(
            name="other", file_path="/f.c", line_start=1, line_end=5,
            source="void other() {}", signature="void()"
        )

        assert hash(func1) == hash(func2)
        assert hash(func1) != hash(func3)

        # Can be used in sets
        s = {func1}
        assert func2 in s
        assert func3 not in s

    def test_function_equality(self):
        """Test function equality comparison."""
        func1 = Function(
            name="test", file_path="/f.c", line_start=1, line_end=5,
            source="void test() {}", signature="void()"
        )
        func2 = Function(
            name="test", file_path="/f.c", line_start=1, line_end=5,
            source="different source", signature="void()"
        )

        # Equal based on name, signature, file_path
        assert func1 == func2


class TestAllocation:
    """Tests for Allocation model."""

    def test_create_allocation(self):
        """Test creating an allocation."""
        alloc = Allocation(
            alloc_type=AllocationType.HEAP,
            source="malloc",
            size_expr="n",
            size_params=["n"],
            returned=True,
            may_be_null=True,
        )

        assert alloc.alloc_type == AllocationType.HEAP
        assert alloc.source == "malloc"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        alloc = Allocation(
            alloc_type=AllocationType.HEAP,
            source="malloc",
            size_expr="n * sizeof(int)",
            size_params=["n"],
            returned=True,
            stored_to=None,
            may_be_null=True,
        )

        d = alloc.to_dict()
        assert d["type"] == "heap"
        assert d["source"] == "malloc"
        assert d["size_expr"] == "n * sizeof(int)"
        assert d["size_params"] == ["n"]
        assert d["returned"] is True
        assert d["may_be_null"] is True


class TestAllocationSummary:
    """Tests for AllocationSummary model."""

    def test_create_summary(self):
        """Test creating a summary."""
        summary = AllocationSummary(
            function_name="create_buffer",
            allocations=[
                Allocation(
                    alloc_type=AllocationType.HEAP,
                    source="malloc",
                    size_expr="n",
                    size_params=["n"],
                    returned=True,
                )
            ],
            parameters={
                "n": ParameterInfo(role="size_indicator", used_in_allocation=True)
            },
            description="Allocates n bytes.",
        )

        assert summary.function_name == "create_buffer"
        assert len(summary.allocations) == 1
        assert "n" in summary.parameters

    def test_to_dict(self):
        """Test conversion to dictionary."""
        summary = AllocationSummary(
            function_name="test",
            allocations=[
                Allocation(
                    alloc_type=AllocationType.HEAP,
                    source="malloc",
                )
            ],
            parameters={
                "size": ParameterInfo(role="size", used_in_allocation=True)
            },
            description="Test function",
        )

        d = summary.to_dict()
        assert d["function"] == "test"
        assert len(d["allocations"]) == 1
        assert d["allocations"][0]["source"] == "malloc"
        assert "size" in d["parameters"]
        assert d["description"] == "Test function"

    def test_empty_summary(self):
        """Test creating an empty summary (non-allocating function)."""
        summary = AllocationSummary(
            function_name="getter",
            description="Does not allocate memory",
        )

        assert len(summary.allocations) == 0
        assert len(summary.parameters) == 0


class TestAllocationType:
    """Tests for AllocationType enum."""

    def test_allocation_types(self):
        """Test allocation type values."""
        assert AllocationType.HEAP.value == "heap"
        assert AllocationType.STACK.value == "stack"
        assert AllocationType.STATIC.value == "static"
        assert AllocationType.UNKNOWN.value == "unknown"

    def test_from_string(self):
        """Test creating from string."""
        assert AllocationType("heap") == AllocationType.HEAP
        assert AllocationType("stack") == AllocationType.STACK
