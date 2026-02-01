"""Tests for the database module."""

import pytest

from llm_summary.db import SummaryDB, compute_source_hash
from llm_summary.models import (
    Allocation,
    AllocationSummary,
    AllocationType,
    CallEdge,
    Function,
    ParameterInfo,
)


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    database = SummaryDB(":memory:")
    yield database
    database.close()


@pytest.fixture
def sample_function():
    """Create a sample function for testing."""
    return Function(
        name="create_buffer",
        file_path="/path/to/file.c",
        line_start=10,
        line_end=20,
        source="char* create_buffer(size_t n) { return malloc(n); }",
        signature="char*(size_t)",
    )


@pytest.fixture
def sample_summary():
    """Create a sample summary for testing."""
    return AllocationSummary(
        function_name="create_buffer",
        allocations=[
            Allocation(
                alloc_type=AllocationType.HEAP,
                source="malloc",
                size_expr="n",
                size_params=["n"],
                returned=True,
                may_be_null=True,
            )
        ],
        parameters={
            "n": ParameterInfo(role="size_indicator", used_in_allocation=True)
        },
        description="Allocates n bytes of heap memory.",
    )


class TestSummaryDB:
    """Tests for SummaryDB class."""

    def test_insert_and_get_function(self, db, sample_function):
        """Test inserting and retrieving a function."""
        func_id = db.insert_function(sample_function)
        assert func_id is not None

        retrieved = db.get_function(func_id)
        assert retrieved is not None
        assert retrieved.name == sample_function.name
        assert retrieved.signature == sample_function.signature
        assert retrieved.file_path == sample_function.file_path

    def test_get_function_by_name(self, db, sample_function):
        """Test looking up function by name."""
        db.insert_function(sample_function)

        functions = db.get_function_by_name("create_buffer")
        assert len(functions) == 1
        assert functions[0].name == "create_buffer"

        functions = db.get_function_by_name("create_buffer", "char*(size_t)")
        assert len(functions) == 1

        functions = db.get_function_by_name("nonexistent")
        assert len(functions) == 0

    def test_insert_functions_batch(self, db):
        """Test batch insertion of functions."""
        functions = [
            Function(
                name=f"func{i}",
                file_path="/path/file.c",
                line_start=i * 10,
                line_end=i * 10 + 5,
                source=f"void func{i}() {{}}",
                signature="void()",
            )
            for i in range(5)
        ]

        result = db.insert_functions_batch(functions)
        assert len(result) == 5

        all_funcs = db.get_all_functions()
        assert len(all_funcs) == 5

    def test_upsert_and_get_summary(self, db, sample_function, sample_summary):
        """Test inserting and retrieving summaries."""
        sample_function.id = db.insert_function(sample_function)

        db.upsert_summary(sample_function, sample_summary, model_used="test")

        retrieved = db.get_summary("create_buffer")
        assert retrieved is not None
        assert retrieved.function_name == "create_buffer"
        assert len(retrieved.allocations) == 1
        assert retrieved.allocations[0].source == "malloc"

    def test_get_summary_by_function_id(self, db, sample_function, sample_summary):
        """Test getting summary by function ID."""
        sample_function.id = db.insert_function(sample_function)
        db.upsert_summary(sample_function, sample_summary)

        retrieved = db.get_summary_by_function_id(sample_function.id)
        assert retrieved is not None
        assert retrieved.function_name == "create_buffer"

    def test_needs_update(self, db, sample_function, sample_summary):
        """Test change detection."""
        # Function not in DB
        assert db.needs_update(sample_function) is True

        # Insert function and summary
        sample_function.id = db.insert_function(sample_function)
        db.upsert_summary(sample_function, sample_summary)

        assert db.needs_update(sample_function) is False

        # Change source
        sample_function.source = "char* create_buffer(size_t n) { return calloc(1, n); }"
        assert db.needs_update(sample_function) is True

    def test_call_edges(self, db):
        """Test call edge operations."""
        func1 = Function(
            name="caller", file_path="/f.c", line_start=1, line_end=5,
            source="void caller() { callee(); }", signature="void()"
        )
        func2 = Function(
            name="callee", file_path="/f.c", line_start=10, line_end=15,
            source="void callee() {}", signature="void()"
        )

        func1.id = db.insert_function(func1)
        func2.id = db.insert_function(func2)

        edge = CallEdge(
            caller_id=func1.id,
            callee_id=func2.id,
            file_path="/f.c",
            line=3,
            column=5,
        )
        db.add_call_edge(edge)

        callees = db.get_callees(func1.id)
        assert func2.id in callees

        callers = db.get_callers(func2.id)
        assert func1.id in callers

        # Test callsite info is preserved
        edges = db.get_call_edges_by_caller(func1.id)
        assert len(edges) == 1
        assert edges[0].file_path == "/f.c"
        assert edges[0].line == 3
        assert edges[0].column == 5

    def test_batch_call_edges(self, db):
        """Test batch insertion of call edges."""
        funcs = []
        for i in range(3):
            func = Function(
                name=f"func{i}", file_path="/f.c", line_start=i * 10, line_end=i * 10 + 5,
                source=f"void func{i}() {{}}", signature="void()"
            )
            func.id = db.insert_function(func)
            funcs.append(func)

        edges = [
            CallEdge(caller_id=funcs[0].id, callee_id=funcs[1].id),
            CallEdge(caller_id=funcs[0].id, callee_id=funcs[2].id),
            CallEdge(caller_id=funcs[1].id, callee_id=funcs[2].id),
        ]
        db.add_call_edges_batch(edges)

        all_edges = db.get_all_call_edges()
        assert len(all_edges) == 3

    def test_invalidate_and_cascade(self, db, sample_function, sample_summary):
        """Test cascade invalidation."""
        # Create function chain: f1 -> f2 -> f3
        funcs = []
        for i in range(3):
            func = Function(
                name=f"func{i}", file_path="/f.c", line_start=i * 10, line_end=i * 10 + 5,
                source=f"void func{i}() {{}}", signature="void()"
            )
            func.id = db.insert_function(func)
            funcs.append(func)

            summary = AllocationSummary(function_name=f"func{i}")
            db.upsert_summary(func, summary)

        db.add_call_edge(CallEdge(caller_id=funcs[0].id, callee_id=funcs[1].id))
        db.add_call_edge(CallEdge(caller_id=funcs[1].id, callee_id=funcs[2].id))

        # Invalidate f2 - should cascade to f0 and f1
        invalidated = db.invalidate_and_cascade(funcs[2].id)
        assert funcs[2].id in invalidated
        assert funcs[1].id in invalidated
        assert funcs[0].id in invalidated

        # Summaries should be deleted
        assert db.get_summary_by_function_id(funcs[0].id) is None
        assert db.get_summary_by_function_id(funcs[1].id) is None
        assert db.get_summary_by_function_id(funcs[2].id) is None

    def test_get_stats(self, db, sample_function, sample_summary):
        """Test database statistics."""
        sample_function.id = db.insert_function(sample_function)
        db.upsert_summary(sample_function, sample_summary)

        stats = db.get_stats()
        assert stats["functions"] == 1
        assert stats["allocation_summaries"] == 1

    def test_clear_all(self, db, sample_function, sample_summary):
        """Test clearing database."""
        sample_function.id = db.insert_function(sample_function)
        db.upsert_summary(sample_function, sample_summary)

        db.clear_all()

        stats = db.get_stats()
        assert stats["functions"] == 0
        assert stats["allocation_summaries"] == 0


class TestSourceHash:
    """Tests for source hash computation."""

    def test_same_source_same_hash(self):
        """Same source should produce same hash."""
        source = "void foo() {}"
        assert compute_source_hash(source) == compute_source_hash(source)

    def test_different_source_different_hash(self):
        """Different source should produce different hash."""
        source1 = "void foo() {}"
        source2 = "void bar() {}"
        assert compute_source_hash(source1) != compute_source_hash(source2)
