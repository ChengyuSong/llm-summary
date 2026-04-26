"""Tests for the database module."""

import pytest

from llm_summary.db import SummaryDB, compute_source_hash
from llm_summary.models import (
    AddressFlow,
    AddressTakenFunction,
    Allocation,
    AllocationSummary,
    AllocationType,
    CallEdge,
    Function,
    IndirectCallsite,
    IndirectCallTarget,
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


class TestImportUnitData:
    """Tests for cross-DB compositional import (`import_unit_data`)."""

    def _make_func(
        self, name: str, file_path: str, line_start: int = 1
    ) -> Function:
        return Function(
            name=name,
            file_path=file_path,
            line_start=line_start,
            line_end=line_start + 5,
            source=f"void {name}() {{}}",
            signature="void()",
        )

    def test_imports_functions_and_typedefs(self, tmp_path):
        """Functions + typedefs from a source DB land in the target DB."""
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        src = SummaryDB(src_path)
        try:
            f_a = self._make_func("a", "/proj/shared.c")
            f_b = self._make_func("b", "/proj/other.c")
            src.insert_function(f_a)
            src.insert_function(f_b)
            src.insert_typedef(
                name="my_t",
                kind="typedef",
                underlying_type="int",
                canonical_type="int",
                file_path="/proj/shared.c",
            )
        finally:
            src.close()

        tgt = SummaryDB(tgt_path)
        try:
            stats = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            assert stats.functions == 1
            assert stats.typedefs == 1

            assert tgt.get_function_by_name("a")
            assert not tgt.get_function_by_name("b")
            assert tgt.get_typedef("my_t") is not None
        finally:
            tgt.close()

    def test_remaps_call_edges_across_dbs(self, tmp_path):
        """Call edges are remapped onto the target DB's function IDs.

        The target gets a pre-existing function so its IDs differ from
        the source — exercises the (name, signature, file_path) join.
        """
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        src = SummaryDB(src_path)
        try:
            f_a = self._make_func("a", "/proj/shared.c", line_start=1)
            f_b = self._make_func("b", "/proj/shared.c", line_start=10)
            a_id = src.insert_function(f_a)
            b_id = src.insert_function(f_b)
            src.add_call_edge(
                CallEdge(
                    caller_id=a_id,
                    callee_id=b_id,
                    is_indirect=0,
                    file_path="/proj/shared.c",
                    line=2,
                    column=0,
                )
            )
        finally:
            src.close()

        tgt = SummaryDB(tgt_path)
        try:
            # Insert a placeholder function so the target's id sequence
            # is offset from the source's.
            tgt.insert_function(self._make_func("placeholder", "/proj/x.c"))

            stats = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            assert stats.functions == 2
            assert stats.call_edges == 1

            a_tgt = tgt.get_function_by_name("a")[0]
            b_tgt = tgt.get_function_by_name("b")[0]
            edges = tgt.get_call_edges_by_caller(a_tgt.id)
            assert len(edges) == 1
            assert edges[0].callee_id == b_tgt.id
        finally:
            tgt.close()

    def test_remaps_indirect_callsite_targets(self, tmp_path):
        """Both callsite_id and target_function_id remap correctly."""
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        src = SummaryDB(src_path)
        try:
            caller = self._make_func("caller", "/proj/shared.c")
            tgt_fn = self._make_func("tgt_fn", "/proj/shared.c", line_start=20)
            caller_id = src.insert_function(caller)
            tgt_id = src.insert_function(tgt_fn)
            cs_id = src.add_indirect_callsite(
                IndirectCallsite(
                    caller_function_id=caller_id,
                    file_path="/proj/shared.c",
                    line_number=3,
                    callee_expr="ctx->h",
                    signature="void()",
                    context_snippet="ctx->h();",
                )
            )
            src.add_indirect_call_target(
                IndirectCallTarget(
                    callsite_id=cs_id,
                    target_function_id=tgt_id,
                    confidence="high",
                    llm_reasoning="",
                )
            )
        finally:
            src.close()

        tgt = SummaryDB(tgt_path)
        try:
            tgt.insert_function(self._make_func("placeholder", "/proj/x.c"))

            stats = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            assert stats.indirect_callsites == 1
            assert stats.indirect_call_targets == 1

            caller_tgt = tgt.get_function_by_name("caller")[0]
            cs_list = tgt.get_indirect_callsites(caller_tgt.id)
            assert len(cs_list) == 1
            targets = tgt.get_indirect_call_targets(cs_list[0].id)
            assert len(targets) == 1
            tgt_fn_tgt = tgt.get_function_by_name("tgt_fn")[0]
            assert targets[0].target_function_id == tgt_fn_tgt.id
        finally:
            tgt.close()

    def test_idempotent_on_repeat_import(self, tmp_path):
        """Re-running with the same inputs leaves the target unchanged."""
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        src = SummaryDB(src_path)
        try:
            a_id = src.insert_function(self._make_func("a", "/proj/shared.c"))
            b_id = src.insert_function(
                self._make_func("b", "/proj/shared.c", line_start=10)
            )
            src.add_call_edge(
                CallEdge(
                    caller_id=a_id, callee_id=b_id, is_indirect=0,
                    file_path="/proj/shared.c", line=2, column=0,
                )
            )
            src.add_address_flow(
                AddressFlow(
                    function_id=a_id,
                    flow_target="g_handler",
                    file_path="/proj/shared.c",
                    line_number=1,
                )
            )
            src.add_address_taken_function(
                AddressTakenFunction(function_id=a_id, signature="void()")
            )
        finally:
            src.close()

        tgt = SummaryDB(tgt_path)
        try:
            stats1 = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            stats2 = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            assert stats1.functions == 2
            assert stats1.call_edges == 1
            assert stats1.address_flows == 1
            assert stats1.address_taken == 1

            assert stats2.functions == 0
            assert stats2.call_edges == 0
            assert stats2.address_flows == 0
            assert stats2.address_taken == 0

            stats_db = tgt.get_stats()
            assert stats_db["functions"] == 2
            assert stats_db["call_edges"] == 1
            assert stats_db["address_flows"] == 1
            assert stats_db["address_taken_functions"] == 1
        finally:
            tgt.close()

    def test_empty_file_paths_is_noop(self, tmp_path):
        """Empty file_paths returns an empty stats record without error."""
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        SummaryDB(src_path).close()
        tgt = SummaryDB(tgt_path)
        try:
            stats = tgt.import_unit_data(src_path, [])
            assert stats.total() == 0
        finally:
            tgt.close()

    def test_summaries_only_when_requested(self, tmp_path):
        """Summary tables copy only when include_summaries=True."""
        src_path = tmp_path / "src.db"
        tgt_path = tmp_path / "tgt.db"
        src = SummaryDB(src_path)
        try:
            func = self._make_func("a", "/proj/shared.c")
            func.id = src.insert_function(func)
            summary = AllocationSummary(
                function_name="a", allocations=[], parameters={},
                description="noop",
            )
            src.upsert_summary(func, summary)
        finally:
            src.close()

        # Without include_summaries: only function lands.
        tgt = SummaryDB(tgt_path)
        try:
            stats = tgt.import_unit_data(src_path, ["/proj/shared.c"])
            assert stats.functions == 1
            assert "allocation_summaries" not in stats.summaries
            assert tgt.get_stats()["allocation_summaries"] == 0
        finally:
            tgt.close()

        # With include_summaries: summary table populated and remapped.
        tgt2_path = tmp_path / "tgt2.db"
        tgt2 = SummaryDB(tgt2_path)
        try:
            stats = tgt2.import_unit_data(
                src_path, ["/proj/shared.c"], include_summaries=True,
            )
            assert stats.functions == 1
            assert stats.summaries.get("allocation_summaries") == 1
            assert tgt2.get_stats()["allocation_summaries"] == 1
        finally:
            tgt2.close()
