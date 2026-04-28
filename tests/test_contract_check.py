"""Tests for the contract-check agent (phase gating, tools, parsing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_summary.agent_tools import ToolExecutor
from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.contract_check import (
    ALLOWED_TRANSITIONS,
    PHASE_TOOLS,
    ContractCheckResult,
    ContractCheckToolExecutor,
    ContractGap,
    parse_public_apis,
)
from llm_summary.db import SummaryDB
from llm_summary.models import Function

# ---------------------------------------------------------------------------
# parse_public_apis
# ---------------------------------------------------------------------------


class TestParsePublicApis:
    def test_png_export_pattern(self) -> None:
        text = """
        PNG_EXPORT(1, void, png_set_sig_bytes, (png_structrp png_ptr, int num_bytes));
        PNG_EXPORTA(2, png_uint_32, png_get_io_ptr, (png_const_structrp png_ptr), PNG_DEPRECATED);
        PNG_EXPORT(3, void, png_destroy_read_struct, (png_structpp png_ptr_ptr));
        """
        names = parse_public_apis(text)
        assert names == [
            "png_set_sig_bytes",
            "png_get_io_ptr",
            "png_destroy_read_struct",
        ]

    def test_png_export_dedup_and_order(self) -> None:
        text = (
            "PNG_EXPORT(1, void, foo, (int));\n"
            "PNG_EXPORT(2, int,  bar, (void));\n"
            "PNG_EXPORT(3, void, foo, (long));\n"  # duplicate name
        )
        assert parse_public_apis(text) == ["foo", "bar"]

    def test_naive_fallback_when_no_png_export(self) -> None:
        text = """
        int do_thing(int x);
        void other_thing(struct S *s);
        // not a function declaration
        typedef int my_t;
        """
        names = parse_public_apis(text)
        assert "do_thing" in names
        assert "other_thing" in names
        # The naive regex uses the last identifier before `(` as the name,
        # which in C declarations IS the function name, but for `typedef`
        # there is no `(` so it should be skipped.
        assert "my_t" not in names

    def test_empty_header(self) -> None:
        assert parse_public_apis("") == []
        assert parse_public_apis("// comment only\n") == []


# ---------------------------------------------------------------------------
# Phase gating in ContractCheckToolExecutor
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_db() -> SummaryDB:
    db = SummaryDB(":memory:")
    yield db
    db.close()


class TestPhaseGating:
    def test_initial_phase_is_search(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        assert ex.phase == "SEARCH"

    def test_search_blocks_db_tools(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        result = ex.execute("get_contracts", {"function_name": "foo"})
        assert "error" in result
        assert "SEARCH" in result["error"]

    def test_search_blocks_submit_gaps(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        result = ex.execute(
            "submit_gaps",
            {"library": "x", "target": "y", "summary": "", "gaps": []},
        )
        assert "error" in result

    def test_check_allows_db_tools(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        ex.execute("transition_phase", {"next_phase": "CHECK"})
        # Tool runs even though function is missing — we just want to see
        # it isn't rejected by phase gating.
        result = ex.execute("get_contracts", {"function_name": "nonexistent"})
        assert "error" in result
        assert "not found" in result["error"]

    def test_check_blocks_submit_gaps(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        ex.execute("transition_phase", {"next_phase": "CHECK"})
        result = ex.execute(
            "submit_gaps",
            {"library": "x", "target": "y", "summary": "", "gaps": []},
        )
        assert "error" in result

    def test_report_only_allows_submit_gaps(self, empty_db: SummaryDB) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        ex.execute("transition_phase", {"next_phase": "CHECK"})
        ex.execute("transition_phase", {"next_phase": "REPORT"})
        # DB tool should be blocked
        result = ex.execute("get_contracts", {"function_name": "foo"})
        assert "error" in result
        # submit_gaps should be allowed (and accepted)
        result = ex.execute(
            "submit_gaps",
            {
                "library": "lib", "target": "tgt",
                "summary": "test", "gaps": [],
            },
        )
        assert result.get("accepted") is True

    def test_invalid_transition_rejected(
        self, empty_db: SummaryDB,
    ) -> None:
        ex = ContractCheckToolExecutor(empty_db)
        # SEARCH -> REPORT is not allowed
        result = ex.execute("transition_phase", {"next_phase": "REPORT"})
        assert "error" in result
        assert ex.phase == "SEARCH"

    def test_allowed_transitions_are_linear(self) -> None:
        assert ALLOWED_TRANSITIONS == {
            "SEARCH": ["CHECK"],
            "CHECK": ["REPORT"],
        }

    def test_phase_tools_are_disjoint_for_report(self) -> None:
        # REPORT is terminal: only submit_gaps
        assert PHASE_TOOLS["REPORT"] == {"submit_gaps"}


# ---------------------------------------------------------------------------
# list_apis_without_contracts handler
# ---------------------------------------------------------------------------


def _insert_func_with_contract(db: SummaryDB, name: str) -> None:
    f = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5,
        source=f"void {name}(void) {{}}",
        signature=f"void {name}(void)",
    )
    f.id = db.insert_function(f)
    summary = CodeContractSummary(function=name, properties=["memsafe"])
    db.store_code_contract_summary(f, summary, model_used="test")


def _insert_func_no_contract(db: SummaryDB, name: str) -> None:
    f = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5,
        source=f"void {name}(void) {{}}",
        signature=f"void {name}(void)",
    )
    db.insert_function(f)


class TestListApisWithoutContracts:
    def test_returns_only_uncontracted(self, empty_db: SummaryDB) -> None:
        _insert_func_with_contract(empty_db, "has_contract_a")
        _insert_func_with_contract(empty_db, "has_contract_b")
        _insert_func_no_contract(empty_db, "no_contract")

        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({
            "api_names": [
                "has_contract_a",
                "no_contract",
                "not_in_db_at_all",
                "has_contract_b",
            ],
        })
        assert result["checked"] == 4
        assert set(result["missing"]) == {"no_contract", "not_in_db_at_all"}
        assert result["missing_count"] == 2

    def test_empty_input(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({"api_names": []})
        assert result == {"checked": 0, "missing_count": 0, "missing": []}

    def test_non_list_input_returns_error(
        self, empty_db: SummaryDB,
    ) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_apis_without_contracts({"api_names": "foo"})
        assert "error" in result


# ---------------------------------------------------------------------------
# list_public_apis handler
# ---------------------------------------------------------------------------


class TestListPublicApisHandler:
    def test_parses_header_relative_to_project(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        header = tmp_path / "lib.h"
        header.write_text(
            "PNG_EXPORT(1, void, foo, (int));\n"
            "PNG_EXPORT(2, int,  bar, (void));\n",
        )
        ex = ToolExecutor(empty_db, project_path=tmp_path)
        result = ex._tool_list_public_apis({"header_path": "lib.h"})
        assert result["api_count"] == 2
        assert result["apis"] == ["foo", "bar"]

    def test_missing_header_returns_error(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        ex = ToolExecutor(empty_db, project_path=tmp_path)
        result = ex._tool_list_public_apis({"header_path": "nope.h"})
        assert "error" in result

    def test_no_project_path_errors(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        result = ex._tool_list_public_apis({"header_path": "lib.h"})
        assert "error" in result

    def test_path_traversal_blocked(
        self, tmp_path: Path, empty_db: SummaryDB,
    ) -> None:
        outside = tmp_path.parent / "outside.h"
        outside.write_text("PNG_EXPORT(1, void, evil, (int));\n")
        try:
            ex = ToolExecutor(empty_db, project_path=tmp_path)
            result = ex._tool_list_public_apis({
                "header_path": "../outside.h",
            })
            assert "error" in result
        finally:
            outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# submit_gaps handler echoes input with accepted=True
# ---------------------------------------------------------------------------


class TestSubmitGaps:
    def test_echoes_input(self, empty_db: SummaryDB) -> None:
        ex = ToolExecutor(empty_db)
        payload = {
            "library": "libpng", "target": "png_static",
            "summary": "audit complete",
            "gaps": [{
                "function": "png_init_io",
                "category": "missing_requires",
                "evidence_source": "png.h:42",
                "evidence_quote": "fp must be a writeable FILE*",
                "suggested_clause": "fp != NULL",
            }],
        }
        result = ex._tool_submit_gaps(payload)
        assert result["accepted"] is True
        assert result["library"] == "libpng"
        assert result["gaps"] == payload["gaps"]


# ---------------------------------------------------------------------------
# ContractGap / ContractCheckResult dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_gap_roundtrip(self) -> None:
        d = {
            "function": "f", "category": "ordering",
            "property": "memsafe",
            "evidence_source": "manual.txt:12",
            "evidence_quote": "must call setup() first",
            "suggested_clause": "state == INITIALIZED",
            "explanation": "ordering w/ setup",
        }
        g = ContractGap.from_dict(d)
        assert g.to_dict() == d

    def test_result_to_dict(self) -> None:
        gaps = [
            ContractGap(
                function="f", category="missing_contract",
                evidence_source="png.h:1", evidence_quote="...",
                suggested_clause="...",
            ),
        ]
        r = ContractCheckResult(
            library="libpng", target="png_static",
            summary="x", gaps=gaps, completed=True,
        )
        d = r.to_dict()
        assert d["library"] == "libpng"
        assert d["gap_count"] == 1
        assert d["completed"] is True
        assert len(d["gaps"]) == 1
