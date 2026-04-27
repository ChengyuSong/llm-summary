"""Tests for the chunked (large-function) path in CodeContractPass.

Covers:
  - DB cache helpers (`get/update_function_block_contract_summary`).
  - `build_skeleton_line_map` — original→skeleton line mapping.
  - `inline_callee_contracts_for_block` — Phase A callee inlining.
  - `inline_callee_contracts_in_skeleton` — skeleton-level callee inlining
    skips callsites collapsed into a block summary.
  - `_build_chunked_source` end-to-end with a mock LLM: Phase A calls land
    on each block, summaries cache, second prop reuses cache.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_summary.code_contract.inliner import (
    inline_callee_contracts_for_block,
    inline_callee_contracts_in_skeleton,
)
from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.code_contract.pass_ import CodeContractPass
from llm_summary.code_contract.prompts import BLOCK_RESPONSE_SCHEMA
from llm_summary.db import SummaryDB
from llm_summary.llm.base import LLMBackend, LLMResponse, make_json_response_format
from llm_summary.models import (
    Function,
    FunctionBlock,
    build_skeleton,
    build_skeleton_line_map,
)

# ── DB helpers ──────────────────────────────────────────────────────────


@pytest.fixture
def db() -> Any:
    database = SummaryDB(":memory:")
    yield database
    database.close()


def _insert_func_with_blocks(
    db: SummaryDB, name: str, source: str, blocks: list[FunctionBlock],
) -> tuple[Function, list[FunctionBlock]]:
    f = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=1 + source.count("\n"),
        source=source, signature=f"void {name}(void)",
    )
    f.id = db.insert_function(f)
    for b in blocks:
        b.function_id = f.id
    db.insert_function_blocks(blocks)
    return f, blocks


class TestContractBlockCache:
    def test_get_returns_none_when_unset(self, db: SummaryDB) -> None:
        f, blocks = _insert_func_with_blocks(
            db, "foo", "void foo(void){}",
            [FunctionBlock(
                function_id=None, kind="block", label="case A:",
                line_start=1, line_end=1, source="case A: break;",
            )],
        )
        block_id = blocks[0].id
        assert block_id is not None
        assert db.get_function_block_contract_summary(block_id, "memsafe") is None

    def test_update_then_get_roundtrip(self, db: SummaryDB) -> None:
        f, blocks = _insert_func_with_blocks(
            db, "foo", "void foo(void){}",
            [FunctionBlock(
                function_id=None, kind="block", label="case A:",
                line_start=1, line_end=1, source="case A: break;",
            )],
        )
        block_id = blocks[0].id
        assert block_id is not None
        db.update_function_block_contract_summary(
            block_id, "memsafe", "no-op",
        )
        assert (
            db.get_function_block_contract_summary(block_id, "memsafe")
            == "no-op"
        )
        # Other props untouched.
        assert db.get_function_block_contract_summary(block_id, "overflow") is None

    def test_update_merges_props(self, db: SummaryDB) -> None:
        f, blocks = _insert_func_with_blocks(
            db, "foo", "void foo(void){}",
            [FunctionBlock(
                function_id=None, kind="block", label="case A:",
                line_start=1, line_end=1, source="case A: break;",
            )],
        )
        block_id = blocks[0].id
        assert block_id is not None
        db.update_function_block_contract_summary(block_id, "memsafe", "ms")
        db.update_function_block_contract_summary(block_id, "overflow", "of")
        assert db.get_function_block_contract_summary(block_id, "memsafe") == "ms"
        assert db.get_function_block_contract_summary(block_id, "overflow") == "of"


# ── build_skeleton_line_map ────────────────────────────────────────────


class TestBuildSkeletonLineMap:
    def test_no_blocks_is_identity(self) -> None:
        m = build_skeleton_line_map(1, 5, [])
        assert m == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    def test_single_block_collapses_body(self) -> None:
        # Function lines 10..16 in file. Block covers file lines 12..15.
        # Source-relative: function starts at 10, so 0-based indices 0..6.
        # Block 0-based: rel_start=2 (label), rel_end=5 (last body line).
        # Skeleton: keep line 0,1 (func head), keep line 2 (label), insert
        # 1 summary line, suppress 3..5, keep line 6 (func tail).
        block = FunctionBlock(
            function_id=1, kind="switch_case", label="case A:",
            line_start=12, line_end=15, source="case A:\n  x;\n  y;\n  z;",
        )
        m = build_skeleton_line_map(10, 7, [block])
        # Lines 0,1,2 unchanged. Line 3..5 suppressed → not in map. Line 6
        # shifts by +1 for the inserted summary minus 3 suppressed = -2.
        # skel layout: 0,1,2,SUMMARY,6 → indices 0,1,2,3,4
        assert m == {0: 0, 1: 1, 2: 2, 6: 4}

    def test_two_blocks(self) -> None:
        # Function lines 1..20 (20 lines). Two blocks:
        #   B1: file 5..7 (rel 4..6) — 1 label + 2 body
        #   B2: file 12..14 (rel 11..13) — 1 label + 2 body
        blocks = [
            FunctionBlock(
                function_id=1, kind="block", label="b1",
                line_start=5, line_end=7, source="b1\n  x;\n  y;",
            ),
            FunctionBlock(
                function_id=1, kind="block", label="b2",
                line_start=12, line_end=14, source="b2\n  a;\n  b;",
            ),
        ]
        m = build_skeleton_line_map(1, 20, blocks)
        # Skeleton layout (build_skeleton output, 18 lines total):
        #   skel 0..3 = orig 0..3
        #   skel 4    = orig 4  (B1 label)
        #   skel 5    = B1 summary
        #   skel 6..9 = orig 7..10
        #   skel 10   = orig 11 (B2 label)
        #   skel 11   = B2 summary
        #   skel 12..17 = orig 14..19
        assert m[0] == 0
        assert m[3] == 3
        assert m[4] == 4  # B1 label
        assert 5 not in m and 6 not in m  # B1 body suppressed
        assert m[7] == 6  # first line after B1
        assert m[10] == 9
        assert m[11] == 10  # B2 label
        assert 12 not in m and 13 not in m
        assert m[14] == 12
        assert m[19] == 17

    def test_consistent_with_build_skeleton(self) -> None:
        """The mapping must agree with what build_skeleton actually emits."""
        source_lines = [f"L{i}" for i in range(20)]
        source = "\n".join(source_lines)
        block = FunctionBlock(
            function_id=1, id=99, kind="block", label="bb",
            line_start=5, line_end=10, source="\n".join(source_lines[4:10]),
        )
        skel = build_skeleton(source, 1, [block], {99: "summary"})
        m = build_skeleton_line_map(1, 20, [block])
        skel_lines = skel.splitlines()
        for orig_idx, skel_idx in m.items():
            assert skel_lines[skel_idx] == source_lines[orig_idx], (
                f"mismatch at orig {orig_idx} → skel {skel_idx}: "
                f"{skel_lines[skel_idx]!r} vs {source_lines[orig_idx]!r}"
            )


# ── Block / skeleton inliners ──────────────────────────────────────────


def _make_func_with_callsites() -> Function:
    """Function whose body has 2 callsites: one inside a block, one outside."""
    source = "\n".join([
        "void f(int *p) {",        # 0  (file line 1)
        "  helper(p);",            # 1  outside-block callsite
        "  switch (op) {",         # 2
        "  case 1:",               # 3  block label
        "    inner(p);",           # 4  inside-block callsite
        "    break;",              # 5
        "  }",                     # 6
        "}",                       # 7
    ])
    f = Function(
        name="f", file_path="/tmp/f.c", line_start=1, line_end=8,
        source=source, signature="void f(int *p)",
        callsites=[
            {"callee": "helper", "line_in_body": 1},
            {"callee": "inner", "line_in_body": 4},
        ],
    )
    return f


def _seed_summaries() -> dict[str, CodeContractSummary]:
    return {
        "helper": CodeContractSummary(
            function="helper", properties=["memsafe"],
            requires={"memsafe": ["p != NULL"]},
            ensures={"memsafe": ["*p initialized"]},
        ),
        "inner": CodeContractSummary(
            function="inner", properties=["memsafe"],
            requires={"memsafe": ["p != NULL"]},
            ensures={"memsafe": []},
        ),
    }


class TestBlockInliner:
    def test_inlines_only_inside_block(self) -> None:
        f = _make_func_with_callsites()
        summaries = _seed_summaries()
        # Block covers file lines 4..6 (rel 3..5). `inner` callsite is at
        # rel 4 (inside block); `helper` is at rel 1 (outside).
        block = FunctionBlock(
            function_id=1, kind="switch_case", label="case 1:",
            line_start=4, line_end=6,
            source="\n".join(f.source.splitlines()[3:6]),
        )
        out = inline_callee_contracts_for_block(
            f, summaries, block, "memsafe",
        )
        assert "inner contract for memsafe" in out
        assert "helper contract for memsafe" not in out
        # The hint should appear ABOVE the inner(p); line.
        lines = out.splitlines()
        inner_hint_idx = next(
            i for i, line in enumerate(lines)
            if "inner contract for memsafe" in line
        )
        inner_call_idx = next(
            i for i, line in enumerate(lines) if "inner(p)" in line
        )
        assert inner_hint_idx < inner_call_idx

    def test_no_inscope_callees_returns_block_unchanged(self) -> None:
        f = _make_func_with_callsites()
        block = FunctionBlock(
            function_id=1, kind="switch_case", label="case 1:",
            line_start=4, line_end=6,
            source="\n".join(f.source.splitlines()[3:6]),
        )
        out = inline_callee_contracts_for_block(
            f, {}, block, "memsafe",
        )
        assert out == block.source


class TestSkeletonInliner:
    def test_skips_inside_block_callsites(self) -> None:
        f = _make_func_with_callsites()
        summaries = _seed_summaries()
        block = FunctionBlock(
            function_id=1, id=42, kind="switch_case", label="case 1:",
            line_start=4, line_end=6,
            source="\n".join(f.source.splitlines()[3:6]),
        )
        skeleton = build_skeleton(
            f.llm_source, f.line_start, [block], {42: "fills *p"},
        )
        line_map = build_skeleton_line_map(
            f.line_start, len(f.source.splitlines()), [block],
        )
        out = inline_callee_contracts_in_skeleton(
            f, summaries, skeleton, line_map, "memsafe",
        )
        # `helper` (outside the block) must appear as a skeleton-level hint.
        assert "helper contract for memsafe" in out
        # `inner` (inside the block) must NOT — its info is in the block summary.
        assert "inner contract for memsafe" not in out


# ── End-to-end: _build_chunked_source ──────────────────────────────────


class _SequencedMockLLM(LLMBackend):
    """Returns canned LLMResponse contents in order; records prompts."""

    def __init__(self, contents: list[str]) -> None:
        super().__init__(model="mock")
        self._contents = list(contents)
        self.prompts: list[str] = []

    @property
    def default_model(self) -> str:
        return "mock"

    def complete(
        self, prompt: str, system: str | None = None,
        cache_system: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        return ""

    def complete_with_metadata(
        self, prompt: str, system: str | None = None,
        cache_system: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        if not self._contents:
            raise AssertionError("mock LLM exhausted")
        content = self._contents.pop(0)
        return LLMResponse(
            content=content, model="mock",
            input_tokens=10, output_tokens=5,
        )


class TestBuildChunkedSource:
    def test_phase_a_calls_one_per_block_and_caches(
        self, db: SummaryDB,
    ) -> None:
        f = _make_func_with_callsites()
        f.id = db.insert_function(f)
        block = FunctionBlock(
            function_id=f.id, kind="switch_case", label="case 1:",
            line_start=4, line_end=6,
            source="\n".join(f.source.splitlines()[3:6]),
        )
        db.insert_function_blocks([block])
        assert block.id is not None

        llm = _SequencedMockLLM([json.dumps({"summary": "fills *p"})])
        block_format = make_json_response_format(
            BLOCK_RESPONSE_SCHEMA, name="block_summary",
        )

        p = CodeContractPass(db=db, model="mock", llm=llm)
        out = p._build_chunked_source(
            f, _seed_summaries(), [block], "memsafe", block_format,
        )

        # Skeleton should mention the block label + summary comment.
        assert "case 1:" in out
        assert "fills *p" in out
        # The block body line `inner(p);` was suppressed.
        assert "inner(p);" not in out
        # Outside-block helper hint is inlined at skeleton level.
        assert "helper contract for memsafe" in out
        # Cache write happened.
        assert (
            db.get_function_block_contract_summary(block.id, "memsafe")
            == "fills *p"
        )
        # Token accounting wired through.
        assert p.calls == 1
        assert p.input_tokens == 10
        assert p.output_tokens == 5

    def test_second_call_reuses_cache(self, db: SummaryDB) -> None:
        f = _make_func_with_callsites()
        f.id = db.insert_function(f)
        block = FunctionBlock(
            function_id=f.id, kind="switch_case", label="case 1:",
            line_start=4, line_end=6,
            source="\n".join(f.source.splitlines()[3:6]),
        )
        db.insert_function_blocks([block])
        assert block.id is not None
        # Pre-seed cache so no LLM calls happen.
        db.update_function_block_contract_summary(
            block.id, "memsafe", "cached summary",
        )

        llm = _SequencedMockLLM([])  # would raise if any call was made
        block_format = make_json_response_format(
            BLOCK_RESPONSE_SCHEMA, name="block_summary",
        )
        p = CodeContractPass(db=db, model="mock", llm=llm)
        out = p._build_chunked_source(
            f, _seed_summaries(), [block], "memsafe", block_format,
        )
        assert "cached summary" in out
        assert p.calls == 0
