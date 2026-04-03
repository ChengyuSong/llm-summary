"""Tests for seed_generator module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_summary.llm.base import LLMBackend
from llm_summary.seed_generator import (
    SEED_CONVENTIONS,
    SEED_SYSTEM_PROMPT,
    SeedExecutor,
    _format_verdict_context,
    generate_seed_tests,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SEED_CODE = """\
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

extern void __ucsan_symbolize_input(void *ptr, unsigned long size, int id);

struct node {
    int value;
    struct node *next;
};

extern int target(struct node *head);

void test() {
    struct node obj2 = { .value = 42, .next = NULL };
    struct node obj1 = { .value = 10, .next = &obj2 };

    __ucsan_symbolize_input(&obj2, sizeof(obj2), 2);
    __ucsan_symbolize_input(&obj1, sizeof(obj1), 1);

    target(&obj1);
}
"""


def _always_compiles(
    code: str, ucsan_config: str, file_path: str | None,
) -> tuple[bool, str]:
    """Compile function that always succeeds."""
    return True, ""


def _never_compiles(
    code: str, ucsan_config: str, file_path: str | None,
) -> tuple[bool, str]:
    """Compile function that always fails."""
    return False, "error: unknown type name 'foo'"


# ---------------------------------------------------------------------------
# SeedExecutor tests
# ---------------------------------------------------------------------------


class TestSeedExecutor:
    """Tests for the SeedExecutor tool dispatcher."""

    def test_submit_seed_stores_code(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\nscope:\n  - target\n",
            file_path=None,
            git_tools=None,
        )
        result = executor._tool_submit_seed({
            "code": SAMPLE_SEED_CODE,
            "description": "basic linked list",
        })
        assert result["accepted"] is True
        assert result["seed_index"] == 0
        assert len(executor.seeds) == 1
        assert executor.seeds[0][0] == SAMPLE_SEED_CODE
        assert executor.seeds[0][1] == "basic linked list"

    def test_submit_multiple_seeds(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        executor._tool_submit_seed({"code": "seed1", "description": "first"})
        executor._tool_submit_seed({"code": "seed2", "description": "second"})
        assert len(executor.seeds) == 2
        assert executor.seeds[1][1] == "second"

    def test_submit_empty_code_rejected(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        result = executor._tool_submit_seed({"code": ""})
        assert "error" in result

    def test_compile_seed_success(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        result = executor._tool_compile_seed({"code": SAMPLE_SEED_CODE})
        assert result["success"] is True

    def test_compile_seed_failure(self) -> None:
        executor = SeedExecutor(
            compile_fn=_never_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        result = executor._tool_compile_seed({"code": "bad code"})
        assert result["success"] is False
        assert "error" in result["errors"]

    def test_compile_attempt_limit(self) -> None:
        executor = SeedExecutor(
            compile_fn=_never_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        for _ in range(6):
            result = executor._tool_compile_seed({"code": "bad"})
        assert "limit" in result["errors"].lower()

    def test_compile_counter_resets_after_submit(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        # Use up some compile attempts
        executor.compile_attempts = 4
        executor._tool_submit_seed({"code": "seed1"})
        assert executor.compile_attempts == 0

    def test_dispatch_unknown_tool(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        result = executor.execute("nonexistent_tool", {})
        assert "error" in result

    def test_dispatch_git_tool_without_git(self) -> None:
        executor = SeedExecutor(
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
            file_path=None,
            git_tools=None,
        )
        result = executor.execute("git_show", {"file_path": "src/main.c"})
        assert "unavailable" in result["error"]


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


class TestFormatVerdictContext:
    def test_basic_format(self) -> None:
        ctx: dict[str, Any] = {
            "hypothesis": "feasible",
            "severity": "high",
            "issue_kind": "buffer_overflow",
            "issue_description": "OOB write in loop",
            "reasoning": "index exceeds buffer size",
            "assumptions": ["palette is non-NULL"],
            "assertions": ["index < buffer_size"],
        }
        text = _format_verdict_context(ctx)
        assert "feasible" in text
        assert "buffer_overflow" in text
        assert "palette is non-NULL" in text
        assert "index < buffer_size" in text

    def test_empty_context(self) -> None:
        text = _format_verdict_context({})
        assert "unknown" in text


# ---------------------------------------------------------------------------
# Prompt content checks
# ---------------------------------------------------------------------------


class TestPromptContent:
    def test_system_prompt_mentions_symbolize(self) -> None:
        assert "__ucsan_symbolize_input" in SEED_SYSTEM_PROMPT

    def test_conventions_has_api_docs(self) -> None:
        assert "void *ptr" in SEED_CONVENTIONS
        assert "leaves first" in SEED_CONVENTIONS

    def test_conventions_has_example(self) -> None:
        assert "void test()" in SEED_CONVENTIONS
        assert "__ucsan_symbolize_input" in SEED_CONVENTIONS


# ---------------------------------------------------------------------------
# generate_seed_tests with mock LLM
# ---------------------------------------------------------------------------


class _MockToolResponse:
    """Minimal mock for LLM tool-use response."""

    def __init__(
        self,
        content: list[Any],
        stop_reason: str = "end_turn",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason


class _MockTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text
        self.thought = False


class _MockToolUseBlock:
    def __init__(self, tool_id: str, name: str, inp: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.id = tool_id
        self.name = name
        self.input = inp


class _MockLLM(LLMBackend):
    """Mock LLM that returns a sequence of tool-use responses."""

    def __init__(self, responses: list[_MockToolResponse]) -> None:
        super().__init__(model="mock")
        self._responses = list(responses)
        self._call_count = 0

    @property
    def default_model(self) -> str:
        return "mock"

    def complete(self, prompt: str, system: str | None = None,
                 cache_system: bool = False) -> str:
        return ""

    def complete_with_metadata(
        self, prompt: str, system: str | None = None,
        cache_system: bool = False, response_format: dict[str, Any] | None = None,
    ) -> Any:
        return None

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> _MockToolResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return _MockToolResponse([], "end_turn")


class TestGenerateSeedTests:
    def test_writes_seed_files(self, tmp_path: Path) -> None:
        """Agent submits one seed -> file written to output_dir."""
        responses = [
            # Turn 1: compile_seed
            _MockToolResponse(
                content=[
                    _MockToolUseBlock("t1", "compile_seed", {"code": SAMPLE_SEED_CODE}),
                ],
                stop_reason="tool_use",
            ),
            # Turn 2: submit_seed
            _MockToolResponse(
                content=[
                    _MockToolUseBlock("t2", "submit_seed", {
                        "code": SAMPLE_SEED_CODE,
                        "description": "linked list seed",
                    }),
                ],
                stop_reason="tool_use",
            ),
            # Turn 3: done
            _MockToolResponse(
                content=[_MockTextBlock("Done.")],
                stop_reason="end_turn",
            ),
        ]
        llm = _MockLLM(responses)

        ctx: dict[str, Any] = {
            "hypothesis": "feasible",
            "issue_kind": "buffer_overflow",
            "issue_description": "OOB",
            "reasoning": "test",
        }

        paths = generate_seed_tests(
            func_name="target",
            shim_code="/* shim */",
            triage_context=ctx,
            output_dir=tmp_path,
            llm=llm,
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
        )

        assert len(paths) == 1
        assert paths[0].name == "seed_target_0.c"
        assert paths[0].read_text() == SAMPLE_SEED_CODE

    def test_no_seeds_when_agent_stops_early(self, tmp_path: Path) -> None:
        """Agent stops without submitting -> no files."""
        responses = [
            _MockToolResponse(
                content=[_MockTextBlock("I cannot generate seeds.")],
                stop_reason="end_turn",
            ),
        ]
        llm = _MockLLM(responses)

        paths = generate_seed_tests(
            func_name="target",
            shim_code="/* shim */",
            triage_context={"hypothesis": "safe"},
            output_dir=tmp_path,
            llm=llm,
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
        )

        assert len(paths) == 0

    def test_multiple_seeds(self, tmp_path: Path) -> None:
        """Agent submits two seeds."""
        responses = [
            _MockToolResponse(
                content=[
                    _MockToolUseBlock("t1", "submit_seed", {
                        "code": "/* seed 0 */",
                        "description": "first",
                    }),
                ],
                stop_reason="tool_use",
            ),
            _MockToolResponse(
                content=[
                    _MockToolUseBlock("t2", "submit_seed", {
                        "code": "/* seed 1 */",
                        "description": "second",
                    }),
                ],
                stop_reason="tool_use",
            ),
            _MockToolResponse(
                content=[_MockTextBlock("Done.")],
                stop_reason="end_turn",
            ),
        ]
        llm = _MockLLM(responses)

        paths = generate_seed_tests(
            func_name="func",
            shim_code="/* shim */",
            triage_context={"hypothesis": "feasible"},
            output_dir=tmp_path,
            llm=llm,
            compile_fn=_always_compiles,
            ucsan_config="entry: test\n",
        )

        assert len(paths) == 2
        assert paths[0].name == "seed_func_0.c"
        assert paths[1].name == "seed_func_1.c"
