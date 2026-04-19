"""Tests for `code_contract.models.CodeContractSummary`."""

from __future__ import annotations

from llm_summary.code_contract.models import (
    PROPERTIES,
    CodeContractSummary,
    is_nontrivial,
)


class TestIsNontrivial:
    def test_trivial_strings(self) -> None:
        for s in ("", "true", "1", "TT", "(no observable effect)",
                  "no resource acquired", "Nothing acquired"):
            assert not is_nontrivial(s)

    def test_real_predicate(self) -> None:
        assert is_nontrivial("ptr != NULL")
        assert is_nontrivial("len <= cap")


class TestCodeContractSummary:
    def test_roundtrip_dict(self) -> None:
        s = CodeContractSummary(
            function="foo",
            properties=list(PROPERTIES),
            requires={"memsafe": ["p != NULL"]},
            ensures={"memsafe": ["result != NULL"]},
            modifies={"memsafe": ["*p"]},
            notes={"memsafe": "writes one byte"},
            origin={"memsafe": ["local"]},
            noreturn=True,
        )
        d = s.to_dict()
        s2 = CodeContractSummary.from_dict(d)
        assert s2.to_dict() == d

    def test_has_requires_skips_trivial(self) -> None:
        s = CodeContractSummary(
            function="foo",
            properties=["memsafe"],
            requires={"memsafe": ["true", "(no observable effect)"]},
        )
        assert not s.has_requires("memsafe")
        s.requires["memsafe"].append("p != NULL")
        assert s.has_requires("memsafe")

    def test_to_annotated_source_skips_trivial(self) -> None:
        s = CodeContractSummary(
            function="foo",
            properties=["memsafe", "memleak"],
            requires={"memsafe": ["p != NULL", "true"], "memleak": []},
            ensures={"memsafe": ["result != NULL"], "memleak": []},
            modifies={"memsafe": ["*p"], "memleak": []},
            notes={"memsafe": "writes one byte", "memleak": ""},
        )
        body = "void foo(char *p) { *p = 1; }"
        out = s.to_annotated_source(body)
        # Non-trivial entries appear; trivial "true" is skipped.
        assert "// @requires[memsafe]: p != NULL" in out
        assert "// @ensures[memsafe]: result != NULL" in out
        assert "// @modifies[memsafe]: *p" in out
        assert "// @notes[memsafe]: writes one byte" in out
        assert "true" not in out.split(body, 1)[0]
        # Body comes after the header, on its own line.
        assert out.endswith(body)
        # memleak (no contents) produces no header lines.
        assert "memleak" not in out

    def test_to_annotated_source_emits_noreturn(self) -> None:
        s = CodeContractSummary(function="foo", noreturn=True)
        out = s.to_annotated_source("void foo(void) { abort(); }")
        assert out.startswith("// @noreturn: true")
