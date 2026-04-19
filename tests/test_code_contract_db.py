"""Tests for `code_contract_summaries` table accessors on `SummaryDB`."""

from __future__ import annotations

import pytest

from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.db import SummaryDB
from llm_summary.models import Function


@pytest.fixture
def db():
    database = SummaryDB(":memory:")
    yield database
    database.close()


def _func(name: str, source: str = "void f(void){}") -> Function:
    return Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5, source=source, signature="void(void)",
    )


class TestCodeContractAccessors:
    def test_store_and_get_roundtrip(self, db: SummaryDB) -> None:
        f = _func("foo")
        f.id = db.insert_function(f)

        summary = CodeContractSummary(
            function="foo", properties=["memsafe", "memleak"],
            requires={"memsafe": ["p != NULL"]},
            ensures={"memsafe": ["result != NULL"], "memleak": []},
            modifies={"memsafe": ["*p"]},
            notes={"memsafe": "writes one byte"},
            origin={"memsafe": ["local"]},
            noreturn=False,
        )
        db.store_code_contract_summary(f, summary, model_used="test")

        got = db.get_code_contract_summary(f.id)
        assert got is not None
        assert got.to_dict() == summary.to_dict()

    def test_upsert_replaces(self, db: SummaryDB) -> None:
        f = _func("foo")
        f.id = db.insert_function(f)

        first = CodeContractSummary(function="foo", properties=["memsafe"])
        db.store_code_contract_summary(f, first, model_used="m1")

        second = CodeContractSummary(
            function="foo", properties=["memsafe"], noreturn=True,
        )
        db.store_code_contract_summary(f, second, model_used="m2")

        got = db.get_code_contract_summary(f.id)
        assert got is not None
        assert got.noreturn is True

    def test_get_missing_returns_none(self, db: SummaryDB) -> None:
        f = _func("foo")
        f.id = db.insert_function(f)
        assert db.get_code_contract_summary(f.id) is None

    def test_needs_update_when_no_summary(self, db: SummaryDB) -> None:
        f = _func("foo", source="void f(void){return;}")
        f.id = db.insert_function(f)
        assert db.needs_code_contract_update(f) is True

    def test_needs_update_false_after_store(self, db: SummaryDB) -> None:
        f = _func("foo", source="void f(void){return;}")
        f.id = db.insert_function(f)
        db.store_code_contract_summary(
            f, CodeContractSummary(function="foo"), model_used="test",
        )
        assert db.needs_code_contract_update(f) is False
