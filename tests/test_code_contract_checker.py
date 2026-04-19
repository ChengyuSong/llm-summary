"""Tests for `code_contract.checker`: entry detection + witness chains."""

from __future__ import annotations

import pytest

from llm_summary.code_contract.checker import (
    Obligation,
    check_entries,
    find_entry_functions,
)
from llm_summary.code_contract.models import CodeContractSummary
from llm_summary.db import SummaryDB
from llm_summary.models import CallEdge, Function


@pytest.fixture
def db():
    database = SummaryDB(":memory:")
    yield database
    database.close()


def _add_func(db: SummaryDB, name: str, source: str = "void f(void){}") -> Function:
    func = Function(
        name=name, file_path=f"/tmp/{name}.c",
        line_start=1, line_end=5, source=source, signature="void(void)",
    )
    func.id = db.insert_function(func)
    return func


class TestFindEntryFunctions:
    def test_no_callers_means_entry(self, db: SummaryDB) -> None:
        a = _add_func(db, "main")
        b = _add_func(db, "helper")
        assert a.id is not None and b.id is not None
        db.add_call_edge(CallEdge(caller_id=a.id, callee_id=b.id))

        entries = find_entry_functions(db)
        assert "main" in entries
        assert "helper" not in entries

    def test_restrict_to_changes_caller_set(self, db: SummaryDB) -> None:
        a = _add_func(db, "outer")
        b = _add_func(db, "mid")
        c = _add_func(db, "leaf")
        assert a.id is not None and b.id is not None and c.id is not None
        db.add_call_edge(CallEdge(caller_id=a.id, callee_id=b.id))
        db.add_call_edge(CallEdge(caller_id=b.id, callee_id=c.id))

        # Restricted to {mid, leaf}: mid has no caller in the restricted set,
        # so mid is the entry; outer is filtered out.
        entries = find_entry_functions(db, restrict_to=["mid", "leaf"])
        assert "mid" in entries
        assert "leaf" not in entries
        assert "outer" not in entries


class TestCheckEntries:
    def _store(self, db: SummaryDB, func: Function, summary: CodeContractSummary) -> None:
        db.store_code_contract_summary(func, summary, model_used="test")

    def test_local_requires_surfaced_at_entry(self, db: SummaryDB) -> None:
        main = _add_func(db, "main")
        self._store(db, main, CodeContractSummary(
            function="main", properties=["memsafe"],
            requires={"memsafe": ["argv != NULL"]},
            origin={"memsafe": ["local"]},
        ))

        obs = check_entries(db, entries=["main"])
        assert len(obs) == 1
        ob: Obligation = obs[0]
        assert ob.entry_function == "main"
        assert ob.property == "memsafe"
        assert ob.predicate == "argv != NULL"
        assert len(ob.witness_chain) == 1
        assert ob.witness_chain[0].origin == "local"

    def test_trivial_requires_filtered_out(self, db: SummaryDB) -> None:
        main = _add_func(db, "main")
        self._store(db, main, CodeContractSummary(
            function="main", properties=["memsafe"],
            requires={"memsafe": ["true", "(no observable effect)"]},
            origin={"memsafe": ["local", "local"]},
        ))
        assert check_entries(db, entries=["main"]) == []

    def test_witness_chain_walks_callee(self, db: SummaryDB) -> None:
        main = _add_func(db, "main")
        helper = _add_func(db, "helper")
        # main propagates helper's first requires verbatim.
        self._store(db, helper, CodeContractSummary(
            function="helper", properties=["memsafe"],
            requires={"memsafe": ["len > 0"]},
            origin={"memsafe": ["local"]},
        ))
        self._store(db, main, CodeContractSummary(
            function="main", properties=["memsafe"],
            requires={"memsafe": ["len > 0"]},
            origin={"memsafe": ["helper:0"]},
        ))

        obs = check_entries(db, entries=["main"])
        assert len(obs) == 1
        chain = obs[0].witness_chain
        # Chain: main → helper (terminating at "local").
        assert [s.function for s in chain] == ["main", "helper"]
        assert chain[-1].origin == "local"

    def test_chain_cycle_is_safe(self, db: SummaryDB) -> None:
        # A pathological case: two functions cite each other as origin.
        a = _add_func(db, "a")
        b = _add_func(db, "b")
        self._store(db, a, CodeContractSummary(
            function="a", properties=["memsafe"],
            requires={"memsafe": ["x != NULL"]},
            origin={"memsafe": ["b:0"]},
        ))
        self._store(db, b, CodeContractSummary(
            function="b", properties=["memsafe"],
            requires={"memsafe": ["x != NULL"]},
            origin={"memsafe": ["a:0"]},
        ))
        # Should terminate, not loop. Both functions are entries (no callers).
        obs = check_entries(db)
        # Each entry surfaces the same requires once; chain stops at the
        # cycle without infinite recursion.
        assert len(obs) == 2
        for ob in obs:
            assert len(ob.witness_chain) <= 4
