"""Tests for link_units.pipeline source-set-aware relation detection."""

from pathlib import Path

import pytest

from llm_summary.link_units.pipeline import (
    compute_unit_source_files,
    detect_source_set_relations,
    source_files_for_objects,
    topo_sort_link_units,
)


def _cc(src: str, output: str, directory: str = "/build") -> dict:
    return {"file": src, "output": output, "directory": directory}


class TestSourceFilesForObjects:
    def test_exact_match(self):
        cc = [_cc("/proj/a.c", "/build/a.o")]
        assert source_files_for_objects(cc, ["/build/a.o"], Path("/build")) == [
            "/proj/a.c"
        ]

    def test_relative_object_resolves_against_build_dir(self):
        cc = [_cc("/proj/a.c", "/build/sub/a.o")]
        result = source_files_for_objects(
            cc, ["sub/a.o"], Path("/build"),
        )
        assert result == ["/proj/a.c"]

    def test_stem_fallback_unique(self):
        cc = [_cc("/proj/foo.c", "/build/static/foo.o")]
        # Object path doesn't match index but stem 'foo' is unique.
        result = source_files_for_objects(
            cc, ["/build/shared/foo.o"], Path("/build"),
        )
        assert result == ["/proj/foo.c"]


class TestDetectSourceSetRelations:
    def _unit(
        self,
        name: str,
        kind: str = "static_library",
        link_deps: list[str] | None = None,
    ) -> dict:
        u: dict = {"name": name, "type": kind, "objects": []}
        if link_deps is not None:
            u["link_deps"] = link_deps
        return u

    def test_equal_sources_pick_static_as_canonical(self):
        units = [
            self._unit("libjpeg_shared", kind="shared_library"),
            self._unit("libjpeg_static", kind="static_library"),
        ]
        files = {
            "libjpeg_shared": {"/p/a.c", "/p/b.c"},
            "libjpeg_static": {"/p/a.c", "/p/b.c"},
        }
        n = detect_source_set_relations(units, files)
        assert n == 1
        assert units[0]["alias_of"] == "libjpeg_static"
        assert "alias_of" not in units[1]

    def test_executables_with_link_deps_not_aliased(self):
        units = [
            self._unit("exe1", kind="executable", link_deps=["liba.a"]),
            self._unit("exe2", kind="executable", link_deps=["liba.a"]),
        ]
        files = {
            "exe1": {"/p/main.c"},
            "exe2": {"/p/main.c"},
        }
        n = detect_source_set_relations(units, files)
        assert n == 0
        assert "alias_of" not in units[0]
        assert "alias_of" not in units[1]

    def test_strict_superset_imports_from_largest_subset(self):
        units = [
            self._unit("libjpeg"),
            self._unit("libturbojpeg"),
        ]
        files = {
            "libjpeg": {f"/p/{i}.c" for i in range(10)},
            "libturbojpeg": {f"/p/{i}.c" for i in range(15)},
        }
        n = detect_source_set_relations(units, files)
        assert n == 1
        assert units[1]["imported_from"] == ["libjpeg"]
        assert units[1]["imported_files"] == sorted(files["libjpeg"])
        assert "imported_from" not in units[0]
        assert "imported_files" not in units[0]

    def test_chain_picks_largest_subset(self):
        units = [self._unit(n) for n in ("A", "B", "C")]
        files = {
            "A": {f"/p/{i}.c" for i in range(5)},
            "B": {f"/p/{i}.c" for i in range(10)},
            "C": {f"/p/{i}.c" for i in range(15)},
        }
        n = detect_source_set_relations(units, files)
        assert n == 2  # B->A, C->B
        assert units[1]["imported_from"] == ["A"]
        assert units[2]["imported_from"] == ["B"]

    def test_idempotent(self):
        units = [
            self._unit("shared", kind="shared_library"),
            self._unit("static", kind="static_library"),
            self._unit("big"),
        ]
        files = {
            "shared": {"/p/a.c"},
            "static": {"/p/a.c"},
            "big": {"/p/a.c", "/p/b.c"},
        }
        n1 = detect_source_set_relations(units, files)
        n2 = detect_source_set_relations(units, files)
        assert n1 > 0
        assert n2 == 0

    def test_aliased_unit_skipped_for_imported_from(self):
        units = [
            self._unit("shared", kind="shared_library"),
            self._unit("static", kind="static_library"),
            self._unit("big"),
        ]
        files = {
            "shared": {"/p/a.c"},
            "static": {"/p/a.c"},
            "big": {"/p/a.c", "/p/b.c"},
        }
        detect_source_set_relations(units, files)
        # 'big' should import_from the canonical 'static', not 'shared'.
        assert units[2]["imported_from"] == ["static"]

    def test_clears_stale_imported_from(self):
        units = [self._unit("A"), self._unit("B")]
        units[1]["imported_from"] = ["A"]
        units[1]["imported_files"] = ["/p/a.c", "/p/b.c"]
        # Now A's source set is no longer a subset of B's.
        files = {
            "A": {"/p/a.c", "/p/b.c"},
            "B": {"/p/c.c"},
        }
        n = detect_source_set_relations(units, files)
        assert n == 1
        assert "imported_from" not in units[1]
        assert "imported_files" not in units[1]


class TestComputeUnitSourceFiles:
    def test_aliased_units_get_empty_set(self):
        units = [
            {"name": "A", "objects": ["/build/a.o"]},
            {"name": "A_alias", "objects": ["/build/a.o"], "alias_of": "A"},
        ]
        cc = [_cc("/proj/a.c", "/build/a.o")]
        out = compute_unit_source_files(units, cc, Path("/build"))
        assert out["A"] == {"/proj/a.c"}
        assert out["A_alias"] == set()

    def test_falls_back_to_bc_files(self):
        units = [{"name": "A", "objects": [], "bc_files": ["/build/a.bc"]}]
        cc = [_cc("/proj/a.c", "/build/a.bc")]
        out = compute_unit_source_files(units, cc, Path("/build"))
        assert out["A"] == {"/proj/a.c"}


class TestTopoSortHonoursImportedFrom:
    def test_imported_from_orders_dep_first(self):
        units = [
            {"name": "B", "output": "B", "imported_from": ["A"]},
            {"name": "A", "output": "A"},
        ]
        ordered = topo_sort_link_units(units)
        names = [u["name"] for u in ordered]
        assert names.index("A") < names.index("B")

    def test_cycle_via_imported_from_raises(self):
        units = [
            {"name": "A", "output": "A", "imported_from": ["B"]},
            {"name": "B", "output": "B", "imported_from": ["A"]},
        ]
        with pytest.raises(ValueError):
            topo_sort_link_units(units)
