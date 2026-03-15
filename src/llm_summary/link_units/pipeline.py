"""Shared helpers for link-unit-aware batch pipeline scripts.

Provides:
  load_link_units(path)           -- load link_units.json, return list of dicts
  topo_sort_link_units(units)     -- DFS topological sort (deps before dependents)
  build_output_index(units)       -- map output filename/basename -> link unit
  resolve_dep_db_paths(lu, idx, project_scan_dir)
                                  -- get dep DB Paths for a target
  update_link_units_file(path, units)
                                  -- write updated link_units.json back to disk
  detect_bc_alias_relations(units)
                                  -- mark bc_files-subset units as alias_of superset
"""

from __future__ import annotations

import json
from pathlib import Path


def load_link_units(link_units_path: Path) -> tuple[dict, list[dict]]:
    """Load link_units.json.

    Returns (raw_data, link_units_list).
    raw_data is the full JSON object (for write-back).
    link_units_list is the array of link unit dicts.
    """
    with open(link_units_path) as f:
        data = json.load(f)
    units = data.get("link_units", data.get("targets", []))
    return data, units


def build_output_index(link_units: list[dict]) -> dict[str, dict]:
    """Map output filename (full and basename) to link unit dict."""
    idx: dict[str, dict] = {}
    for lu in link_units:
        idx[lu["output"]] = lu
        idx[Path(lu["output"]).name] = lu
    return idx


def topo_sort_link_units(link_units: list[dict]) -> list[dict]:
    """Return link units in dependency order (deps before dependents).

    Uses iterative DFS to avoid recursion limits on large graphs.
    Raises ValueError on cycles.
    """
    by_output = build_output_index(link_units)
    visited: set[str] = set()
    in_stack: set[str] = set()
    order: list[dict] = []

    def visit(lu: dict) -> None:
        name = lu["name"]
        if name in visited:
            return
        if name in in_stack:
            raise ValueError(f"Cycle detected involving link unit '{name}'")
        in_stack.add(name)
        for dep_output in lu.get("link_deps", []):
            dep_lu = by_output.get(dep_output)
            if dep_lu:
                visit(dep_lu)
        in_stack.discard(name)
        visited.add(name)
        order.append(lu)

    for lu in link_units:
        visit(lu)

    return order


def resolve_dep_db_paths(
    lu: dict,
    output_index: dict[str, dict],
    project_scan_dir: Path,
) -> list[Path]:
    """Return existing DB paths for the direct link_deps of a link unit.

    Prefers lu["db_path"] written by batch_call_graph_gen; falls back to
    func-scans/<project>/<target>/functions.db.
    """
    dep_paths: list[Path] = []
    for dep_output in lu.get("link_deps", []):
        dep_lu = output_index.get(dep_output)
        if not dep_lu:
            continue
        db_str = dep_lu.get("db_path")
        db_path = Path(db_str) if db_str else project_scan_dir / dep_lu["name"] / "functions.db"
        if db_path.exists():
            dep_paths.append(db_path)
    return dep_paths


def update_link_units_file(link_units_path: Path, data: dict) -> None:
    """Write the (possibly mutated) link_units.json data back to disk."""
    with open(link_units_path, "w") as f:
        json.dump(data, f, indent=2)


def detect_bc_alias_relations(link_units: list[dict]) -> int:
    """Detect alias relationships from bc_files subset overlap.

    When unit A's bc_files are a strict subset of unit B's bc_files, and
    neither has objects or link_deps recorded, A is redundant: its functions
    and call graph are fully covered by B. A is marked ``alias_of: B``
    (choosing the smallest-extra-files superset as the target).

    Only considers units not already marked alias_of. Idempotent.

    Returns the number of new aliases detected.

    Typical trigger: libjpeg-turbo pattern where libjpeg.a (101 bc files)
    is a strict subset of libturbojpeg.a (113 bc files).
    """
    # Eligible: bc_files populated, no objects, no link_deps, not yet aliased
    eligible: dict[str, set[str]] = {
        u["name"]: set(u["bc_files"])
        for u in link_units
        if u.get("bc_files")
        and not u.get("objects")
        and not u.get("link_deps")
        and not u.get("alias_of")
    }

    new_aliases = 0
    for u in link_units:
        name = u["name"]
        if name not in eligible:
            continue
        bc_a = eligible[name]
        if not bc_a:
            continue

        best_target: str | None = None
        best_extra: int | None = None
        for name_b, bc_b in eligible.items():
            if name_b == name:
                continue
            if bc_a < bc_b:  # strict subset
                extra = len(bc_b) - len(bc_a)
                if best_target is None or extra < best_extra:  # type: ignore[operator]
                    best_target = name_b
                    best_extra = extra

        if best_target:
            u["alias_of"] = best_target
            new_aliases += 1

    return new_aliases


def propagate_alias_db_paths(link_units: list[dict]) -> None:
    """Copy db_path from each superset unit to its alias units.

    Call after the main processing loop once superset db_paths are known,
    so that downstream consumers (import-dep-summaries, gen-harness) resolve
    alias units to the correct DB without special-casing.
    """
    by_name = {u["name"]: u for u in link_units}
    for u in link_units:
        alias_target = u.get("alias_of")
        if alias_target and not u.get("db_path"):
            target_lu = by_name.get(alias_target)
            if target_lu and target_lu.get("db_path"):
                u["db_path"] = target_lu["db_path"]
