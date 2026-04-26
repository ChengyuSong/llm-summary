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
  source_files_for_objects(cc_entries, objects, build_dir)
                                  -- resolve object/bc paths to source files
  compute_unit_source_files(units, cc_entries, build_dir)
                                  -- per-unit set of source files (alias-aware)
  detect_source_set_relations(units, source_files_by_unit)
                                  -- mark equivalent/superset units (alias_of, imported_from)
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_summary.asm_extractor import ASM_EXTENSIONS
from llm_summary.extractor import C_EXTENSIONS

SCAN_EXTENSIONS = C_EXTENSIONS | ASM_EXTENSIONS


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

    Honours both ``link_deps`` (linker dependency) and ``imported_from``
    (source-set superset relation, see ``detect_source_set_relations``).
    The latter ensures importers run after the smaller unit whose data
    they will copy.

    Uses recursive DFS over the merged dep graph. Raises ValueError on
    cycles.
    """
    by_output = build_output_index(link_units)
    by_name = {lu["name"]: lu for lu in link_units}
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
        for dep_name in lu.get("imported_from", []):
            dep_lu = by_name.get(dep_name)
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


def _bare_stem(p: Path) -> str:
    """Strip all extensions: adler32.c.o -> adler32, adler32.bc -> adler32."""
    s = p.stem
    while "." in s:
        s = Path(s).stem
    return s


def source_files_for_objects(
    cc_entries: list[dict],
    objects: list[str],
    build_dir: Path,
) -> list[str]:
    """Return source files whose compiled output matches an object file path.

    Used by both the scan pipeline and source-set-aware link unit relation
    detection. Matching strategy (in order):
      1. Exact absolute output path match.
      2. Resolve relative object against build_dir.
      3. Stem-based fallback: strip mangled prefixes from the object name
         (e.g. ``libtestutil-lib-opt.o -> opt``) and match against source
         stems.
    """
    idx: dict[str, str] = {}
    stem_idx: dict[str, list[str]] = {}
    for entry in cc_entries:
        output = entry.get("output", "")
        src = entry.get("file", "")
        if not (src and Path(src).suffix.lower() in SCAN_EXTENSIONS):
            continue
        if output:
            out_path = Path(output)
            if not out_path.is_absolute():
                directory = entry.get("directory", "")
                if directory:
                    out_path = Path(directory) / out_path
                else:
                    out_path = build_dir / out_path
            idx[str(out_path)] = src
        src_stem = Path(src).stem
        stem_idx.setdefault(src_stem, []).append(src)

    sources: list[str] = []
    seen: set[str] = set()
    for obj in objects:
        src = idx.get(obj)
        if not src and not Path(obj).is_absolute():
            src = idx.get(str(build_dir / obj))
        if not src:
            obj_stem = _bare_stem(Path(obj))
            if "-lib-" in obj_stem:
                obj_stem = obj_stem.split("-lib-", 1)[1]
            candidates = stem_idx.get(obj_stem, [])
            if len(candidates) == 1:
                src = candidates[0]
        if src and src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


def compute_unit_source_files(
    link_units: list[dict],
    cc_entries: list[dict],
    build_dir: Path,
) -> dict[str, set[str]]:
    """Resolve each link unit to its set of source files.

    Per unit, prefers ``objects`` (linker command, exact output match) and
    falls back to ``bc_files`` (pure-LTO builds where only .bc paths are
    captured). Aliased units inherit the empty set — callers should not
    process them.
    """
    out: dict[str, set[str]] = {}
    for u in link_units:
        if u.get("alias_of"):
            out[u["name"]] = set()
            continue
        objects = u.get("objects", [])
        bc_files = u.get("bc_files", [])
        resolve_from = objects if objects else bc_files
        sources = source_files_for_objects(cc_entries, resolve_from, build_dir)
        out[u["name"]] = set(sources)
    return out


def _kind_rank(unit: dict) -> int:
    """Canonical preference for choosing the alias target.

    Lower rank wins. Static libs are preferred (they are usually built
    first and have the most stable scan), then shared libs, then
    executables, then anything else.
    """
    kind = unit.get("type") or unit.get("target_type") or ""
    if "static" in kind:
        return 0
    if "shared" in kind:
        return 1
    if "exec" in kind:
        return 2
    return 3


def detect_source_set_relations(
    link_units: list[dict],
    source_files_by_unit: dict[str, set[str]],
) -> int:
    """Mark equivalent and strict-superset units from per-unit source sets.

    Complements ``detect_bc_alias_relations`` (which is gated on no-objects
    units and compares bc_files paths — too strict for static/shared splits
    where CMake puts the .o files in different directories). Operates on
    the resolved source-file sets, so static and shared variants of the
    same library are caught even though their .o paths differ.

    First pass — equality:
      Pairs of units with equal non-empty source sets and no ``link_deps``
      get one marked ``alias_of`` the other. Canonical pick: static
      library > shared library > executable, ties broken by name.
      Units with ``link_deps`` are excluded because their linker-resolved
      structure may differ even when source sets match (e.g. executables).

    Second pass — strict superset:
      For each non-aliased unit B, find the LARGEST A whose source set is
      a strict subset of B's. Set ``B.imported_from = [A.name]`` so that
      the scan / summarize pipeline can copy A's analysis instead of
      re-doing the shared portion. Largest-subset ensures import chains
      compound efficiently when multiple subsets exist (A ⊂ B ⊂ C → C
      imports from B, not A).

    Idempotent: clears stale ``imported_from`` whose subset relation no
    longer holds; re-applies ``alias_of`` consistently.

    Returns the number of relations added or changed.
    """
    by_name = {u["name"]: u for u in link_units}
    new_relations = 0

    # ---- Pass 1: equality (alias_of) ----
    groups: dict[frozenset[str], list[dict]] = {}
    for name, files in source_files_by_unit.items():
        if not files:
            continue
        u = by_name.get(name)
        if u is None:
            continue
        if u.get("link_deps"):
            continue
        # Allow alias overlap: re-canonicalise even when one is set, so the
        # detection stays idempotent across re-runs.
        groups.setdefault(frozenset(files), []).append(u)

    for members in groups.values():
        if len(members) < 2:
            continue
        members_sorted = sorted(
            members, key=lambda u: (_kind_rank(u), u["name"])
        )
        canonical = members_sorted[0]
        # The canonical itself must not be aliased away.
        if canonical.get("alias_of"):
            canonical.pop("alias_of", None)
            new_relations += 1
        for u in members_sorted[1:]:
            if u.get("alias_of") != canonical["name"]:
                u["alias_of"] = canonical["name"]
                new_relations += 1

    # ---- Pass 2: strict superset (imported_from) ----
    eligible_files: dict[str, frozenset[str]] = {}
    for u in link_units:
        if u.get("alias_of"):
            continue
        u_files = source_files_by_unit.get(u["name"])
        if u_files:
            eligible_files[u["name"]] = frozenset(u_files)

    for u in link_units:
        if u.get("alias_of"):
            continue
        name = u["name"]
        my_files = eligible_files.get(name)
        if not my_files:
            continue

        best_name: str | None = None
        best_size = -1
        for other_name, other_files in eligible_files.items():
            if other_name == name:
                continue
            if other_files < my_files:  # strict subset
                if len(other_files) > best_size:
                    best_name = other_name
                    best_size = len(other_files)

        if best_name is not None:
            new_value = [best_name]
            shared_files = sorted(eligible_files[best_name])
            changed = False
            if u.get("imported_from") != new_value:
                u["imported_from"] = new_value
                changed = True
            if u.get("imported_files") != shared_files:
                u["imported_files"] = shared_files
                changed = True
            if changed:
                new_relations += 1
        elif u.get("imported_from") or u.get("imported_files"):
            # Subset relation no longer holds — drop the stale fields.
            u.pop("imported_from", None)
            u.pop("imported_files", None)
            new_relations += 1

    return new_relations
