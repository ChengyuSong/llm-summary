#!/usr/bin/env python3
"""Batch scan all built projects for indirect call targets.

Iterates over build-scripts/*/compile_commands.json and runs the Phase-1
scanner (extract functions, scan targets, find callsites) on each project.

By default, creates func-scans/<project>/functions.db for each project.
Use --dry-run to skip DB storage (in-memory only).

Produces a JSON report with per-project and aggregate statistics.

Usage:
    python scripts/batch_scan_targets.py [--verbose] [--dry-run] [-j4] [--tier 1]
"""

import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Add src to path so we can import llm_summary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpr_utils import resolve_compile_commands

import subprocess

from llm_summary.asm_extractor import ASM_EXTENSIONS, extract_asm_functions
from llm_summary.compile_commands import CompileCommandsDB
from llm_summary.db import SummaryDB
from llm_summary.extern_headers import extract_extern_headers
from llm_summary.extractor import C_EXTENSIONS, FunctionExtractor
from llm_summary.indirect.callsites import IndirectCallsiteFinder
from llm_summary.indirect.scanner import AddressTakenScanner
from llm_summary.link_units.pipeline import (
    compute_unit_source_files,
    detect_bc_alias_relations,
    detect_source_set_relations,
    load_link_units,
    source_files_for_objects,
    topo_sort_link_units,
    update_link_units_file,
)
from llm_summary.models import TargetType

SCAN_EXTENSIONS = C_EXTENSIONS | ASM_EXTENSIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BUILD_SCRIPTS_DIR = REPO_ROOT / "build-scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"


def _load_tier_map() -> dict[str, int]:
    """Load project name/dir -> tier mapping from gpr_projects.json."""
    if not GPR_PROJECTS_PATH.exists():
        return {}
    with open(GPR_PROJECTS_PATH) as f:
        projects = json.load(f)
    result: dict[str, int] = {}
    for p in projects:
        if "project_dir" in p:
            result[p["project_dir"]] = p["tier"]
        # Also map by name (for monorepo sub-projects where artifact name != project_dir)
        result[p["name"]] = p["tier"]
    return result


def _load_project_root(project_dir: Path) -> Path | None:
    """Read project_path from config.json if present."""
    config_path = project_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            p = config.get("project_path")
            if p:
                return Path(p)
        except Exception:
            pass
    return None


def _git_head_commit(repo_path: Path) -> str | None:
    """Return the short SHA of HEAD in the given repo."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_changed_files(
    repo_path: Path, old_commit: str, new_commit: str = "HEAD",
) -> tuple[set[str], set[str]]:
    """Return (changed_or_added, deleted) file sets between two commits.

    Paths are relative to repo root.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "diff", "--name-status",
             f"{old_commit}..{new_commit}"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return set(), set()
    except Exception:
        return set(), set()

    changed: set[str] = set()
    deleted: set[str] = set()
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[1]
        if status.startswith("D"):
            deleted.add(path)
        else:
            changed.add(path)
    return changed, deleted


# Re-export for back-compat: callers used to import _source_files_for_objects
# from this module. The implementation lives in link_units.pipeline so that
# the source-set-aware relation detector can share it.
_source_files_for_objects = source_files_for_objects


def _build_asm_obj_map(
    cc_path: str | Path, build_root: Path | None,
) -> dict[str, str]:
    """Map resolved asm source path -> compiled object path from compile_commands.

    The asm extractor needs the .o file to enumerate symbols via `nm`.
    """
    out: dict[str, str] = {}
    try:
        with open(cc_path) as fh:
            entries = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return out
    for entry in entries:
        src = entry.get("file", "")
        obj = entry.get("output", "")
        if not (src and obj):
            continue
        if Path(src).suffix.lower() not in ASM_EXTENSIONS:
            continue
        directory = entry.get("directory", "")
        src_path = Path(src) if Path(src).is_absolute() else Path(directory, src)
        obj_path = Path(obj)
        if not obj_path.is_absolute():
            base = Path(directory) if directory else build_root
            if base:
                obj_path = base / obj_path
        try:
            out[str(src_path.resolve())] = str(obj_path)
        except OSError:
            pass
    return out


def _scan_files(
    source_files: list[str],
    cc: CompileCommandsDB,
    db: SummaryDB,
    project_root: Path | None,
    verbose: bool,
    preprocess: bool = False,
    cc_path: str | Path | None = None,
    build_root: Path | None = None,
) -> tuple[int, int, int, list[str]]:
    """Extract functions, scan address-taken, find callsites. Returns (funcs, targets, callsites).

    Parses each source file once with libclang and runs all passes on the
    same translation unit.
    """
    extractor = FunctionExtractor(
        compile_commands=cc, project_root=project_root,
        build_root=build_root,
        enable_preprocessing=preprocess,
    )

    # Split sources by kind. Asm files don't go through libclang.
    c_files = [f for f in source_files
               if Path(f).suffix.lower() in C_EXTENSIONS]
    asm_files = [f for f in source_files
                 if Path(f).suffix.lower() in ASM_EXTENSIONS]

    # Phase 1: parse each C/C++ file once, extract functions + typedefs
    all_functions = []
    all_typedefs = []
    parsed_tus: list[tuple] = []  # (tu, file_path) for reuse

    for f in c_files:
        try:
            tu = extractor.parse_file(f)
            funcs = extractor.extract_from_tu(tu, f)
            all_functions.extend(funcs)
            all_typedefs.extend(extractor.extract_typedefs_from_tu(tu, f))
            parsed_tus.append((tu, f))
        except Exception:
            pass

    # Phase 1b: extract assembly functions (e.g. musl arch-specific .s/.S).
    # Symbols come from `nm` on the compiled object; source from label
    # boundaries in the .s file.
    if asm_files:
        asm_obj_map = _build_asm_obj_map(cc_path, build_root) if cc_path else {}
        for f in asm_files:
            try:
                obj = asm_obj_map.get(str(Path(f).resolve()))
                all_functions.extend(
                    extract_asm_functions(Path(f),
                                          Path(obj) if obj else None)
                )
            except Exception:
                pass

    # Relativize file paths before DB insert so paths are portable.
    if project_root:
        root_prefix = str(project_root) + "/"
        # Migrate existing absolute-path entries to relative so ON CONFLICT
        # matches them on re-scan instead of creating duplicates.
        db.relativize_file_paths(root_prefix)

        for func in all_functions:
            if func.file_path.startswith(root_prefix):
                func.file_path = func.file_path[len(root_prefix):]
        for td in all_typedefs:
            fp = td.get("file_path", "")
            if fp.startswith(root_prefix):
                td["file_path"] = fp[len(root_prefix):]

    db.insert_functions_batch(all_functions)
    db.insert_typedefs_batch(all_typedefs)

    # Phase 2: reuse parsed TUs for address-taken scan + callsite finding
    scanner = AddressTakenScanner(db, compile_commands=cc)
    # Build function map once (normally done per scan_files call)
    for func in db.get_all_functions():
        if func.id is not None:
            scanner._function_map[func.name] = func.id

    finder = IndirectCallsiteFinder(db, compile_commands=cc)
    for func in db.get_all_functions():
        if func.id is not None:
            key = (func.file_path, func.name, func.line_start)
            finder._function_map[key] = func.id

    callsites = []
    for tu, f in parsed_tus:
        try:
            scanner.scan_tu(tu, f)
        except Exception:
            pass
        try:
            callsites.extend(finder.find_in_tu(tu, f))
        except Exception:
            pass

    atfs = db.get_address_taken_functions()

    # Extract extern declaration headers (for import-dep)
    preprocess_failed: list[str] = []
    if cc_path:
        try:
            header_map, failed = extract_extern_headers(
                compile_commands_path=cc_path,
                project_root=project_root,
                source_files=source_files,
                verbose=verbose,
            )
            preprocess_failed = failed
            if header_map:
                updated = db.update_decl_headers(header_map)
                if verbose:
                    print(
                        f"      Extern headers: {len(header_map)} mapped, "
                        f"{updated} DB rows updated"
                    )
                # Persist header map so callgraph import can re-apply
                # it after creating stubs for sourceless functions.
                hm_path = Path(db.db_path).parent / "extern_headers.json"
                import json as _json
                hm_path.write_text(_json.dumps(header_map, sort_keys=True))
        except Exception as e:
            if verbose:
                print(f"      Extern headers: failed ({e})")

    return len(all_functions), len(atfs), len(callsites), preprocess_failed


def _scan_one_target(args: tuple) -> dict[str, Any]:
    """Scan a single link-unit target. Designed for ProcessPoolExecutor."""
    (target, source_files, match_desc, cc_entries_json_path,
     db_path_str, project_root, verbose, preprocess, build_root) = args

    target_start = time.monotonic()
    target_result: dict[str, Any] = {
        "target": target,
        "source_files": len(source_files),
        "functions": 0,
        "total_targets": 0,
        "callsites": 0,
        "preprocess_failed": [],
        "error": None,
        "timing_seconds": 0.0,
    }

    if not source_files:
        target_result["error"] = "no_source_files_matched"
        target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
        return target_result

    cc = CompileCommandsDB(Path(cc_entries_json_path))
    db = SummaryDB(db_path_str)
    try:
        n_funcs, n_targets, n_callsites, preprocess_failed = _scan_files(
            source_files, cc, db, project_root, verbose,
            preprocess=preprocess,
            cc_path=cc_entries_json_path,
            build_root=build_root,
        )
    finally:
        db.close()

    target_result["functions"] = n_funcs
    target_result["total_targets"] = n_targets
    target_result["callsites"] = n_callsites
    target_result["preprocess_failed"] = preprocess_failed
    target_result["timing_seconds"] = round(time.monotonic() - target_start, 2)
    return target_result


def scan_project_link_units(
    project_dir: Path,
    func_scans_dir: Path,
    cc_entries: list[dict],
    project_root: Path | None,
    link_units_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    preprocess: bool = False,
    jobs: int = 0,
    incremental: bool = False,
) -> dict[str, Any]:
    """Scan a project with link_units.json — one DB per target.

    Args:
        jobs: Number of parallel target workers. 0 = cpu_count, 1 = sequential.
        incremental: Only re-scan files that changed since last scan (git diff).
    """
    project_name = project_dir.name
    result: dict[str, Any] = {
        "project": project_name,
        "link_unit_mode": True,
        "targets": [],
        "source_files": 0,
        "functions": 0,
        "total_targets": 0,
        "callsites": 0,
        "timing_seconds": 0.0,
        "error": None,
    }
    start = time.monotonic()

    lu_data, raw_units = load_link_units(link_units_path)
    if not raw_units:
        result["error"] = "link_units.json has no targets"
        result["timing_seconds"] = round(time.monotonic() - start, 2)
        return result

    detect_bc_alias_relations(raw_units)
    project_scan_dir = func_scans_dir / project_name
    build_dir = Path(lu_data.get("build_dir", f"/data/csong/build-artifacts/{project_name}"))

    # Source-set-aware relation detection: after bc-alias, look for units
    # whose source-file sets are equal (alias_of) or whose source set is a
    # strict subset of another (imported_from). The latter lets the larger
    # unit copy the shared portion from the smaller unit's DB instead of
    # re-analyzing it.
    # The scan layer relativizes file_path against project_root before insert
    # (DBs stay portable). For cross-DB import to JOIN on file_path, every
    # downstream file-path comparison — unit_files, imported_files, the
    # wave-2 shared list — must use the same relative form.
    def _relativize(files_by_unit: dict[str, set[str]]) -> dict[str, set[str]]:
        if not project_root:
            return files_by_unit
        root_prefix = str(project_root) + "/"
        out: dict[str, set[str]] = {}
        for name, files in files_by_unit.items():
            out[name] = {
                p[len(root_prefix):] if p.startswith(root_prefix) else p
                for p in files
            }
        return out

    unit_files = _relativize(
        compute_unit_source_files(raw_units, cc_entries, build_dir)
    )
    n_rel = detect_source_set_relations(raw_units, unit_files)
    if n_rel > 0:
        update_link_units_file(link_units_path, lu_data)
        unit_files = _relativize(
            compute_unit_source_files(raw_units, cc_entries, build_dir)
        )
        if verbose:
            print(f"    Source-set relations: {n_rel} added/changed")

    # Pre-populate db_path so the importer can resolve its dep DB without
    # special-casing. Aliased units intentionally get no db_path here —
    # propagate_alias_db_paths handles that downstream.
    for lu in raw_units:
        if not lu.get("alias_of") and not lu.get("db_path"):
            lu["db_path"] = str(project_scan_dir / lu["name"] / "functions.db")
    if any(not u.get("alias_of") for u in raw_units):
        update_link_units_file(link_units_path, lu_data)

    link_units = topo_sort_link_units(raw_units)

    # Incremental: compute changed files via git diff
    incr_changed_rel: set[str] | None = None
    incr_deleted_rel: set[str] | None = None
    new_commit: str | None = None
    if incremental and project_root:
        new_commit = _git_head_commit(project_root)
        # Read stored commit from any existing target DB
        old_commit: str | None = None
        for lu in link_units:
            if lu.get("alias_of"):
                continue
            candidate_db = project_scan_dir / lu["name"] / "functions.db"
            if candidate_db.exists():
                tmp_db = SummaryDB(str(candidate_db))
                old_commit = tmp_db.get_scan_meta("project_commit")
                tmp_db.close()
                break

        if old_commit and new_commit and old_commit != new_commit:
            changed_rel, deleted_rel = _git_changed_files(
                project_root, old_commit, new_commit,
            )
            incr_changed_rel = {f for f in changed_rel
                                if Path(f).suffix.lower() in SCAN_EXTENSIONS}
            incr_deleted_rel = {f for f in deleted_rel
                                if Path(f).suffix.lower() in SCAN_EXTENSIONS}
            if verbose:
                print(f"    Incremental: {old_commit}..{new_commit} — "
                      f"{len(incr_changed_rel)} changed, "
                      f"{len(incr_deleted_rel)} deleted source files")
        elif old_commit and old_commit == new_commit:
            if verbose:
                print(f"    Incremental: no changes since {old_commit}")
            result["timing_seconds"] = round(time.monotonic() - start, 2)
            return result
        else:
            if verbose:
                print("    Incremental: no stored commit — full scan")

    # Write resolved compile_commands to a temp file (shared across workers)
    import tempfile as _tempfile
    tmp = _tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    import json as _json
    _json.dump(cc_entries, tmp)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    try:
        # Prepare work items for each target (skip alias units). Items
        # whose unit has `imported_from` go into wave 2 — their dep DB must
        # finish wave-1 scanning first, then we copy the shared portion
        # before scanning the residual files.
        work_items_w1: list[tuple] = []
        work_items_w2: list[tuple] = []
        # unit_name -> (source_db_path, sorted shared file list)
        import_directives: dict[str, tuple[str, list[str]]] = {}
        # unit_name -> full absolute source list (so we can fall back to
        # scanning everything if validation drops the import directive).
        full_source_by_target: dict[str, list[str]] = {}
        for lu in link_units:
            target = lu["name"]
            if lu.get("alias_of"):
                if verbose:
                    print(f"    [{target}] Skipped: alias of {lu['alias_of']}")
                continue
            objects = lu.get("objects", [])
            bc_files = lu.get("bc_files", [])
            # Prefer objects (linker command, exact output path match).
            # Fall back to bc_files stem matching when no objects recorded
            # (e.g. pure-LTO builds where only .bc paths are captured).
            resolve_from = objects if objects else bc_files
            source_files = _source_files_for_objects(cc_entries, resolve_from, build_dir)
            match_desc = f"{len(resolve_from)} {'objects' if objects else 'bc_files'}"

            if dry_run:
                db_path_str = ":memory:"
            else:
                target_dir = project_scan_dir / target
                target_dir.mkdir(parents=True, exist_ok=True)
                db_path_str = str(target_dir / "functions.db")

            # Source-set superset: split files into shared (imported from
            # the smaller unit) and residual (this unit will scan). unit_files
            # is in *relative* form (relativized above) to match the dep DB's
            # stored file_path. source_files stays absolute because the scan
            # layer needs to open them.
            imported_from = lu.get("imported_from") or []
            if imported_from and not dry_run:
                dep_name = imported_from[0]
                dep_lu = next(
                    (u for u in link_units if u["name"] == dep_name), None,
                )
                if dep_lu:
                    if project_root:
                        root_pfx = str(project_root) + "/"

                        def _rel(p: str) -> str:
                            return p[len(root_pfx):] if p.startswith(root_pfx) else p
                    else:
                        def _rel(p: str) -> str:
                            return p
                    dep_files = unit_files.get(dep_name, set())
                    rel_to_abs = {_rel(p): p for p in source_files}
                    shared_rel = sorted(set(rel_to_abs) & dep_files)
                    residual = sorted(
                        rel_to_abs[r]
                        for r in (set(rel_to_abs) - dep_files)
                    )
                    dep_db_str = (
                        dep_lu.get("db_path")
                        or str(project_scan_dir / dep_name / "functions.db")
                    )
                    import_directives[target] = (dep_db_str, shared_rel)
                    full_source_by_target[target] = list(rel_to_abs.values())
                    if verbose:
                        print(
                            f"    [{target}] importing {len(shared_rel)} "
                            f"shared files from {dep_name}, scanning "
                            f"{len(residual)} residual"
                        )
                    source_files = residual
                    match_desc = (
                        f"{len(residual)} residual after import from {dep_name}"
                    )

            # Incremental: filter to changed files, delete stale entries
            if incr_changed_rel is not None and project_root:
                root_str = str(project_root) + "/"
                # source_files are absolute; convert to relative for matching
                def _to_rel(p: str) -> str:
                    return p[len(root_str):] if p.startswith(root_str) else p

                changed_src = [f for f in source_files
                               if _to_rel(f) in incr_changed_rel]
                deleted_rels = [f for f in
                                (incr_deleted_rel or set())
                                if _to_rel(f) in (incr_deleted_rel or set())]

                if db_path_str != ":memory:" and Path(db_path_str).exists():
                    tgt_db = SummaryDB(db_path_str)
                    # Delete entries for changed/deleted files (both relative
                    # and absolute forms for backward compat with old DBs)
                    to_delete = []
                    for rel in [_to_rel(f) for f in changed_src] + deleted_rels:
                        to_delete.append(rel)
                        to_delete.append(str(project_root / rel))
                    if to_delete:
                        n_del = tgt_db.delete_functions_by_files(to_delete)
                        if verbose and n_del:
                            print(f"    [{target}] Deleted {n_del} stale function(s)")
                    tgt_db.close()

                source_files = changed_src
                if verbose:
                    print(f"    [{target}] Incremental: {len(source_files)} files to re-scan")

            item = (
                target, source_files, match_desc, str(tmp_path),
                db_path_str, project_root, verbose, preprocess, build_dir,
            )
            if target in import_directives:
                work_items_w2.append(item)
            else:
                work_items_w1.append(item)

        num_workers = jobs if jobs > 0 else (os.cpu_count() or 1)

        def _process_wave(items: list[tuple]) -> None:
            if not items:
                return
            scannable_now = [w for w in items if w[1]]
            wave_parallel = num_workers > 1 and len(scannable_now) > 1
            if wave_parallel:
                if verbose:
                    print(
                        f"    Scanning {len(scannable_now)} targets with "
                        f"{num_workers} workers"
                    )
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(_scan_one_target, item): item[0]
                        for item in items
                    }
                    for future in as_completed(futures):
                        target_name = futures[future]
                        target_result = future.result()
                        result["targets"].append(target_result)
                        if verbose:
                            tr = target_result
                            if tr["error"]:
                                desc = next(
                                    (w[2] for w in items if w[0] == target_name),
                                    "",
                                )
                                print(
                                    f"    [{target_name}] SKIP: no source "
                                    f"files matched {desc}"
                                )
                            else:
                                print(
                                    f"    [{target_name}] {tr['functions']} funcs, "
                                    f"{tr['callsites']} callsites "
                                    f"({tr['timing_seconds']}s)"
                                )
            else:
                for item in items:
                    target_name = item[0]
                    src_files = item[1]
                    desc = item[2]
                    if not src_files:
                        target_result = {
                            "target": target_name,
                            "source_files": 0,
                            "functions": 0,
                            "total_targets": 0,
                            "callsites": 0,
                            "error": "no_source_files_matched",
                            "timing_seconds": 0.0,
                        }
                        result["targets"].append(target_result)
                        if verbose:
                            print(
                                f"    [{target_name}] SKIP: no source "
                                f"files matched {desc}"
                            )
                        continue
                    if verbose:
                        print(
                            f"    [{target_name}] {len(src_files)} source "
                            f"files from {desc}"
                        )
                    target_result = _scan_one_target(item)
                    result["targets"].append(target_result)

        # Wave 1: units with no source-set dep.
        _process_wave(work_items_w1)

        # Validation: confirm the dep DB's stored file_path representation
        # matches our shared list before committing to the import. Mismatch
        # happens when scan stores a clang-resolved path (e.g.
        # ``src/wrapper/../jcapistd.c``) while compile_commands "file" — the
        # source of imported_files — is a wrapper TU
        # (``src/wrapper/jcapistd-12.c``). If the dep DB has substantial
        # content we wouldn't pick up via the JOIN, drop the import and
        # scan the full source set instead so functions aren't silently lost.
        COVERAGE_THRESHOLD = 0.9
        validated_w2: list[tuple] = []
        for item in work_items_w2:
            target_name = item[0]
            directive = import_directives.get(target_name)
            if directive is None:
                validated_w2.append(item)
                continue
            dep_db_str, shared_files = directive
            dep_db_path = Path(dep_db_str)
            if not dep_db_path.exists():
                validated_w2.append(item)
                continue
            try:
                with sqlite3.connect(str(dep_db_path)) as dep_conn:
                    n_total = dep_conn.execute(
                        "SELECT COUNT(*) FROM functions"
                    ).fetchone()[0]
                    if n_total == 0 or not shared_files:
                        n_match = 0
                    else:
                        # Chunk to stay under SQLite's variable limit.
                        n_match = 0
                        chunk = 500
                        for i in range(0, len(shared_files), chunk):
                            sub = shared_files[i:i + chunk]
                            placeholders = ",".join("?" for _ in sub)
                            n_match += dep_conn.execute(
                                f"SELECT COUNT(*) FROM functions WHERE file_path IN ({placeholders})",
                                sub,
                            ).fetchone()[0]
            except sqlite3.Error as e:
                if verbose:
                    print(
                        f"    [{target_name}] dep-DB probe failed ({e}); "
                        f"keeping import directive"
                    )
                validated_w2.append(item)
                continue

            coverage = (n_match / n_total) if n_total else 1.0
            if n_total > 0 and coverage < COVERAGE_THRESHOLD:
                full_files = full_source_by_target.get(target_name) or []
                if verbose:
                    print(
                        f"    [{target_name}] WARNING: dropping imported_from "
                        f"({n_match}/{n_total} = {coverage:.0%} of dep DB "
                        f"matches imported_files; file_path representation "
                        f"diverges). Falling back to full scan of "
                        f"{len(full_files)} files."
                    )
                import_directives.pop(target_name, None)
                fallback = (
                    item[0], full_files,
                    f"{len(full_files)} files (import skipped)",
                    item[3], item[4], item[5], item[6], item[7], item[8],
                )
                validated_w2.append(fallback)
            else:
                validated_w2.append(item)
        work_items_w2 = validated_w2

        # Wave 2: importer units. Copy shared functions from the dep DB
        # before scanning the residual files.
        for item in work_items_w2:
            target_name = item[0]
            db_path_str = item[4]
            directive = import_directives.get(target_name)
            if directive and db_path_str != ":memory:":
                dep_db_str, shared_files = directive
                if shared_files and Path(dep_db_str).exists():
                    Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)
                    tgt_db = SummaryDB(db_path_str)
                    try:
                        stats_imp = tgt_db.import_unit_data(
                            dep_db_str, shared_files,
                        )
                    finally:
                        tgt_db.close()
                    if verbose:
                        print(
                            f"    [{target_name}] imported "
                            f"{stats_imp.functions} functions, "
                            f"{stats_imp.call_edges} call edges from "
                            f"{Path(dep_db_str).parent.name}"
                        )
        _process_wave(work_items_w2)

        preprocess_failed: list[str] = []
        for tr in result["targets"]:
            result["source_files"] += tr["source_files"]
            result["functions"] += tr["functions"]
            result["total_targets"] += tr["total_targets"]
            result["callsites"] += tr["callsites"]
            preprocess_failed.extend(tr.get("preprocess_failed", []))
        result["preprocess_failed"] = preprocess_failed

        # Update stored commit in all target DBs
        if new_commit and not dry_run:
            for item in (*work_items_w1, *work_items_w2):
                db_path_str = item[4]
                if db_path_str != ":memory:" and Path(db_path_str).exists():
                    tgt_db = SummaryDB(db_path_str)
                    tgt_db.set_scan_meta("project_commit", new_commit)
                    tgt_db.close()

    finally:
        tmp_path.unlink(missing_ok=True)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def scan_project(
    project_dir: Path,
    func_scans_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
    preprocess: bool = False,
    jobs: int = 0,
    incremental: bool = False,
) -> dict[str, Any]:
    """Scan a single project and return statistics.

    Auto-routes to scan_project_link_units when link_units.json is present.
    """
    cc_path = project_dir / "compile_commands.json"
    project_name = project_dir.name

    result: dict[str, Any] = {
        "project": project_name,
        "source_files": 0,
        "functions": 0,
        "targets_by_type": {},
        "total_targets": 0,
        "callsites": 0,
        "preprocess_failed": [],
        "db_path": None,
        "timing_seconds": 0.0,
        "error": None,
    }

    start = time.monotonic()

    try:
        project_root = _load_project_root(project_dir)

        # Resolve Docker /workspace/ paths if needed
        default_build_root = Path("/data/csong/build-artifacts")
        build_dir = default_build_root / project_dir.name
        entries = resolve_compile_commands(
            cc_path,
            project_source_dir=project_root or (project_dir / "src"),
            build_dir=build_dir,
        )

        # Auto-route to link-unit mode when link_units.json is present
        link_units_path = func_scans_dir / project_name / "link_units.json"
        if link_units_path.exists():
            if verbose:
                print("  link_units.json found — scanning per target")
            return scan_project_link_units(
                project_dir=project_dir,
                func_scans_dir=func_scans_dir,
                cc_entries=entries,
                project_root=project_root,
                link_units_path=link_units_path,
                dry_run=dry_run,
                verbose=verbose,
                preprocess=preprocess,
                jobs=jobs,
                incremental=incremental,
            )

        # --- Legacy single-DB mode ---
        tmp_cc = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(entries, tmp_cc)
        tmp_cc.flush()
        tmp_cc_path = Path(tmp_cc.name)
        tmp_cc.close()

        try:
            cc = CompileCommandsDB(tmp_cc_path)
            all_files = cc.get_all_files()
            source_files = [f for f in all_files if Path(f).suffix.lower() in C_EXTENSIONS]
        finally:
            tmp_cc_path.unlink(missing_ok=True)

        if not source_files:
            result["error"] = "no_c_cpp_files"
            return result

        result["source_files"] = len(source_files)

        if verbose and project_root:
            print(f"  project_root: {project_root} (header functions will be extracted)")

        if dry_run:
            db_path_str = ":memory:"
        else:
            project_scan_dir = func_scans_dir / project_name
            project_scan_dir.mkdir(parents=True, exist_ok=True)
            db_path_str = str(project_scan_dir / "functions.db")
        result["db_path"] = None if dry_run else db_path_str

        tmp_cc2 = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump(entries, tmp_cc2)
        tmp_cc2.flush()
        tmp_cc2_path = Path(tmp_cc2.name)
        tmp_cc2.close()
        try:
            cc = CompileCommandsDB(tmp_cc2_path)
            db = SummaryDB(db_path_str)
            try:
                n_funcs, n_targets, n_callsites, preprocess_failed = _scan_files(
                    source_files, cc, db, project_root, verbose,
                    preprocess=preprocess,
                    cc_path=tmp_cc2_path,
                    build_root=build_dir,
                )
                result["preprocess_failed"] = preprocess_failed
                atfs = db.get_address_taken_functions()
                type_counts: Counter[str] = Counter()
                for atf in atfs:
                    type_counts[atf.target_type] += 1
                result["targets_by_type"] = dict(type_counts)
            finally:
                db.close()
        finally:
            tmp_cc2_path.unlink(missing_ok=True)

        result["functions"] = n_funcs
        result["total_targets"] = n_targets
        result["callsites"] = n_callsites

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _scan_worker(args: tuple) -> dict[str, Any]:
    """Worker wrapper for ProcessPoolExecutor (needs top-level picklable callable)."""
    project_dir, func_scans_dir, dry_run, verbose = args[:4]
    preprocess = args[4] if len(args) > 4 else False
    jobs = args[5] if len(args) > 5 else 0
    incremental = args[6] if len(args) > 6 else False
    return scan_project(
        Path(project_dir), Path(func_scans_dir),
        dry_run=dry_run, verbose=verbose, preprocess=preprocess, jobs=jobs,
        incremental=incremental,
    )


def _format_result(result: dict[str, Any]) -> str:
    """Format a single result for printing."""
    if result["error"]:
        return f"SKIP ({result['error']})"
    suffix = f"({result['timing_seconds']}s)"
    if result.get("link_unit_mode"):
        n_targets = len(result.get("targets", []))
        return (
            f"{n_targets} targets, "
            f"{result['functions']} funcs, "
            f"{result['callsites']} callsites {suffix}"
        )
    return (
        f"{result['functions']} funcs, "
        f"{result['total_targets']} addr-taken, "
        f"{result['callsites']} callsites {suffix}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch scan projects for indirect call targets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use in-memory DBs only (don't create func-scans/<project>/functions.db)",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Number of parallel workers (default: 1, 0 = cpu count)",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only scan projects with this tier (1, 2, or 3) from gpr_projects.json",
    )
    parser.add_argument(
        "--skip-list", type=str, default=None,
        help="File with project names to skip (one per line)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only scan projects whose name contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to at most N projects",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Run clang -E to expand macros and store preprocessed source (pp_source)",
    )
    parser.add_argument(
        "--auto-rebuild", action="store_true",
        help="When --preprocess fails (missing build artifacts), automatically rebuild "
             "the project via batch_rebuild.py and re-scan. Requires --preprocess.",
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only re-scan source files that changed since the last scan (git diff).",
    )
    args = parser.parse_args()

    if not BUILD_SCRIPTS_DIR.exists():
        print(f"Error: {BUILD_SCRIPTS_DIR} not found")
        sys.exit(1)

    # Find all projects with compile_commands.json
    projects = sorted(
        d for d in BUILD_SCRIPTS_DIR.iterdir()
        if d.is_dir() and (d / "compile_commands.json").exists()
    )

    # Filter by name
    if args.filter:
        filter_str = args.filter.lower()
        before = len(projects)
        projects = [p for p in projects if filter_str in p.name.lower()]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    # Filter by tier if requested
    if args.tier is not None:
        tier_map = _load_tier_map()
        if not tier_map:
            print(f"Warning: {GPR_PROJECTS_PATH} not found, --tier ignored")
        else:
            before = len(projects)
            projects = [p for p in projects if tier_map.get(p.name) == args.tier]
            print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    # Filter by skip list if provided
    if args.skip_list is not None:
        skip_path = Path(args.skip_list)
        if not skip_path.exists():
            print(f"Error: skip list file not found: {args.skip_list}")
            sys.exit(1)
        skip_names = set()
        with open(skip_path) as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    skip_names.add(name)
        before = len(projects)
        projects = [p for p in projects if p.name not in skip_names]
        print(f"Skip list: skipped {before - len(projects)}/{before} projects")

    # Limit number of projects
    if args.limit is not None:
        projects = projects[: args.limit]

    num_workers = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)

    print(f"Found {len(projects)} projects with compile_commands.json")
    if args.dry_run:
        print("Dry run: using in-memory DBs (no files written)")
    else:
        print(f"DB output: {FUNC_SCANS_DIR}/<project>/[<target>/]functions.db")
    if num_workers > 1:
        print(f"Parallel workers: {num_workers}")
    print()

    # Build work items: (project_dir, func_scans_dir, dry_run, verbose, preprocess, jobs, incremental)
    work_items = [
        (
            str(project_dir), str(FUNC_SCANS_DIR),
            args.dry_run, args.verbose, args.preprocess, args.jobs,
            args.incremental,
        )
        for project_dir in projects
    ]

    # Run scans
    results_by_name: dict[str, dict[str, Any]] = {}

    if num_workers <= 1:
        # Sequential
        for i, item in enumerate(work_items, 1):
            name = Path(item[0]).name
            print(f"[{i}/{len(projects)}] {name}...", end=" ", flush=True)
            result = _scan_worker(item)
            results_by_name[name] = result
            print(_format_result(result))
    else:
        # Parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_name = {}
            for item in work_items:
                name = Path(item[0]).name
                future = executor.submit(_scan_worker, item)
                future_to_name[future] = name

            for future in as_completed(future_to_name):
                completed += 1
                name = future_to_name[future]
                result = future.result()
                results_by_name[name] = result
                print(
                    f"[{completed}/{len(projects)}] {name}... "
                    f"{_format_result(result)}"
                )

    # Collect results in sorted project order
    results: list[dict[str, Any]] = []
    totals: dict[str, Any] = {
        "projects_scanned": 0,
        "projects_skipped": 0,
        "total_source_files": 0,
        "total_functions": 0,
        "total_targets": 0,
        "total_callsites": 0,
        "targets_by_type": Counter(),
    }

    for project_dir in projects:
        name = project_dir.name
        result = results_by_name[name]
        results.append(result)

        if result["error"]:
            totals["projects_skipped"] += 1
        else:
            totals["projects_scanned"] += 1
            totals["total_source_files"] += result["source_files"]
            totals["total_functions"] += result["functions"]
            totals["total_targets"] += result["total_targets"]
            totals["total_callsites"] += result["callsites"]
            totals["targets_by_type"].update(result.get("targets_by_type", {}))

    # Check for preprocessing failures
    if args.preprocess:
        failed_projects = [
            name for name, r in results_by_name.items()
            if r.get("preprocess_failed")
        ]
        if failed_projects:
            # Count total source files vs failed to determine severity
            total_sources = sum(r.get("source_files", 0) for r in results_by_name.values())
            total_failed = sum(
                len(r.get("preprocess_failed", []))
                for r in results_by_name.values()
            )
            failure_rate = total_failed / max(total_sources, 1)

            if failure_rate > 0.5 and args.auto_rebuild:
                n = len(failed_projects)
                print(f"\nPreprocessing failed for {n} project(s) "
                      f"({total_failed}/{total_sources} files). Auto-rebuilding...")
                rebuild_script = SCRIPTS_DIR / "batch_rebuild.py"
                for proj in failed_projects:
                    print(f"  Rebuilding {proj}...")
                    import subprocess as _sp
                    rc = _sp.run(
                        [sys.executable, str(rebuild_script), "--filter", proj],
                        check=False,
                    ).returncode
                    if rc != 0:
                        print(f"  ERROR: rebuild failed for {proj}, aborting.")
                        sys.exit(1)
                print(f"\nRe-scanning {len(failed_projects)} project(s) with --preprocess...")
                for proj in failed_projects:
                    proj_dir = BUILD_SCRIPTS_DIR / proj
                    re_result = scan_project(
                        proj_dir, FUNC_SCANS_DIR,
                        dry_run=args.dry_run, verbose=args.verbose,
                        preprocess=True, jobs=args.jobs,
                    )
                    results_by_name[proj] = re_result
                    still_failed = re_result.get("preprocess_failed", [])
                    if still_failed:
                        print(f"  ERROR: preprocessing still failing for {proj} after rebuild.")
                        sys.exit(1)
                    print(f"  {proj}: OK")
            elif failure_rate > 0.5:
                print(f"\nERROR: Preprocessing failed for {len(failed_projects)} project(s) "
                      f"({total_failed}/{total_sources} files):")
                for proj in failed_projects:
                    print(f"  - {proj}")
                print(
                    "\nBuild artifacts are likely missing or stale. To fix, run:\n"
                    "  python scripts/batch_rebuild.py --filter <project>\n"
                    "Then re-run this scan with --preprocess.\n"
                    "Or use --auto-rebuild to rebuild automatically."
                )
                sys.exit(1)
            else:
                print(f"\nWARNING: Preprocessing failed for {total_failed}/{total_sources} "
                      f"files in {len(failed_projects)} project(s) "
                      "(likely test/benchmark files — extern headers skipped for those)")

    # Print summary
    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects scanned: {totals['projects_scanned']}")
    print(f"  Projects skipped: {totals['projects_skipped']}")
    print(f"  Source files: {totals['total_source_files']}")
    print(f"  Functions: {totals['total_functions']}")
    print(f"  Indirect call targets: {totals['total_targets']}")
    for tt in TargetType:
        count = totals["targets_by_type"].get(tt.value, 0)
        if count > 0:
            print(f"    {tt.value}: {count}")
    print(f"  Indirect callsites: {totals['total_callsites']}")

    # Write report
    report = {
        "projects": results,
        "totals": {
            "projects_scanned": totals["projects_scanned"],
            "projects_skipped": totals["projects_skipped"],
            "total_source_files": totals["total_source_files"],
            "total_functions": totals["total_functions"],
            "total_targets": totals["total_targets"],
            "targets_by_type": dict(totals["targets_by_type"]),
            "total_callsites": totals["total_callsites"],
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"scan_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
