#!/usr/bin/env python3
"""Batch scan all built projects for indirect call targets.

Iterates over build-scripts/*/compile_commands.json and runs the Phase-1
scanner (extract functions, scan targets, find callsites) on each project.

By default, creates func-scans/<project>/functions.db for each project.
Use --dry-run to skip DB storage (in-memory only).

Produces a JSON report with per-project and aggregate statistics.

Usage:
    python scripts/batch_scan_targets.py [--verbose] [--dry-run] [-j4] [--tier 1] [--skip-list skip.txt]
"""

import json
import os
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src to path so we can import llm_summary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.compile_commands import CompileCommandsDB
from llm_summary.db import SummaryDB
from llm_summary.extractor import FunctionExtractor
from llm_summary.indirect.callsites import IndirectCallsiteFinder
from llm_summary.indirect.scanner import AddressTakenScanner
from llm_summary.models import TargetType
from gpr_utils import resolve_compile_commands


C_EXTENSIONS = {".c", ".cpp", ".cc", ".cxx", ".c++"}

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BUILD_SCRIPTS_DIR = REPO_ROOT / "build-scripts"
FUNC_SCANS_DIR = REPO_ROOT / "func-scans"
GPR_PROJECTS_PATH = SCRIPTS_DIR / "gpr_projects.json"


def _load_tier_map() -> dict[str, int]:
    """Load project_dir -> tier mapping from gpr_projects.json."""
    if not GPR_PROJECTS_PATH.exists():
        return {}
    with open(GPR_PROJECTS_PATH) as f:
        projects = json.load(f)
    return {
        p["project_dir"]: p["tier"]
        for p in projects
        if "project_dir" in p
    }


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


def scan_project(
    project_dir: Path,
    db_path: str = ":memory:",
    verbose: bool = False,
) -> dict:
    """Scan a single project and return statistics."""
    cc_path = project_dir / "compile_commands.json"
    project_name = project_dir.name

    result = {
        "project": project_name,
        "source_files": 0,
        "functions": 0,
        "targets_by_type": {},
        "total_targets": 0,
        "callsites": 0,
        "db_path": None if db_path == ":memory:" else db_path,
        "timing_seconds": 0.0,
        "error": None,
    }

    start = time.monotonic()

    try:
        project_root = _load_project_root(project_dir)

        # Resolve Docker /workspace/ paths if needed
        DEFAULT_BUILD_ROOT = Path("/data/csong/build-artifacts")
        build_dir = DEFAULT_BUILD_ROOT / project_dir.name
        entries = resolve_compile_commands(
            cc_path,
            project_source_dir=project_root or (project_dir / "src"),
            build_dir=build_dir,
        )

        # Write resolved entries to a temporary file for CompileCommandsDB
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

        db = SummaryDB(db_path)

        if verbose and project_root:
            print(f"  project_root: {project_root} (header functions will be extracted)")

        try:
            # Extract functions
            extractor = FunctionExtractor(compile_commands=cc, project_root=project_root)
            all_functions = []
            for f in source_files:
                try:
                    functions = extractor.extract_from_file(f)
                    all_functions.extend(functions)
                except Exception:
                    pass

            db.insert_functions_batch(all_functions)
            result["functions"] = len(all_functions)

            # Scan targets
            scanner = AddressTakenScanner(db, compile_commands=cc)
            for f in source_files:
                try:
                    scanner.scan_files([f])
                except Exception:
                    pass

            atfs = db.get_address_taken_functions()
            type_counts: Counter[str] = Counter()
            for atf in atfs:
                type_counts[atf.target_type] += 1

            result["targets_by_type"] = dict(type_counts)
            result["total_targets"] = len(atfs)

            # Find callsites
            finder = IndirectCallsiteFinder(db, compile_commands=cc)
            callsites = []
            for f in source_files:
                try:
                    cs = finder.find_in_files([f])
                    callsites.extend(cs)
                except Exception:
                    pass

            result["callsites"] = len(callsites)

        finally:
            db.close()

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _scan_worker(args: tuple) -> dict:
    """Worker wrapper for ProcessPoolExecutor (needs top-level picklable callable)."""
    project_dir, db_path, verbose = args
    return scan_project(Path(project_dir), db_path=db_path, verbose=verbose)


def _format_result(result: dict) -> str:
    """Format a single result for printing."""
    if result["error"]:
        return f"SKIP ({result['error']})"
    return (
        f"{result['functions']} funcs, "
        f"{result['total_targets']} targets, "
        f"{result['callsites']} callsites "
        f"({result['timing_seconds']}s)"
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

    num_workers = args.jobs if args.jobs > 0 else os.cpu_count()

    print(f"Found {len(projects)} projects with compile_commands.json")
    if args.dry_run:
        print("Dry run: using in-memory DBs (no files written)")
    else:
        print(f"DB output: {FUNC_SCANS_DIR}/<project>/functions.db")
    if num_workers > 1:
        print(f"Parallel workers: {num_workers}")
    print()

    # Build work items: (project_dir, db_path, verbose)
    work_items = []
    for project_dir in projects:
        if args.dry_run:
            db_path = ":memory:"
        else:
            project_scan_dir = FUNC_SCANS_DIR / project_dir.name
            project_scan_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(project_scan_dir / "functions.db")
        work_items.append((str(project_dir), db_path, args.verbose))

    # Run scans
    results_by_name: dict[str, dict] = {}

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
    results = []
    totals = {
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
            totals["targets_by_type"].update(result["targets_by_type"])

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
