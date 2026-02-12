#!/usr/bin/env python3
"""Batch container function detection across all scanned projects.

Iterates over func-scans/<project>/functions.db and runs the container
detector (heuristic pre-filter + LLM confirmation) on each project.

Usage:
    python scripts/batch_container_detect.py --backend llamacpp --llm-host 192.168.1.11 --llm-port 8001
    python scripts/batch_container_detect.py --heuristic-only
    python scripts/batch_container_detect.py --backend llamacpp -j4 --min-score 7
"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src to path so we can import llm_summary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.container import ContainerDetector
from llm_summary.db import SummaryDB

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
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


def detect_project(
    db_path: str,
    project_name: str,
    backend: str | None = None,
    model: str | None = None,
    llm_host: str = "localhost",
    llm_port: int | None = None,
    min_score: int = 5,
    force: bool = False,
    verbose: bool = False,
    log_dir: str | None = None,
    disable_thinking: bool = False,
) -> dict:
    """Run container detection on a single project DB. Returns statistics."""
    result = {
        "project": project_name,
        "functions": 0,
        "candidates": 0,
        "containers_found": 0,
        "llm_calls": 0,
        "cache_hits": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "containers_by_type": {},
        "timing_seconds": 0.0,
        "error": None,
    }

    start = time.monotonic()

    try:
        db = SummaryDB(db_path)

        try:
            functions = db.get_all_functions()
            result["functions"] = len(functions)

            if not functions:
                result["error"] = "no_functions"
                return result

            # Create LLM backend if requested
            llm = None
            if backend is not None:
                from llm_summary.llm import create_backend

                backend_kwargs = {}
                if backend == "llamacpp":
                    backend_kwargs["host"] = llm_host
                    backend_kwargs["port"] = llm_port if llm_port is not None else 8080
                elif backend == "ollama":
                    port = llm_port if llm_port is not None else 11434
                    backend_kwargs["base_url"] = f"http://{llm_host}:{port}"

                if disable_thinking:
                    backend_kwargs["enable_thinking"] = False

                llm = create_backend(backend, model=model, **backend_kwargs)

            log_file = None
            if log_dir:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                log_file = str(log_path / f"{project_name}_container.log")

            detector = ContainerDetector(
                db, llm=llm, verbose=verbose,
                log_file=log_file, min_score=min_score,
                project_name=project_name,
            )

            results_map = detector.detect_all(force=force)

            stats = detector.stats
            result["candidates"] = stats["candidates"]
            result["containers_found"] = stats["containers_found"]
            result["llm_calls"] = stats["llm_calls"]
            result["cache_hits"] = stats["cache_hits"]
            result["errors"] = stats["errors"]
            result["input_tokens"] = stats["input_tokens"]
            result["output_tokens"] = stats["output_tokens"]

            # Count by container type
            type_counts: Counter[str] = Counter()
            for cs in results_map.values():
                type_counts[cs.container_type] += 1
            result["containers_by_type"] = dict(type_counts)

        finally:
            db.close()

    except Exception as e:
        result["error"] = str(e)

    result["timing_seconds"] = round(time.monotonic() - start, 2)
    return result


def _detect_worker(args: tuple) -> dict:
    """Worker wrapper for ProcessPoolExecutor."""
    return detect_project(*args)


def _format_result(result: dict) -> str:
    """Format a single result for printing."""
    if result["error"]:
        return f"SKIP ({result['error']})"
    parts = [
        f"{result['functions']} funcs",
        f"{result['candidates']} candidates",
        f"{result['containers_found']} containers",
    ]
    if result["llm_calls"]:
        parts.append(f"{result['llm_calls']} LLM calls")
    if result["cache_hits"]:
        parts.append(f"{result['cache_hits']} cached")
    total_tok = result.get("input_tokens", 0) + result.get("output_tokens", 0)
    if total_tok:
        parts.append(f"{total_tok:,} tok")
    if result["errors"]:
        parts.append(f"{result['errors']} errors")
    parts.append(f"({result['timing_seconds']}s)")
    return ", ".join(parts)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch container function detection across scanned projects"
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        help="LLM backend (omit for heuristic-only)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument(
        "--llm-host", type=str, default="localhost",
        help="LLM server hostname (default: localhost)",
    )
    parser.add_argument(
        "--llm-port", type=int, default=None,
        help="LLM server port (llamacpp: 8080, ollama: 11434)",
    )
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="Disable thinking/reasoning mode",
    )
    parser.add_argument(
        "--min-score", type=int, default=5,
        help="Minimum heuristic score for LLM confirmation (default: 5)",
    )
    parser.add_argument(
        "--heuristic-only", action="store_true",
        help="Only run heuristic scoring, skip LLM (overrides --backend)",
    )
    parser.add_argument("--force", "-f", action="store_true", help="Force re-analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON report")
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Directory for per-project LLM logs",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Parallel workers (default: 1, 0 = cpu count)",
    )
    parser.add_argument(
        "--tier", type=int, default=None,
        help="Only scan projects with this tier from gpr_projects.json",
    )
    parser.add_argument(
        "--skip-list", type=str, default=None,
        help="File with project names to skip (one per line)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only process projects matching this substring",
    )
    args = parser.parse_args()

    if args.heuristic_only:
        args.backend = None

    if not FUNC_SCANS_DIR.exists():
        print(f"Error: {FUNC_SCANS_DIR} not found")
        sys.exit(1)

    # Find all projects with functions.db
    projects = sorted(
        d for d in FUNC_SCANS_DIR.iterdir()
        if d.is_dir() and (d / "functions.db").exists()
    )

    # Filter by tier
    if args.tier is not None:
        tier_map = _load_tier_map()
        if not tier_map:
            print(f"Warning: {GPR_PROJECTS_PATH} not found, --tier ignored")
        else:
            before = len(projects)
            projects = [p for p in projects if tier_map.get(p.name) == args.tier]
            print(f"Tier {args.tier} filter: {len(projects)}/{before} projects")

    # Filter by name substring
    if args.filter:
        before = len(projects)
        projects = [p for p in projects if args.filter in p.name]
        print(f"Filter '{args.filter}': {len(projects)}/{before} projects")

    # Filter by skip list
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

    mode = "heuristic-only" if args.backend is None else f"{args.backend}"
    print(f"Found {len(projects)} projects with functions.db")
    print(f"Mode: {mode}, min_score: {args.min_score}")
    if args.backend:
        host_port = f"{args.llm_host}:{args.llm_port or 'default'}"
        print(f"LLM server: {host_port}")
    if num_workers > 1:
        print(f"Parallel workers: {num_workers}")
    print()

    # Build work items
    work_items = []
    for project_dir in projects:
        db_path = str(project_dir / "functions.db")
        work_items.append((
            db_path,
            project_dir.name,
            args.backend,
            args.model,
            args.llm_host,
            args.llm_port,
            args.min_score,
            args.force,
            args.verbose,
            args.log_dir,
            args.disable_thinking,
        ))

    # Run detection
    results_by_name: dict[str, dict] = {}

    if num_workers <= 1:
        for i, item in enumerate(work_items, 1):
            name = item[1]
            print(f"[{i}/{len(projects)}] {name}...", end=" ", flush=True)
            result = _detect_worker(item)
            results_by_name[name] = result
            print(_format_result(result))
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_name = {}
            for item in work_items:
                name = item[1]
                future = executor.submit(_detect_worker, item)
                future_to_name[future] = name

            for future in as_completed(future_to_name):
                completed += 1
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "project": name,
                        "functions": 0, "candidates": 0,
                        "containers_found": 0, "llm_calls": 0,
                        "cache_hits": 0, "errors": 1,
                        "input_tokens": 0, "output_tokens": 0,
                        "containers_by_type": {},
                        "timing_seconds": 0.0,
                        "error": str(e),
                    }
                results_by_name[name] = result
                print(
                    f"[{completed}/{len(projects)}] {name}... "
                    f"{_format_result(result)}"
                )

    # Aggregate results
    results = []
    totals = {
        "projects_scanned": 0,
        "projects_skipped": 0,
        "total_functions": 0,
        "total_candidates": 0,
        "total_containers": 0,
        "total_llm_calls": 0,
        "total_cache_hits": 0,
        "total_errors": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "containers_by_type": Counter(),
    }

    for project_dir in projects:
        name = project_dir.name
        result = results_by_name[name]
        results.append(result)

        if result["error"]:
            totals["projects_skipped"] += 1
        else:
            totals["projects_scanned"] += 1
            totals["total_functions"] += result["functions"]
            totals["total_candidates"] += result["candidates"]
            totals["total_containers"] += result["containers_found"]
            totals["total_llm_calls"] += result["llm_calls"]
            totals["total_cache_hits"] += result["cache_hits"]
            totals["total_errors"] += result["errors"]
            totals["total_input_tokens"] += result.get("input_tokens", 0)
            totals["total_output_tokens"] += result.get("output_tokens", 0)
            totals["containers_by_type"].update(result["containers_by_type"])

    # Print summary
    print()
    print("=" * 60)
    print("AGGREGATE TOTALS")
    print("=" * 60)
    print(f"  Projects scanned: {totals['projects_scanned']}")
    print(f"  Projects skipped: {totals['projects_skipped']}")
    print(f"  Functions: {totals['total_functions']}")
    print(f"  Candidates (score >= {args.min_score}): {totals['total_candidates']}")
    print(f"  Containers found: {totals['total_containers']}")
    if totals["total_llm_calls"]:
        print(f"  LLM calls: {totals['total_llm_calls']}")
    if totals["total_cache_hits"]:
        print(f"  Cache hits: {totals['total_cache_hits']}")
    total_tok = totals["total_input_tokens"] + totals["total_output_tokens"]
    if total_tok:
        print(f"  Tokens: {total_tok:,} ({totals['total_input_tokens']:,} in + {totals['total_output_tokens']:,} out)")
    if totals["total_errors"]:
        print(f"  Errors: {totals['total_errors']}")
    if totals["containers_by_type"]:
        print("  By type:")
        for ctype, count in sorted(
            totals["containers_by_type"].items(), key=lambda x: -x[1]
        ):
            print(f"    {ctype}: {count}")

    # Write report
    report = {
        "mode": mode,
        "min_score": args.min_score,
        "projects": results,
        "totals": {
            "projects_scanned": totals["projects_scanned"],
            "projects_skipped": totals["projects_skipped"],
            "total_functions": totals["total_functions"],
            "total_candidates": totals["total_candidates"],
            "total_containers": totals["total_containers"],
            "total_llm_calls": totals["total_llm_calls"],
            "total_cache_hits": totals["total_cache_hits"],
            "total_errors": totals["total_errors"],
            "total_input_tokens": totals["total_input_tokens"],
            "total_output_tokens": totals["total_output_tokens"],
            "containers_by_type": dict(totals["containers_by_type"]),
        },
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"container_report_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
