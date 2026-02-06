#!/usr/bin/env python3
"""Batch build-learn script for processing multiple projects."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from gpr_utils import find_project_dir


def run_build_learn(
    project_path: Path,
    build_dir: Path,
    backend: str,
    model: str | None,
    verbose: bool,
    log_file: Path | None,
    llm_host: str | None = None,
    llm_port: int | None = None,
) -> tuple[bool, str, float, bool]:
    """
    Run build-learn for a single project.

    Returns:
        (success, error_message, duration_seconds, is_connection_error)
    """
    cmd = [
        "llm-summary",
        "build-learn",
        "--project-path", str(project_path),
        "--build-dir", str(build_dir),
        "--backend", backend,
    ]

    if model:
        cmd.extend(["--model", model])

    if verbose:
        cmd.append("--verbose")

    if log_file:
        cmd.extend(["--log-llm", str(log_file)])

    # Add LLM host/port if specified (for llamacpp, ollama, etc.)
    if llm_host:
        cmd.extend(["--llm-host", llm_host])
    if llm_port:
        cmd.extend(["--llm-port", str(llm_port)])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 10 minute timeout per project
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            return True, "", duration, False
        else:
            error_msg = f"Exit code {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}"

            # Check if this is a connection/server error for the LLM
            combined_output = (result.stderr or "") + (result.stdout or "")
            connection_error_patterns = [
                "connection refused",
                "connection error",
                "failed to connect",
                "cannot connect",
                "network is unreachable",
                "connection timeout",
                "no route to host",
                "name or service not known",
                "temporarily unavailable",
                "http error 400",  # Bad request to LLM server
                "http error 500",  # Internal server error
                "http error 502",  # Bad gateway
                "http error 503",  # Service unavailable
                "http error 504",  # Gateway timeout
                "errno 111",  # Connection refused errno
                "urlopen error",  # urllib connection errors
            ]

            combined_lower = combined_output.lower()
            is_connection_error = any(pattern in combined_lower for pattern in connection_error_patterns)

            # Debug: print which pattern matched
            if is_connection_error:
                for pattern in connection_error_patterns:
                    if pattern in combined_lower:
                        print(f"[DEBUG] Connection error detected: pattern '{pattern}' found in output")
                        break

            return False, error_msg, duration, is_connection_error

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        return False, "Timeout after 10 minutes", duration, False
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return False, f"Exception: {str(e)}", duration, False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process projects with build-learn"
    )
    parser.add_argument(
        "--projects-json",
        type=Path,
        default=Path("scripts/gpr_projects.json"),
        help="Path to gpr_projects.json file",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/data/csong/opensource"),
        help="Root directory where projects are cloned",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("/data/csong/build-artifacts"),
        help="Root directory for build outputs (each project gets a subdirectory)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="llamacpp",
        choices=["claude", "openai", "ollama", "llamacpp", "gemini"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (optional, uses backend default if not specified)",
    )
    parser.add_argument(
        "--tier",
        type=int,
        help="Only process projects from this tier (1 or 2)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="Only process projects whose name contains this string (case-insensitive)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of projects to process",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N projects",
    )
    parser.add_argument(
        "--skip-list",
        type=Path,
        help="File with project names to skip (one per line, based on directory name)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for build-learn",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory to store individual project log files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build_learn_report.json"),
        help="Output JSON report file",
    )
    parser.add_argument(
        "--llm-host",
        type=str,
        help="llama.cpp/ollama server host (default: localhost)",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        help="llama.cpp/ollama server port (default: 8080 for llamacpp, 11434 for ollama)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.projects_json.exists():
        print(f"Error: Projects JSON file not found: {args.projects_json}")
        sys.exit(1)

    if not args.source_dir.exists():
        print(f"Error: Source directory not found: {args.source_dir}")
        sys.exit(1)

    # Create build root if needed
    args.build_root.mkdir(parents=True, exist_ok=True)

    # Create log directory if specified
    if args.log_dir:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    # Load skip list if provided
    skip_set = set()
    if args.skip_list:
        if not args.skip_list.exists():
            print(f"Error: Skip list file not found: {args.skip_list}")
            sys.exit(1)
        with open(args.skip_list) as f:
            skip_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(skip_set)} projects to skip from {args.skip_list}")

    # Load projects
    with open(args.projects_json) as f:
        projects = json.load(f)

    print(f"Loaded {len(projects)} projects from {args.projects_json}")

    # Apply filters
    if args.tier:
        projects = [p for p in projects if p.get("tier") == args.tier]
        print(f"Filtered to {len(projects)} projects in tier {args.tier}")

    if args.filter:
        filter_str = args.filter.lower()
        projects = [p for p in projects if filter_str in p["name"].lower()]
        print(f"Filtered to {len(projects)} projects matching '{args.filter}'")

    # Apply skip and limit
    if args.skip > 0:
        projects = projects[args.skip:]
        print(f"Skipped first {args.skip} projects, {len(projects)} remaining")

    if args.limit:
        projects = projects[:args.limit]
        print(f"Limited to {args.limit} projects")

    # Track results
    results = {
        "timestamp": datetime.now().isoformat(),
        "backend": args.backend,
        "model": args.model,
        "llm_host": args.llm_host,
        "llm_port": args.llm_port,
        "total": len(projects),
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "projects": [],
    }

    # Process each project
    for i, project in enumerate(projects, 1):
        project_name = project["name"]

        print(f"\n[{i}/{len(projects)}] Processing: {project_name}")

        # Find the project directory
        project_path = find_project_dir(project, args.source_dir)

        if project_path:
            print(f"  Directory: {project_path.name}")

        # Check skip list
        if project_path and project_path.name in skip_set:
            print(f"  ⏭️  Skipping (in skip list)")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "in_skip_list",
                "path": str(project_path),
            })
            continue

        # If directory not found
        if not project_path:
            print(f"  ⚠️  Project directory not found")
            print(f"  Skipping...")
            results["skipped"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "skipped",
                "reason": "directory_not_found",
                "path": str(args.source_dir / project_name),
            })
            continue

        # Setup build directory (use directory name, not project name from JSON)
        build_dir = args.build_root / project_path.name
        build_dir.mkdir(parents=True, exist_ok=True)

        # Setup log file if requested
        log_file = None
        if args.log_dir:
            log_file = args.log_dir / f"{project_name}.log"

        # Run build-learn
        success, error_msg, duration, is_connection_error = run_build_learn(
            project_path=project_path,
            build_dir=build_dir,
            backend=args.backend,
            model=args.model,
            verbose=args.verbose,
            log_file=log_file,
            llm_host=args.llm_host,
            llm_port=args.llm_port,
        )

        # Read build_system from generated config.json if available
        config_path = Path("build-scripts") / project_path.name / "config.json"
        build_system = "unknown"
        if config_path.exists():
            try:
                config_data = json.loads(config_path.read_text())
                build_system = config_data.get("build_system", "unknown")
            except (json.JSONDecodeError, OSError):
                pass

        if success:
            print(f"  ✅ Success! ({duration:.1f}s)")
            results["succeeded"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "success",
                "build_system": build_system,
                "duration": duration,
                "path": str(project_path),
                "build_dir": str(build_dir),
            })
        else:
            print(f"  ❌ Failed ({duration:.1f}s)")
            print(f"  Error: {error_msg[:200]}...")
            results["failed"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "failed",
                "build_system": build_system,
                "duration": duration,
                "error": error_msg,
                "path": str(project_path),
                "build_dir": str(build_dir),
            })

            # Stop immediately if LLM server is not responding
            if is_connection_error:
                print(f"\n{'='*80}")
                print("❌ STOPPING: LLM server connection error detected")
                print(f"{'='*80}")
                print(f"Cannot connect to LLM server at {args.llm_host}:{args.llm_port or 'default'}")
                print(f"Please check that the server is running and accessible.")
                print(f"\nProcessed {i} of {len(projects)} projects before stopping.")
                break

    # Save report
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total projects: {results['total']}")
    print(f"  ✅ Succeeded: {results['succeeded']}")
    print(f"  ❌ Failed: {results['failed']}")
    print(f"  ⚠️  Skipped: {results['skipped']}")
    print(f"\nDetailed report saved to: {args.report}")

    if results["failed"] > 0:
        print(f"\nFailed projects:")
        for proj in results["projects"]:
            if proj["status"] == "failed":
                print(f"  - {proj['name']}: {proj['error'][:100]}...")

    # Exit with error if any failures
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
