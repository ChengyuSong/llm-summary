#!/usr/bin/env python3
"""Batch rebuild script for rebuilding projects with debug info enabled.

This script invokes build.sh scripts from build-scripts/ directory to rebuild
projects with debug information (-g flag) enabled.

Usage examples:
    # Rebuild all projects with build scripts, save success list
    ./scripts/batch_rebuild.py --only-with-scripts --success-list done.txt

    # Clean rebuild of tier 1 projects
    ./scripts/batch_rebuild.py --tier 1 --only-with-scripts --clean --success-list tier1_done.txt

    # Rebuild specific project(s) with verbose output
    ./scripts/batch_rebuild.py --filter libpng --verbose

    # Clean rebuild first 5 projects, save success list
    ./scripts/batch_rebuild.py --limit 5 --only-with-scripts --clean --success-list batch1.txt

    # Continue from previous run, skipping already built projects
    ./scripts/batch_rebuild.py --skip-list batch1.txt --only-with-scripts --clean --success-list batch2.txt
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from gpr_utils import find_project_dir


def run_build_script(
    project_name: str,
    build_script: Path,
    project_path: Path,
    build_dir: Path,
    verbose: bool,
) -> tuple[bool, str, float]:
    """
    Run build.sh script for a single project.

    Args:
        build_script: Path to build.sh
        project_path: Source directory (arg 1)
        build_dir: Build directory (arg 3) - passed as arg 2, artifacts defaults to build-scripts/<project>/artifacts

    Returns:
        (success, error_message, duration_seconds)
    """
    # ARTIFACTS_DIR should default to build-scripts/<project>/artifacts
    # Convert to absolute path for Docker volume mounting
    artifacts_default = build_script.parent / "artifacts"
    artifacts_default = artifacts_default.resolve()

    cmd = [
        "bash",
        str(build_script),
        str(project_path),
        str(artifacts_default),
        str(build_dir),
    ]

    print(f"\n{'='*80}")
    print(f"Building: {project_name}")
    print(f"  Script: {build_script}")
    print(f"  Project: {project_path}")
    print(f"  Build: {build_dir}")
    print(f"  Artifacts: {artifacts_default}")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=3600,  # 60 minute timeout per project
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            return True, "", duration
        else:
            error_msg = f"Exit code {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}"
            return False, error_msg, duration

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        return False, "Timeout after 60 minutes", duration
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        return False, f"Exception: {str(e)}", duration


def main():
    parser = argparse.ArgumentParser(
        description="Batch rebuild projects with debug info enabled"
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
        "--build-scripts-dir",
        type=Path,
        default=Path("build-scripts"),
        help="Directory containing build.sh scripts for each project",
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
        help="Show build output in real-time",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("rebuild_report.json"),
        help="Output JSON report file",
    )
    parser.add_argument(
        "--success-list",
        type=Path,
        help="Output file for successful project names (one per line, can be used with --skip-list)",
    )
    parser.add_argument(
        "--only-with-scripts",
        action="store_true",
        help="Only process projects that have build.sh scripts",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove build directories (BUILD_ROOT/<project>/) before rebuilding to ensure clean builds",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.projects_json.exists():
        print(f"Error: Projects JSON file not found: {args.projects_json}")
        sys.exit(1)

    if not args.source_dir.exists():
        print(f"Error: Source directory not found: {args.source_dir}")
        sys.exit(1)

    if not args.build_scripts_dir.exists():
        print(f"Error: Build scripts directory not found: {args.build_scripts_dir}")
        sys.exit(1)

    # Create build root if needed
    args.build_root.mkdir(parents=True, exist_ok=True)

    # Load skip list if provided
    skip_set = set()
    if args.skip_list:
        if not args.skip_list.exists():
            print(f"Error: Skip list file not found: {args.skip_list}")
            sys.exit(1)
        with open(args.skip_list) as f:
            skip_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(skip_set)} projects to skip from {args.skip_list}")

    # Initialize success list file (clear if exists)
    if args.success_list:
        with open(args.success_list, "w") as f:
            pass  # Create empty file

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
            if args.only_with_scripts:
                print(f"  Skipping...")
                results["skipped"] += 1
                results["projects"].append({
                    "name": project_name,
                    "status": "skipped",
                    "reason": "directory_not_found",
                    "path": str(args.source_dir / project_name),
                })
                continue
            else:
                print(f"  Skipping...")
                results["skipped"] += 1
                results["projects"].append({
                    "name": project_name,
                    "status": "skipped",
                    "reason": "directory_not_found",
                    "path": str(args.source_dir / project_name),
                })
                continue

        # Find build script (use directory name, not project name from JSON)
        build_script = args.build_scripts_dir / project_path.name / "build.sh"

        if not build_script.exists():
            print(f"  ⚠️  Build script not found: {build_script}")
            if args.only_with_scripts:
                print(f"  Skipping...")
                results["skipped"] += 1
                results["projects"].append({
                    "name": project_name,
                    "status": "skipped",
                    "reason": "no_build_script",
                    "path": str(project_path),
                    "build_script": str(build_script),
                })
                continue
            else:
                print(f"  Skipping...")
                results["skipped"] += 1
                results["projects"].append({
                    "name": project_name,
                    "status": "skipped",
                    "reason": "no_build_script",
                    "path": str(project_path),
                    "build_script": str(build_script),
                })
                continue

        # Setup build directory
        build_dir = args.build_root / project_path.name

        # Clean build directory if requested
        if args.clean and build_dir.exists():
            print(f"  🧹 Cleaning build directory: {build_dir}")
            import shutil
            try:
                shutil.rmtree(build_dir)
                print(f"  ✓ Cleaned successfully")
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to clean build directory: {e}")

        build_dir.mkdir(parents=True, exist_ok=True)

        # Run build script (artifacts will default to build-scripts/<project>/artifacts)
        success, error_msg, duration = run_build_script(
            project_name=project_name,
            build_script=build_script,
            project_path=project_path,
            build_dir=build_dir,
            verbose=args.verbose,
        )

        if success:
            print(f"  ✅ Success! ({duration:.1f}s)")
            results["succeeded"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "success",
                "duration": duration,
                "path": str(project_path),
                "build_dir": str(build_dir),
                "build_script": str(build_script),
            })

            # Append to success list immediately (incremental save)
            if args.success_list:
                with open(args.success_list, "a") as f:
                    f.write(f"{project_path.name}\n")
        else:
            print(f"  ❌ Failed ({duration:.1f}s)")
            if error_msg:
                # Print first 500 chars of error
                error_preview = error_msg[:500]
                if len(error_msg) > 500:
                    error_preview += "..."
                print(f"  Error: {error_preview}")
            results["failed"] += 1
            results["projects"].append({
                "name": project_name,
                "status": "failed",
                "duration": duration,
                "error": error_msg,
                "path": str(project_path),
                "build_dir": str(build_dir),
                "build_script": str(build_script),
            })

        # Update report incrementally after each project
        with open(args.report, "w") as f:
            json.dump(results, f, indent=2)

    # Save final report
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)

    # Success list already written incrementally during builds

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total projects: {results['total']}")
    print(f"  ✅ Succeeded: {results['succeeded']}")
    print(f"  ❌ Failed: {results['failed']}")
    print(f"  ⏭️  Skipped: {results['skipped']}")
    print(f"\nDetailed report: {args.report}")
    if args.success_list:
        if results["succeeded"] > 0:
            print(f"Success list: {args.success_list} ({results['succeeded']} projects)")
        else:
            print(f"Success list: {args.success_list} (no successes)")

    if results["succeeded"] > 0:
        print(f"\n✅ Successfully rebuilt projects:")
        for proj in results["projects"]:
            if proj["status"] == "success":
                duration = proj.get("duration", 0)
                print(f"  - {proj['name']} ({duration:.1f}s)")

    if results["failed"] > 0:
        print(f"\n❌ Failed projects:")
        for proj in results["projects"]:
            if proj["status"] == "failed":
                error = proj.get("error", "Unknown error")
                # Show first 100 chars
                error_preview = error[:100]
                if len(error) > 100:
                    error_preview += "..."
                print(f"  - {proj['name']}: {error_preview}")

    # Exit with error if any failures
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
