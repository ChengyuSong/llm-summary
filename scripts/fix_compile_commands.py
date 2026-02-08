#!/usr/bin/env python3
"""Check and fix paths in compile_commands.json files under build-scripts/.

Issues handled:
  1. Container paths: /workspace/src/... and /workspace/build/... not translated
     to host paths (project_path and build_dir).
  2. In-source build aliasing: projects like ffmpeg create a symlink (e.g. src/)
     inside the build dir pointing back to the source tree. After naive
     /workspace/build -> build_dir translation the file paths land under
     build_dir but the real files live under project_path.
  3. Relative file paths that need resolving against the "directory" field.

Usage:
    python scripts/fix_compile_commands.py                  # check all projects
    python scripts/fix_compile_commands.py --fix             # apply fixes in-place
    python scripts/fix_compile_commands.py --project ffmpeg  # single project
    python scripts/fix_compile_commands.py --verbose         # show per-entry details
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DOCKER_WORKSPACE_SRC = "/workspace/src"
DOCKER_WORKSPACE_BUILD = "/workspace/build"

BUILD_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "build-scripts"

# Default build-artifacts root (inferred from existing configs)
DEFAULT_BUILD_ARTIFACTS_ROOT = "/data/csong/build-artifacts"


def load_config(project_dir: Path) -> dict:
    cfg_path = project_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def infer_build_dir(project_name: str, config: dict) -> str | None:
    """Infer the build directory for a project."""
    # Some configs have use_build_dir; if True, build dir is under build-artifacts
    use_build_dir = config.get("use_build_dir")
    if use_build_dir is False:
        # Build happened in the source dir
        return config.get("project_path")
    # Default: assume build-artifacts/<project>
    return f"{DEFAULT_BUILD_ARTIFACTS_ROOT}/{project_name}"


def replace_in_string(s: str, project_path: str, build_dir: str) -> str:
    """Replace Docker container paths with host paths in a string."""
    # Order matters: replace /workspace/src first (more specific if build is a subpath)
    s = s.replace(DOCKER_WORKSPACE_SRC, project_path)
    s = s.replace(DOCKER_WORKSPACE_BUILD, build_dir)
    return s


def fix_entry(
    entry: dict, project_path: str, build_dir: str
) -> tuple[dict, list[str]]:
    """Fix paths in a single compile_commands entry.

    Returns (fixed_entry, list_of_fixes_applied).
    """
    fixes = []
    entry = dict(entry)  # shallow copy

    # --- Phase 1: Replace remaining Docker container paths ---
    for field in ("file", "directory", "output"):
        val = entry.get(field, "")
        if not val:
            continue
        new_val = replace_in_string(val, project_path, build_dir)
        if new_val != val:
            fixes.append(f"{field}: container path -> host ({val} -> {new_val})")
            entry[field] = new_val

    if "command" in entry:
        old = entry["command"]
        new = replace_in_string(old, project_path, build_dir)
        if new != old:
            fixes.append("command: container paths -> host")
            entry["command"] = new

    if "arguments" in entry:
        old_args = entry["arguments"]
        new_args = [replace_in_string(a, project_path, build_dir) for a in old_args]
        if new_args != old_args:
            fixes.append("arguments: container paths -> host")
            entry["arguments"] = new_args

    # --- Phase 2: Resolve relative file paths ---
    # Relative paths were recorded inside the Docker container, so resolve them
    # in Docker-space first, then translate to host paths.
    file_path = entry.get("file", "")
    directory = entry.get("directory", "")
    if file_path and not os.path.isabs(file_path) and directory:
        # Convert the host directory back to Docker-space
        docker_dir = directory.replace(build_dir, DOCKER_WORKSPACE_BUILD).replace(
            project_path, DOCKER_WORKSPACE_SRC
        )
        docker_resolved = str((Path(docker_dir) / file_path).resolve())
        # Translate Docker path back to host
        resolved = replace_in_string(docker_resolved, project_path, build_dir)
        fixes.append(f"file: resolved relative path ({file_path} -> {resolved})")
        entry["file"] = resolved
        file_path = resolved

    # --- Phase 3: Fix in-source build aliasing ---
    # If the file is under build_dir but doesn't exist, try project_path
    if file_path and not os.path.exists(file_path) and file_path.startswith(build_dir + "/"):
        rel = file_path[len(build_dir) + 1 :]

        candidates = []
        # Direct mapping: build_dir/X -> project_path/X
        candidates.append(os.path.join(project_path, rel))
        # Symlink prefix stripping: build_dir/src/X -> project_path/X
        # (ffmpeg creates a src/ symlink in the build dir)
        for prefix in ("src/", "source/"):
            if rel.startswith(prefix):
                candidates.append(os.path.join(project_path, rel[len(prefix) :]))

        for candidate in candidates:
            if os.path.exists(candidate):
                fixes.append(
                    f"file: build-dir alias -> source ({file_path} -> {candidate})"
                )
                entry["file"] = candidate
                # Also fix in command/arguments if present
                if "command" in entry:
                    entry["command"] = entry["command"].replace(file_path, candidate)
                if "arguments" in entry:
                    entry["arguments"] = [
                        a.replace(file_path, candidate) for a in entry["arguments"]
                    ]
                break

    return entry, fixes


def check_project(
    project_dir: Path, fix: bool = False, verbose: bool = False
) -> dict:
    """Check and optionally fix a single project's compile_commands.json.

    Returns a summary dict.
    """
    project_name = project_dir.name
    cc_path = project_dir / "compile_commands.json"

    if not cc_path.exists():
        return {"project": project_name, "status": "no_compile_commands"}

    config = load_config(project_dir)
    project_path = config.get("project_path", "")
    build_dir = infer_build_dir(project_name, config) or ""

    with open(cc_path) as f:
        data = json.load(f)

    total = len(data)
    container_path_fixes = 0
    relative_path_fixes = 0
    alias_fixes = 0
    removed = 0
    fixed_data = []

    for entry in data:
        fixed_entry, fixes = fix_entry(entry, project_path, build_dir)

        for desc in fixes:
            if "container path" in desc:
                container_path_fixes += 1
            elif "resolved relative" in desc:
                relative_path_fixes += 1
            elif "build-dir alias" in desc:
                alias_fixes += 1

        # Phase 4: drop entries whose source file still can't be found
        fp = fixed_entry.get("file", "")
        if fp and not os.path.exists(fp):
            removed += 1
            if verbose:
                print(f"  REMOVED: {fp}")
            continue

        if verbose and fixes:
            print(f"  {fp}:")
            for desc in fixes:
                print(f"    - {desc}")

        fixed_data.append(fixed_entry)

    total_fixes = container_path_fixes + relative_path_fixes + alias_fixes
    changed = total_fixes > 0 or removed > 0

    if fix and changed:
        with open(cc_path, "w") as f:
            json.dump(fixed_data, f, indent=2)
            f.write("\n")

    return {
        "project": project_name,
        "total_entries": total,
        "container_path_fixes": container_path_fixes,
        "relative_path_fixes": relative_path_fixes,
        "alias_fixes": alias_fixes,
        "total_fixes": total_fixes,
        "removed": removed,
        "kept": len(fixed_data),
        "applied": fix and changed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check and fix paths in compile_commands.json files"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply fixes in-place"
    )
    parser.add_argument(
        "--project", type=str, help="Check a single project (directory name)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-entry details"
    )
    args = parser.parse_args()

    if args.project:
        project_dirs = [BUILD_SCRIPTS_DIR / args.project]
        if not project_dirs[0].is_dir():
            print(f"Error: {project_dirs[0]} not found", file=sys.stderr)
            sys.exit(1)
    else:
        project_dirs = sorted(
            p for p in BUILD_SCRIPTS_DIR.iterdir() if p.is_dir()
        )

    results = []
    for project_dir in project_dirs:
        result = check_project(project_dir, fix=args.fix, verbose=args.verbose)
        if result.get("status") == "no_compile_commands":
            continue
        results.append(result)

    # Print summary table
    print()
    print(
        f"{'Project':<25} {'Total':>6} {'Container':>10} {'Relative':>9} "
        f"{'Alias':>6} {'Fixed':>6} {'Removed':>8} {'Kept':>6}"
    )
    print("-" * 86)

    total_fixes_all = 0
    total_removed_all = 0
    for r in results:
        if r["total_fixes"] > 0 or r["removed"] > 0:
            applied = " *" if r.get("applied") else ""
            print(
                f"{r['project']:<25} {r['total_entries']:>6} "
                f"{r['container_path_fixes']:>10} {r['relative_path_fixes']:>9} "
                f"{r['alias_fixes']:>6} {r['total_fixes']:>6} "
                f"{r['removed']:>8} {r['kept']:>6}{applied}"
            )
            total_fixes_all += r["total_fixes"]
            total_removed_all += r["removed"]

    print("-" * 86)
    print(
        f"{'TOTAL':<25} {'':>6} {'':>10} {'':>9} {'':>6} "
        f"{total_fixes_all:>6} {total_removed_all:>8}"
    )
    print()

    if not args.fix:
        print("Run with --fix to apply changes in-place.")
    else:
        print("Fixes applied. Entries marked with * were updated.")


if __name__ == "__main__":
    main()
