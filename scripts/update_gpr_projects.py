#!/usr/bin/env python3
"""
Update Git repositories from GPR projects JSON file.
For each project, finds the repo directory, checks out default branch, and pulls.
Updates the JSON file with project_dir field for future runs.
"""

import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
from gpr_utils import find_project_dir, store_project_dir


def get_default_branch(repo_path):
    """
    Get the default branch of a git repository.
    Returns the branch name or None on error.
    """
    try:
        # Try to get the default branch from remote HEAD
        result = subprocess.run(
            ['git', '-C', repo_path, 'symbolic-ref', 'refs/remotes/origin/HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        # Output format: refs/remotes/origin/main
        default_ref = result.stdout.strip()
        branch_name = default_ref.split('/')[-1]
        return branch_name
    except subprocess.CalledProcessError:
        # If that fails, try common default branches
        for branch in ['main', 'master', 'develop']:
            try:
                subprocess.run(
                    ['git', '-C', repo_path, 'rev-parse', '--verify', f'origin/{branch}'],
                    capture_output=True,
                    check=True
                )
                return branch
            except subprocess.CalledProcessError:
                continue
        return None


def get_current_branch(repo_path):
    """Get the current branch name."""
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def has_uncommitted_changes(repo_path):
    """Check if repo has uncommitted changes to tracked files."""
    try:
        # Only check for changes to tracked files (not untracked files)
        # Format: XY PATH, where X is staged, Y is working tree
        # We only care about modifications/deletions to tracked files
        result = subprocess.run(
            ['git', '-C', repo_path, 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if not line:
                continue
            # Check if it's a modification to a tracked file
            # Untracked files start with '??'
            if not line.startswith('??'):
                return True
        return False
    except subprocess.CalledProcessError:
        return True  # Assume changes if we can't check


def update_repo(repo_path, repo_name, dry_run=False, verbose=False, force=False):
    """
    Update a git repository by checking out default branch and pulling.
    Returns: ('updated'|'skipped'|'error', message)
    """
    # Check for uncommitted changes
    if not force and has_uncommitted_changes(repo_path):
        return 'error', f"Uncommitted changes in {repo_name}, use --force to override"

    # Get default branch
    default_branch = get_default_branch(repo_path)
    if not default_branch:
        return 'error', f"Cannot determine default branch for {repo_name}"

    current_branch = get_current_branch(repo_path)

    if dry_run:
        if current_branch != default_branch:
            return 'dry-run', f"Would checkout {default_branch} and pull in {repo_name} (currently on {current_branch})"
        else:
            return 'dry-run', f"Would pull {default_branch} in {repo_name}"

    try:
        # Fetch latest changes
        if verbose:
            print(f"  Fetching {repo_name}...")
        subprocess.run(
            ['git', '-C', repo_path, 'fetch', 'origin'],
            check=True,
            timeout=300  # 5 minute timeout
        )

        # Checkout default branch if not already on it
        if current_branch != default_branch:
            if verbose:
                print(f"  Checking out {default_branch}...")
            subprocess.run(
                ['git', '-C', repo_path, 'checkout', default_branch],
                check=True,
                timeout=60  # 1 minute timeout
            )

        # Pull latest changes
        if verbose:
            print(f"  Pulling {default_branch}...")
        result = subprocess.run(
            ['git', '-C', repo_path, 'pull'],
            capture_output=not verbose,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )

        # Check if anything was updated (only check if we captured output)
        if not verbose:
            if result.stdout and ('Already up to date' in result.stdout or 'Already up-to-date' in result.stdout):
                return 'skipped', f"Already up to date: {repo_name}"
            else:
                return 'updated', f"Updated: {repo_name} ({default_branch})"
        else:
            # In verbose mode, just report success
            return 'updated', f"Updated: {repo_name} ({default_branch})"

    except subprocess.TimeoutExpired:
        return 'error', f"Timeout while updating {repo_name}"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, 'stderr') and e.stderr else str(e)
        return 'error', f"Failed to update {repo_name}: {error_msg}"


def update_project(project, base_dir, dry_run=False, verbose=False, force=False):
    """
    Update a single project from the JSON.
    Returns: (status, message, project_dir_found)
    """
    name = project['name']

    # Find the project directory
    project_dir = find_project_dir(project, base_dir)

    if not project_dir:
        return 'not_found', f"Directory not found for {name}", None

    # Update the repository
    status, message = update_repo(
        str(project_dir),
        name,
        dry_run=dry_run,
        verbose=verbose,
        force=force
    )

    return status, message, str(project_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Update GPR projects from JSON file'
    )
    parser.add_argument(
        '--projects-file',
        default='projects.json',
        help='Path to projects JSON file (default: scripts/gpr_projects.json)'
    )
    parser.add_argument(
        '--target-dir',
        default='.',
        help='Base directory containing git repositories (default: current directory)'
    )
    parser.add_argument(
        '--tier',
        type=int,
        choices=[1, 2],
        help='Only process projects from this tier'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually updating'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Update even if there are uncommitted changes (may cause conflicts)'
    )
    parser.add_argument(
        '--no-update-json',
        action='store_true',
        help='Do not update the JSON file with project_dir fields'
    )

    args = parser.parse_args()

    # Load projects
    if not os.path.exists(args.projects_file):
        print(f"Error: Projects file not found: {args.projects_file}", file=sys.stderr)
        return 1

    with open(args.projects_file) as f:
        projects = json.load(f)

    # Filter by tier if requested
    if args.tier:
        projects = [p for p in projects if p.get('tier') == args.tier]

    # Expand target directory
    target_dir = os.path.abspath(os.path.expanduser(args.target_dir))

    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}", file=sys.stderr)
        return 1

    print(f"Processing {len(projects)} projects from {args.projects_file}")
    print(f"Base directory: {target_dir}")
    if args.dry_run:
        print("[DRY RUN MODE - no changes will be made]\n")
    print()

    # Track results
    results = {
        'updated': [],
        'skipped': [],
        'error': [],
        'not_found': [],
        'dry-run': []
    }

    # Track whether we need to update the JSON
    json_modified = False

    # Process each project
    for project in projects:
        if args.verbose:
            print(f"\nProcessing {project['name']}...")

        status, message, project_dir = update_project(
            project,
            target_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
            force=args.force
        )
        results[status].append(message)

        # Update project_dir in the JSON if found and not already set
        if project_dir and not args.no_update_json:
            stored_path = store_project_dir(Path(project_dir), Path(target_dir))
            if project.get('project_dir') != stored_path:
                project['project_dir'] = stored_path
                json_modified = True

        # Print as we go
        if status == 'error':
            print(f"✗ {message}", file=sys.stderr)
        elif status == 'updated':
            print(f"✓ {message}")
        elif status == 'dry-run':
            print(f"→ {message}")
        elif args.verbose and status == 'not_found':
            print(f"? {message}")
        elif args.verbose and status == 'skipped':
            print(f"- {message}")

    # Update JSON file if modified
    if json_modified and not args.dry_run:
        try:
            with open(args.projects_file, 'w') as f:
                json.dump(projects, f, indent=2)
            print(f"\n✓ Updated {args.projects_file} with project_dir fields")
        except Exception as e:
            print(f"\n✗ Failed to update {args.projects_file}: {e}", file=sys.stderr)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total projects: {len(projects)}")
    print(f"Updated: {len(results['updated'])}")
    print(f"Skipped (already up to date): {len(results['skipped'])}")
    print(f"Not found: {len(results['not_found'])}")
    print(f"Errors: {len(results['error'])}")

    if results['not_found'] and args.verbose:
        print("\nNOT FOUND:")
        for msg in results['not_found']:
            print(f"  - {msg}")

    if results['error']:
        print("\nERRORS:")
        for msg in results['error']:
            print(f"  - {msg}")

    return 0 if not results['error'] else 1


if __name__ == '__main__':
    sys.exit(main())
