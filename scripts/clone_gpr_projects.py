#!/usr/bin/env python3
"""
Clone Google Patch Rewards projects to a local directory.
Checks existing repos for URL matches before cloning.
"""

import json
import os
import subprocess
import sys
import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse


def extract_git_url(url):
    """
    Convert project URLs to git clone URLs.
    Returns (git_url, needs_manual_check)
    """
    # Direct git URLs
    if url.endswith('.git'):
        return url, False

    # GitHub repos
    if 'github.com' in url:
        # Clean up the URL
        url = url.rstrip('/')
        # Extract owner/repo pattern
        if '/blob/' in url or '/tree/' in url:
            # Strip branch/path info
            parts = url.split('/')
            idx = parts.index('github.com')
            owner_repo = '/'.join(parts[idx+1:idx+3])
            return f"https://github.com/{owner_repo}.git", False
        return f"{url}.git", False

    # GitLab repos
    if 'gitlab' in url.lower() and '/tree/' not in url and '/blob/' not in url:
        url = url.rstrip('/')
        if not url.endswith('.git'):
            return f"{url}.git", False
        return url, False

    # Chromium/Googlesource repos
    if 'googlesource.com' in url:
        url = url.rstrip('/')
        # Remove /+/HEAD or similar suffixes
        if '/+/' in url:
            url = url.split('/+/')[0]
        return url, False

    # Git protocol URLs
    if url.startswith('git://'):
        return url, False

    # Known git.* domains
    if url.startswith('https://git.'):
        if not url.endswith('.git'):
            return f"{url}.git", False
        return url, False

    # SourceForge, project landing pages, etc.
    # These need manual mapping or special handling
    return None, True


def get_remote_url(repo_path):
    """Get the remote origin URL of a git repo."""
    try:
        result = subprocess.run(
            ['git', '-C', repo_path, 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def normalize_url(url):
    """Normalize git URLs for comparison."""
    if not url:
        return None

    # Remove .git suffix
    url = url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]

    # Normalize protocol
    url = url.replace('git://', 'https://')
    url = url.replace('http://', 'https://')

    return url.lower()


def clone_project(project, base_dir, dry_run=False, verbose=False, interactive=False):
    """
    Clone a project or verify existing clone.
    Returns: ('cloned'|'skipped'|'skipped_non_git'|'error'|'replaced', message)
    """
    name = project['name']
    url = project['url']

    git_url, needs_manual = extract_git_url(url)

    if needs_manual or not git_url:
        if verbose:
            return 'skipped_non_git', f"Skipping non-git URL: {name} ({url})"
        return 'skipped_non_git', f"Skipping: {name} (non-git URL)"

    # Determine target directory name
    # Use the repo name from the URL
    if git_url.endswith('.git'):
        repo_name = git_url.rstrip('/').split('/')[-1][:-4]
    else:
        repo_name = git_url.rstrip('/').split('/')[-1]

    # For some projects, use the project name from our list if it's cleaner
    if len(name.split()) == 1 and name.lower() != repo_name.lower():
        # Single word name, might be cleaner
        target_name = name.lower().replace(' ', '-')
    else:
        target_name = repo_name

    target_dir = os.path.join(base_dir, target_name)

    # Check if directory exists
    if os.path.exists(target_dir):
        if not os.path.isdir(target_dir):
            return 'error', f"Path exists but is not a directory: {target_dir}"

        # Check if it's a git repo
        if not os.path.exists(os.path.join(target_dir, '.git')):
            return 'error', f"Directory exists but is not a git repo: {target_dir}"

        # Get existing remote URL
        existing_url = get_remote_url(target_dir)
        if not existing_url:
            return 'error', f"Cannot get remote URL for: {target_dir}"

        # Compare URLs
        if normalize_url(existing_url) == normalize_url(git_url):
            if verbose:
                return 'skipped', f"Already cloned: {target_name} ({git_url})"
            return 'skipped', f"Already exists: {target_name}"
        else:
            # URL mismatch - ask user what to do
            mismatch_msg = (
                f"URL mismatch in {target_name}:\n"
                f"  Expected: {git_url}\n"
                f"  Found:    {existing_url}"
            )

            if dry_run:
                return 'error', f"[DRY RUN] {mismatch_msg}"

            if interactive:
                print(f"\n{mismatch_msg}")
                response = input("Replace with new URL? [y/N]: ").strip().lower()
                if response in ('y', 'yes'):
                    try:
                        if verbose:
                            print(f"Removing {target_dir}...")
                        shutil.rmtree(target_dir)

                        if verbose:
                            print(f"Cloning {name} from {git_url}...")
                        subprocess.run(
                            ['git', 'clone', git_url, target_dir],
                            check=True,
                            capture_output=not verbose
                        )
                        return 'replaced', f"Replaced: {target_name}"
                    except Exception as e:
                        return 'error', f"Failed to replace {name}: {e}"
                else:
                    return 'skipped', f"Keeping existing: {target_name}"
            else:
                return 'error', mismatch_msg

    # Clone the repo
    if dry_run:
        return 'dry-run', f"Would clone: {git_url} -> {target_dir}"

    try:
        if verbose:
            print(f"Cloning {name} from {git_url}...")

        subprocess.run(
            ['git', 'clone', git_url, target_dir],
            check=True,
            capture_output=not verbose
        )
        return 'cloned', f"Cloned: {target_name}"
    except subprocess.CalledProcessError as e:
        return 'error', f"Failed to clone {name}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description='Clone Google Patch Rewards projects'
    )
    parser.add_argument(
        '--projects-file',
        default='gpr_projects.json',
        help='Path to projects JSON file (default: gpr_projects.json)'
    )
    parser.add_argument(
        '--target-dir',
        default='.',
        help='Target directory for clones (default: /data/csong/opensource)'
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
        help='Show what would be done without actually cloning'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Ask before replacing repos with URL mismatches'
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

    # Create target directory if needed
    if not args.dry_run:
        os.makedirs(args.target_dir, exist_ok=True)

    # Track results
    results = {
        'cloned': [],
        'replaced': [],
        'skipped': [],
        'skipped_non_git': [],
        'error': [],
        'dry-run': []
    }

    # Process each project
    for project in projects:
        status, message = clone_project(
            project,
            args.target_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
            interactive=args.interactive
        )
        results[status].append(message)

        # Print as we go
        if status == 'error':
            print(f"ERROR: {message}", file=sys.stderr)
        elif status == 'cloned':
            print(f"✓ {message}")
        elif status == 'replaced':
            print(f"↻ {message}")
        elif status == 'dry-run':
            print(f"[DRY RUN] {message}")
        elif status == 'skipped_non_git':
            if args.verbose:
                print(f"⊘ {message}")
        elif args.verbose and status == 'skipped':
            print(f"- {message}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total projects: {len(projects)}")
    print(f"Cloned: {len(results['cloned'])}")
    print(f"Replaced: {len(results['replaced'])}")
    print(f"Skipped (already exist): {len(results['skipped'])}")
    print(f"Skipped (non-git URL): {len(results['skipped_non_git'])}")
    print(f"Errors: {len(results['error'])}")

    if results['skipped_non_git'] and args.verbose:
        print("\nSKIPPED (Non-git URLs):")
        for msg in results['skipped_non_git']:
            print(f"  - {msg}")

    if results['error']:
        print("\nERRORS:")
        for msg in results['error']:
            print(f"  - {msg}")

    return 0 if not results['error'] else 1


if __name__ == '__main__':
    sys.exit(main())
