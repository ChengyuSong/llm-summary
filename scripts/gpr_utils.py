"""Common utilities for GPR project scripts."""

import json
import os
import shlex
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Docker container path translation
# ---------------------------------------------------------------------------

def is_docker_path(path: str) -> bool:
    """Check if a path uses Docker container conventions (/workspace/...)."""
    return path.startswith("/workspace/")


def resolve_host_path(
    container_path: str,
    project_source_dir: Path,
    build_dir: Path,
) -> Path:
    """Translate a Docker /workspace/ path to the host equivalent.

    Docker volume mapping convention:
      /workspace/src   -> project_source_dir
      /workspace/build -> build_dir
    """
    if not is_docker_path(container_path):
        return Path(container_path)

    remainder = container_path[len("/workspace/"):]

    if remainder.startswith("src/"):
        return project_source_dir / remainder[len("src/"):]
    elif remainder.startswith("src"):
        return project_source_dir
    elif remainder.startswith("build/"):
        return build_dir / remainder[len("build/"):]
    elif remainder.startswith("build"):
        return build_dir
    else:
        return build_dir / remainder


def _translate_command_paths(
    command: str,
    project_source_dir: Path,
    build_dir: Path,
) -> str:
    """Translate /workspace/ paths inside a compile command string."""
    parts = shlex.split(command)
    translated = [
        str(resolve_host_path(p, project_source_dir, build_dir))
        if is_docker_path(p) else p
        for p in parts
    ]
    return " ".join(shlex.quote(p) for p in translated)


def resolve_compile_commands(
    cc_path: Path,
    project_source_dir: Path,
    build_dir: Path,
) -> list[dict]:
    """Load compile_commands.json and resolve all Docker /workspace/ paths to host paths."""
    with open(cc_path) as f:
        entries = json.load(f)

    needs_translation = any(
        is_docker_path(e.get("directory", "")) or is_docker_path(e.get("file", ""))
        for e in entries[:5]
    )

    if not needs_translation:
        return entries

    resolved = []
    for entry in entries:
        e = dict(entry)
        if is_docker_path(e.get("directory", "")):
            e["directory"] = str(resolve_host_path(e["directory"], project_source_dir, build_dir))
        if is_docker_path(e.get("file", "")):
            e["file"] = str(resolve_host_path(e["file"], project_source_dir, build_dir))
        if "output" in e and is_docker_path(e["output"]):
            e["output"] = str(resolve_host_path(e["output"], project_source_dir, build_dir))
        if "command" in e:
            e["command"] = _translate_command_paths(e["command"], project_source_dir, build_dir)
        if "arguments" in e:
            e["arguments"] = [
                str(resolve_host_path(a, project_source_dir, build_dir))
                if is_docker_path(a) else a
                for a in e["arguments"]
            ]
        resolved.append(e)

    return resolved


def derive_dir_name_from_url(url: str) -> str | None:
    """
    Derive directory name from URL.
    Returns the last path component, with .git removed if present.

    Examples:
        https://github.com/owner/repo.git -> repo
        https://github.com/owner/repo -> repo
    """
    if not url:
        return None

    # Remove trailing slash
    url = url.rstrip('/')

    # Get last component
    last_component = url.split('/')[-1]

    # Remove .git suffix
    if last_component.endswith('.git'):
        last_component = last_component[:-4]

    return last_component


def get_remote_url(repo_path: str | Path) -> str | None:
    """Get the remote origin URL of a git repo."""
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_path), 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def normalize_url(url: str) -> str | None:
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


def find_project_dir(project: dict, base_dir: str | Path) -> Path | None:
    """
    Find the directory for a project.
    Tries in order:
    1. project['project_dir'] if it exists (relative to base_dir or absolute)
    2. Lowercase project name with spaces replaced by hyphens
    3. Project name as-is
    4. Directory name derived from URL (lowercase)
    5. Directory name derived from URL (as-is)

    Returns the absolute Path if found, None otherwise.
    """
    base_dir = Path(base_dir)

    # Try existing project_dir field
    if 'project_dir' in project and project['project_dir']:
        project_dir = Path(project['project_dir'])
        # Convert to absolute path if needed
        if not project_dir.is_absolute():
            project_dir = base_dir / project_dir
        if project_dir.exists() and (project_dir / '.git').exists():
            return project_dir

    # Try lowercase project name with hyphens
    name_lower = project['name'].lower().replace(' ', '-')
    candidate = base_dir / name_lower
    if candidate.exists() and (candidate / '.git').exists():
        return candidate

    # Try project name as-is
    candidate = base_dir / project['name']
    if candidate.exists() and (candidate / '.git').exists():
        return candidate

    # Try name derived from URL
    if 'url' in project and project['url']:
        url_name = derive_dir_name_from_url(project['url'])
        if url_name:
            # Try lowercase first
            candidate = base_dir / url_name.lower()
            if candidate.exists() and (candidate / '.git').exists():
                return candidate

            # Try as-is
            candidate = base_dir / url_name
            if candidate.exists() and (candidate / '.git').exists():
                return candidate

    return None


def store_project_dir(project_dir: Path, base_dir: Path) -> str:
    """
    Convert project_dir to a storable path.
    Returns relative path if under base_dir, otherwise absolute path.
    """
    try:
        rel_path = project_dir.relative_to(base_dir)
        return str(rel_path)
    except ValueError:
        # Not relative to base_dir, use absolute
        return str(project_dir)
