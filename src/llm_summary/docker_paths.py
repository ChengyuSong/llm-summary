"""Unified Docker container ↔ host path remapping.

Docker builds mount the host source tree at /workspace/src and the build
directory at /workspace/build.  This module provides helpers to detect and
translate those container paths back to host paths.
"""

from __future__ import annotations

from pathlib import Path

# Canonical prefixes — used both for Docker mounts (forward) and for
# remapping container paths back to host (reverse).
DOCKER_SRC_PREFIX = "/workspace/src"
DOCKER_BUILD_PREFIX = "/workspace/build"
DOCKER_CCACHE_DIR = "/ccache"


def is_docker_path(path: str) -> bool:
    """Return True if *path* looks like a Docker container path."""
    return path.startswith("/workspace/")


def remap_path(
    path: str,
    project_path: Path,
    build_dir: Path | None = None,
) -> str:
    """Remap a Docker container path to the corresponding host path.

    * ``/workspace/src/…``   → ``project_path/…``
    * ``/workspace/build/…`` → ``build_dir/…``  (falls back to *project_path*)
    * Anything else under ``/workspace/`` → ``build_dir/…``

    Returns *path* unchanged when it is not a Docker path.
    """
    if not is_docker_path(path):
        return path
    effective_build = build_dir or project_path
    remainder = path[len("/workspace/"):]
    if remainder.startswith("src/"):
        return str(project_path / remainder[len("src/"):])
    if remainder == "src":
        return str(project_path)
    if remainder.startswith("build/"):
        return str(effective_build / remainder[len("build/"):])
    if remainder == "build":
        return str(effective_build)
    # Unknown sub-path — best guess is build dir
    return str(effective_build / remainder)


def strip_docker_prefix(path: str) -> str:
    """Strip ``/workspace/src/`` or ``/workspace/build/`` prefix.

    Returns the relative remainder, useful for suffix-matching against
    host DB entries when we don't know the host root.
    """
    for prefix in (DOCKER_SRC_PREFIX + "/", DOCKER_BUILD_PREFIX + "/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def translate_compiler_arg(
    arg: str,
    project_path: Path,
    build_dir: Path | None = None,
) -> str:
    """Remap Docker paths inside a single compiler argument.

    Handles bare paths as well as paths prefixed with ``-I``, ``-isystem``,
    ``-include``, etc.
    """
    if is_docker_path(arg):
        return remap_path(arg, project_path, build_dir)
    for flag in ("-I", "-isystem", "-isysroot", "-include", "-iprefix",
                 "-iwithprefix", "-iwithprefixbefore", "-iquote"):
        if arg.startswith(flag) and is_docker_path(arg[len(flag):]):
            return f"{flag}{remap_path(arg[len(flag):], project_path, build_dir)}"
    return arg
