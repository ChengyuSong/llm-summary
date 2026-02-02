"""Build system detection for projects."""

from enum import Enum
from pathlib import Path


class BuildSystem(Enum):
    """Supported build systems."""

    CMAKE = "cmake"
    AUTOTOOLS = "autotools"
    MAKE = "make"
    MESON = "meson"
    UNKNOWN = "unknown"


def detect_build_system(project_path: Path) -> BuildSystem:
    """
    Detect the build system used by a project.

    Args:
        project_path: Path to the project directory

    Returns:
        BuildSystem enum indicating the detected build system
    """
    project_path = Path(project_path)

    if not project_path.exists() or not project_path.is_dir():
        raise ValueError(f"Invalid project path: {project_path}")

    # Check for CMake
    if (project_path / "CMakeLists.txt").exists():
        return BuildSystem.CMAKE

    # Check for Meson
    if (project_path / "meson.build").exists():
        return BuildSystem.MESON

    # Check for Autotools
    if (project_path / "configure.ac").exists() or (project_path / "configure.in").exists():
        return BuildSystem.AUTOTOOLS

    # Check for plain Makefile (without CMake or Autotools indicators)
    if (project_path / "Makefile").exists():
        return BuildSystem.MAKE

    # Could also check for:
    # - GNUmakefile
    # - makefile (lowercase)
    # - *.mk files
    for makefile_name in ["GNUmakefile", "makefile"]:
        if (project_path / makefile_name).exists():
            return BuildSystem.MAKE

    return BuildSystem.UNKNOWN
