"""Assembly checking utilities for the builder module."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assembly_checker import AssemblyCheckResult


def check_assembly(
    compile_commands_path: Path,
    build_dir: Path,
    project_path: Path,
    unavoidable_asm_path: Path | None = None,
    verbose: bool = False,
    log_prefix: str = "[build]",
) -> "AssemblyCheckResult | None":
    """
    Run assembly verification after successful build.

    Args:
        compile_commands_path: Path to compile_commands.json
        build_dir: Build output directory
        project_path: Source project directory
        unavoidable_asm_path: Path to store unavoidable assembly findings
        verbose: If True, print status messages
        log_prefix: Prefix for log messages (e.g., "[cmake_build]")

    Returns:
        AssemblyCheckResult or None if check could not be performed
    """
    try:
        from .assembly_checker import AssemblyChecker

        if not compile_commands_path.exists():
            if verbose:
                print(f"{log_prefix} No compile_commands.json, skipping assembly check")
            return None

        checker = AssemblyChecker(
            compile_commands_path=compile_commands_path,
            build_dir=build_dir,
            project_path=project_path,
            unavoidable_asm_path=unavoidable_asm_path,
            verbose=verbose,
        )
        return checker.check(scan_ir=True)
    except Exception as e:
        if verbose:
            print(f"{log_prefix} Assembly check failed: {e}")
        return None
