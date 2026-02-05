"""Tools for the build agent to explore projects."""

from pathlib import Path
from typing import Any


class BuildTools:
    """Tools available to the build agent for exploring projects."""

    def __init__(self, project_path: Path, build_dir: Path | None = None):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir).resolve() if build_dir else None

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve path, ensuring it's within project or build directory.

        Args:
            path: Requested path (should be relative to project root or "build/..." for build dir)

        Returns:
            Resolved absolute path within allowed directories

        Raises:
            ValueError: If path escapes sandbox or is invalid
        """
        # Convert to Path
        requested = Path(path)

        # Reject absolute paths
        if requested.is_absolute():
            raise ValueError(f"Absolute paths not allowed: {path}")

        # Special handling for build directory paths
        # If path starts with "build/" and we have a separate build_dir, use it
        parts = requested.parts
        if parts and parts[0] == "build" and self.build_dir:
            # Strip "build/" prefix and resolve relative to build_dir
            relative_to_build = Path(*parts[1:]) if len(parts) > 1 else Path(".")
            base_dir = self.build_dir
            full_path = (base_dir / relative_to_build).resolve()
            dir_name = "build directory"
        else:
            base_dir = self.project_path
            full_path = (base_dir / requested).resolve()
            dir_name = "project directory"

        # Security: check for symlink escape attacks
        # Walk through path components and ensure no symlink points outside sandbox
        self._check_symlink_escape(base_dir, requested, dir_name)

        # Security: ensure resolved path is within allowed directory
        try:
            full_path.relative_to(base_dir)
            return full_path
        except ValueError:
            raise ValueError(f"Path escapes {dir_name}: {path}")

    def _check_symlink_escape(self, base_dir: Path, relative_path: Path, dir_name: str) -> None:
        """
        Check that no symlink in the path points outside the sandbox.

        This prevents attacks where a symlink inside the project points to
        sensitive files outside the sandbox (e.g., /etc/passwd).

        Args:
            base_dir: The sandbox root directory (project or build dir)
            relative_path: The relative path being accessed
            dir_name: Name of the directory for error messages

        Raises:
            ValueError: If a symlink escape is detected
        """
        current = base_dir
        for part in relative_path.parts:
            current = current / part

            # Skip if path doesn't exist yet (will be caught later)
            if not current.exists():
                break

            # Check if this component is a symlink
            if current.is_symlink():
                # Resolve just this symlink and check if it escapes
                resolved = current.resolve()
                try:
                    resolved.relative_to(base_dir)
                except ValueError:
                    raise ValueError(
                        f"Symlink escape detected: {part} points outside {dir_name}"
                    )

    def read_file(self, file_path: str, max_lines: int = 200, start_line: int = 1) -> dict[str, Any]:
        """
        Read a file from the project directory.

        Args:
            file_path: Relative path from project root
            max_lines: Maximum lines to read (default: 200)
            start_line: Line number to start reading from (1-indexed, default: 1)

        Returns:
            Dict with 'content' (file contents) or 'error' (error message)
        """
        try:
            full_path = self._validate_path(file_path)

            if not full_path.exists():
                return {"error": f"File not found: {file_path}"}

            if not full_path.is_file():
                return {"error": f"Not a file: {file_path}"}

            # Validate start_line
            if start_line < 1:
                return {"error": f"start_line must be >= 1, got {start_line}"}

            # Read file with line limit and offset
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                current_line = 0
                lines_read = 0

                for line in f:
                    current_line += 1

                    # Skip lines before start_line
                    if current_line < start_line:
                        continue

                    # Check if we've read enough lines
                    if lines_read >= max_lines:
                        lines.append(
                            f"\n... (truncated after {max_lines} lines, use start_line={current_line} to continue)"
                        )
                        break

                    # Add line number prefix for easier reference
                    lines.append(f"{current_line:4d}: {line.rstrip()}")
                    lines_read += 1

                content = "\n".join(lines)

                # Check if we skipped past the end
                if current_line < start_line:
                    return {
                        "error": f"start_line {start_line} exceeds file length ({current_line} lines)"
                    }

            return {
                "content": content,
                "path": file_path,
                "start_line": start_line,
                "end_line": start_line + lines_read - 1,
                "lines_read": lines_read,
                "truncated": lines_read >= max_lines,
            }

        except ValueError as e:
            # Security validation error
            return {"error": str(e)}
        except UnicodeDecodeError:
            return {"error": f"Cannot read {file_path}: binary file or unsupported encoding"}
        except PermissionError:
            return {"error": f"Permission denied: {file_path}"}
        except Exception as e:
            return {"error": f"Error reading {file_path}: {str(e)}"}

    def list_dir(self, dir_path: str = ".", pattern: str | None = None) -> dict[str, Any]:
        """
        List files and directories in a project directory.

        Args:
            dir_path: Relative path from project root (default: ".")
            pattern: Optional glob pattern (e.g., "*.cmake", "CMake*")

        Returns:
            Dict with 'files', 'directories' lists or 'error' message
        """
        try:
            full_path = self._validate_path(dir_path)

            if not full_path.exists():
                return {"error": f"Directory not found: {dir_path}"}

            if not full_path.is_dir():
                return {"error": f"Not a directory: {dir_path}"}

            # List contents
            if pattern:
                items = list(full_path.glob(pattern))
            else:
                items = list(full_path.iterdir())

            # Separate files and directories
            files = []
            directories = []

            # Determine which base path to use for relative paths
            # Check if we're listing inside build_dir or project_path
            if self.build_dir:
                try:
                    full_path.relative_to(self.build_dir)
                    base_path = self.build_dir
                    path_prefix = "build"
                except ValueError:
                    base_path = self.project_path
                    path_prefix = None
            else:
                base_path = self.project_path
                path_prefix = None

            for item in sorted(items):
                try:
                    rel_path = item.relative_to(base_path)
                    if path_prefix:
                        rel_path = Path(path_prefix) / rel_path
                except ValueError:
                    # Fallback: just use the name
                    rel_path = Path(item.name)
                name = item.name

                if item.is_file():
                    size = item.stat().st_size
                    files.append({"name": name, "path": str(rel_path), "size": size})
                elif item.is_dir():
                    directories.append({"name": name, "path": str(rel_path)})

            return {
                "path": dir_path,
                "files": files,
                "directories": directories,
                "total": len(files) + len(directories),
            }

        except ValueError as e:
            # Security validation error
            return {"error": str(e)}
        except PermissionError:
            return {"error": f"Permission denied: {dir_path}"}
        except Exception as e:
            return {"error": f"Error listing {dir_path}: {str(e)}"}


# Tool definitions for LLM (Anthropic tool use format)
TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the project or build directory. Use this to examine build scripts, "
            "source files, configuration files, or build artifacts (like compile_commands.json). "
            "Paths must be RELATIVE to project root. Use 'build/' prefix to access build directory. "
            "If errors reference specific line numbers, use start_line to jump to that section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path from project root. Use 'build/' prefix for build artifacts. Examples: 'configure.ac', 'src/main.c', 'build/compile_commands.json'. Absolute paths are NOT allowed.",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default: 200)",
                    "default": 200,
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed, default: 1). Use this to jump to specific sections when errors mention line numbers.",
                    "default": 1,
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files and directories in the project or build directory. Use this to explore the "
            "project structure, discover source files, inspect build artifacts, or understand the "
            "layout of the codebase. Paths must be RELATIVE. Use 'build/' prefix for build directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Relative path from project root. Use 'build/' prefix for build directory. Examples: '.', 'src', 'build', 'build/src'. Absolute paths are NOT allowed.",
                    "default": ".",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern to filter results (e.g., '*.cmake', 'CMake*')",
                },
            },
        },
    },
]
