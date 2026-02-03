"""Tools for the build agent to explore projects."""

from pathlib import Path
from typing import Any


class BuildTools:
    """Tools available to the build agent for exploring projects."""

    def __init__(self, project_path: Path):
        self.project_path = Path(project_path).resolve()

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve path, ensuring it's within project directory.

        Args:
            path: Requested path (should be relative to project root)

        Returns:
            Resolved absolute path within project directory

        Raises:
            ValueError: If path escapes sandbox or is invalid
        """
        # Convert to Path
        requested = Path(path)

        # Reject absolute paths
        if requested.is_absolute():
            raise ValueError(f"Absolute paths not allowed: {path}")

        # Resolve relative to project root
        full_path = (self.project_path / requested).resolve()

        # Security: ensure resolved path is within project
        try:
            full_path.relative_to(self.project_path)
        except ValueError:
            raise ValueError(f"Path escapes project directory: {path}")

        return full_path

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

            for item in sorted(items):
                rel_path = item.relative_to(self.project_path)
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
            "Read a file from the project directory. Use this to examine CMake files, "
            "included modules, source files, configuration files, or any other project files "
            "that might help understand the build configuration or debug errors. "
            "If errors reference specific line numbers, use start_line to jump to that section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Relative path from project root (e.g., 'cmake/FindZLIB.cmake', 'src/config.h.in')",
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
            "List files and directories in a project directory. Use this to explore the "
            "project structure, find CMake modules, discover included files, or understand "
            "the layout of the codebase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "Relative path from project root (default: '.' for root)",
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
