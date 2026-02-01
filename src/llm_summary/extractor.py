"""Function extraction from C/C++ source files using libclang."""

import os
from pathlib import Path

from clang.cindex import (
    Config,
    Cursor,
    CursorKind,
    Index,
    TranslationUnit,
    TypeKind,
)

from .models import Function


def configure_libclang(libclang_path: str | None = None) -> None:
    """Configure libclang library path if needed."""
    # Check if already loaded - can't reconfigure
    if Config.loaded:
        return

    if libclang_path:
        Config.set_library_file(libclang_path)
    else:
        # Try common paths (prefer newer versions)
        common_paths = [
            "/usr/lib/x86_64-linux-gnu/libclang-18.so.1",
            "/usr/lib/llvm-18/lib/libclang-18.so.1",
            "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
            "/usr/lib/llvm-14/lib/libclang-14.so.1",
            "/usr/local/lib/libclang.dylib",  # macOS
            "/opt/homebrew/opt/llvm/lib/libclang.dylib",  # macOS ARM
        ]
        for path in common_paths:
            if os.path.exists(path):
                Config.set_library_file(path)
                break


def get_type_spelling(cursor: Cursor) -> str:
    """Get a normalized type spelling for a function."""
    result_type = cursor.result_type.spelling
    params = []

    for child in cursor.get_children():
        if child.kind == CursorKind.PARM_DECL:
            param_type = child.type.spelling
            params.append(param_type)

    return f"{result_type}({', '.join(params)})"


def get_function_source(cursor: Cursor, file_contents: dict[str, str]) -> str:
    """Extract source code for a function from file contents."""
    file_path = str(cursor.location.file)
    if file_path not in file_contents:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                file_contents[file_path] = f.read()
        except (OSError, IOError):
            return ""

    content = file_contents[file_path]
    lines = content.splitlines()

    start_line = cursor.extent.start.line - 1
    end_line = cursor.extent.end.line

    if start_line < 0 or end_line > len(lines):
        return ""

    source_lines = lines[start_line:end_line]

    # Handle partial first/last lines
    if source_lines:
        start_col = cursor.extent.start.column - 1
        end_col = cursor.extent.end.column

        if len(source_lines) == 1:
            source_lines[0] = source_lines[0][start_col:end_col]
        else:
            source_lines[0] = source_lines[0][start_col:]
            source_lines[-1] = source_lines[-1][:end_col]

    return "\n".join(source_lines)


class FunctionExtractor:
    """Extracts function definitions from C/C++ source files."""

    def __init__(
        self,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
    ):
        """
        Initialize the extractor.

        Args:
            compile_args: Additional compiler arguments (e.g., ["-I/path/to/include"])
            libclang_path: Path to libclang shared library
        """
        configure_libclang(libclang_path)
        self.index = Index.create()
        self.compile_args = compile_args or []
        self._file_contents: dict[str, str] = {}

    def extract_from_file(self, file_path: str | Path) -> list[Function]:
        """
        Extract all function definitions from a source file.

        Args:
            file_path: Path to the C/C++ source file

        Returns:
            List of Function objects
        """
        file_path = Path(file_path).resolve()

        # Determine language from extension
        ext = file_path.suffix.lower()
        args = list(self.compile_args)

        if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
            args.extend(["-x", "c++", "-std=c++17"])
        else:
            args.extend(["-x", "c", "-std=c11"])

        try:
            tu = self.index.parse(
                str(file_path),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}")

        functions = []
        self._extract_functions_recursive(
            tu.cursor, str(file_path), functions
        )

        return functions

    def extract_from_files(self, file_paths: list[str | Path]) -> list[Function]:
        """Extract functions from multiple files."""
        all_functions = []
        for path in file_paths:
            try:
                functions = self.extract_from_file(path)
                all_functions.extend(functions)
            except Exception as e:
                print(f"Warning: Failed to extract from {path}: {e}")
        return all_functions

    def extract_from_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp"),
        recursive: bool = True,
    ) -> list[Function]:
        """Extract functions from all C/C++ files in a directory."""
        directory = Path(directory)

        if recursive:
            files = [
                f for f in directory.rglob("*") if f.suffix.lower() in extensions
            ]
        else:
            files = [
                f for f in directory.glob("*") if f.suffix.lower() in extensions
            ]

        return self.extract_from_files(files)

    def _extract_functions_recursive(
        self,
        cursor: Cursor,
        main_file: str,
        functions: list[Function],
    ) -> None:
        """Recursively extract functions from AST."""
        for child in cursor.get_children():
            # Skip if not from the main file
            if child.location.file and str(child.location.file) != main_file:
                continue

            if child.kind == CursorKind.FUNCTION_DECL:
                # Only extract function definitions (not declarations)
                if child.is_definition():
                    func = self._cursor_to_function(child)
                    if func:
                        functions.append(func)

            elif child.kind == CursorKind.CXX_METHOD:
                # C++ method definition
                if child.is_definition():
                    func = self._cursor_to_function(child)
                    if func:
                        functions.append(func)

            # Recurse into namespaces and classes
            elif child.kind in (
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
                CursorKind.CLASS_TEMPLATE,
            ):
                self._extract_functions_recursive(child, main_file, functions)

    def _cursor_to_function(self, cursor: Cursor) -> Function | None:
        """Convert a clang cursor to a Function object."""
        if not cursor.location.file:
            return None

        # Get the full source
        source = self._get_full_source(cursor)
        if not source:
            return None

        # Get qualified name for C++ methods
        name = self._get_qualified_name(cursor)
        signature = get_type_spelling(cursor)

        return Function(
            name=name,
            file_path=str(cursor.location.file),
            line_start=cursor.extent.start.line,
            line_end=cursor.extent.end.line,
            source=source,
            signature=signature,
        )

    def _get_qualified_name(self, cursor: Cursor) -> str:
        """Get the fully qualified name for a function/method."""
        parts = []
        c = cursor
        while c:
            if c.kind in (
                CursorKind.FUNCTION_DECL,
                CursorKind.CXX_METHOD,
                CursorKind.CONSTRUCTOR,
                CursorKind.DESTRUCTOR,
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
            ):
                if c.spelling:
                    parts.append(c.spelling)
            c = c.semantic_parent

        parts.reverse()
        return "::".join(parts) if len(parts) > 1 else (parts[0] if parts else "")

    def _get_full_source(self, cursor: Cursor) -> str:
        """Get the full source code of a function including body."""
        if not cursor.location.file:
            return ""

        file_path = str(cursor.location.file)

        # Load file contents if not cached
        if file_path not in self._file_contents:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    self._file_contents[file_path] = f.read()
            except (OSError, IOError):
                return ""

        content = self._file_contents[file_path]
        lines = content.splitlines(keepends=True)

        start_line = cursor.extent.start.line - 1
        end_line = cursor.extent.end.line

        if start_line < 0 or end_line > len(lines):
            return ""

        source_lines = lines[start_line:end_line]
        return "".join(source_lines).strip()


class FunctionExtractorWithBodies(FunctionExtractor):
    """Function extractor that also parses function bodies for call extraction."""

    def __init__(
        self,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
    ):
        super().__init__(compile_args, libclang_path)

    def extract_from_file(self, file_path: str | Path) -> list[Function]:
        """Extract functions with full body parsing."""
        file_path = Path(file_path).resolve()

        ext = file_path.suffix.lower()
        args = list(self.compile_args)

        if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
            args.extend(["-x", "c++", "-std=c++17"])
        else:
            args.extend(["-x", "c", "-std=c11"])

        try:
            # Don't skip function bodies
            tu = self.index.parse(
                str(file_path),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse {file_path}: {e}")

        functions = []
        self._extract_functions_recursive(tu.cursor, str(file_path), functions)

        return functions

    def get_translation_unit(self, file_path: str | Path) -> TranslationUnit:
        """Get the translation unit for advanced analysis."""
        file_path = Path(file_path).resolve()

        ext = file_path.suffix.lower()
        args = list(self.compile_args)

        if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
            args.extend(["-x", "c++", "-std=c++17"])
        else:
            args.extend(["-x", "c", "-std=c11"])

        return self.index.parse(
            str(file_path),
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
