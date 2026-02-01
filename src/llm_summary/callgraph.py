"""Call graph construction from C/C++ source files."""

from pathlib import Path

from clang.cindex import (
    Cursor,
    CursorKind,
    TranslationUnit,
)

from .db import SummaryDB
from .extractor import FunctionExtractorWithBodies
from .models import CallEdge, Function


class CallGraphBuilder:
    """Builds a call graph from C/C++ source files."""

    def __init__(
        self,
        db: SummaryDB,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
    ):
        self.db = db
        self.extractor = FunctionExtractorWithBodies(compile_args, libclang_path)
        self._function_map: dict[str, list[int]] = {}  # name -> [function_ids]

    def build_from_files(self, file_paths: list[str | Path]) -> list[CallEdge]:
        """
        Build call graph from source files.

        Returns list of call edges (direct calls only).
        """
        # First pass: extract and store all functions
        all_functions = []
        for path in file_paths:
            try:
                functions = self.extractor.extract_from_file(path)
                all_functions.extend(functions)
            except Exception as e:
                print(f"Warning: Failed to extract from {path}: {e}")

        # Store functions and build name -> id mapping
        func_ids = self.db.insert_functions_batch(all_functions)
        for func, func_id in func_ids.items():
            if func.name not in self._function_map:
                self._function_map[func.name] = []
            self._function_map[func.name].append(func_id)

        # Second pass: extract call edges
        all_edges = []
        for path in file_paths:
            try:
                edges = self._extract_calls_from_file(path)
                all_edges.extend(edges)
            except Exception as e:
                print(f"Warning: Failed to extract calls from {path}: {e}")

        # Store edges
        self.db.add_call_edges_batch(all_edges)

        return all_edges

    def build_from_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".c", ".cpp", ".cc", ".cxx"),
        recursive: bool = True,
    ) -> list[CallEdge]:
        """Build call graph from all files in a directory."""
        directory = Path(directory)

        if recursive:
            files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in directory.glob("*") if f.suffix.lower() in extensions]

        return self.build_from_files(files)

    def _extract_calls_from_file(self, file_path: str | Path) -> list[CallEdge]:
        """Extract call edges from a single file."""
        file_path = Path(file_path).resolve()
        tu = self.extractor.get_translation_unit(file_path)

        edges = []
        self._find_calls_recursive(tu.cursor, str(file_path), edges)
        return edges

    def _find_calls_recursive(
        self,
        cursor: Cursor,
        main_file: str,
        edges: list[CallEdge],
        current_function_id: int | None = None,
    ) -> None:
        """Recursively find call expressions in the AST."""
        for child in cursor.get_children():
            # Skip if not from the main file
            if child.location.file and str(child.location.file) != main_file:
                continue

            # Track current function context
            func_id = current_function_id
            if child.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD):
                if child.is_definition():
                    # Find this function's ID
                    func_id = self._find_function_id(child, main_file)

            # Found a call expression
            if child.kind == CursorKind.CALL_EXPR:
                if func_id is not None:
                    callee_id = self._resolve_callee(child)
                    if callee_id is not None:
                        edges.append(CallEdge(
                            caller_id=func_id,
                            callee_id=callee_id,
                            file_path=main_file,
                            line=child.location.line,
                            column=child.location.column,
                        ))

            # Recurse
            self._find_calls_recursive(child, main_file, edges, func_id)

    def _find_function_id(self, cursor: Cursor, file_path: str) -> int | None:
        """Find the database ID for a function cursor."""
        name = cursor.spelling

        # Look up by name
        if name in self._function_map:
            ids = self._function_map[name]
            if len(ids) == 1:
                return ids[0]

            # Multiple matches - try to disambiguate by file and line
            for func_id in ids:
                func = self.db.get_function(func_id)
                if func and func.file_path == file_path:
                    if func.line_start == cursor.extent.start.line:
                        return func_id

            # Return first match as fallback
            return ids[0] if ids else None

        return None

    def _resolve_callee(self, call_cursor: Cursor) -> int | None:
        """Resolve the callee of a call expression to a function ID."""
        # Get the referenced function
        referenced = call_cursor.referenced
        if referenced is None:
            return None

        callee_name = referenced.spelling
        if not callee_name:
            return None

        # Look up in our function map
        if callee_name in self._function_map:
            ids = self._function_map[callee_name]
            if ids:
                # TODO: Better overload resolution
                return ids[0]

        return None

    def get_call_graph(self) -> dict[int, list[int]]:
        """Get the call graph as an adjacency list."""
        edges = self.db.get_all_call_edges()
        graph: dict[int, list[int]] = {}

        for edge in edges:
            if edge.caller_id not in graph:
                graph[edge.caller_id] = []
            graph[edge.caller_id].append(edge.callee_id)

        return graph

    def get_reverse_call_graph(self) -> dict[int, list[int]]:
        """Get the reverse call graph (callee -> callers)."""
        edges = self.db.get_all_call_edges()
        graph: dict[int, list[int]] = {}

        for edge in edges:
            if edge.callee_id not in graph:
                graph[edge.callee_id] = []
            graph[edge.callee_id].append(edge.caller_id)

        return graph


class DirectCallExtractor:
    """Extracts direct function calls from a function body."""

    def __init__(self, db: SummaryDB):
        self.db = db

    def get_callees_from_source(
        self, func: Function, all_functions: dict[str, list[Function]]
    ) -> list[Function]:
        """
        Extract direct callees from function source code.

        This is a simpler regex-based approach when full AST parsing isn't available.
        """
        import re

        callees = []
        source = func.source

        # Simple pattern to find function calls
        # Matches: identifier followed by (
        call_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

        for match in call_pattern.finditer(source):
            callee_name = match.group(1)

            # Skip common keywords and control structures
            if callee_name in (
                "if",
                "while",
                "for",
                "switch",
                "return",
                "sizeof",
                "typeof",
                "alignof",
                "offsetof",
                "static_cast",
                "dynamic_cast",
                "const_cast",
                "reinterpret_cast",
            ):
                continue

            # Look up the callee
            if callee_name in all_functions:
                # Add all overloads - caller should disambiguate if needed
                callees.extend(all_functions[callee_name])

        return list(set(callees))
