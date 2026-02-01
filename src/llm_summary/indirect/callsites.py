"""Finder for indirect call sites (function pointer calls)."""

from pathlib import Path

from clang.cindex import Cursor, CursorKind, TypeKind

from ..compile_commands import CompileCommandsDB
from ..db import SummaryDB
from ..extractor import FunctionExtractorWithBodies
from ..models import IndirectCallsite


class IndirectCallsiteFinder:
    """
    Finds indirect call sites (function pointer calls, virtual calls).

    Identifies:
    1. Calls through function pointers: ptr(args), struct->callback(args)
    2. Calls through arrays of function pointers: handlers[i](args)
    3. Virtual method calls (C++): obj->virtual_method()
    """

    def __init__(
        self,
        db: SummaryDB,
        compile_args: list[str] | None = None,
        libclang_path: str | None = None,
        compile_commands: CompileCommandsDB | None = None,
    ):
        self.db = db
        self.compile_commands = compile_commands
        self.extractor = FunctionExtractorWithBodies(
            compile_args, libclang_path, compile_commands
        )
        self._function_map: dict[tuple[str, str, int], int] = {}  # (file, name, line) -> id
        self._file_contents: dict[str, list[str]] = {}

    def find_in_files(self, file_paths: list[str | Path]) -> list[IndirectCallsite]:
        """Find indirect call sites in files."""
        # Build function map from database
        for func in self.db.get_all_functions():
            if func.id is not None:
                key = (func.file_path, func.name, func.line_start)
                self._function_map[key] = func.id

        all_callsites = []
        for path in file_paths:
            try:
                callsites = self._find_in_file(path)
                all_callsites.extend(callsites)
            except Exception as e:
                print(f"Warning: Failed to find indirect calls in {path}: {e}")

        return all_callsites

    def find_in_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".c", ".cpp", ".cc", ".cxx"),
        recursive: bool = True,
    ) -> list[IndirectCallsite]:
        """Find indirect call sites in all files in a directory."""
        directory = Path(directory)

        if recursive:
            files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in directory.glob("*") if f.suffix.lower() in extensions]

        return self.find_in_files(files)

    def _find_in_file(self, file_path: str | Path) -> list[IndirectCallsite]:
        """Find indirect call sites in a single file."""
        file_path = Path(file_path).resolve()
        tu = self.extractor.get_translation_unit(file_path)

        # Load file contents
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                self._file_contents[str(file_path)] = f.read().splitlines()
        except (OSError, IOError):
            self._file_contents[str(file_path)] = []

        callsites = []
        self._find_recursive(tu.cursor, str(file_path), callsites, None)
        return callsites

    def _find_recursive(
        self,
        cursor: Cursor,
        main_file: str,
        callsites: list[IndirectCallsite],
        current_func_id: int | None,
    ) -> None:
        """Recursively find indirect call expressions."""
        for child in cursor.get_children():
            # Skip if not from main file
            if child.location.file and str(child.location.file) != main_file:
                continue

            func_id = current_func_id

            # Track which function we're in
            if child.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD):
                if child.is_definition():
                    key = (main_file, child.spelling, child.extent.start.line)
                    func_id = self._function_map.get(key)

            # Check for call expressions
            if child.kind == CursorKind.CALL_EXPR:
                if func_id is not None:
                    callsite = self._check_indirect_call(child, main_file, func_id)
                    if callsite:
                        callsites.append(callsite)
                        # Store in database
                        self.db.add_indirect_callsite(callsite)

            self._find_recursive(child, main_file, callsites, func_id)

    def _check_indirect_call(
        self, call_cursor: Cursor, file_path: str, caller_func_id: int
    ) -> IndirectCallsite | None:
        """Check if this call expression is an indirect call."""
        # Get the callee expression
        children = list(call_cursor.get_children())
        if not children:
            return None

        callee = children[0]

        # Unwrap UNEXPOSED_EXPR which wraps implicit casts
        actual_callee = self._unwrap_unexposed(callee)

        # Direct function call - skip
        if actual_callee.kind == CursorKind.DECL_REF_EXPR:
            referenced = actual_callee.referenced
            if referenced and referenced.kind == CursorKind.FUNCTION_DECL:
                return None  # Direct call

        # Check if this is a call through a pointer
        if self._is_indirect_call(actual_callee):
            callee_expr = self._get_callee_expr(actual_callee)
            signature = self._get_call_signature(call_cursor)
            context_snippet = self._get_context_snippet(
                file_path, call_cursor.location.line
            )

            return IndirectCallsite(
                caller_function_id=caller_func_id,
                file_path=file_path,
                line_number=call_cursor.location.line,
                callee_expr=callee_expr,
                signature=signature,
                context_snippet=context_snippet,
            )

        return None

    def _unwrap_unexposed(self, cursor: Cursor) -> Cursor:
        """Unwrap UNEXPOSED_EXPR nodes to get the actual expression."""
        while cursor.kind == CursorKind.UNEXPOSED_EXPR:
            children = list(cursor.get_children())
            if not children:
                break
            cursor = children[0]
        return cursor

    def _is_indirect_call(self, callee: Cursor) -> bool:
        """Determine if the callee represents an indirect call."""
        # Member expression (ptr->callback or obj.callback)
        if callee.kind == CursorKind.MEMBER_REF_EXPR:
            # Check if the member is a function pointer
            member_type = callee.type
            if member_type.kind == TypeKind.POINTER:
                pointee = member_type.get_pointee()
                if pointee.kind == TypeKind.FUNCTIONPROTO:
                    return True
            # Also check for direct function pointer type
            if member_type.kind == TypeKind.FUNCTIONPROTO:
                return True
            return True  # Assume indirect for member calls

        # Variable reference that's a function pointer
        if callee.kind == CursorKind.DECL_REF_EXPR:
            var_type = callee.type
            if var_type.kind == TypeKind.POINTER:
                pointee = var_type.get_pointee()
                if pointee.kind == TypeKind.FUNCTIONPROTO:
                    return True
            # Typedef'd function pointer
            if var_type.kind == TypeKind.TYPEDEF:
                canonical = var_type.get_canonical()
                if canonical.kind == TypeKind.POINTER:
                    pointee = canonical.get_pointee()
                    if pointee.kind == TypeKind.FUNCTIONPROTO:
                        return True

        # Array subscript (handlers[i])
        if callee.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            return True

        # Parenthesized expression
        if callee.kind == CursorKind.PAREN_EXPR:
            inner_children = list(callee.get_children())
            if inner_children:
                return self._is_indirect_call(inner_children[0])

        # Unary operator (dereference)
        if callee.kind == CursorKind.UNARY_OPERATOR:
            return True

        return False

    def _get_callee_expr(self, callee: Cursor) -> str:
        """Get a string representation of the callee expression."""
        if callee.kind == CursorKind.MEMBER_REF_EXPR:
            # Try to get the base expression
            children = list(callee.get_children())
            if children:
                base = self._get_callee_expr(children[0])
                member = callee.spelling
                # Determine if arrow or dot
                return f"{base}->{member}" if base else member
            return callee.spelling

        elif callee.kind == CursorKind.DECL_REF_EXPR:
            return callee.spelling

        elif callee.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(callee.get_children())
            if len(children) >= 2:
                base = self._get_callee_expr(children[0])
                # Index expression - just use [i] as placeholder
                return f"{base}[i]"
            return "unknown[i]"

        elif callee.kind == CursorKind.PAREN_EXPR:
            children = list(callee.get_children())
            if children:
                inner = self._get_callee_expr(children[0])
                return f"({inner})"
            return "()"

        elif callee.kind == CursorKind.UNARY_OPERATOR:
            children = list(callee.get_children())
            if children:
                operand = self._get_callee_expr(children[0])
                return f"*{operand}"
            return "*ptr"

        return callee.spelling or "unknown"

    def _get_call_signature(self, call_cursor: Cursor) -> str:
        """Get the expected signature for the call."""
        result_type = call_cursor.type.spelling

        # Count arguments
        children = list(call_cursor.get_children())
        arg_count = len(children) - 1  # First child is callee

        # Try to get argument types
        arg_types = []
        for child in children[1:]:  # Skip callee
            arg_types.append(child.type.spelling)

        if arg_types:
            return f"{result_type}({', '.join(arg_types)})"
        else:
            return f"{result_type}()"

    def _get_context_snippet(
        self, file_path: str, line: int, context_lines: int = 5
    ) -> str:
        """Get surrounding lines for context."""
        lines = self._file_contents.get(file_path, [])
        if not lines:
            return ""

        start = max(0, line - 1 - context_lines)
        end = min(len(lines), line + context_lines)

        return "\n".join(lines[start:end])
