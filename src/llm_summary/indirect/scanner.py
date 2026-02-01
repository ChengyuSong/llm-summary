"""Scanner for address-taken functions."""

from pathlib import Path

from clang.cindex import Cursor, CursorKind

from ..compile_commands import CompileCommandsDB
from ..db import SummaryDB
from ..extractor import FunctionExtractorWithBodies, get_type_spelling
from ..models import AddressFlow, AddressTakenFunction


class AddressTakenScanner:
    """
    Scans codebase to find functions whose addresses are taken.

    This identifies:
    1. Functions whose address is taken (&foo, callback = foo, passed as argument)
    2. Where these addresses flow to (struct fields, globals, parameters)
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
        self._function_map: dict[str, int] = {}  # name -> function_id
        self._file_contents: dict[str, list[str]] = {}

    def scan_files(self, file_paths: list[str | Path]) -> None:
        """Scan files for address-taken functions."""
        # Build function map from database
        for func in self.db.get_all_functions():
            if func.id is not None:
                self._function_map[func.name] = func.id

        # Scan each file
        for path in file_paths:
            try:
                self._scan_file(path)
            except Exception as e:
                print(f"Warning: Failed to scan {path}: {e}")

    def scan_directory(
        self,
        directory: str | Path,
        extensions: tuple[str, ...] = (".c", ".cpp", ".cc", ".cxx"),
        recursive: bool = True,
    ) -> None:
        """Scan all files in a directory."""
        directory = Path(directory)

        if recursive:
            files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in directory.glob("*") if f.suffix.lower() in extensions]

        self.scan_files(files)

    def _scan_file(self, file_path: str | Path) -> None:
        """Scan a single file for address-taken functions."""
        file_path = Path(file_path).resolve()
        tu = self.extractor.get_translation_unit(file_path)

        # Load file contents for context extraction
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                self._file_contents[str(file_path)] = f.read().splitlines()
        except (OSError, IOError):
            self._file_contents[str(file_path)] = []

        self._scan_cursor(tu.cursor, str(file_path))

    def _scan_cursor(
        self,
        cursor: Cursor,
        main_file: str,
        context: str | None = None,
        meaningful_parent: Cursor | None = None,
    ) -> None:
        """Recursively scan for address-taken expressions."""
        for child in cursor.get_children():
            # Skip if not from main file
            if child.location.file and str(child.location.file) != main_file:
                continue

            # Look for unary operator & (address-of)
            if child.kind == CursorKind.UNARY_OPERATOR:
                self._check_address_of(child, main_file, context)

            # Look for function references in assignments and calls
            elif child.kind == CursorKind.DECL_REF_EXPR:
                # Use meaningful_parent (skipping UNEXPOSED_EXPR wrappers)
                parent_to_check = meaningful_parent if meaningful_parent else cursor
                self._check_function_reference(child, main_file, parent_to_check)

            # Track context for flow analysis
            new_context = context
            if child.kind == CursorKind.VAR_DECL:
                new_context = child.spelling
            elif child.kind == CursorKind.FIELD_DECL:
                new_context = f"field:{child.spelling}"
            elif child.kind == CursorKind.BINARY_OPERATOR:
                # Try to get the LHS for assignment context
                lhs = self._get_assignment_target(child)
                if lhs:
                    new_context = lhs

            # Determine the meaningful parent to pass down
            # Skip UNEXPOSED_EXPR nodes which are implicit casts
            new_meaningful_parent = None
            if child.kind == CursorKind.UNEXPOSED_EXPR:
                # Keep the previous meaningful parent
                new_meaningful_parent = meaningful_parent if meaningful_parent else cursor
            elif child.kind in (
                CursorKind.CALL_EXPR,
                CursorKind.BINARY_OPERATOR,
                CursorKind.VAR_DECL,
                CursorKind.INIT_LIST_EXPR,
            ):
                # This is a meaningful parent
                new_meaningful_parent = child

            self._scan_cursor(child, main_file, new_context, new_meaningful_parent)

    def _check_address_of(
        self, cursor: Cursor, file_path: str, context: str | None
    ) -> None:
        """Check if this is address-of a function."""
        for child in cursor.get_children():
            if child.kind == CursorKind.DECL_REF_EXPR:
                referenced = child.referenced
                if referenced and referenced.kind == CursorKind.FUNCTION_DECL:
                    self._record_address_taken(referenced, file_path, cursor, context)

    def _check_function_reference(
        self, cursor: Cursor, file_path: str, parent: Cursor
    ) -> None:
        """Check if a declaration reference is a function being used as a value."""
        referenced = cursor.referenced
        if not referenced:
            return

        if referenced.kind != CursorKind.FUNCTION_DECL:
            return

        # Check if this is in a context where address is being taken
        # (assignment, initialization, argument passing)
        parent_kind = parent.kind

        if parent_kind == CursorKind.CALL_EXPR:
            # For call expressions, check if this function is in the callee position
            # or in an argument position. Only arguments have their address taken.
            children = list(parent.get_children())
            if children:
                # The first child (after skipping UNEXPOSED_EXPR) is the callee
                first_child = children[0]
                # Check if cursor is the callee by comparing locations
                if self._is_callee_position(cursor, first_child):
                    return  # This is a direct call, not address-taken

            # This is an argument - the function's address is being passed
            flow_target = self._determine_flow_target(cursor, parent)
            self._record_address_taken(
                referenced, file_path, cursor, flow_target
            )

        elif parent_kind in (
            CursorKind.VAR_DECL,
            CursorKind.INIT_LIST_EXPR,
            CursorKind.BINARY_OPERATOR,
        ):
            # Determine the flow target
            flow_target = self._determine_flow_target(cursor, parent)
            self._record_address_taken(
                referenced, file_path, cursor, flow_target
            )

    def _is_callee_position(self, func_ref: Cursor, first_child: Cursor) -> bool:
        """Check if func_ref is in the callee position of a call expression."""
        # The callee is typically an UNEXPOSED_EXPR wrapping a DECL_REF_EXPR
        # Compare by location since cursor identity doesn't work reliably
        if first_child.kind == CursorKind.UNEXPOSED_EXPR:
            for child in first_child.get_children():
                if child.kind == CursorKind.DECL_REF_EXPR:
                    if (child.location.line == func_ref.location.line and
                        child.location.column == func_ref.location.column):
                        return True
        elif first_child.kind == CursorKind.DECL_REF_EXPR:
            if (first_child.location.line == func_ref.location.line and
                first_child.location.column == func_ref.location.column):
                return True
        return False

    def _record_address_taken(
        self,
        func_cursor: Cursor,
        file_path: str,
        context_cursor: Cursor,
        flow_target: str | None,
    ) -> None:
        """Record that a function's address is taken."""
        func_name = func_cursor.spelling
        if func_name not in self._function_map:
            return  # External function

        func_id = self._function_map[func_name]

        # Get signature
        signature = get_type_spelling(func_cursor)

        # Record as address-taken
        atf = AddressTakenFunction(function_id=func_id, signature=signature)
        try:
            self.db.add_address_taken_function(atf)
        except Exception:
            pass  # Already recorded

        # Record the flow if we have a target
        if flow_target:
            context_snippet = self._get_context_snippet(
                file_path, context_cursor.location.line
            )
            flow = AddressFlow(
                function_id=func_id,
                flow_target=flow_target,
                file_path=file_path,
                line_number=context_cursor.location.line,
                context_snippet=context_snippet,
            )
            self.db.add_address_flow(flow)

    def _determine_flow_target(self, cursor: Cursor, parent: Cursor) -> str | None:
        """Determine where a function address flows to."""
        if parent.kind == CursorKind.VAR_DECL:
            return f"var:{parent.spelling}"

        elif parent.kind == CursorKind.CALL_EXPR:
            # Passed as argument
            func_name = self._extract_callee_name(parent)
            param_idx = -1

            for i, child in enumerate(parent.get_children()):
                if i == 0:
                    continue  # Skip the callee (first child)
                if self._cursors_match(child, cursor) or self._contains_cursor(child, cursor):
                    param_idx = i - 1  # Subtract 1 because first child is the callee
                    break

            if func_name and param_idx >= 0:
                return f"param:{func_name}[{param_idx}]"

        elif parent.kind == CursorKind.BINARY_OPERATOR:
            # Assignment
            target = self._get_assignment_target(parent)
            if target:
                return target

        elif parent.kind == CursorKind.INIT_LIST_EXPR:
            return "init_list"

        return None

    def _extract_callee_name(self, call_expr: Cursor) -> str:
        """Extract the function name from a CALL_EXPR, handling UNEXPOSED_EXPR wrappers."""
        children = list(call_expr.get_children())
        if not children:
            return ""

        callee = children[0]

        # Unwrap UNEXPOSED_EXPR if present
        if callee.kind == CursorKind.UNEXPOSED_EXPR:
            for child in callee.get_children():
                if child.kind == CursorKind.DECL_REF_EXPR:
                    return child.spelling
        elif callee.kind == CursorKind.DECL_REF_EXPR:
            return callee.spelling

        return ""

    def _cursors_match(self, c1: Cursor, c2: Cursor) -> bool:
        """Check if two cursors refer to the same location."""
        # Direct equality
        if c1 == c2:
            return True
        # Compare by location (more reliable)
        if (c1.location.file and c2.location.file and
            c1.location.line == c2.location.line and
            c1.location.column == c2.location.column):
            return True
        return False

    def _contains_cursor(self, parent: Cursor, target: Cursor) -> bool:
        """Check if parent contains target cursor."""
        for child in parent.get_children():
            if self._cursors_match(child, target):
                return True
            if self._contains_cursor(child, target):
                return True
        return False

    def _get_assignment_target(self, binary_op: Cursor) -> str | None:
        """Get the target of an assignment operation."""
        children = list(binary_op.get_children())
        if len(children) >= 1:
            lhs = children[0]
            if lhs.kind == CursorKind.DECL_REF_EXPR:
                return f"var:{lhs.spelling}"
            elif lhs.kind == CursorKind.MEMBER_REF_EXPR:
                return f"field:{lhs.spelling}"
            elif lhs.kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
                # Array assignment
                base = list(lhs.get_children())[0] if list(lhs.get_children()) else None
                if base and base.kind == CursorKind.DECL_REF_EXPR:
                    return f"array:{base.spelling}"
        return None

    def _get_context_snippet(self, file_path: str, line: int, context_lines: int = 3) -> str:
        """Get surrounding lines for context."""
        lines = self._file_contents.get(file_path, [])
        if not lines:
            return ""

        start = max(0, line - 1 - context_lines)
        end = min(len(lines), line + context_lines)

        return "\n".join(lines[start:end])
