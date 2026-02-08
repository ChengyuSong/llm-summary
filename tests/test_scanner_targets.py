"""Tests for extended AddressTakenScanner target type detection."""

import pytest
from pathlib import Path


def _check_libclang_available():
    """Check if libclang is available, compatible, and functional."""
    try:
        import os as _os
        import tempfile
        from clang.cindex import Config, Index

        if not Config.loaded:
            common_paths = [
                "/usr/lib/x86_64-linux-gnu/libclang-18.so.1",
                "/usr/lib/llvm-18/lib/libclang.so.1",
                "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
                "/usr/lib/llvm-14/lib/libclang.so.1",
                "/usr/local/lib/libclang.dylib",
                "/opt/homebrew/opt/llvm/lib/libclang.dylib",
            ]
            for path in common_paths:
                if _os.path.exists(path):
                    Config.set_library_file(path)
                    break

        _test_index = Index.create()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write('void test_func(void) {}')
            f.flush()
            temp_path = f.name

        try:
            tu = _test_index.parse(temp_path, args=['-x', 'c'])
            found = False
            for cursor in tu.cursor.get_children():
                if cursor.spelling == 'test_func':
                    found = True
                    break
            if not found:
                return False, "libclang failed to parse test function"
            return True, None
        finally:
            _os.unlink(temp_path)

    except Exception as e:
        return False, str(e)


LIBCLANG_AVAILABLE, LIBCLANG_ERROR = _check_libclang_available()

if LIBCLANG_AVAILABLE:
    from llm_summary.db import SummaryDB
    from llm_summary.extractor import FunctionExtractor
    from llm_summary.indirect.scanner import AddressTakenScanner
    from llm_summary.models import TargetType


pytestmark = pytest.mark.skipif(
    not LIBCLANG_AVAILABLE,
    reason=LIBCLANG_ERROR or "libclang not available"
)


@pytest.fixture
def fixtures_dir():
    """Get the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def db():
    """Create an in-memory database."""
    return SummaryDB(":memory:")


def _extract_and_scan(db, file_path):
    """Helper: extract functions and scan for targets."""
    extractor = FunctionExtractor()
    functions = extractor.extract_from_file(file_path)
    db.insert_functions_batch(functions)
    scanner = AddressTakenScanner(db)
    scanner.scan_files([file_path])
    return functions


class TestAttributeDetection:
    """Tests for __attribute__ detection in C code."""

    def test_constructor_detected(self, db, fixtures_dir):
        """Test that __attribute__((constructor)) is detected."""
        _extract_and_scan(db, fixtures_dir / "attributes.c")
        atfs = db.get_address_taken_functions(target_type=TargetType.CONSTRUCTOR_ATTR.value)
        names = self._get_func_names(db, atfs)
        assert "init_subsystem" in names

    def test_destructor_detected(self, db, fixtures_dir):
        """Test that __attribute__((destructor)) is detected."""
        _extract_and_scan(db, fixtures_dir / "attributes.c")
        atfs = db.get_address_taken_functions(target_type=TargetType.DESTRUCTOR_ATTR.value)
        names = self._get_func_names(db, atfs)
        assert "cleanup_subsystem" in names

    def test_weak_detected(self, db, fixtures_dir):
        """Test that __attribute__((weak)) is detected."""
        _extract_and_scan(db, fixtures_dir / "attributes.c")
        atfs = db.get_address_taken_functions(target_type=TargetType.WEAK_SYMBOL.value)
        names = self._get_func_names(db, atfs)
        assert "default_handler" in names

    def test_section_detected(self, db, fixtures_dir):
        """Test that __attribute__((section(".init..."))) is detected."""
        _extract_and_scan(db, fixtures_dir / "attributes.c")
        atfs = db.get_address_taken_functions(target_type=TargetType.SECTION_PLACED.value)
        names = self._get_func_names(db, atfs)
        assert "early_init" in names

    def test_regular_functions_not_detected(self, db, fixtures_dir):
        """Test that regular functions are NOT marked with special types."""
        _extract_and_scan(db, fixtures_dir / "attributes.c")
        all_atfs = db.get_address_taken_functions()
        names = self._get_func_names(db, all_atfs)
        assert "regular_function" not in names
        assert "compute" not in names

    def _get_func_names(self, db, atfs):
        """Get function names from AddressTakenFunction list."""
        names = set()
        for atf in atfs:
            func = db.get_function(atf.function_id)
            if func:
                names.add(func.name)
        return names


class TestVirtualMethodDetection:
    """Tests for C++ virtual method detection."""

    def test_virtual_method_detected(self, db, fixtures_dir):
        """Test that virtual methods are detected."""
        _extract_and_scan(db, fixtures_dir / "virtual.cpp")
        atfs = db.get_address_taken_functions(target_type=TargetType.VIRTUAL_METHOD.value)
        names = self._get_func_names(db, atfs)
        # Base::on_event and Derived::on_event are virtual
        assert any("on_event" in n for n in names)

    def test_non_virtual_not_detected(self, db, fixtures_dir):
        """Test that non-virtual methods are NOT detected as virtual."""
        _extract_and_scan(db, fixtures_dir / "virtual.cpp")
        atfs = db.get_address_taken_functions(target_type=TargetType.VIRTUAL_METHOD.value)
        names = self._get_func_names(db, atfs)
        assert not any("non_virtual_method" in n for n in names)
        assert not any("derived_only" in n for n in names)

    def _get_func_names(self, db, atfs):
        """Get function names from AddressTakenFunction list."""
        names = set()
        for atf in atfs:
            func = db.get_function(atf.function_id)
            if func:
                names.add(func.name)
        return names


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing address-taken detection."""

    def test_address_taken_still_works(self, db, fixtures_dir):
        """Test that regular address-taken detection still works."""
        _extract_and_scan(db, fixtures_dir / "callbacks.c")
        atfs = db.get_address_taken_functions(target_type=TargetType.ADDRESS_TAKEN.value)
        names = self._get_func_names(db, atfs)
        # These should be detected as address-taken from callbacks.c
        assert "log_event" in names
        assert "handle_click" in names
        assert "default_alloc" in names

    def test_default_target_type(self, db):
        """Test that AddressTakenFunction defaults to 'address_taken'."""
        from llm_summary.models import AddressTakenFunction
        atf = AddressTakenFunction(function_id=1, signature="void()")
        assert atf.target_type == "address_taken"

    def test_multiple_target_types_same_function(self, db):
        """Test that a function can have multiple target types."""
        from llm_summary.models import Function, AddressTakenFunction
        func = Function(
            name="multi", file_path="test.c",
            line_start=1, line_end=5, source="void multi() {}",
            signature="void()",
        )
        func.id = db.insert_function(func)

        # Add as address_taken
        atf1 = AddressTakenFunction(
            function_id=func.id, signature="void()",
            target_type="address_taken",
        )
        db.add_address_taken_function(atf1)

        # Add as weak_symbol
        atf2 = AddressTakenFunction(
            function_id=func.id, signature="void()",
            target_type="weak_symbol",
        )
        db.add_address_taken_function(atf2)

        # Should have both
        all_atfs = db.get_address_taken_functions()
        func_atfs = [a for a in all_atfs if a.function_id == func.id]
        types = {a.target_type for a in func_atfs}
        assert "address_taken" in types
        assert "weak_symbol" in types

    def _get_func_names(self, db, atfs):
        """Get function names from AddressTakenFunction list."""
        names = set()
        for atf in atfs:
            func = db.get_function(atf.function_id)
            if func:
                names.add(func.name)
        return names
