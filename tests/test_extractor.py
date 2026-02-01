"""Tests for the function extractor."""

import pytest
from pathlib import Path


def _check_libclang_available():
    """Check if libclang is available, compatible, and functional."""
    try:
        import os as _os
        import tempfile
        from clang.cindex import Config, Index

        # Configure libclang path before any usage
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

        # Create a test index
        _test_index = Index.create()

        # Try to actually parse something to ensure full compatibility
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write('void test_func(void) {}')
            f.flush()
            temp_path = f.name

        try:
            tu = _test_index.parse(temp_path, args=['-x', 'c'])
            # Check if we can find the function
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

# Conditionally import - only if libclang works
if LIBCLANG_AVAILABLE:
    from llm_summary.extractor import FunctionExtractor, FunctionExtractorWithBodies


pytestmark = pytest.mark.skipif(
    not LIBCLANG_AVAILABLE,
    reason=LIBCLANG_ERROR or "libclang not available"
)


@pytest.fixture
def fixtures_dir():
    """Get the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def extractor():
    """Create a function extractor."""
    return FunctionExtractor()


class TestFunctionExtractor:
    """Tests for FunctionExtractor class."""

    def test_extract_from_simple_file(self, extractor, fixtures_dir):
        """Test extracting functions from simple.c."""
        simple_c = fixtures_dir / "simple.c"
        if not simple_c.exists():
            pytest.skip("simple.c fixture not found")

        functions = extractor.extract_from_file(simple_c)

        # Should extract multiple functions
        assert len(functions) > 0

        # Check for expected functions
        func_names = {f.name for f in functions}
        assert "create_buffer" in func_names
        assert "buffer_new" in func_names
        assert "buffer_free" in func_names

    def test_function_attributes(self, extractor, fixtures_dir):
        """Test that extracted functions have correct attributes."""
        simple_c = fixtures_dir / "simple.c"
        if not simple_c.exists():
            pytest.skip("simple.c fixture not found")

        functions = extractor.extract_from_file(simple_c)
        create_buffer = next((f for f in functions if f.name == "create_buffer"), None)

        assert create_buffer is not None
        assert create_buffer.file_path == str(simple_c)
        assert create_buffer.line_start > 0
        assert create_buffer.line_end >= create_buffer.line_start
        assert "malloc" in create_buffer.source
        assert "size_t" in create_buffer.signature

    def test_extract_from_callbacks(self, extractor, fixtures_dir):
        """Test extracting from file with function pointers."""
        callbacks_c = fixtures_dir / "callbacks.c"
        if not callbacks_c.exists():
            pytest.skip("callbacks.c fixture not found")

        functions = extractor.extract_from_file(callbacks_c)

        func_names = {f.name for f in functions}
        assert "dispatch_event" in func_names
        assert "register_handler" in func_names
        assert "log_event" in func_names

    def test_extract_from_recursive(self, extractor, fixtures_dir):
        """Test extracting recursive functions."""
        recursive_c = fixtures_dir / "recursive.c"
        if not recursive_c.exists():
            pytest.skip("recursive.c fixture not found")

        functions = extractor.extract_from_file(recursive_c)

        func_names = {f.name for f in functions}
        assert "factorial" in func_names
        assert "is_even" in func_names
        assert "is_odd" in func_names
        assert "build_tree" in func_names

    def test_extract_nonexistent_file(self, extractor):
        """Test handling of nonexistent file."""
        with pytest.raises(RuntimeError):
            extractor.extract_from_file("/nonexistent/file.c")

    def test_extract_from_directory(self, extractor, fixtures_dir):
        """Test extracting from a directory."""
        functions = extractor.extract_from_directory(fixtures_dir, recursive=False)

        # Should find functions from all .c files
        assert len(functions) > 0

        # Functions from multiple files
        files = {f.file_path for f in functions}
        assert len(files) >= 1


class TestFunctionExtractorWithBodies:
    """Tests for FunctionExtractorWithBodies class."""

    def test_extract_preserves_body(self, fixtures_dir):
        """Test that function bodies are preserved."""
        extractor = FunctionExtractorWithBodies()
        simple_c = fixtures_dir / "simple.c"
        if not simple_c.exists():
            pytest.skip("simple.c fixture not found")

        functions = extractor.extract_from_file(simple_c)
        create_buffer = next((f for f in functions if f.name == "create_buffer"), None)

        assert create_buffer is not None
        # Should have the full function body
        assert "malloc" in create_buffer.source
        assert "return" in create_buffer.source
