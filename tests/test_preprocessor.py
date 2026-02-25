"""Tests for the source preprocessor module."""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_summary.preprocessor import PreprocessedFile, SourcePreprocessor, _LineMapping


class TestPreprocessedFile:
    """Tests for PreprocessedFile.extract_pp_source()."""

    def test_extract_matching_lines(self):
        """Lines mapping to the requested range are returned."""
        pf = PreprocessedFile(
            source_file="/tmp/test.c",
            mappings=[
                _LineMapping("int x = 1;", "/tmp/test.c", 5),
                _LineMapping("int y = 2;", "/tmp/test.c", 6),
                _LineMapping("int z = 3;", "/tmp/test.c", 7),
                _LineMapping("int w = 4;", "/tmp/test.c", 10),
            ],
        )
        result = pf.extract_pp_source("/tmp/test.c", 5, 7)
        assert result == "int x = 1;\nint y = 2;\nint z = 3;"

    def test_extract_no_match(self):
        """Returns None when no lines map to the range."""
        pf = PreprocessedFile(
            source_file="/tmp/test.c",
            mappings=[
                _LineMapping("int x = 1;", "/tmp/test.c", 5),
            ],
        )
        result = pf.extract_pp_source("/tmp/test.c", 100, 200)
        assert result is None

    def test_extract_filters_by_file(self):
        """Only lines from the requested file are included."""
        pf = PreprocessedFile(
            source_file="/tmp/test.c",
            mappings=[
                _LineMapping("from_header", "/usr/include/stdlib.h", 5),
                _LineMapping("from_source", "/tmp/test.c", 5),
            ],
        )
        result = pf.extract_pp_source("/tmp/test.c", 5, 5)
        assert result == "from_source"

    def test_extract_empty_mappings(self):
        """Returns None with empty mappings."""
        pf = PreprocessedFile(source_file="/tmp/test.c")
        result = pf.extract_pp_source("/tmp/test.c", 1, 10)
        assert result is None


class TestSourcePreprocessorParseOutput:
    """Tests for SourcePreprocessor._parse_output()."""

    def test_parse_simple_output(self):
        """Parses line markers and tracks line numbers."""
        output = textwrap.dedent("""\
            # 1 "/tmp/test.c"
            # 1 "<built-in>" 1
            # 1 "<built-in>" 3
            # 1 "/tmp/test.c" 2
            # 5 "/tmp/test.c"
            int create_buffer ( size_t n ) {
                char * buf = malloc ( n + 1 ) ;
                if ( ! buf ) return ((void *)0) ;
                buf [ n ] = '\\0' ;
                return buf ;
            }
        """)
        mappings = SourcePreprocessor._parse_output(output)
        # Should have 6 non-empty lines from test.c (lines 5-10)
        assert len(mappings) == 6
        assert mappings[0].orig_file == "/tmp/test.c"
        assert mappings[0].orig_line == 5
        assert "create_buffer" in mappings[0].pp_line

    def test_parse_skips_blank_lines(self):
        """Blank lines are skipped but line numbers still increment."""
        output = textwrap.dedent("""\
            # 10 "/tmp/test.c"
            int a;

            int b;
        """)
        mappings = SourcePreprocessor._parse_output(output)
        assert len(mappings) == 2
        assert mappings[0].orig_line == 10
        # Blank line increments counter, so b is at line 12
        assert mappings[1].orig_line == 12


class TestSourcePreprocessorIntegration:
    """Integration tests that run clang -E on actual files."""

    @pytest.fixture
    def simple_c(self):
        """Path to the simple.c test fixture."""
        return Path(__file__).parent / "fixtures" / "simple.c"

    def test_preprocess_simple_c(self, simple_c):
        """clang -E successfully preprocesses simple.c."""
        pp = SourcePreprocessor()
        result = pp.preprocess(simple_c)

        assert result.error is None, f"Preprocessing failed: {result.error}"
        assert len(result.mappings) > 0

        # Extract pp_source for create_buffer (lines 7-12)
        pp_src = result.extract_pp_source(str(simple_c.resolve()), 7, 12)
        assert pp_src is not None
        assert "create_buffer" in pp_src or "buf" in pp_src

    def test_preprocess_nonexistent_file(self):
        """Gracefully handles nonexistent files."""
        pp = SourcePreprocessor()
        result = pp.preprocess("/nonexistent/file.c")
        assert result.error is not None
        assert result.mappings == []

    def test_preprocess_with_macro_expansion(self, tmp_path):
        """Macros are expanded in the preprocessed output."""
        src = tmp_path / "macro_test.c"
        src.write_text(textwrap.dedent("""\
            #define ADD(a, b) ((a) + (b))
            #define CONST_VAL 42

            int test_func(int x) {
                return ADD(x, CONST_VAL);
            }
        """))

        pp = SourcePreprocessor()
        result = pp.preprocess(src)

        assert result.error is None
        # Extract the function body (lines 4-6)
        pp_src = result.extract_pp_source(str(src.resolve()), 4, 6)
        assert pp_src is not None
        # The macro should be expanded — we should NOT see ADD or CONST_VAL
        assert "ADD" not in pp_src
        assert "CONST_VAL" not in pp_src
        # We should see the expansion
        assert "42" in pp_src

    def test_preprocess_clang_not_found(self):
        """Handles missing clang binary gracefully."""
        pp = SourcePreprocessor(clang_binary="/nonexistent/clang")
        result = pp.preprocess("/tmp/test.c")
        assert result.error is not None
        assert "not found" in result.error
