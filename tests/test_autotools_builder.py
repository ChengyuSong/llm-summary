"""Tests for the AutotoolsBuilder class."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_summary.builder.autotools_builder import AutotoolsBuilder, AUTOTOOLS_TOOL_DEFINITIONS
from llm_summary.builder.llm_utils import (
    deduplicate_tool_result,
    filter_warnings,
    truncate_messages,
)


class TestAutotoolsBuilderToolDefinitions:
    """Tests for autotools tool definitions in the builder."""

    def test_file_tools_included(self):
        """Test that file tools are included in tool definitions."""
        tool_names = [t["name"] for t in AUTOTOOLS_TOOL_DEFINITIONS]
        assert "read_file" in tool_names
        assert "list_dir" in tool_names

    def test_autotools_tools_included(self):
        """Test that autotools action tools are included."""
        tool_names = [t["name"] for t in AUTOTOOLS_TOOL_DEFINITIONS]
        assert "autoreconf" in tool_names
        assert "autotools_configure" in tool_names
        assert "autotools_build" in tool_names
        assert "autotools_clean" in tool_names
        assert "autotools_distclean" in tool_names

    def test_cmake_tools_not_included(self):
        """Test that CMake tools are NOT included."""
        tool_names = [t["name"] for t in AUTOTOOLS_TOOL_DEFINITIONS]
        assert "cmake_configure" not in tool_names
        assert "cmake_build" not in tool_names


class TestAutotoolsBuilder:
    """Tests for AutotoolsBuilder class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM backend."""
        llm = MagicMock()
        llm.model = "test-model"
        return llm

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary autotools project structure."""
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()

        # Create configure.ac
        configure_ac = project_dir / "configure.ac"
        configure_ac.write_text("""
AC_INIT([myproject], [1.0])
AM_INIT_AUTOMAKE([foreign])
AC_PROG_CC
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
""")

        # Create configure script
        configure = project_dir / "configure"
        configure.write_text("#!/bin/bash\necho 'Configure script'\n")
        configure.chmod(0o755)

        # Create Makefile.am
        makefile_am = project_dir / "Makefile.am"
        makefile_am.write_text("bin_PROGRAMS = hello\nhello_SOURCES = hello.c\n")

        return project_dir

    def test_builder_initialization(self, mock_llm):
        """Test builder initializes with correct defaults."""
        builder = AutotoolsBuilder(
            llm=mock_llm,
            verbose=True,
        )

        assert builder.llm is mock_llm
        assert builder.enable_lto is True
        assert builder.prefer_static is True
        assert builder.max_retries == 3

    def test_get_default_config(self, mock_llm):
        """Test default configuration includes static linking."""
        builder = AutotoolsBuilder(
            llm=mock_llm,
            prefer_static=True,
        )

        flags = builder._get_default_config()
        assert "--disable-shared" in flags
        assert "--enable-static" in flags

    def test_get_default_config_no_static(self, mock_llm):
        """Test default configuration without static preference."""
        builder = AutotoolsBuilder(
            llm=mock_llm,
            prefer_static=False,
        )

        flags = builder._get_default_config()
        assert "--disable-shared" not in flags
        assert "--enable-static" not in flags

    def test_learn_and_build_no_configure(self, mock_llm, tmp_path):
        """Test that missing configure.ac raises error."""
        empty_project = tmp_path / "empty"
        empty_project.mkdir()

        builder = AutotoolsBuilder(llm=mock_llm)

        with pytest.raises(ValueError, match="Neither configure nor configure.ac"):
            builder.learn_and_build(empty_project)

    def test_execute_tool_safe_read_file(self, mock_llm, temp_project):
        """Test _execute_tool_safe routes read_file correctly."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "read_file",
            {"file_path": "configure.ac"}
        )

        assert "content" in result
        assert "AC_INIT" in result["content"]

    def test_execute_tool_safe_list_dir(self, mock_llm, temp_project):
        """Test _execute_tool_safe routes list_dir correctly."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "list_dir",
            {"dir_path": "."}
        )

        assert "files" in result
        file_names = [f["name"] for f in result["files"]]
        assert "configure.ac" in file_names
        assert "configure" in file_names

    @patch("subprocess.run")
    def test_execute_tool_safe_autoreconf(self, mock_run, mock_llm, temp_project):
        """Test _execute_tool_safe routes autoreconf correctly."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "autoreconf", {}
        )

        assert result["success"] is True

    @patch("subprocess.run")
    def test_execute_tool_safe_configure(self, mock_run, mock_llm, temp_project):
        """Test _execute_tool_safe routes autotools_configure correctly."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "autotools_configure",
            {"configure_flags": ["--disable-shared"], "use_build_dir": True}
        )

        assert result["success"] is True

    def test_execute_tool_safe_unknown_tool(self, mock_llm, temp_project):
        """Test _execute_tool_safe returns error for unknown tool."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "unknown_tool", {}
        )

        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_execute_tool_safe_finish(self, mock_llm, temp_project):
        """Test _execute_tool_safe handles finish tool correctly."""
        from llm_summary.builder.tools import BuildTools
        from llm_summary.builder.actions import AutotoolsActions

        builder = AutotoolsBuilder(llm=mock_llm)
        file_tools = BuildTools(temp_project, temp_project / "build")
        actions = AutotoolsActions(temp_project, temp_project / "build")

        result = builder._execute_tool_safe(
            file_tools, actions, "finish",
            {"status": "success", "summary": "Build completed successfully"}
        )

        assert result["acknowledged"] is True
        assert result["status"] == "success"
        assert result["summary"] == "Build completed successfully"

    def test_truncate_messages(self, mock_llm):
        """Test message truncation for large contexts."""
        # Create messages that exceed token limit
        messages = [
            {"role": "user", "content": "Initial request"},
            {"role": "assistant", "content": "x" * 50000},  # Large message
            {"role": "user", "content": "Tool result"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Final"},
        ]

        truncated = truncate_messages(messages, max_tokens=10000)

        # Should keep first and some recent messages
        assert len(truncated) < len(messages)
        assert truncated[0]["content"] == "Initial request"

    def test_deduplicate_tool_result_tracks_reads(self, mock_llm):
        """Test that file reads are tracked (compression happens separately)."""
        history = {}
        result1 = {"content": "file contents", "path": "test.c", "start_line": 1, "end_line": 50}
        result2 = {"content": "file contents", "path": "test.c", "start_line": 1, "end_line": 50}

        # First read - tracked and returned unchanged
        deduped1 = deduplicate_tool_result(
            "read_file", {"file_path": "test.c"}, result1, history, current_turn=0
        )
        assert deduped1["content"] == "file contents"

        # Second read - also returned unchanged (compression is separate)
        deduped2 = deduplicate_tool_result(
            "read_file", {"file_path": "test.c"}, result2, history, current_turn=1
        )
        assert deduped2["content"] == "file contents"

        # Both reads should be tracked
        assert len(history["reads"]["test.c"]) == 2

    def test_filter_warnings_small_output(self, mock_llm):
        """Test that small outputs are not filtered."""
        output = "Building...\nCompiling foo.c\nDone."
        filtered = filter_warnings(output)

        assert filtered == output

    def test_filter_warnings_large_output(self, mock_llm):
        """Test that large outputs are filtered to show errors."""
        # Create large output with warnings and errors (>10000 chars)
        lines = ["warning: unused variable 'x' in function 'some_long_function_name'"] * 200
        lines.append("error: undefined reference to 'foo'")
        lines.extend(["warning: something else happened here with more text"] * 200)
        output = "\n".join(lines)

        # Ensure output is large enough to trigger filtering
        assert len(output) > 10000

        filtered = filter_warnings(output)

        # Should contain the error
        assert "undefined reference" in filtered
        # Should be smaller than original
        assert len(filtered) < len(output)


class TestExtractCompileCommands:
    """Tests for compile_commands.json extraction."""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.model = "test-model"
        return llm

    def test_extract_compile_commands_out_of_source(self, mock_llm, tmp_path):
        """Test extracting compile_commands from out-of-source build."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create compile_commands.json with Docker paths
        compile_commands = [
            {
                "directory": "/workspace/build",
                "file": "/workspace/src/main.c",
                "command": "clang -c /workspace/src/main.c -o main.o",
            }
        ]
        (build_dir / "compile_commands.json").write_text(json.dumps(compile_commands))

        builder = AutotoolsBuilder(
            llm=mock_llm,
            build_dir=build_dir,
            verbose=True,
        )

        result_path = builder.extract_compile_commands(
            project_dir, output_dir=output_dir, use_build_dir=True
        )

        # Verify file was created
        assert result_path.exists()

        # Verify paths were translated
        with open(result_path) as f:
            fixed = json.load(f)

        assert str(project_dir) in fixed[0]["file"]
        assert str(build_dir) in fixed[0]["directory"]
        assert "/workspace/src" not in fixed[0]["file"]

    def test_extract_compile_commands_in_source(self, mock_llm, tmp_path):
        """Test extracting compile_commands from in-source build."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create compile_commands.json in project dir (in-source build)
        compile_commands = [
            {
                "directory": "/workspace/src",
                "file": "/workspace/src/main.c",
                "command": "clang -c main.c",
            }
        ]
        (project_dir / "compile_commands.json").write_text(json.dumps(compile_commands))

        builder = AutotoolsBuilder(
            llm=mock_llm,
            build_dir=project_dir,  # Same as project dir for in-source
            verbose=True,
        )

        result_path = builder.extract_compile_commands(
            project_dir, output_dir=output_dir, use_build_dir=False
        )

        assert result_path.exists()

        with open(result_path) as f:
            fixed = json.load(f)

        assert str(project_dir) in fixed[0]["directory"]

    def test_extract_compile_commands_not_found(self, mock_llm, tmp_path):
        """Test error when compile_commands.json doesn't exist."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        builder = AutotoolsBuilder(
            llm=mock_llm,
            build_dir=build_dir,
        )

        with pytest.raises(FileNotFoundError):
            builder.extract_compile_commands(project_dir, use_build_dir=True)
