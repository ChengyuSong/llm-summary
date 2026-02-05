"""Tests for the AutotoolsActions class."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_summary.builder.autotools_actions import AutotoolsActions, TOOL_DEFINITIONS


class TestAutotoolsActionsMocked:
    """Tests for AutotoolsActions using mocked subprocess."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure for testing."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        # Create a minimal configure.ac
        configure_ac = src_dir / "configure.ac"
        configure_ac.write_text("AC_INIT([test], [1.0])\nAC_OUTPUT\n")

        # Create a configure script
        configure = src_dir / "configure"
        configure.write_text("#!/bin/bash\necho 'Configuring...'\n")
        configure.chmod(0o755)

        return src_dir, build_dir

    def test_get_default_env_flags(self, temp_project):
        """Test that default env flags include clang-18 and LTO."""
        src_dir, build_dir = temp_project
        actions = AutotoolsActions(src_dir, build_dir)

        env = actions._get_default_env_flags()

        assert env["CC"] == "clang-18"
        assert env["CXX"] == "clang++-18"
        assert "-flto=full" in env["CFLAGS"]
        assert "-save-temps=obj" in env["CFLAGS"]
        assert "-flto=full" in env["LDFLAGS"]
        assert "-fuse-ld=lld" in env["LDFLAGS"]
        assert env["LD"] == "ld.lld-18"
        assert env["AR"] == "llvm-ar-18"

    @patch("subprocess.run")
    def test_autoreconf_success(self, mock_run, temp_project):
        """Test autoreconf command construction and success."""
        src_dir, build_dir = temp_project

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="autoreconf: configuring...",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autoreconf()

        assert result["success"] is True
        assert "error" in result and result["error"] == ""

        # Verify Docker command
        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        assert "docker" in docker_cmd
        assert "run" in docker_cmd
        assert "/workspace/src" in str(docker_cmd)
        assert "autoreconf -fi" in docker_cmd[-1]

    @patch("subprocess.run")
    def test_autoreconf_failure(self, mock_run, temp_project):
        """Test autoreconf failure handling."""
        src_dir, build_dir = temp_project

        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="autoreconf: error: configure.ac missing",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autoreconf()

        assert result["success"] is False
        assert "exit code 1" in result["error"]

    @patch("subprocess.run")
    def test_configure_out_of_source(self, mock_run, temp_project):
        """Test configure with out-of-source build."""
        src_dir, build_dir = temp_project

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="checking for gcc... clang-18",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_configure(
            configure_flags=["--disable-shared", "--enable-static"],
            use_build_dir=True,
        )

        assert result["success"] is True
        assert result["use_build_dir"] is True

        # Verify Docker command
        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]

        # Check environment variables are set
        assert 'CC="clang-18"' in cmd_str
        assert 'CXX="clang++-18"' in cmd_str
        assert "-flto=full" in cmd_str

        # Check configure path is correct for out-of-source
        assert "/workspace/src/configure" in cmd_str
        assert "--disable-shared" in cmd_str
        assert "--enable-static" in cmd_str

    @patch("subprocess.run")
    def test_configure_in_source(self, mock_run, temp_project):
        """Test configure with in-source build."""
        src_dir, build_dir = temp_project

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="checking for gcc... clang-18",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_configure(
            configure_flags=["--prefix=/usr"],
            use_build_dir=False,
        )

        assert result["success"] is True

        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]

        # Check configure path is correct for in-source
        assert "./configure" in cmd_str
        assert "--prefix=/usr" in cmd_str

    @patch("subprocess.run")
    def test_build_with_bear(self, mock_run, temp_project):
        """Test build command uses bear to capture compile commands."""
        src_dir, build_dir = temp_project

        # Create Makefile to simulate successful configure
        (build_dir / "Makefile").write_text("all:\n\techo building\n")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Building project...",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_build(use_build_dir=True)

        assert result["success"] is True

        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]

        # Verify bear is used
        assert "bear -- make" in cmd_str
        assert "-j$(nproc)" in cmd_str

    @patch("subprocess.run")
    def test_build_with_target(self, mock_run, temp_project):
        """Test build with specific make target."""
        src_dir, build_dir = temp_project

        (build_dir / "Makefile").write_text("all lib:\n\techo target\n")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Building lib...",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_build(make_target="lib", use_build_dir=True)

        assert result["success"] is True

        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]

        assert "bear -- make" in cmd_str
        assert "lib" in cmd_str

    @patch("subprocess.run")
    def test_build_failure_retries_j1(self, mock_run, temp_project):
        """Test that build failure triggers retry with -j1."""
        src_dir, build_dir = temp_project

        (build_dir / "Makefile").write_text("all:\n\tfalse\n")

        # First call fails (parallel), second call also fails (j1)
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout="", stderr="parallel build error"),
            MagicMock(returncode=1, stdout="", stderr="clear error: undefined reference"),
        ]

        actions = AutotoolsActions(src_dir, build_dir, verbose=True)
        result = actions.autotools_build(use_build_dir=True)

        assert result["success"] is False
        assert "undefined reference" in result["output"]

        # Verify j1 was used in retry
        assert mock_run.call_count == 2
        second_call = mock_run.call_args_list[1]
        assert "-j1" in second_call[0][0][-1]

    def test_build_no_makefile_error(self, temp_project):
        """Test build fails gracefully when Makefile doesn't exist."""
        src_dir, build_dir = temp_project

        # Don't create Makefile
        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_build(use_build_dir=True)

        assert result["success"] is False
        assert "Makefile not found" in result["error"]

    @patch("subprocess.run")
    def test_clean_success(self, mock_run, temp_project):
        """Test make clean command."""
        src_dir, build_dir = temp_project

        # Create Makefile
        (build_dir / "Makefile").write_text("clean:\n\trm -f *.o\n")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Cleaning...",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_clean(use_build_dir=True)

        assert result["success"] is True

        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]
        assert "make clean" in cmd_str

    @patch("subprocess.run")
    def test_distclean_success(self, mock_run, temp_project):
        """Test make distclean command."""
        src_dir, build_dir = temp_project

        # Create Makefile
        (build_dir / "Makefile").write_text("distclean:\n\trm -rf *\n")

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Distcleaning...",
            stderr="",
        )

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_distclean(use_build_dir=True)

        assert result["success"] is True

        call_args = mock_run.call_args
        docker_cmd = call_args[0][0]
        cmd_str = docker_cmd[-1]
        assert "make distclean" in cmd_str

    def test_clean_no_makefile_error(self, temp_project):
        """Test clean fails gracefully when Makefile doesn't exist."""
        src_dir, build_dir = temp_project

        actions = AutotoolsActions(src_dir, build_dir)
        result = actions.autotools_clean(use_build_dir=True)

        assert result["success"] is False
        assert "Makefile not found" in result["error"]


class TestToolDefinitions:
    """Tests for autotools tool definitions."""

    def test_autoreconf_tool_exists(self):
        """Test autoreconf tool is defined."""
        tool = next((t for t in TOOL_DEFINITIONS if t["name"] == "autoreconf"), None)
        assert tool is not None
        assert "autoreconf" in tool["description"]
        assert "configure.ac" in tool["description"]

    def test_configure_tool_exists(self):
        """Test autotools_configure tool is defined."""
        tool = next((t for t in TOOL_DEFINITIONS if t["name"] == "autotools_configure"), None)
        assert tool is not None
        assert "configure_flags" in tool["input_schema"]["properties"]
        assert "use_build_dir" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["properties"]["use_build_dir"]["default"] is True

    def test_build_tool_exists(self):
        """Test autotools_build tool is defined."""
        tool = next((t for t in TOOL_DEFINITIONS if t["name"] == "autotools_build"), None)
        assert tool is not None
        assert "bear" in tool["description"]
        assert "compile_commands" in tool["description"]
        assert "make_target" in tool["input_schema"]["properties"]

    def test_tool_count(self):
        """Test correct number of tools are defined."""
        assert len(TOOL_DEFINITIONS) == 5  # autoreconf, configure, build, clean, distclean

    def test_clean_tool_exists(self):
        """Test autotools_clean tool is defined."""
        tool = next((t for t in TOOL_DEFINITIONS if t["name"] == "autotools_clean"), None)
        assert tool is not None
        assert "make clean" in tool["description"]

    def test_distclean_tool_exists(self):
        """Test autotools_distclean tool is defined."""
        tool = next((t for t in TOOL_DEFINITIONS if t["name"] == "autotools_distclean"), None)
        assert tool is not None
        assert "make distclean" in tool["description"]
