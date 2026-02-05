"""Tests for build tools module."""

import os
import pytest
from pathlib import Path

from llm_summary.builder.tools import BuildTools


class TestValidatePath:
    """Tests for _validate_path method."""

    def test_rejects_absolute_paths(self, tmp_path):
        """Should reject absolute paths."""
        tools = BuildTools(tmp_path)
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            tools._validate_path("/etc/passwd")

    def test_accepts_relative_paths(self, tmp_path):
        """Should accept valid relative paths."""
        (tmp_path / "file.txt").write_text("test")
        tools = BuildTools(tmp_path)
        result = tools._validate_path("file.txt")
        assert result == tmp_path / "file.txt"

    def test_rejects_parent_directory_escape(self, tmp_path):
        """Should reject paths that escape via .."""
        tools = BuildTools(tmp_path)
        with pytest.raises(ValueError, match="Path escapes project directory"):
            tools._validate_path("../etc/passwd")

    def test_handles_build_dir_prefix(self, tmp_path):
        """Should route build/ prefix to separate build directory."""
        project = tmp_path / "project"
        build = tmp_path / "build"
        project.mkdir()
        build.mkdir()
        (build / "output.txt").write_text("build output")

        tools = BuildTools(project, build)
        result = tools._validate_path("build/output.txt")
        assert result == build / "output.txt"

    def test_rejects_build_dir_escape(self, tmp_path):
        """Should reject paths that escape build directory."""
        project = tmp_path / "project"
        build = tmp_path / "build"
        project.mkdir()
        build.mkdir()

        tools = BuildTools(project, build)
        with pytest.raises(ValueError, match="Path escapes build directory"):
            tools._validate_path("build/../project/file.txt")


class TestSymlinkEscape:
    """Tests for symlink escape detection."""

    @pytest.fixture
    def project_with_symlink(self, tmp_path):
        """Create a project with a symlink pointing outside."""
        project = tmp_path / "project"
        project.mkdir()

        # Create a file inside the project
        (project / "safe.txt").write_text("safe content")

        # Create a symlink pointing outside
        external = tmp_path / "external"
        external.mkdir()
        (external / "secret.txt").write_text("secret content")

        symlink = project / "escape"
        symlink.symlink_to(external)

        return project, external

    def test_detects_symlink_escape(self, project_with_symlink):
        """Should detect symlink that points outside project."""
        project, external = project_with_symlink
        tools = BuildTools(project)

        with pytest.raises(ValueError, match="Symlink escape detected"):
            tools._validate_path("escape/secret.txt")

    def test_allows_internal_symlinks(self, tmp_path):
        """Should allow symlinks that stay within project."""
        project = tmp_path / "project"
        project.mkdir()

        # Create directories
        (project / "src").mkdir()
        (project / "src" / "file.txt").write_text("content")

        # Create symlink pointing to another location inside project
        (project / "link").symlink_to(project / "src")

        tools = BuildTools(project)
        result = tools._validate_path("link/file.txt")
        assert result == project / "src" / "file.txt"

    def test_allows_safe_paths_without_symlinks(self, tmp_path):
        """Should allow normal paths without symlinks."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "subdir").mkdir()
        (project / "subdir" / "file.txt").write_text("content")

        tools = BuildTools(project)
        result = tools._validate_path("subdir/file.txt")
        assert result == project / "subdir" / "file.txt"

    def test_handles_nonexistent_paths(self, tmp_path):
        """Should handle paths that don't exist (validation passes, file check fails later)."""
        project = tmp_path / "project"
        project.mkdir()

        tools = BuildTools(project)
        # Should not raise - the path doesn't exist but isn't escaping
        result = tools._validate_path("nonexistent/path/file.txt")
        assert result == project / "nonexistent" / "path" / "file.txt"

    def test_detects_symlink_in_middle_of_path(self, tmp_path):
        """Should detect symlink escape in the middle of a path."""
        project = tmp_path / "project"
        project.mkdir()
        (project / "subdir").mkdir()

        # Create external directory
        external = tmp_path / "external"
        external.mkdir()
        (external / "secrets").mkdir()
        (external / "secrets" / "file.txt").write_text("secret")

        # Create symlink inside project pointing outside
        (project / "subdir" / "escape").symlink_to(external)

        tools = BuildTools(project)
        with pytest.raises(ValueError, match="Symlink escape detected"):
            tools._validate_path("subdir/escape/secrets/file.txt")


class TestReadFile:
    """Tests for read_file method."""

    def test_reads_file_content(self, tmp_path):
        """Should read file content correctly."""
        (tmp_path / "test.txt").write_text("line1\nline2\nline3")
        tools = BuildTools(tmp_path)
        result = tools.read_file("test.txt")

        assert "content" in result
        assert "line1" in result["content"]
        assert "line2" in result["content"]
        assert result["lines_read"] == 3

    def test_returns_error_for_missing_file(self, tmp_path):
        """Should return error for missing files."""
        tools = BuildTools(tmp_path)
        result = tools.read_file("nonexistent.txt")

        assert "error" in result
        assert "File not found" in result["error"]

    def test_returns_error_for_directory(self, tmp_path):
        """Should return error when path is a directory."""
        (tmp_path / "subdir").mkdir()
        tools = BuildTools(tmp_path)
        result = tools.read_file("subdir")

        assert "error" in result
        assert "Not a file" in result["error"]

    def test_respects_max_lines(self, tmp_path):
        """Should respect max_lines parameter."""
        content = "\n".join(f"line{i}" for i in range(100))
        (tmp_path / "large.txt").write_text(content)
        tools = BuildTools(tmp_path)
        result = tools.read_file("large.txt", max_lines=10)

        assert result["lines_read"] == 10
        assert result["truncated"] is True

    def test_respects_start_line(self, tmp_path):
        """Should start reading from specified line."""
        content = "\n".join(f"line{i}" for i in range(1, 11))
        (tmp_path / "test.txt").write_text(content)
        tools = BuildTools(tmp_path)
        result = tools.read_file("test.txt", start_line=5)

        assert result["start_line"] == 5
        assert "line5" in result["content"]
        assert "line4" not in result["content"]


class TestListDir:
    """Tests for list_dir method."""

    def test_lists_directory_contents(self, tmp_path):
        """Should list files and directories."""
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "subdir").mkdir()
        tools = BuildTools(tmp_path)
        result = tools.list_dir(".")

        assert len(result["files"]) == 1
        assert len(result["directories"]) == 1
        assert result["files"][0]["name"] == "file.txt"
        assert result["directories"][0]["name"] == "subdir"

    def test_returns_error_for_missing_dir(self, tmp_path):
        """Should return error for missing directories."""
        tools = BuildTools(tmp_path)
        result = tools.list_dir("nonexistent")

        assert "error" in result
        assert "Directory not found" in result["error"]

    def test_supports_glob_pattern(self, tmp_path):
        """Should filter by glob pattern."""
        (tmp_path / "file.txt").write_text("content")
        (tmp_path / "file.py").write_text("content")
        (tmp_path / "file.js").write_text("content")
        tools = BuildTools(tmp_path)
        result = tools.list_dir(".", pattern="*.py")

        assert len(result["files"]) == 1
        assert result["files"][0]["name"] == "file.py"
