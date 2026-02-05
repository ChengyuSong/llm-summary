"""Tests for assembly_utils module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_summary.builder.assembly_utils import check_assembly


class TestCheckAssembly:
    """Tests for check_assembly function."""

    def test_returns_none_when_compile_commands_missing(self, tmp_path):
        """Should return None when compile_commands.json doesn't exist."""
        result = check_assembly(
            compile_commands_path=tmp_path / "compile_commands.json",
            build_dir=tmp_path,
            project_path=tmp_path,
        )
        assert result is None

    def test_prints_message_when_verbose_and_missing(self, tmp_path, capsys):
        """Should print message when verbose and file is missing."""
        check_assembly(
            compile_commands_path=tmp_path / "compile_commands.json",
            build_dir=tmp_path,
            project_path=tmp_path,
            verbose=True,
            log_prefix="[test]",
        )
        captured = capsys.readouterr()
        assert "[test] No compile_commands.json" in captured.out

    def test_uses_custom_log_prefix(self, tmp_path, capsys):
        """Should use custom log prefix in messages."""
        check_assembly(
            compile_commands_path=tmp_path / "compile_commands.json",
            build_dir=tmp_path,
            project_path=tmp_path,
            verbose=True,
            log_prefix="[custom_prefix]",
        )
        captured = capsys.readouterr()
        assert "[custom_prefix]" in captured.out

    @patch("llm_summary.builder.assembly_checker.AssemblyChecker")
    def test_calls_checker_with_correct_args(self, mock_checker_class, tmp_path):
        """Should create AssemblyChecker with correct arguments."""
        # Create compile_commands.json so the check proceeds
        compile_commands = tmp_path / "compile_commands.json"
        compile_commands.write_text("[]")

        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker
        mock_result = MagicMock()
        mock_checker.check.return_value = mock_result

        unavoidable_path = tmp_path / "unavoidable.json"
        result = check_assembly(
            compile_commands_path=compile_commands,
            build_dir=tmp_path / "build",
            project_path=tmp_path / "src",
            unavoidable_asm_path=unavoidable_path,
            verbose=True,
        )

        mock_checker_class.assert_called_once_with(
            compile_commands_path=compile_commands,
            build_dir=tmp_path / "build",
            project_path=tmp_path / "src",
            unavoidable_asm_path=unavoidable_path,
            verbose=True,
        )
        mock_checker.check.assert_called_once_with(scan_ir=True)
        assert result == mock_result

    @patch("llm_summary.builder.assembly_checker.AssemblyChecker")
    def test_returns_none_on_exception(self, mock_checker_class, tmp_path):
        """Should return None and print message on exception."""
        compile_commands = tmp_path / "compile_commands.json"
        compile_commands.write_text("[]")

        mock_checker_class.side_effect = RuntimeError("test error")

        result = check_assembly(
            compile_commands_path=compile_commands,
            build_dir=tmp_path,
            project_path=tmp_path,
            verbose=False,
        )
        assert result is None

    @patch("llm_summary.builder.assembly_checker.AssemblyChecker")
    def test_prints_error_when_verbose_and_exception(self, mock_checker_class, tmp_path, capsys):
        """Should print error message when verbose and exception occurs."""
        compile_commands = tmp_path / "compile_commands.json"
        compile_commands.write_text("[]")

        mock_checker_class.side_effect = RuntimeError("test error")

        check_assembly(
            compile_commands_path=compile_commands,
            build_dir=tmp_path,
            project_path=tmp_path,
            verbose=True,
            log_prefix="[test]",
        )
        captured = capsys.readouterr()
        assert "[test] Assembly check failed" in captured.out
        assert "test error" in captured.out
