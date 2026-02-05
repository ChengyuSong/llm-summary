"""Tests for the AssemblyChecker class."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_summary.builder.assembly_checker import AssemblyChecker
from llm_summary.models import AssemblyCheckResult, AssemblyFinding, AssemblyType


class TestAssemblyCheckResult:
    """Tests for AssemblyCheckResult data model."""

    def test_no_assembly(self):
        """Test result with no assembly detected."""
        result = AssemblyCheckResult(has_assembly=False)
        assert not result.has_assembly
        assert result.standalone_asm_files == []
        assert result.inline_asm_sources == []
        assert result.inline_asm_ir == []
        assert "No assembly" in result.summary()

    def test_with_standalone(self):
        """Test result with standalone assembly files."""
        result = AssemblyCheckResult(
            has_assembly=True,
            standalone_asm_files=[
                AssemblyFinding(
                    asm_type=AssemblyType.STANDALONE_FILE,
                    file_path="/path/to/crc.S",
                )
            ],
        )
        assert result.has_assembly
        assert len(result.standalone_asm_files) == 1
        assert "crc.S" in result.summary()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AssemblyCheckResult(
            has_assembly=True,
            standalone_asm_files=[
                AssemblyFinding(
                    asm_type=AssemblyType.STANDALONE_FILE,
                    file_path="/path/to/startup.S",
                )
            ],
            inline_asm_sources=[
                AssemblyFinding(
                    asm_type=AssemblyType.INLINE_SOURCE,
                    file_path="/path/to/simd.c",
                    line_number=42,
                    snippet="__asm__ volatile(...)",
                    pattern_matched="__asm__ __volatile__(...)",
                )
            ],
        )

        d = result.to_dict()
        assert d["has_assembly"] is True
        assert len(d["standalone_asm_files"]) == 1
        assert d["standalone_asm_files"][0]["file_path"] == "/path/to/startup.S"
        assert len(d["inline_asm_sources"]) == 1
        assert d["inline_asm_sources"][0]["line_number"] == 42


class TestAssemblyFinding:
    """Tests for AssemblyFinding data model."""

    def test_standalone_finding(self):
        """Test standalone assembly finding."""
        finding = AssemblyFinding(
            asm_type=AssemblyType.STANDALONE_FILE,
            file_path="/path/to/asm.S",
        )
        assert finding.asm_type == AssemblyType.STANDALONE_FILE
        assert finding.line_number is None

        d = finding.to_dict()
        assert d["type"] == "standalone_file"
        assert "line_number" not in d

    def test_inline_finding(self):
        """Test inline assembly finding with all fields."""
        finding = AssemblyFinding(
            asm_type=AssemblyType.INLINE_SOURCE,
            file_path="/path/to/code.c",
            line_number=100,
            snippet="asm volatile()",
            pattern_matched="asm volatile(...)",
        )

        d = finding.to_dict()
        assert d["type"] == "inline_source"
        assert d["line_number"] == 100
        assert d["snippet"] == "asm volatile()"
        assert d["pattern_matched"] == "asm volatile(...)"


class TestAssemblyChecker:
    """Tests for AssemblyChecker class."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure for testing."""
        # Create project directory
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        return tmp_path, src_dir, build_dir

    def _create_compile_commands(self, build_dir, files):
        """Helper to create compile_commands.json."""
        commands = []
        for f in files:
            commands.append({
                "directory": str(build_dir),
                "file": f,
                "command": f"clang -c {f}",
            })

        compile_commands_path = build_dir / "compile_commands.json"
        with open(compile_commands_path, "w") as fp:
            json.dump(commands, fp)

        return compile_commands_path

    def test_detect_standalone_asm(self, temp_project):
        """Test detection of standalone assembly files."""
        project_path, src_dir, build_dir = temp_project

        # Create assembly file
        asm_file = src_dir / "crc_folding.S"
        asm_file.write_text("; Assembly code\n.global crc32\n")

        # Create compile_commands.json that includes the .S file
        compile_commands = self._create_compile_commands(
            build_dir, [str(asm_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert result.has_assembly
        assert len(result.standalone_asm_files) == 1
        assert result.standalone_asm_files[0].file_path == str(asm_file)
        assert result.standalone_asm_files[0].asm_type == AssemblyType.STANDALONE_FILE

    def test_detect_inline_asm_gcc_style(self, temp_project):
        """Test detection of GCC-style inline assembly."""
        project_path, src_dir, build_dir = temp_project

        # Create C file with inline asm
        c_file = src_dir / "simd.c"
        c_file.write_text("""
#include <stdint.h>

uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
""")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert result.has_assembly
        assert len(result.inline_asm_sources) == 1
        assert result.inline_asm_sources[0].file_path == str(c_file)
        assert result.inline_asm_sources[0].line_number == 6  # Line 6 in the file (after blank line at top)
        assert "__asm__" in result.inline_asm_sources[0].pattern_matched

    def test_detect_inline_asm_simple(self, temp_project):
        """Test detection of simple asm() inline assembly."""
        project_path, src_dir, build_dir = temp_project

        c_file = src_dir / "simple.c"
        c_file.write_text("""
void pause(void) {
    asm("pause");
}
""")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert result.has_assembly
        assert len(result.inline_asm_sources) == 1
        assert 'asm("...")' in result.inline_asm_sources[0].pattern_matched

    def test_detect_llvm_ir_asm(self, temp_project):
        """Test detection of inline assembly in LLVM bitcode files."""
        import subprocess

        project_path, src_dir, build_dir = temp_project

        # Create empty compile_commands (no source files with asm)
        c_file = src_dir / "clean.c"
        c_file.write_text("int main() { return 0; }\n")
        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        # Create LLVM IR file with inline asm, then convert to .bc
        ll_file = build_dir / "simd.ll"
        ll_file.write_text("""
define i64 @rdtsc() {
entry:
  %0 = call i64 asm sideeffect "rdtsc", "=A,~{dirflag},~{fpsr},~{flags}"()
  ret i64 %0
}
""")
        bc_file = build_dir / "simd.bc"

        # Convert .ll to .bc using llvm-as
        try:
            result = subprocess.run(
                ["llvm-as", str(ll_file), "-o", str(bc_file)],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                pytest.skip("llvm-as failed to create .bc file")
        except FileNotFoundError:
            pytest.skip("llvm-as not available")

        # Remove the .ll file to ensure we're testing .bc scanning
        ll_file.unlink()

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=True)

        assert result.has_assembly
        assert len(result.inline_asm_ir) == 1
        assert result.inline_asm_ir[0].asm_type == AssemblyType.INLINE_LLVM_IR
        # Pattern matching may match "call asm" first (before "asm sideeffect")
        assert "asm" in result.inline_asm_ir[0].pattern_matched

    def test_clean_build_no_assembly(self, temp_project):
        """Test that clean build with no assembly is correctly identified."""
        project_path, src_dir, build_dir = temp_project

        # Create clean C file
        c_file = src_dir / "clean.c"
        c_file.write_text("""
#include <stdio.h>

int main() {
    printf("Hello, world!\\n");
    return 0;
}
""")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=True)

        assert not result.has_assembly
        assert len(result.standalone_asm_files) == 0
        assert len(result.inline_asm_sources) == 0
        assert len(result.inline_asm_ir) == 0

    def test_asm_in_comment_not_detected(self, temp_project):
        """Test that asm in comments is not falsely detected."""
        project_path, src_dir, build_dir = temp_project

        c_file = src_dir / "commented.c"
        c_file.write_text("""
// This is a comment about asm("nop")
/* Another comment with __asm__ inside */
int main() { return 0; }
""")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert not result.has_assembly

    def test_asm_variable_name_not_detected(self, temp_project):
        """Test that variable names containing 'asm' are not detected."""
        project_path, src_dir, build_dir = temp_project

        c_file = src_dir / "variable.c"
        c_file.write_text("""
int asmEnabled = 1;
int use_asm_optimization = 0;
void process_asm_file(void) { }
""")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert not result.has_assembly

    def test_multiple_extensions(self, temp_project):
        """Test detection across multiple assembly file extensions."""
        project_path, src_dir, build_dir = temp_project

        # Create files with different extensions
        (src_dir / "file1.s").write_text(".global func1\n")
        (src_dir / "file2.S").write_text(".global func2\n")
        (src_dir / "file3.asm").write_text("; NASM syntax\n")

        compile_commands = self._create_compile_commands(
            build_dir,
            [
                str(src_dir / "file1.s"),
                str(src_dir / "file2.S"),
                str(src_dir / "file3.asm"),
            ],
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        assert result.has_assembly
        assert len(result.standalone_asm_files) == 3

    def test_summary_format(self, temp_project):
        """Test that summary format is human-readable."""
        project_path, src_dir, build_dir = temp_project

        # Create multiple assembly sources
        (src_dir / "a.S").write_text(".global a\n")
        (src_dir / "b.S").write_text(".global b\n")
        c_file = src_dir / "c.c"
        c_file.write_text("void f() { __asm__(\"\"); }\n")

        compile_commands = self._create_compile_commands(
            build_dir,
            [str(src_dir / "a.S"), str(src_dir / "b.S"), str(c_file)],
        )

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
        )

        result = checker.check(scan_ir=False)

        summary = result.summary()
        assert "Assembly detected" in summary
        assert "2 standalone" in summary
        assert "1 C/C++" in summary

    def test_unavoidable_filtering(self, temp_project):
        """Test that known unavoidable findings are filtered out."""
        project_path, src_dir, build_dir = temp_project

        # Create assembly file
        asm_file = src_dir / "crypto.S"
        asm_file.write_text(".global aes_encrypt\n")

        # Create C file with inline asm
        c_file = src_dir / "simd.c"
        c_file.write_text("void f() { __asm__(\"nop\"); }\n")

        compile_commands = self._create_compile_commands(
            build_dir, [str(asm_file), str(c_file)]
        )

        # Create unavoidable_asm.json marking crypto.S as unavoidable
        unavoidable_path = build_dir / "unavoidable_asm.json"
        unavoidable_path.write_text(json.dumps({
            "unavoidable": [
                {"type": "standalone_file", "file_path": str(asm_file)}
            ]
        }))

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
            unavoidable_asm_path=unavoidable_path,
        )

        result = checker.check(scan_ir=False)

        # crypto.S should be filtered out, only simd.c should remain
        assert result.has_assembly  # Still has assembly overall
        assert len(result.standalone_asm_files) == 0  # Filtered out
        assert len(result.inline_asm_sources) == 1  # simd.c still there
        assert len(result.known_unavoidable) == 1  # crypto.S moved here
        assert result.known_unavoidable[0].file_path == str(asm_file)

    def test_save_unavoidable(self, temp_project):
        """Test saving findings as unavoidable."""
        project_path, src_dir, build_dir = temp_project

        # Create C file with inline asm
        c_file = src_dir / "critical.c"
        c_file.write_text("void spin() { __asm__(\"pause\"); }\n")

        compile_commands = self._create_compile_commands(
            build_dir, [str(c_file)]
        )

        unavoidable_path = build_dir / "unavoidable_asm.json"

        checker = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
            unavoidable_asm_path=unavoidable_path,
        )

        # First check - should find inline asm
        result1 = checker.check(scan_ir=False)
        assert len(result1.inline_asm_sources) == 1

        # Save as unavoidable
        checker.save_unavoidable(result1.inline_asm_sources)
        assert unavoidable_path.exists()

        # Create new checker that loads the unavoidable file
        checker2 = AssemblyChecker(
            compile_commands_path=compile_commands,
            build_dir=build_dir,
            project_path=project_path,
            unavoidable_asm_path=unavoidable_path,
        )

        # Second check - should be filtered out
        result2 = checker2.check(scan_ir=False)
        assert len(result2.inline_asm_sources) == 0  # Filtered
        assert len(result2.known_unavoidable) == 1  # Now known
