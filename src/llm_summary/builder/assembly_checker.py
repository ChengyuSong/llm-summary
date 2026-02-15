"""Assembly code detection for build artifacts."""

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..compile_commands import CompileCommandsDB
from ..models import AssemblyCheckResult, AssemblyFinding, AssemblyType

# Default filename for unavoidable assembly records
UNAVOIDABLE_ASM_FILENAME = "unavoidable_asm.json"


class AssemblyChecker:
    """
    Detects assembly code in build artifacts.

    Checks for:
    1. Standalone assembly files (.s, .S, .asm) in compile_commands.json
    2. Inline assembly in C/C++ source files (__asm__, asm(), etc.)
    3. Inline assembly in LLVM bitcode files (.bc) using llvm-dis
    """

    # Extensions for standalone assembly files
    ASM_EXTENSIONS = {".s", ".S", ".asm"}

    # Patterns for inline assembly in C/C++ source files
    # These are carefully crafted to avoid false positives from comments/strings
    INLINE_ASM_PATTERNS = [
        (r'\b__asm__\s*\(', "__asm__(...)"),
        (r'\b__asm\s*\(', "__asm(...)"),
        (r'\basm\s*\(\s*["\']', 'asm("...")'),
        (r'\b__asm\s*\{', "__asm {...}"),
        (r'\b__asm__\s+__volatile__\s*\(', "__asm__ __volatile__(...)"),
        (r'\basm\s+volatile\s*\(', "asm volatile(...)"),
    ]

    # Patterns for inline assembly in LLVM IR
    LLVM_IR_ASM_PATTERNS = [
        (r'\bcall\s+.*\basm\b', "call asm"),
        (r'\basm\s+sideeffect\b', "asm sideeffect"),
        (r'module\s+asm\b', "module asm"),
    ]

    # Docker container path prefixes to translate
    DOCKER_SRC_PREFIX = "/workspace/src"
    DOCKER_BUILD_PREFIX = "/workspace/build"

    def __init__(
        self,
        compile_commands_path: Path | str,
        build_dir: Path | str,
        project_path: Path | str,
        unavoidable_asm_path: Path | str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize assembly checker.

        Args:
            compile_commands_path: Path to compile_commands.json
            build_dir: Path to build directory (for LLVM IR files)
            project_path: Path to project source (for resolving relative paths)
            unavoidable_asm_path: Path to unavoidable_asm.json (optional)
            verbose: Enable verbose output
        """
        self.compile_commands_path = Path(compile_commands_path)
        self.build_dir = Path(build_dir).resolve()
        self.project_path = Path(project_path).resolve()
        self.unavoidable_asm_path = Path(unavoidable_asm_path) if unavoidable_asm_path else None
        self.verbose = verbose

        # Compile regex patterns once for efficiency
        self._inline_patterns = [
            (re.compile(pattern), desc) for pattern, desc in self.INLINE_ASM_PATTERNS
        ]
        self._ir_patterns = [
            (re.compile(pattern), desc) for pattern, desc in self.LLVM_IR_ASM_PATTERNS
        ]

        # Load compile commands database
        self._db = CompileCommandsDB(self.compile_commands_path)

        # Load known unavoidable findings
        self._unavoidable_keys: set[str] = set()
        self._load_unavoidable()

    def _load_unavoidable(self) -> None:
        """Load known unavoidable assembly findings from JSON file."""
        if not self.unavoidable_asm_path or not self.unavoidable_asm_path.exists():
            return

        try:
            with open(self.unavoidable_asm_path) as f:
                data = json.load(f)

            for item in data.get("unavoidable", []):
                finding = AssemblyFinding.from_dict(item)
                self._unavoidable_keys.add(finding.stable_key())

            if self.verbose and self._unavoidable_keys:
                print(f"[AssemblyChecker] Loaded {len(self._unavoidable_keys)} known unavoidable findings")

        except (json.JSONDecodeError, OSError) as e:
            if self.verbose:
                print(f"[AssemblyChecker] Could not load unavoidable_asm.json: {e}")

    def save_unavoidable(self, findings: list[AssemblyFinding], append: bool = True) -> None:
        """
        Save assembly findings as unavoidable to JSON file.

        Args:
            findings: List of findings to mark as unavoidable
            append: If True, add to existing; if False, replace
        """
        if not self.unavoidable_asm_path:
            raise ValueError("unavoidable_asm_path not set")

        existing: list[dict] = []
        if append and self.unavoidable_asm_path.exists():
            try:
                with open(self.unavoidable_asm_path) as f:
                    data = json.load(f)
                    existing = data.get("unavoidable", [])
            except (json.JSONDecodeError, OSError):
                pass

        # Add new findings, avoiding duplicates by stable_key
        existing_keys = {AssemblyFinding.from_dict(item).stable_key() for item in existing}
        for finding in findings:
            if finding.stable_key() not in existing_keys:
                existing.append(finding.to_dict())
                existing_keys.add(finding.stable_key())

        # Ensure parent directory exists
        self.unavoidable_asm_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.unavoidable_asm_path, "w") as f:
            json.dump({"unavoidable": existing}, f, indent=2)

        if self.verbose:
            print(f"[AssemblyChecker] Saved {len(existing)} unavoidable findings to {self.unavoidable_asm_path}")

    def _translate_docker_path(self, docker_path: str) -> str:
        """Translate Docker container paths to host paths."""
        if docker_path.startswith(self.DOCKER_SRC_PREFIX):
            return str(self.project_path / docker_path[len(self.DOCKER_SRC_PREFIX) + 1:])
        elif docker_path.startswith(self.DOCKER_BUILD_PREFIX):
            return str(self.build_dir / docker_path[len(self.DOCKER_BUILD_PREFIX) + 1:])
        return docker_path

    def check(self, scan_ir: bool = True, max_findings_per_type: int = 20) -> AssemblyCheckResult:
        """
        Run assembly detection on build artifacts.

        Filters out known unavoidable findings so the agent only sees new ones.
        Stops checking after finding max_findings_per_type in each category to
        avoid overwhelming the LLM context.

        Args:
            scan_ir: Whether to scan LLVM IR files for inline assembly
            max_findings_per_type: Maximum findings to collect per type before
                                   stopping (like compiler error limits)

        Returns:
            AssemblyCheckResult with new findings (known unavoidable filtered out)
        """
        from itertools import islice

        # Collect findings, but stop at max_findings_per_type (like compiler error limit)
        standalone_gen = self._check_standalone_asm_files()
        inline_sources_gen = self._check_inline_asm_in_sources()
        inline_ir_gen = self._check_inline_asm_in_ir() if scan_ir else iter([])

        # Use islice to limit findings, then check if there were more
        all_standalone = list(islice(standalone_gen, max_findings_per_type + 1))
        standalone_truncated = len(all_standalone) > max_findings_per_type
        if standalone_truncated:
            all_standalone = all_standalone[:max_findings_per_type]

        all_inline_sources = list(islice(inline_sources_gen, max_findings_per_type + 1))
        inline_sources_truncated = len(all_inline_sources) > max_findings_per_type
        if inline_sources_truncated:
            all_inline_sources = all_inline_sources[:max_findings_per_type]

        all_inline_ir = list(islice(inline_ir_gen, max_findings_per_type + 1))
        inline_ir_truncated = len(all_inline_ir) > max_findings_per_type
        if inline_ir_truncated:
            all_inline_ir = all_inline_ir[:max_findings_per_type]

        # Filter out known unavoidable findings
        standalone, standalone_known = self._filter_unavoidable(all_standalone)
        inline_sources, inline_known = self._filter_unavoidable(all_inline_sources)
        inline_ir, ir_known = self._filter_unavoidable(all_inline_ir)

        known_unavoidable = standalone_known + inline_known + ir_known
        has_assembly = bool(standalone or inline_sources or inline_ir or known_unavoidable)

        result = AssemblyCheckResult(
            has_assembly=has_assembly,
            standalone_asm_files=standalone,
            inline_asm_sources=inline_sources,
            inline_asm_ir=inline_ir,
            known_unavoidable=known_unavoidable,
            standalone_truncated=standalone_truncated,
            inline_sources_truncated=inline_sources_truncated,
            inline_ir_truncated=inline_ir_truncated,
        )

        if self.verbose:
            print(f"[AssemblyChecker] {result.summary()}")

        return result

    def _filter_unavoidable(
        self, findings: list[AssemblyFinding]
    ) -> tuple[list[AssemblyFinding], list[AssemblyFinding]]:
        """
        Separate findings into new and known unavoidable.

        Returns:
            Tuple of (new_findings, known_unavoidable)
        """
        new_findings = []
        known = []
        for f in findings:
            if f.stable_key() in self._unavoidable_keys:
                known.append(f)
            else:
                new_findings.append(f)
        return new_findings, known

    def _check_standalone_asm_files(self) -> Iterator[AssemblyFinding]:
        """Check for standalone assembly files in compile_commands.json."""
        for file_path in self._db.get_all_files():
            path = Path(file_path)
            if path.suffix in self.ASM_EXTENSIONS:
                if self.verbose:
                    print(f"[AssemblyChecker] Found standalone asm: {file_path}")
                yield AssemblyFinding(
                    asm_type=AssemblyType.STANDALONE_FILE,
                    file_path=file_path,
                )

    def _check_inline_asm_in_sources(self) -> Iterator[AssemblyFinding]:
        """Check for inline assembly in C/C++ source files."""
        c_extensions = {".c", ".cc", ".cpp", ".cxx", ".C", ".h", ".hpp", ".hxx"}

        for file_path in self._db.get_all_files():
            path = Path(file_path)
            if path.suffix not in c_extensions:
                continue

            # Translate Docker paths to host paths
            host_path = self._translate_docker_path(file_path)
            finding = self._scan_file_for_inline_asm(host_path, original_path=file_path)
            if finding:
                yield finding

    def _check_inline_asm_in_ir(self) -> Iterator[AssemblyFinding]:
        """Check for inline assembly in LLVM bitcode (.bc) files using llvm-dis."""
        if not self.build_dir.exists():
            return

        # Find all .bc files in build directory
        bc_files = list(self.build_dir.rglob("*.bc"))
        if not bc_files:
            if self.verbose:
                print("[AssemblyChecker] No .bc files found in build directory")
            return

        if self.verbose:
            print(f"[AssemblyChecker] Found {len(bc_files)} .bc files to scan")

        for bc_file in bc_files:
            finding = self._scan_bc_file_for_asm(bc_file)
            if finding:
                yield finding

    def _scan_file_for_inline_asm(
        self, file_path: str, original_path: str | None = None
    ) -> AssemblyFinding | None:
        """
        Scan a single C/C++ file for inline assembly.

        Args:
            file_path: Host path to the file
            original_path: Original path (e.g., Docker path) to report in findings

        Returns first match to avoid noise. Returns None if no assembly found.
        """
        report_path = original_path or file_path
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    # Skip comment lines (simple heuristic)
                    stripped = line.strip()
                    if stripped.startswith("//") or stripped.startswith("/*"):
                        continue

                    for pattern, desc in self._inline_patterns:
                        if pattern.search(line):
                            if self.verbose:
                                print(
                                    f"[AssemblyChecker] Found inline asm in "
                                    f"{report_path}:{line_num}: {desc}"
                                )
                            return AssemblyFinding(
                                asm_type=AssemblyType.INLINE_SOURCE,
                                file_path=report_path,
                                line_number=line_num,
                                snippet=line.strip()[:100],
                                pattern_matched=desc,
                            )
        except OSError as e:
            if self.verbose:
                print(f"[AssemblyChecker] Could not read {file_path}: {e}")

        return None

    def _scan_bc_file_for_asm(self, bc_path: Path) -> AssemblyFinding | None:
        """
        Scan a single LLVM bitcode file for inline assembly using llvm-dis.

        Runs llvm-dis to disassemble .bc to text and searches for asm patterns.
        Returns first match to avoid noise. Returns None if no assembly found.
        """
        try:
            # Run llvm-dis to convert .bc to text IR (output to stdout)
            result = subprocess.run(
                ["llvm-dis", "-o", "-", str(bc_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                if self.verbose:
                    print(f"[AssemblyChecker] llvm-dis failed for {bc_path}: {result.stderr[:100]}")
                return None

            # Search the disassembled IR for asm patterns
            for line_num, line in enumerate(result.stdout.split("\n"), start=1):
                for pattern, desc in self._ir_patterns:
                    if pattern.search(line):
                        if self.verbose:
                            print(
                                f"[AssemblyChecker] Found IR asm in "
                                f"{bc_path}:{line_num}: {desc}"
                            )
                        return AssemblyFinding(
                            asm_type=AssemblyType.INLINE_LLVM_IR,
                            file_path=str(bc_path),
                            line_number=line_num,
                            snippet=line.strip()[:100],
                            pattern_matched=desc,
                        )

        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"[AssemblyChecker] llvm-dis timed out for {bc_path}")
        except FileNotFoundError:
            if self.verbose:
                print("[AssemblyChecker] llvm-dis not found, skipping IR scan")
        except OSError as e:
            if self.verbose:
                print(f"[AssemblyChecker] Error scanning {bc_path}: {e}")

        return None
