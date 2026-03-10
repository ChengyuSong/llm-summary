"""Source preprocessor: runs clang -E and maps expanded output back to original lines."""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .compile_commands import CompileCommandsDB

logger = logging.getLogger(__name__)

# Matches GCC/Clang line markers: # linenum "filename" [flags...]
_LINE_MARKER_RE = re.compile(r'^#\s+(\d+)\s+"([^"]*)"')


@dataclass
class _LineMapping:
    """A preprocessed output line mapped back to its origin."""

    pp_line: str  # the preprocessed text
    orig_file: str  # originating source file (from line marker)
    orig_line: int  # 1-based line number in the original file


@dataclass
class PreprocessedFile:
    """Result of preprocessing a single source file.

    Holds the expanded output with per-line origin mappings so that callers
    can extract the preprocessed source for any function given its original
    file path and line range.
    """

    source_file: str
    mappings: list[_LineMapping] = field(default_factory=list)
    error: str | None = None
    # Lazy index: norm_file -> sorted list of (orig_line, pp_line)
    _index: dict[str, list[tuple[int, str]]] | None = field(
        default=None, repr=False
    )

    def _build_index(self) -> dict[str, list[tuple[int, str]]]:
        """Build a per-file index sorted by orig_line for fast extraction."""
        idx: dict[str, list[tuple[int, str]]] = {}
        for m in self.mappings:
            norm = str(Path(m.orig_file).resolve())
            idx.setdefault(norm, []).append((m.orig_line, m.pp_line))
        # Sort each file's entries by line number
        for v in idx.values():
            v.sort(key=lambda x: x[0])
        return idx

    def extract_pp_source(
        self, file_path: str, start_line: int, end_line: int
    ) -> str | None:
        """Extract preprocessed source for an original file+line range.

        Args:
            file_path: The original source file path (as stored in Function.file_path).
            start_line: 1-based start line in the original file.
            end_line: 1-based end line (inclusive) in the original file.

        Returns:
            The concatenated preprocessed lines that map back to the given range,
            or None if no lines matched.
        """
        if self._index is None:
            self._index = self._build_index()

        norm = str(Path(file_path).resolve())
        file_entries = self._index.get(norm)
        if not file_entries:
            return None

        # Binary search for start_line
        import bisect
        lo = bisect.bisect_left(file_entries, (start_line,))
        hi = bisect.bisect_right(file_entries, (end_line + 1,))

        lines = [entry[1] for entry in file_entries[lo:hi]
                 if start_line <= entry[0] <= end_line]

        if not lines:
            return None

        return "\n".join(lines)


class SourcePreprocessor:
    """Runs ``clang -E`` on source files and parses line directives."""

    def __init__(
        self,
        compile_commands: CompileCommandsDB | None = None,
        extra_args: list[str] | None = None,
        clang_binary: str = "clang",
        verbose: bool = False,
    ):
        self.compile_commands = compile_commands
        self.extra_args = extra_args or []
        self.clang_binary = clang_binary
        self.verbose = verbose

    def preprocess(self, file_path: str | Path) -> PreprocessedFile:
        """Preprocess a single source file and return mapped output.

        Falls back gracefully: on any failure the returned ``PreprocessedFile``
        has an empty mappings list and a populated ``error`` field.
        """
        file_path = Path(file_path).resolve()
        result = PreprocessedFile(source_file=str(file_path))

        cmd = self._build_command(file_path)

        if self.verbose:
            logger.info("Preprocessing: %s", " ".join(cmd))

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError:
            result.error = f"{self.clang_binary} not found"
            logger.warning("Preprocessor: %s", result.error)
            return result
        except subprocess.TimeoutExpired:
            result.error = f"clang -E timed out for {file_path}"
            logger.warning("Preprocessor: %s", result.error)
            return result

        if proc.returncode != 0:
            # clang -E can still produce useful partial output on warnings;
            # only treat as error if there's truly no output.
            if not proc.stdout.strip():
                result.error = f"clang -E failed (rc={proc.returncode}): {proc.stderr[:200]}"
                logger.warning("Preprocessor: %s", result.error)
                return result

        result.mappings = self._parse_output(proc.stdout)
        return result

    def _build_command(self, file_path: Path) -> list[str]:
        """Build the clang -E command line."""
        cmd = [self.clang_binary, "-E"]

        # Add per-file flags from compile_commands.json
        if self.compile_commands and self.compile_commands.has_file(file_path):
            cmd.extend(self.compile_commands.get_compile_flags(file_path))
        else:
            # Default language detection by extension
            ext = file_path.suffix.lower()
            if ext in (".cpp", ".cxx", ".cc", ".hpp", ".hxx"):
                cmd.extend(["-x", "c++", "-std=c++17"])
            else:
                cmd.extend(["-x", "c", "-std=c11"])

        cmd.extend(self.extra_args)
        cmd.append(str(file_path))
        return cmd

    @staticmethod
    def _parse_output(output: str) -> list[_LineMapping]:
        """Parse clang -E output into line mappings.

        The preprocessor output contains line markers of the form:
            # 42 "/path/to/file.c"
        followed by the expanded source lines. We track the current file/line
        as we scan, incrementing the line counter for each non-marker line.
        """
        mappings: list[_LineMapping] = []
        current_file: str | None = None
        current_line: int = 0

        for raw_line in output.splitlines():
            m = _LINE_MARKER_RE.match(raw_line)
            if m:
                current_line = int(m.group(1))
                current_file = m.group(2)
                continue

            if current_file is None:
                continue

            # Skip blank lines from the preprocessor
            stripped = raw_line.strip()
            if not stripped:
                current_line += 1
                continue

            mappings.append(
                _LineMapping(
                    pp_line=raw_line,
                    orig_file=current_file,
                    orig_line=current_line,
                )
            )
            current_line += 1

        return mappings
