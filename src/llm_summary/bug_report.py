"""Bug report generation for feasible-confirmed verdicts.

Given a confirmed-feasible verdict (triage says bug exists, validation confirms),
this module:

1. Searches the project for existing harnesses (OSS-Fuzz, unit tests) as references
2. Asks LLM to generate a standalone PoC harness (C main() reading from file/stdin)
3. Compiles with ASan/UBSan
4. Runs to confirm crash
5. Generates a markdown bug report
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .db import SummaryDB
from .docker_paths import remap_path as _remap_path
from .git_tools import GitTools
from .llm.base import LLMBackend

# Patterns to search for existing harnesses in a project repo
_FUZZ_PATTERNS = [
    "LLVMFuzzerTestOneInput",
    "AFL_INIT_BUFS",
    "afl_fuzz",
]
_FUZZ_DIRS = [
    "contrib/oss-fuzz",
    "fuzz",
    "fuzzing",
    "tests/fuzz",
    "test/fuzz",
]
_TEST_DIRS = [
    "tests",
    "test",
    "contrib/libtests",
    "contrib/testpngs",
]

POC_HARNESS_PROMPT = """\
You are a security engineer writing a standalone PoC (proof-of-concept) harness \
to trigger a confirmed bug.

## Bug Details

- **Function**: `{func_name}`
- **Issue**: [{severity}] {issue_kind} — {description}
- **Location**: {location}

## Triage Reasoning

{reasoning}

## Relevant Functions

{relevant_functions}

## Source Code

{source_code}

## Existing Harness References

{references}

## Task

Write a standalone C program (with `main()`) that:

1. Reads input from a file passed as `argv[1]` (or stdin if no arg)
2. Sets up the minimal program state needed to reach the vulnerable function
3. Calls the function chain that triggers the bug
4. Compiles with: `clang -fsanitize=address,undefined -g -O0`

Requirements:
- Include all necessary headers
- The harness should be self-contained (no external deps beyond the project)
- Use the input data to control relevant parameters
- Keep it minimal — just enough to trigger the bug
- Add a comment explaining what input triggers the bug
- If the function requires complex struct setup, initialize fields to realistic \
values from the triage reasoning

Output ONLY the C code, no explanation. Start with `#include` and end with the \
closing brace of `main()`.
"""


def find_reference_harnesses(
    project_path: Path,
    func_name: str,
    *,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Search a project repo for existing fuzz/test harnesses as references.

    Uses GitTools to search tracked files only. Returns a list of dicts:
    [{"path": "contrib/oss-fuzz/foo.cc", "kind": "oss-fuzz", "snippet": "..."}]
    """
    if not (project_path / ".git").exists():
        return []

    git = GitTools(project_path)
    results: list[dict[str, Any]] = []

    # Search for fuzz harness files
    for fuzz_dir in _FUZZ_DIRS:
        tree = git.git_ls_tree(fuzz_dir)
        if tree.get("error"):
            continue
        for f in tree.get("files", []):
            fname = f.split("  (")[0]  # strip size suffix
            if fname.endswith((".c", ".cc", ".cpp")):
                fpath = f"{fuzz_dir}/{fname}"
                content = git.git_show(fpath, max_lines=100)
                if not content.get("error"):
                    results.append({
                        "path": fpath,
                        "kind": "oss-fuzz",
                        "snippet": content.get("content", ""),
                    })
                    if verbose:
                        print(f"  Found fuzz harness: {fpath}")

    # Search for test files that reference the target function
    grep_result = git.git_grep(
        func_name,
        glob="*.c",
        max_results=10,
    )
    if not grep_result.get("error"):
        for match in grep_result.get("matches", []):
            # match format: "path:line:content"
            parts = match.split(":", 2)
            if len(parts) < 2:
                continue
            mpath = parts[0]
            # Skip if already found as fuzz harness
            if any(r["path"] == mpath for r in results):
                continue
            # Only include test-like files
            if any(td in mpath for td in _TEST_DIRS) or "test" in mpath.lower():
                content = git.git_show(mpath, max_lines=100)
                if not content.get("error"):
                    results.append({
                        "path": mpath,
                        "kind": "test",
                        "snippet": content.get("content", ""),
                    })
                    if verbose:
                        print(f"  Found test file: {mpath}")

    return results


def _get_source_code(
    db: SummaryDB,
    func_name: str,
    relevant_functions: list[str],
    project_path: Path,
) -> str:
    """Get source code for the target and relevant functions."""
    git = GitTools(project_path)
    blocks: list[str] = []

    for rname in [func_name] + [
        f for f in relevant_functions if f != func_name
    ]:
        funcs = db.get_function_by_name(rname)
        if not funcs:
            continue
        func = funcs[0]
        if not func.file_path or not func.line_start or not func.line_end:
            continue
        # Read from git (path relative to repo)
        try:
            rel_path = str(Path(func.file_path).relative_to(project_path))
        except ValueError:
            rel_path = func.file_path
        content = git.git_show(
            rel_path,
            start_line=max(1, func.line_start - 5),
            max_lines=func.line_end - func.line_start + 10,
        )
        if content.get("error"):
            continue
        blocks.append(
            f"### `{rname}` ({rel_path}:"
            f"{func.line_start}-{func.line_end})\n\n"
            f"```c\n{content['content']}\n```"
        )

    return "\n\n".join(blocks) if blocks else "_Source code not available._"


def generate_poc_harness(
    verdict: dict[str, Any],
    db: SummaryDB,
    llm: LLMBackend,
    project_path: Path,
    references: list[dict[str, Any]] | None = None,
    verbose: bool = False,
) -> str:
    """Ask LLM to generate a standalone PoC harness C file.

    Returns the generated C source code.
    """
    func_name = verdict["function_name"]
    issue = verdict.get("issue", {})
    relevant = verdict.get("relevant_functions", [func_name])

    source_code = _get_source_code(db, func_name, relevant, project_path)

    refs_text = "_No existing harnesses found._"
    if references:
        ref_blocks = []
        for ref in references[:3]:  # limit to 3 references
            ref_blocks.append(
                f"### {ref['kind']}: `{ref['path']}`\n\n"
                f"```c\n{ref['snippet']}\n```"
            )
        refs_text = "\n\n".join(ref_blocks)

    relevant_text = ", ".join(f"`{f}`" for f in relevant)

    prompt = POC_HARNESS_PROMPT.format(
        func_name=func_name,
        severity=issue.get("severity", "medium"),
        issue_kind=issue.get("issue_kind", "unknown"),
        description=issue.get("description", ""),
        location=issue.get("location", "unknown"),
        reasoning=verdict.get("reasoning", "N/A"),
        relevant_functions=relevant_text,
        source_code=source_code,
        references=refs_text,
    )

    if verbose:
        print(f"[BugReport] Generating PoC harness for {func_name}...")

    response = llm.complete(prompt)

    # Extract C code from response (strip markdown fences if present)
    code_match = re.search(r"```c\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1)
    # If no fences, assume entire response is code
    return response.strip()


def _extract_include_dirs(
    compile_commands_path: Path,
    project_path: Path,
    build_dir: Path | None = None,
) -> list[str]:
    """Extract unique -I dirs from compile_commands.json, remapping Docker paths."""
    with open(compile_commands_path) as f:
        cc = json.load(f)
    include_dirs: set[str] = set()
    for entry in cc:
        cmd = entry.get("command", "") or " ".join(
            entry.get("arguments", []),
        )
        for m in re.finditer(r"-I\s*(\S+)", cmd):
            remapped = _remap_path(m.group(1), project_path, build_dir)
            include_dirs.add(remapped)
    return sorted(include_dirs)


def _extract_define_flags(
    compile_commands_path: Path,
) -> list[str]:
    """Extract unique -D flags from compile_commands.json."""
    with open(compile_commands_path) as f:
        cc = json.load(f)
    defines: set[str] = set()
    for entry in cc:
        cmd = entry.get("command", "") or " ".join(
            entry.get("arguments", []),
        )
        for m in re.finditer(r"(-D\S+)", cmd):
            defines.add(m.group(1))
    return sorted(defines)


def _find_static_lib(
    project_name: str,
    target_name: str | None,
) -> tuple[Path | None, Path | None]:
    """Find static library and build_dir from link_units.json.

    Returns (lib_path, build_dir) or (None, None).
    """
    lu_path = Path("func-scans") / project_name / "link_units.json"
    if not lu_path.exists():
        return None, None
    with open(lu_path) as f:
        data = json.load(f)
    build_dir = Path(data.get("build_dir", ""))
    for unit in data.get("link_units", []):
        if target_name and unit.get("name") != target_name:
            continue
        output = unit.get("output", "")
        if output:
            lib = build_dir / output
            if lib.exists():
                return lib, build_dir
    # Fallback: try any .a in build_dir
    if build_dir.exists():
        for a in sorted(build_dir.glob("*.a")):
            return a, build_dir
    return None, build_dir if build_dir.exists() else None


def _build_compile_cmd(
    harness_source: Path,
    project_path: Path,
    output_path: Path,
    compile_commands_path: Path | None = None,
    extra_cflags: list[str] | None = None,
    build_dir: Path | None = None,
    static_lib: Path | None = None,
) -> list[str]:
    """Build the clang command line for compiling a PoC."""
    cflags = [
        "-fsanitize=address,undefined",
        "-fno-sanitize-recover=all",
        "-fuse-ld=lld",
        "-g", "-O0",
    ]

    if compile_commands_path and compile_commands_path.exists():
        for d in _extract_include_dirs(
            compile_commands_path, project_path, build_dir,
        ):
            cflags.extend(["-I", d])
        cflags.extend(_extract_define_flags(compile_commands_path))

    cflags.extend(["-I", str(project_path)])
    if build_dir:
        cflags.extend(["-I", str(build_dir)])

    if extra_cflags:
        cflags.extend(extra_cflags)

    cmd = [
        "clang", *cflags,
        str(harness_source),
    ]

    if static_lib and static_lib.exists():
        cmd.append(str(static_lib))

    cmd.extend(["-o", str(output_path), "-lm", "-lz"])

    return cmd


def write_build_script(
    script_path: Path,
    compile_cmd: list[str],
    binary_path: Path,
) -> None:
    """Write a reproducible build script for the PoC."""
    import shlex
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Build PoC harness with ASan/UBSan",
        shlex.join(compile_cmd),
        "",
        f'echo "Built: {binary_path}"',
        f'echo "Run:   {binary_path} [input_file]"',
        "",
    ]
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)


def compile_poc(
    harness_source: Path,
    project_path: Path,
    output_path: Path,
    compile_commands_path: Path | None = None,
    extra_cflags: list[str] | None = None,
    build_dir: Path | None = None,
    static_lib: Path | None = None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Compile a PoC harness with ASan/UBSan.

    Returns (success, error_message).
    """
    cmd = _build_compile_cmd(
        harness_source, project_path, output_path,
        compile_commands_path, extra_cflags,
        build_dir, static_lib,
    )

    if verbose:
        print(f"[BugReport] Compiling: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
    )

    if proc.returncode != 0:
        return False, proc.stderr
    return True, ""


def run_poc(
    binary: Path,
    input_file: Path | None = None,
    timeout: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a compiled PoC harness and capture crash output.

    Returns dict with keys: crashed, exit_code, stdout, stderr, signal.
    """
    cmd = [str(binary)]
    if input_file:
        cmd.append(str(input_file))

    stdin_data = None
    if not input_file:
        # Provide empty stdin
        stdin_data = b""

    if verbose:
        print(f"[BugReport] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "crashed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "timeout",
            "signal": None,
        }

    # ASan/UBSan crashes typically have non-zero exit and specific stderr
    crashed = proc.returncode != 0
    signal_name = None
    if proc.returncode < 0:
        import signal
        try:
            signal_name = signal.Signals(-proc.returncode).name
        except (ValueError, AttributeError):
            signal_name = f"signal {-proc.returncode}"

    return {
        "crashed": crashed,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.decode("utf-8", errors="replace")[:2000],
        "stderr": proc.stderr.decode("utf-8", errors="replace")[:4000],
        "signal": signal_name,
    }


def generate_report(
    verdict: dict[str, Any],
    reflection: dict[str, Any] | None,
    poc_result: dict[str, Any] | None,
    harness_path: str | None = None,
    harness_tier: str = "synthetic",
) -> str:
    """Generate a markdown bug report.

    Args:
        verdict: Triage verdict dict.
        reflection: Reflection result dict (optional).
        poc_result: Result from run_poc() (optional).
        harness_path: Path to the PoC harness source.
        harness_tier: "oss-fuzz", "test", or "synthetic".
    """
    func_name = verdict["function_name"]
    issue = verdict.get("issue", {})
    relevant = verdict.get("relevant_functions", [func_name])

    confidence_map = {
        "oss-fuzz": "High (reuses project's OSS-Fuzz harness)",
        "test": "Medium-High (based on project unit test)",
        "synthetic": "Medium (LLM-generated, may not reflect real usage)",
    }

    lines = [
        f"# Bug Report: {issue.get('issue_kind', 'unknown')} "
        f"in `{func_name}`\n",
        "## Summary\n",
        f"{verdict.get('reasoning', 'N/A')}\n",
        "## Issue\n",
        f"- **Type**: {issue.get('issue_kind', 'unknown')}",
        f"- **Location**: {issue.get('location', 'unknown')}",
        f"- **Severity**: {issue.get('severity', 'medium')}",
        "- **Confirmed by**: ucsan concolic execution\n",
        "## Call Chain\n",
        " -> ".join(f"`{f}`" for f in relevant),
        "",
    ]

    if reflection:
        lines.extend([
            "## Validation\n",
            f"- **Hypothesis**: {reflection.get('hypothesis', '?')}",
            f"- **Confidence**: {reflection.get('confidence', '?')}",
            f"- **Action**: {reflection.get('action', '?')}",
            f"- **Reasoning**: {reflection.get('reasoning', 'N/A')}\n",
        ])

    if poc_result:
        lines.append("## Proof of Concept\n")
        if harness_path:
            lines.append(f"- **Harness**: `{harness_path}`")
        lines.append(
            f"- **Crashed**: {'Yes' if poc_result['crashed'] else 'No'}",
        )
        lines.append(f"- **Exit code**: {poc_result['exit_code']}")
        if poc_result.get("signal"):
            lines.append(f"- **Signal**: {poc_result['signal']}")
        if poc_result.get("stderr"):
            stderr = poc_result["stderr"]
            # Extract ASan/UBSan summary lines
            summary_lines = [
                ln for ln in stderr.splitlines()
                if any(
                    k in ln for k in
                    ["SUMMARY:", "ERROR:", "runtime error:"]
                )
            ]
            if summary_lines:
                lines.append("\n### Sanitizer Output\n")
                lines.append("```")
                lines.extend(summary_lines[:10])
                lines.append("```")
        lines.append("")

    lines.extend([
        "## Harness Confidence\n",
        f"- **Tier**: {harness_tier}",
        f"- **Confidence**: {confidence_map.get(harness_tier, 'Unknown')}\n",
    ])

    if harness_tier == "synthetic":
        lines.extend([
            "> **Note**: This harness was generated by an LLM. The bug may",
            "> not be triggerable through normal program usage. Manual review",
            "> of the harness and input setup is recommended.\n",
        ])

    return "\n".join(lines)


def run_bug_report(
    verdict: dict[str, Any],
    db: SummaryDB,
    llm: LLMBackend,
    project_path: Path,
    output_dir: Path,
    compile_commands_path: Path | None = None,
    reflection: dict[str, Any] | None = None,
    project_name: str | None = None,
    target_name: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """End-to-end bug report generation.

    1. Find reference harnesses
    2. Generate PoC harness
    3. Compile (link against project static lib)
    4. Run
    5. Write report

    Returns dict with keys: report_path, harness_path, poc_result, success.
    """
    func_name = verdict["function_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve project name for link_units lookup
    if not project_name:
        project_name = project_path.name

    static_lib, build_dir = _find_static_lib(project_name, target_name)
    if verbose:
        if static_lib:
            print(f"[BugReport] Static lib: {static_lib}")
        if build_dir:
            print(f"[BugReport] Build dir: {build_dir}")

    # Step 1: find references
    if verbose:
        print(f"[BugReport] Searching for reference harnesses in {project_path}")
    references = find_reference_harnesses(project_path, func_name, verbose=verbose)

    harness_tier = "synthetic"
    if any(r["kind"] == "oss-fuzz" for r in references):
        harness_tier = "oss-fuzz"
    elif any(r["kind"] == "test" for r in references):
        harness_tier = "test"

    # Step 2: generate PoC
    poc_source = generate_poc_harness(
        verdict, db, llm, project_path, references, verbose=verbose,
    )

    harness_path = output_dir / f"poc_{func_name}.c"
    harness_path.write_text(poc_source)
    if verbose:
        print(f"[BugReport] Wrote harness: {harness_path}")

    # Step 3: write build script and compile
    binary_path = output_dir / f"poc_{func_name}"
    compile_cmd = _build_compile_cmd(
        harness_path, project_path, binary_path,
        compile_commands_path, build_dir=build_dir,
        static_lib=static_lib,
    )
    script_path = output_dir / f"build_poc_{func_name}.sh"
    write_build_script(script_path, compile_cmd, binary_path)
    if verbose:
        print(f"[BugReport] Wrote build script: {script_path}")

    ok, err = compile_poc(
        harness_path, project_path, binary_path,
        compile_commands_path=compile_commands_path,
        build_dir=build_dir,
        static_lib=static_lib,
        verbose=verbose,
    )

    poc_result: dict[str, Any] | None = None
    if ok:
        # Step 4: run
        poc_result = run_poc(binary_path, verbose=verbose)
        if verbose:
            print(
                f"[BugReport] PoC result: crashed={poc_result['crashed']}, "
                f"exit={poc_result['exit_code']}",
            )
    elif verbose:
        print(f"[BugReport] Compile failed: {err[:300]}")

    # Step 5: write report
    report = generate_report(
        verdict, reflection, poc_result,
        harness_path=str(harness_path),
        harness_tier=harness_tier,
    )
    report_path = output_dir / f"bug_report_{func_name}.md"
    report_path.write_text(report)
    if verbose:
        print(f"[BugReport] Report: {report_path}")

    return {
        "report_path": str(report_path),
        "harness_path": str(harness_path),
        "build_script": str(script_path),
        "poc_result": poc_result,
        "compile_error": err if not ok else None,
        "harness_tier": harness_tier,
        "success": ok and poc_result is not None and poc_result["crashed"],
    }
