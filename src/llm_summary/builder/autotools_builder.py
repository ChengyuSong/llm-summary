"""Autotools builder with LLM-powered incremental learning."""

import json
import shutil
from pathlib import Path
from typing import Any

from ..llm.base import LLMBackend
from .actions import AutotoolsActions
from .actions import AUTOTOOLS_TOOL_DEFINITIONS as AUTOTOOLS_ACTION_TOOLS
from .constants import (
    DOCKER_WORKSPACE_BUILD,
    DOCKER_WORKSPACE_SRC,
    MAX_CONTEXT_TOKENS,
    MAX_TURNS_AUTOTOOLS,
)
from .llm_utils import (
    compress_stale_reads,
    deduplicate_tool_result,
    truncate_messages,
)
from .tools import BuildTools
from .tools import TOOL_DEFINITIONS as FILE_TOOLS


# Tool definitions for autotools builds
AUTOTOOLS_TOOL_DEFINITIONS = FILE_TOOLS + AUTOTOOLS_ACTION_TOOLS


class AutotoolsBuilder:
    """Build autotools projects with incremental learning from failures."""

    def __init__(
        self,
        llm: LLMBackend,
        container_image: str = "llm-summary-builder:latest",
        build_dir: Path | None = None,
        build_scripts_dir: Path | None = None,
        max_retries: int = 3,
        enable_lto: bool = True,
        prefer_static: bool = True,
        generate_ir: bool = True,
        verbose: bool = False,
        log_file: str | None = None,
    ):
        self.llm = llm
        self.container_image = container_image
        self.build_dir = build_dir
        self.build_scripts_dir = Path(build_scripts_dir) if build_scripts_dir else None
        self.max_retries = max_retries
        self.enable_lto = enable_lto
        self.prefer_static = prefer_static
        self.generate_ir = generate_ir
        self.verbose = verbose
        self.log_file = log_file

    def _get_unavoidable_asm_path(self, project_name: str) -> Path | None:
        """Get path to unavoidable_asm.json for a project."""
        if not self.build_scripts_dir:
            return None
        return self.build_scripts_dir / project_name.lower() / "unavoidable_asm.json"

    def learn_and_build(
        self, project_path: Path
    ) -> dict[str, Any]:
        """
        Learn how to build an autotools project through iterative attempts.

        Returns a dict with:
        - success: bool
        - configure_flags: list[str]
        - attempts: int
        - build_log: str
        - error_messages: list[str]
        """
        project_path = Path(project_path).resolve()

        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Clean build directory to ensure fresh build
        build_dir = Path(self.build_dir) if self.build_dir else project_path / "build"
        if build_dir.exists():
            if self.verbose:
                print(f"[Cleanup] Removing existing build directory: {build_dir}")
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Check for configure or configure.ac
        configure_path = project_path / "configure"
        configure_ac_path = project_path / "configure.ac"

        if not configure_path.exists() and not configure_ac_path.exists():
            raise ValueError(f"Neither configure nor configure.ac found in {project_path}")

        # Read configure.ac if available for LLM analysis
        if configure_ac_path.exists():
            configure_ac_content = configure_ac_path.read_text()
        else:
            configure_ac_content = "(configure.ac not found, only configure script exists)"

        # Get initial configuration from LLM using ReAct loop
        if self.verbose:
            print("\n[1/3] Analyzing project with LLM...")

        result = self._get_initial_config(project_path, configure_ac_content, configure_path.exists())

        # Handle both list[str] (simple mode) and dict (ReAct mode with metadata)
        if isinstance(result, dict):
            configure_flags = result.get("configure_flags", self._get_default_config())
            react_build_succeeded = result.get("build_succeeded", False)
            react_terminated = result.get("react_terminated", False)

            if react_build_succeeded:
                if self.verbose:
                    print("\n[ReAct] Build already succeeded in exploration phase!")
                    print(f"[3/3] Build successful after {result.get('attempts', 1)} turns!")

                return {
                    "success": True,
                    "configure_flags": configure_flags,
                    "attempts": 1,
                    "build_log": "Build succeeded during ReAct exploration",
                    "error_messages": [],
                }

            if react_terminated:
                if self.verbose:
                    print("\n[ReAct] Build terminated - LLM identified unresolvable blockers")
                    print("[FAILED] Cannot proceed with available tools")

                return {
                    "success": False,
                    "configure_flags": configure_flags,
                    "attempts": 1,
                    "build_log": "ReAct loop terminated without successful build",
                    "error_messages": ["LLM identified blockers that cannot be resolved with available tools"],
                }
        else:
            configure_flags = result
            react_build_succeeded = False

        if self.verbose:
            print(f"[Initial Config] {len(configure_flags)} flags:")
            for flag in configure_flags:
                print(f"  {flag}")

        # All retries exhausted or simple mode without ReAct
        return {
            "success": False,
            "configure_flags": configure_flags,
            "attempts": 1,
            "build_log": "Initial configuration returned but no build attempted",
            "error_messages": [],
        }

    def _get_initial_config(
        self, project_path: Path, configure_ac_content: str, has_configure: bool
    ) -> list[str] | dict:
        """Get initial configuration using LLM analysis with tool support."""
        if hasattr(self.llm, "complete_with_tools"):
            return self._get_initial_config_with_tools(
                project_path, configure_ac_content, has_configure
            )
        else:
            return self._get_default_config()

    def _get_initial_config_with_tools(
        self, project_path: Path, configure_ac_content: str, has_configure: bool
    ) -> list[str] | dict:
        """
        ReAct approach: Allow LLM to explore and build iteratively.

        Returns:
            - list[str]: Configure flags (simple mode)
            - dict: Full result including flags and build status (ReAct mode)
        """
        project_name = project_path.name
        build_dir = Path(self.build_dir) if self.build_dir else project_path / "build"

        # Initialize tools
        file_tools = BuildTools(project_path, build_dir)
        unavoidable_asm_path = self._get_unavoidable_asm_path(project_name)
        actions = AutotoolsActions(
            project_path, self.build_dir, self.container_image,
            unavoidable_asm_path=unavoidable_asm_path, verbose=self.verbose
        )

        configure_exists_msg = "configure script EXISTS" if has_configure else "configure script DOES NOT exist (need autoreconf)"

        system = f"""You are a build configuration expert for autotools projects. Build this project iteratively using the available tools.

**Available Tools:**
- read_file: Read project files (configure.ac, Makefile.am, etc.) and build artifacts
- list_dir: Explore project structure and build directory
- bootstrap: Run bootstrap/autogen.sh script to prepare build system (if project has one)
- autoreconf: Run autoreconf -fi to regenerate configure script (only if configure doesn't exist)
- autotools_configure: Run ./configure with flags
- autotools_build: Run bear -- make to build and capture compile commands
- finish: Signal completion (MUST call this when done instead of calling build tools again)

**IMPORTANT**: All file/directory paths in tools must be RELATIVE to project root (e.g., ".", "src/", "configure.ac"). The build directory is accessible at "build/". Absolute paths are not allowed.

**Build Requirements:**
- Generate compile_commands.json (via bear -- make)
- Use clang-18 with LTO (automatically injected via env vars)
- Prefer static linking (--disable-shared --enable-static)
- Disable SIMD/hardware optimizations to minimize assembly code

**Assembly Verification:**
After each successful autotools_build(), an assembly check runs automatically. The result includes:
- `assembly_check.has_assembly`: true if any assembly code was detected
- `assembly_check.standalone_asm_files`: List of .s/.S/.asm files
- `assembly_check.inline_asm_sources`: List of C/C++ files with inline assembly

**If assembly is detected after a successful build:**
1. Review the findings to understand the source of assembly
2. Look for configure options to disable assembly (e.g., --disable-asm, --disable-simd)
3. Try autotools_configure with new flags and autotools_build again
4. If assembly is unavoidable (e.g., critical code), note it and continue

**CRITICAL - When to STOP:**
- You CANNOT install packages or dependencies
- If missing system dependencies are identified, call finish(status="failure", summary="...") immediately
- Once build succeeds with acceptable assembly results, call finish(status="success", summary="...")
- Do NOT call autotools_build again after a successful build - call finish instead

You are working on: {project_name}
{configure_exists_msg}
If you recognize this project, leverage your knowledge of its typical build requirements.

Goal: Successfully build the project while minimizing assembly code."""

        messages = [{
            "role": "user",
            "content": f"""Build this autotools project: {project_name}

configure.ac:
```
{configure_ac_content[:5000]}
```

Note: All file paths must be relative to the project root. Build directory is at "build/".

Proceed with the build using the available tools."""
        }]

        if self.verbose:
            print("[LLM] Requesting initial configuration (ReAct mode)...")
            print(f"[LLM] Project: {project_name}")
            print("[LLM] Available tools: read_file, list_dir, bootstrap, autoreconf, autotools_configure, autotools_build")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("AUTOTOOLS BUILD-LEARN - INITIAL REQUEST\n")
                f.write(f"Project: {project_name}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"SYSTEM PROMPT:\n{system}\n\n")
                f.write(f"USER MESSAGE:\n{messages[0]['content']}\n\n")
                f.write(f"TOOLS:\n{json.dumps(AUTOTOOLS_TOOL_DEFINITIONS, indent=2)}\n\n")

        # First turn
        response = self.llm.complete_with_tools(
            messages=messages,
            tools=AUTOTOOLS_TOOL_DEFINITIONS,
            system=system,
        )

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("RESPONSE:\n")
                f.write(f"Stop reason: {response.stop_reason}\n")
                for i, block in enumerate(response.content):
                    if hasattr(block, 'text'):
                        f.write(f"[Block {i}] Text: {block.text}\n")
                    elif hasattr(block, 'name'):
                        f.write(f"[Block {i}] Tool use: {block.name}({block.input})\n")
                f.write("\n")

        if response.stop_reason in ("end_turn", "stop"):
            # Simple mode or immediate termination
            if self.verbose:
                print("[Mode] LLM provided config directly or terminated")
            return self._get_default_config()

        elif response.stop_reason == "tool_use":
            # ReAct mode
            if self.verbose:
                print("[Mode] ReAct workflow (LLM exploring with tools)")

            result = self._execute_react_loop(
                messages, response, file_tools, actions, system
            )
            return result

        else:
            if self.verbose:
                print(f"[Mode] Unexpected stop_reason: {response.stop_reason}, using default")
            return self._get_default_config()

    def _execute_react_loop(
        self,
        messages: list,
        initial_response,
        file_tools: BuildTools,
        actions: AutotoolsActions,
        system: str,
    ) -> dict:
        """
        ReAct-style build loop where LLM uses tools iteratively.

        Returns:
            Dict with 'success', 'configure_flags', 'attempts', 'configure_succeeded', 'build_succeeded', 'react_terminated'
        """
        max_turns = MAX_TURNS_AUTOTOOLS
        configure_succeeded = False
        build_succeeded = False
        react_terminated = False
        finished = False  # Set when LLM calls finish tool
        finish_status = None
        finish_summary = None
        final_flags = []
        use_build_dir = True
        seen_reads = {}  # Track file reads for context compression

        response = initial_response

        for turn in range(max_turns):
            if response.stop_reason in ("end_turn", "stop"):
                if not build_succeeded:
                    react_terminated = True
                    if self.verbose:
                        print(f"[ReAct] LLM terminated without successful build after {turn + 1} turns")
                else:
                    if self.verbose:
                        print(f"[ReAct] LLM finished after {turn + 1} turns")
                break

            elif response.stop_reason == "tool_use":
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_content.append({"type": "text", "text": block.text})
                        if self.verbose:
                            print(f"[LLM] {block.text}")

                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                        result = self._execute_tool_safe(
                            file_tools, actions, block.name, block.input
                        )

                        # Track file reads for context compression
                        result = deduplicate_tool_result(
                            block.name, block.input, result, seen_reads, turn
                        )

                        if self.verbose:
                            print(f"[Tool] {block.name}({json.dumps(block.input)})")
                            if "success" in result:
                                print(f"  Success: {result['success']}")
                            if "error" in result and result["error"]:
                                print(f"  Error: {result['error'][:200]}")

                        # Track configure/build status
                        if block.name == "autotools_configure" and result.get("success"):
                            configure_succeeded = True
                            final_flags = block.input.get("configure_flags", [])
                            use_build_dir = block.input.get("use_build_dir", True)
                        if block.name == "autotools_build" and result.get("success"):
                            build_succeeded = True
                        if block.name == "finish":
                            finished = True
                            finish_status = block.input.get("status")
                            finish_summary = block.input.get("summary")
                            if self.verbose:
                                print(f"[ReAct] LLM called finish: status={finish_status}")
                                print(f"[ReAct] Summary: {finish_summary}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

                # Check if LLM called finish tool
                if finished:
                    if self.verbose:
                        print(f"[ReAct] Finished after {turn + 1} turns")
                    break

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"REACT TURN {turn + 1} - TOOL RESULTS\n")
                        f.write(f"{'='*80}\n\n")
                        for tr in tool_results:
                            f.write(f"Tool result for {tr.get('tool_use_id')}:\n")
                            f.write(f"{tr.get('content')}\n\n")

                if turn < max_turns - 1:
                    # Compress stale file reads (replace with forward refs to newer reads)
                    compressed_messages = compress_stale_reads(messages, seen_reads)

                    truncated_messages = truncate_messages(compressed_messages, max_tokens=MAX_CONTEXT_TOKENS)

                    if self.verbose and len(truncated_messages) < len(compressed_messages):
                        print(f"[Context] Truncated {len(compressed_messages) - len(truncated_messages)} messages")

                    response = self.llm.complete_with_tools(
                        messages=truncated_messages,
                        tools=AUTOTOOLS_TOOL_DEFINITIONS,
                        system=system,
                    )

                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write("LLM RESPONSE:\n")
                            f.write(f"Stop reason: {response.stop_reason}\n")
                            for i, block in enumerate(response.content):
                                if hasattr(block, 'text'):
                                    f.write(f"[Block {i}] Text: {block.text}\n")
                                elif hasattr(block, 'name'):
                                    f.write(f"[Block {i}] Tool use: {block.name}({block.input})\n")
                            f.write("\n")
                else:
                    break
            else:
                if self.verbose:
                    print(f"[ReAct] Unexpected stop reason: {response.stop_reason}")
                break

        if not final_flags:
            final_flags = self._get_default_config()

        return {
            "success": build_succeeded,
            "configure_flags": final_flags,
            "attempts": turn + 1,
            "configure_succeeded": configure_succeeded,
            "build_succeeded": build_succeeded,
            "react_terminated": react_terminated,
            "use_build_dir": use_build_dir,
        }

    def _execute_tool_safe(
        self,
        file_tools: BuildTools,
        actions: AutotoolsActions,
        tool_name: str,
        tool_input: dict,
    ) -> dict:
        """Execute tool with security checks."""
        try:
            if tool_name == "read_file":
                return file_tools.read_file(
                    file_path=tool_input.get("file_path"),
                    max_lines=tool_input.get("max_lines", 200),
                    start_line=tool_input.get("start_line", 1),
                )
            elif tool_name == "list_dir":
                return file_tools.list_dir(
                    dir_path=tool_input.get("dir_path", "."),
                    pattern=tool_input.get("pattern"),
                )
            elif tool_name == "bootstrap":
                return actions.bootstrap(
                    script_path=tool_input.get("script_path", "bootstrap"),
                )
            elif tool_name == "autoreconf":
                return actions.autoreconf()
            elif tool_name == "autotools_configure":
                return actions.autotools_configure(
                    configure_flags=tool_input.get("configure_flags", []),
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "autotools_build":
                return actions.autotools_build(
                    make_target=tool_input.get("make_target", ""),
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "autotools_clean":
                return actions.autotools_clean(
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "autotools_distclean":
                return actions.autotools_distclean(
                    use_build_dir=tool_input.get("use_build_dir", True),
                )
            elif tool_name == "finish":
                # Finish tool is handled in the ReAct loop, just return acknowledgement
                return {
                    "acknowledged": True,
                    "status": tool_input.get("status"),
                    "summary": tool_input.get("summary"),
                }
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except ValueError as e:
            return {"error": f"Security error: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool error: {str(e)}"}

    def _get_default_config(self) -> list[str]:
        """Get default configure flags (fallback if LLM fails)."""
        flags = []

        if self.prefer_static:
            flags.extend(["--disable-shared", "--enable-static"])

        return flags

    def extract_compile_commands(self, project_path: Path, output_dir: Path | None = None, use_build_dir: bool = True) -> Path:
        """
        Extract compile_commands.json from build directory to output directory.
        Fixes Docker container paths to host paths.

        Args:
            project_path: Path to the project source
            output_dir: Directory to save compile_commands.json (default: build-scripts/<project>/)
            use_build_dir: Whether out-of-source build was used

        Returns the path to the extracted file.
        """
        if use_build_dir:
            if self.build_dir:
                build_dir = Path(self.build_dir).resolve()
            else:
                build_dir = project_path / "build"
            compile_commands_src = build_dir / "compile_commands.json"
        else:
            build_dir = project_path
            compile_commands_src = project_path / "compile_commands.json"

        if output_dir is None:
            output_dir = Path("build-scripts") / project_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

        compile_commands_dst = output_dir / "compile_commands.json"

        if not compile_commands_src.exists():
            raise FileNotFoundError(
                f"compile_commands.json not found in {compile_commands_src.parent}"
            )

        with open(compile_commands_src, "r") as f:
            compile_commands = json.load(f)

        # Replace Docker container paths with host paths
        for entry in compile_commands:
            if "file" in entry:
                entry["file"] = entry["file"].replace(DOCKER_WORKSPACE_SRC, str(project_path))
                entry["file"] = entry["file"].replace(DOCKER_WORKSPACE_BUILD, str(build_dir))
            if "directory" in entry:
                entry["directory"] = entry["directory"].replace(DOCKER_WORKSPACE_BUILD, str(build_dir))
                entry["directory"] = entry["directory"].replace(DOCKER_WORKSPACE_SRC, str(project_path))
            if "command" in entry:
                entry["command"] = entry["command"].replace(DOCKER_WORKSPACE_SRC, str(project_path))
                entry["command"] = entry["command"].replace(DOCKER_WORKSPACE_BUILD, str(build_dir))
            if "arguments" in entry:
                entry["arguments"] = [
                    arg.replace(DOCKER_WORKSPACE_SRC, str(project_path)).replace(DOCKER_WORKSPACE_BUILD, str(build_dir))
                    for arg in entry["arguments"]
                ]

        with open(compile_commands_dst, "w") as f:
            json.dump(compile_commands, f, indent=2)

        if self.verbose:
            print(f"[Extract] Copied and fixed compile_commands.json to {compile_commands_dst}")
            print(f"[Extract] Fixed Docker paths: {DOCKER_WORKSPACE_SRC} -> {project_path}")
            print(f"[Extract] Fixed Docker paths: {DOCKER_WORKSPACE_BUILD} -> {build_dir}")

        return compile_commands_dst
