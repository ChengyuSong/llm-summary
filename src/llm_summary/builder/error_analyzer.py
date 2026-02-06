"""Error analysis and retry logic for build failures."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..llm.base import LLMBackend
from .constants import MAX_TURNS_ERROR_ANALYSIS
from .json_utils import parse_llm_json
from .prompts import BUILD_FAILURE_PROMPT, ERROR_ANALYSIS_PROMPT

if TYPE_CHECKING:
    from .tools import BuildTools


class BuildError(Exception):
    """Exception raised when a build fails."""

    pass


class ErrorAnalyzer:
    """Analyzes build errors and suggests fixes using LLM."""

    def __init__(self, llm: LLMBackend, verbose: bool = False, log_file: str | None = None):
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file

    def analyze_cmake_error(
        self,
        error_output: str,
        current_flags: list[str],
        project_path: Path,
        cmakelists_content: str | None = None,
    ) -> dict:
        """
        Analyze a CMake configuration error and suggest fixes.

        Returns a dict with:
        - diagnosis: str
        - suggested_flags: list[str]
        - missing_dependencies: list[str]
        - confidence: str
        """
        # Extract relevant excerpt from CMakeLists.txt if available
        cmakelists_excerpt = ""
        if cmakelists_content:
            # Take first 100 lines or full content if shorter
            lines = cmakelists_content.split("\n")
            cmakelists_excerpt = "\n".join(lines[:100])

        prompt = ERROR_ANALYSIS_PROMPT.format(
            current_flags="\n".join(current_flags),
            error_output=error_output,
            cmakelists_excerpt=cmakelists_excerpt,
        )

        if self.verbose:
            print("\n[LLM] Analyzing CMake error...")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("CMAKE ERROR ANALYSIS\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"PROMPT:\n{prompt}\n\n")

        response = self.llm.complete(prompt)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"RESPONSE:\n{response}\n\n")

        if self.verbose:
            print(f"[LLM] Response: {response[:500]}...")

        result = parse_llm_json(
            response,
            default_response={
                "diagnosis": "Failed to parse LLM response",
                "suggested_flags": [],
                "missing_dependencies": [],
                "confidence": "low",
            },
            verbose=self.verbose,
        )
        return {
            "diagnosis": result.get("diagnosis", "Unknown error"),
            "suggested_flags": result.get("suggested_flags", []),
            "missing_dependencies": result.get("missing_dependencies", []),
            "confidence": result.get("confidence", "low"),
        }

    def analyze_build_error(
        self,
        error_output: str,
        current_flags: list[str],
    ) -> dict:
        """
        Analyze a compilation/build error and suggest fixes.

        Returns a dict with:
        - diagnosis: str
        - suggested_flags: list[str]
        - compiler_flag_changes: dict[str, str]
        - confidence: str
        - notes: str
        """
        prompt = BUILD_FAILURE_PROMPT.format(
            current_flags="\n".join(current_flags),
            error_output=error_output,
        )

        if self.verbose:
            print("\n[LLM] Analyzing build error...")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("BUILD ERROR ANALYSIS\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"PROMPT:\n{prompt}\n\n")

        response = self.llm.complete(prompt)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"RESPONSE:\n{response}\n\n")

        if self.verbose:
            print(f"[LLM] Response: {response[:500]}...")

        result = parse_llm_json(
            response,
            default_response={
                "diagnosis": "Failed to parse LLM response",
                "suggested_flags": [],
                "compiler_flag_changes": {},
                "confidence": "low",
                "notes": "",
            },
            verbose=self.verbose,
        )
        return {
            "diagnosis": result.get("diagnosis", "Unknown build error"),
            "suggested_flags": result.get("suggested_flags", []),
            "compiler_flag_changes": result.get("compiler_flag_changes", {}),
            "confidence": result.get("confidence", "low"),
            "notes": result.get("notes", ""),
        }

    def analyze_error_with_tools(
        self,
        error_output: str,
        current_flags: list[str],
        project_path: Path,
        build_tools: "BuildTools",
        is_cmake_error: bool = True,
    ) -> dict:
        """
        Analyze an error using ReAct approach with tools.

        The model can explore files, read specific sections, and iteratively
        investigate the error before suggesting fixes.

        Returns a dict with:
        - diagnosis: str
        - suggested_flags: list[str]
        - missing_dependencies: list[str]
        - confidence: str
        """
        # Check if backend supports tools
        if not hasattr(self.llm, "complete_with_tools"):
            # Fall back to simple analysis
            if is_cmake_error:
                return self.analyze_cmake_error(
                    error_output, current_flags, project_path, None
                )
            else:
                return self.analyze_build_error(error_output, current_flags)

        # Tool definitions for error analysis
        from .tool_definitions import TOOL_DEFINITIONS_READ_ONLY

        error_type = "CMake configuration" if is_cmake_error else "build/compilation"

        system = f"""You are a build error diagnostic expert. Analyze the {error_type} error and suggest fixes.

You have tools to explore the project:
- read_file: Read any project file (CMakeLists.txt, source files, config files, etc.) or build artifact
- list_dir: Explore project structure and build directory

**IMPORTANT**: All file/directory paths must be RELATIVE to project root (e.g., ".", "CMakeLists.txt", "cmake/FindZLIB.cmake", "build/compile_commands.json"). The build directory is accessible at "build/".

**CRITICAL - No install tool available**:
- You CANNOT install packages or dependencies
- If the error is due to missing system dependencies (libraries, headers, packages), you MUST stop and report this
- Dependencies can only be fixed by updating the Docker image, which you cannot do

Context:
- Build runs in Docker container (source at /workspace/src, build at /workspace/build)
- Error messages reference container paths
- Using Clang 18, LTO enabled, static linking preferred

Your task:
1. Read the error output carefully
2. Use tools to investigate (read relevant CMakeLists.txt sections, config files, etc.)
3. Identify the root cause
4. If the issue can be fixed with CMake flags → suggest them
5. If the issue requires missing dependencies → report them and STOP (no further investigation needed)

When done investigating, return ONLY valid JSON:
{{
  "diagnosis": "Brief description of the problem",
  "suggested_flags": ["-DFLAG=VALUE", ...],
  "missing_dependencies": ["package-name", ...],
  "confidence": "high|medium|low"
}}

If missing_dependencies is non-empty, the error CANNOT be fixed with available tools."""

        messages = [{
            "role": "user",
            "content": f"""A {error_type} error occurred. Please investigate and suggest fixes.

Current CMake flags:
{chr(10).join(current_flags)}

Error output:
{error_output}

Use the available tools to explore the project and understand the error.
When ready, return your analysis as JSON."""
        }]

        if self.verbose:
            print("[LLM] Starting ReAct error analysis...")
            print(f"[LLM] Error type: {error_type}")
            print("[LLM] Available tools: read_file, list_dir")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"REACT ERROR ANALYSIS - {error_type.upper()}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"SYSTEM PROMPT:\n{system}\n\n")
                f.write(f"USER MESSAGE:\n{messages[0]['content']}\n\n")

        # ReAct loop
        max_turns = MAX_TURNS_ERROR_ANALYSIS
        for turn in range(max_turns):
            # Log the request
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(f"\nTURN {turn + 1} REQUEST:\n")
                    f.write(f"Messages count: {len(messages)}\n")
                    # Estimate tokens (rough: 4 chars per token)
                    msg_text = str(messages)
                    estimated_tokens = len(msg_text) // 4
                    f.write(f"Estimated tokens: ~{estimated_tokens}\n")
                    f.write(f"System prompt: {len(system)} chars\n\n")

            response = self.llm.complete_with_tools(
                messages=messages,
                tools=TOOL_DEFINITIONS_READ_ONLY,
                system=system,
            )

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(f"TURN {turn + 1} RESPONSE:\n")
                    f.write(f"Stop reason: {response.stop_reason}\n")
                    for i, block in enumerate(response.content):
                        if hasattr(block, 'text'):
                            f.write(f"[Block {i}] Text: {block.text}\n")
                        elif hasattr(block, 'name'):
                            f.write(f"[Block {i}] Tool: {block.name}({block.input})\n")
                    f.write("\n")

            # Check if done
            if response.stop_reason in ("end_turn", "stop"):
                # Extract text and try to parse JSON
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text

                if self.verbose:
                    print(f"[LLM] Analysis complete after {turn + 1} turns")

                return self._parse_analysis_json(text)

            elif response.stop_reason == "tool_use":
                # Execute tools
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if hasattr(block, "text") and block.type == "text":
                        text_entry = {"type": "text", "text": block.text}
                        if getattr(block, "thought", False):
                            text_entry["thought"] = True
                        if getattr(block, "thought_signature", None):
                            text_entry["thought_signature"] = block.thought_signature
                        assistant_content.append(text_entry)
                        if self.verbose and not getattr(block, "thought", False):
                            print(f"[LLM] {block.text}")

                    elif block.type == "tool_use":
                        tool_use_entry = {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        if getattr(block, "thought_signature", None):
                            tool_use_entry["thought_signature"] = block.thought_signature
                        assistant_content.append(tool_use_entry)

                        # Execute tool
                        result = self._execute_tool_safe(
                            build_tools, block.name, block.input
                        )

                        if self.verbose:
                            print(f"[Tool] {block.name}({json.dumps(block.input)})")
                            if "error" in result and result["error"]:
                                print(f"  Error: {result['error'][:200]}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

            else:
                if self.verbose:
                    print(f"[LLM] Unexpected stop reason: {response.stop_reason}")
                break

        # Max turns reached without JSON response
        if self.verbose:
            print("[LLM] Max turns reached without analysis")

        return {
            "diagnosis": "Error analysis did not complete",
            "suggested_flags": [],
            "missing_dependencies": [],
            "confidence": "low",
        }

    def _execute_tool_safe(
        self, build_tools: "BuildTools", tool_name: str, tool_input: dict
    ) -> dict:
        """Execute tool with security checks."""
        try:
            if tool_name == "read_file":
                return build_tools.read_file(
                    file_path=tool_input.get("file_path"),
                    max_lines=tool_input.get("max_lines", 200),
                    start_line=tool_input.get("start_line", 1),
                )
            elif tool_name == "list_dir":
                return build_tools.list_dir(
                    dir_path=tool_input.get("dir_path", "."),
                    pattern=tool_input.get("pattern"),
                )
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except ValueError as e:
            # Security violation
            return {"error": f"Security error: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool error: {str(e)}"}

    def _parse_analysis_json(self, text: str) -> dict:
        """Parse analysis JSON from LLM response."""
        result = parse_llm_json(
            text,
            default_response={
                "diagnosis": "Failed to parse LLM analysis",
                "suggested_flags": [],
                "missing_dependencies": [],
                "confidence": "low",
            },
            verbose=self.verbose,
        )
        return {
            "diagnosis": result.get("diagnosis", "Unknown error"),
            "suggested_flags": result.get("suggested_flags", []),
            "missing_dependencies": result.get("missing_dependencies", []),
            "confidence": result.get("confidence", "low"),
        }
