"""CMake builder with LLM-powered incremental learning."""

import json
import shutil
from pathlib import Path
from typing import Any

from ..llm.base import LLMBackend
from .actions import CMakeActions
from .error_analyzer import BuildError, ErrorAnalyzer
from .prompts import INITIAL_CONFIG_PROMPT
from .tool_definitions import ALL_TOOL_DEFINITIONS
from .tools import BuildTools


class CMakeBuilder:
    """Build CMake projects with incremental learning from failures."""

    def __init__(
        self,
        llm: LLMBackend,
        container_image: str = "llm-summary-builder:latest",
        build_dir: Path | None = None,
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
        self.max_retries = max_retries
        self.enable_lto = enable_lto
        self.prefer_static = prefer_static
        self.generate_ir = generate_ir
        self.verbose = verbose
        self.log_file = log_file
        self.error_analyzer = ErrorAnalyzer(llm, verbose, log_file=log_file)

    def learn_and_build(
        self, project_path: Path
    ) -> dict[str, Any]:
        """
        Learn how to build a CMake project through iterative attempts.

        Returns a dict with:
        - success: bool
        - cmake_flags: list[str]
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

        # Read CMakeLists.txt for LLM analysis
        cmakelists_path = project_path / "CMakeLists.txt"
        if not cmakelists_path.exists():
            raise ValueError(f"CMakeLists.txt not found in {project_path}")

        cmakelists_content = cmakelists_path.read_text()

        # Get initial configuration from LLM
        if self.verbose:
            print("\n[1/3] Analyzing CMakeLists.txt with LLM...")

        result = self._get_initial_config(project_path, cmakelists_content)

        # Handle both list[str] (simple mode) and dict (ReAct mode with metadata)
        if isinstance(result, dict):
            cmake_flags = result.get("cmake_flags", self._get_default_config())
            react_build_succeeded = result.get("build_succeeded", False)

            if react_build_succeeded:
                if self.verbose:
                    print("\n[ReAct] Build already succeeded in exploration phase!")
                    print(f"[3/3] Build successful after {result.get('attempts', 1)} turns!")

                return {
                    "success": True,
                    "cmake_flags": cmake_flags,
                    "attempts": 1,  # Count as single attempt
                    "build_log": "Build succeeded during ReAct exploration",
                    "error_messages": [],
                }
        else:
            cmake_flags = result
            react_build_succeeded = False

        if self.verbose:
            print(f"[Initial Config] {len(cmake_flags)} flags:")
            for flag in cmake_flags:
                print(f"  {flag}")

        # Attempt build with retries
        build_log = []
        error_messages = []
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            if self.verbose:
                print(f"\n[2/3] Build attempt {attempt}/{self.max_retries}...")

            try:
                log = self._attempt_build(project_path, cmake_flags)
                build_log.append(log)

                # Success!
                if self.verbose:
                    print(f"\n[3/3] Build successful after {attempt} attempts!")

                return {
                    "success": True,
                    "cmake_flags": cmake_flags,
                    "attempts": attempts,
                    "build_log": "\n\n".join(build_log),
                    "error_messages": error_messages,
                }

            except BuildError as e:
                error_msg = str(e)
                error_messages.append(error_msg)
                build_log.append(f"Attempt {attempt} failed:\n{error_msg}")

                if self.verbose:
                    print(f"[ERROR] Attempt {attempt} failed:")
                    print(f"  {error_msg[:500]}...")

                # If this was the last retry, give up
                if attempt >= self.max_retries:
                    if self.verbose:
                        print(f"\n[FAILED] Max retries ({self.max_retries}) reached.")
                    break

                # Analyze error and adjust configuration
                if self.verbose:
                    print("\n[LLM] Analyzing error to adjust configuration...")

                # Determine if this is a CMake config error or build error
                is_cmake_error = "CMake Error" in error_msg or "cmake" in error_msg.lower()

                # Use ReAct-enabled error analysis if backend supports tools
                if hasattr(self.llm, "complete_with_tools"):
                    file_tools = BuildTools(project_path)
                    analysis = self.error_analyzer.analyze_error_with_tools(
                        error_msg,
                        cmake_flags,
                        project_path,
                        file_tools,
                        is_cmake_error=is_cmake_error,
                    )
                else:
                    # Fall back to simple analysis
                    if is_cmake_error:
                        analysis = self.error_analyzer.analyze_cmake_error(
                            error_msg,
                            cmake_flags,
                            project_path,
                            cmakelists_content,
                        )
                    else:
                        analysis = self.error_analyzer.analyze_build_error(
                            error_msg,
                            cmake_flags,
                        )

                if self.verbose:
                    print(f"[Analysis] {analysis['diagnosis']}")
                    print(f"[Confidence] {analysis['confidence']}")

                # Apply suggested changes
                cmake_flags = self._apply_suggestions(cmake_flags, analysis)

                if self.verbose:
                    print(f"[Updated Config] {len(cmake_flags)} flags")

        # All retries exhausted
        return {
            "success": False,
            "cmake_flags": cmake_flags,
            "attempts": attempts,
            "build_log": "\n\n".join(build_log),
            "error_messages": error_messages,
        }

    def _get_initial_config(
        self, project_path: Path, cmakelists_content: str
    ) -> list[str]:
        """Get initial CMake configuration using LLM analysis with tool support."""
        # Check if backend supports tool use
        if hasattr(self.llm, "complete_with_tools"):
            return self._get_initial_config_with_tools(project_path, cmakelists_content)
        else:
            return self._get_initial_config_simple(project_path, cmakelists_content)

    def _get_initial_config_with_tools(
        self, project_path: Path, cmakelists_content: str
    ) -> list[str] | dict:
        """
        Hybrid approach: Allow LLM to choose between simple (JSON) or ReAct (tools) mode.

        Returns:
            - list[str]: CMake flags (simple mode)
            - dict: Full result including flags and build status (ReAct mode)
        """
        # Extract project name as hint for the LLM
        project_name = project_path.name

        # Initialize tools
        file_tools = BuildTools(project_path)
        actions = CMakeActions(
            project_path, self.build_dir, self.container_image, self.verbose
        )

        # System prompt offering both options
        system = f"""You are a build configuration expert. You can build this CMake project in two ways:

**Option 1 - Simple (Recommended for straightforward projects):**
If the CMakeLists.txt is straightforward and standard, immediately return JSON configuration:
{{
  "cmake_flags": ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ...],
  "reasoning": "...",
  "dependencies": [...]
}}

**Option 2 - ReAct (For complex projects needing exploration):**
Use available tools to explore and build iteratively:
- read_file: Read project files (CMake modules, configs, etc.)
- list_dir: Explore project structure
- cmake_configure: Run cmake configure with specific flags
- cmake_build: Run ninja build after successful configure

**IMPORTANT**: All file/directory paths in tools must be RELATIVE to project root (e.g., ".", "cmake/", "src/config.h"). Absolute paths are not allowed.

Requirements (both modes):
- Generate compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Enable LLVM LTO (CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
- Use Clang 18 (CMAKE_C_COMPILER=clang-18, CMAKE_CXX_COMPILER=clang++-18)
- Prefer static linking (BUILD_SHARED_LIBS=OFF)
- Disable SIMD/hardware optimizations to minimize assembly code
- Enable LLVM IR generation (CMAKE_C_FLAGS=-flto=full -save-temps=obj)

You are working on the project: {project_name}
If you recognize this project, leverage your knowledge of its typical build requirements.

Choose based on project complexity. Standard CMakeLists.txt → Option 1. Complex/unusual builds → Option 2."""

        # Initial user message
        messages = [{
            "role": "user",
            "content": f"""Build this CMake project: {project_name}

CMakeLists.txt:
```
{cmakelists_content[:5000]}
```

If you recognize this project ({project_name}), use your knowledge to inform the build configuration.

Note: If using tools, all file paths must be relative to the project root (e.g., "." for root directory, "cmake/FindZLIB.cmake" for a file in the cmake subdirectory).

Choose your approach (simple JSON or ReAct with tools) and proceed."""
        }]

        if self.verbose:
            print("[LLM] Requesting initial configuration (hybrid mode)...")
            print(f"[LLM] Project: {project_name}")
            print("[LLM] Available tools: read_file, list_dir, cmake_configure, cmake_build")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("BUILD-LEARN HYBRID MODE - INITIAL REQUEST\n")
                f.write(f"Project: {project_name}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"SYSTEM PROMPT:\n{system}\n\n")
                f.write(f"USER MESSAGE:\n{messages[0]['content']}\n\n")
                import json as json_module
                f.write(f"TOOLS:\n{json_module.dumps(ALL_TOOL_DEFINITIONS, indent=2)}\n\n")

        # First turn: LLM decides mode
        response = self.llm.complete_with_tools(
            messages=messages,
            tools=ALL_TOOL_DEFINITIONS,
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

        # OpenAI format returns "stop", Anthropic returns "end_turn"
        if response.stop_reason in ("end_turn", "stop"):
            # Simple mode: LLM returned text (likely JSON config)
            if self.verbose:
                print("[Mode] Simple workflow (LLM provided config directly)")

            text = self._extract_text(response)
            config = self._parse_config_response(text)

            if config:
                return config
            else:
                # Failed to parse, use default
                if self.verbose:
                    print("[Mode] Failed to parse JSON, using default config")
                return self._get_default_config()

        elif response.stop_reason == "tool_use":
            # ReAct mode: LLM chose to use tools
            if self.verbose:
                print("[Mode] ReAct workflow (LLM exploring with tools)")

            result = self._execute_react_loop(
                messages, response, file_tools, actions, system
            )

            # Return the full result dict (includes build status)
            return result

        else:
            # Unexpected
            if self.verbose:
                print(f"[Mode] Unexpected stop_reason: {response.stop_reason}, using default")
            return self._get_default_config()

    def _get_initial_config_simple(
        self, project_path: Path, cmakelists_content: str
    ) -> list[str]:
        """Get initial config without tool support (fallback)."""
        project_name = project_path.name

        prompt = INITIAL_CONFIG_PROMPT.format(
            project_name=project_name,
            cmakelists_content=cmakelists_content,
        )

        if self.verbose:
            print("[LLM] Requesting initial configuration...")
            print(f"[LLM] Project: {project_name}")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write("BUILD-LEARN INITIAL CONFIG REQUEST\n")
                f.write(f"Project: {project_name}\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"PROMPT:\n{prompt}\n\n")

        response = self.llm.complete(prompt)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"RESPONSE:\n{response}\n\n")

        if self.verbose:
            print(f"[LLM] Response length: {len(response)} chars")

        return self._parse_config_response(response)

    def _parse_config_response(self, response: str) -> list[str]:
        """Parse LLM response to extract CMake configuration."""
        try:
            # Try to parse JSON response, stripping markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]  # Remove ```json
            if json_str.startswith("```"):
                json_str = json_str[3:]  # Remove ```
            if json_str.endswith("```"):
                json_str = json_str[:-3]  # Remove trailing ```
            json_str = json_str.strip()

            result = json.loads(json_str)
            flags = result.get("cmake_flags", [])

            if self.verbose and result.get("reasoning"):
                print(f"[LLM Reasoning] {result['reasoning'][:200]}...")

            if result.get("dependencies"):
                print(f"[Dependencies] LLM identified {len(result['dependencies'])} dependencies:")
                for dep in result["dependencies"]:
                    # Handle both string format ["zlib"] and dict format [{"package": "zlib", "reason": "..."}]
                    if isinstance(dep, str):
                        print(f"  - {dep}")
                    elif isinstance(dep, dict):
                        pkg = dep.get("package", "unknown")
                        reason = dep.get("reason", "")
                        print(f"  - {pkg}: {reason}")
                    else:
                        print(f"  - {dep}")  # Fallback for unexpected format
                print("[Note] Ensure these are installed in the Docker image or build may fail")

            if self.verbose and result.get("potential_issues"):
                print("[Potential Issues]")
                for issue in result["potential_issues"]:
                    print(f"  - {issue}")

            return flags
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[ERROR] Failed to parse LLM response: {e}")
                print(f"[ERROR] Response: {response[:500]}...")

            # Fall back to default configuration
            return self._get_default_config()

    def _extract_text(self, response) -> str:
        """Extract all text blocks from response."""
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (4 chars per token)."""
        return len(text) // 4

    def _filter_warnings(self, output: str) -> str:
        """Filter out warnings, keep only errors when output is large."""
        lines = output.split('\n')

        # If output is small, keep everything
        if len(output) < 10000:
            return output

        # Otherwise, keep only error lines and some context
        error_keywords = ['error:', 'Error:', 'ERROR:', 'failed', 'Failed', 'FAILED']
        error_ranges = []

        for i, line in enumerate(lines):
            # Keep errors
            if any(kw in line for kw in error_keywords):
                # Include 2 lines before and after for context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                error_ranges.append((start, end))

        if not error_ranges:
            # No errors found, keep first 20 and last 30 lines
            if len(lines) <= 50:
                return output
            return '\n'.join(lines[:20] + ['...'] + lines[-30:])

        # Merge overlapping ranges
        merged_ranges = []
        error_ranges.sort()
        current_start, current_end = error_ranges[0]

        for start, end in error_ranges[1:]:
            if start <= current_end:
                # Overlapping, merge
                current_end = max(current_end, end)
            else:
                # Non-overlapping, save current and start new
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end

        merged_ranges.append((current_start, current_end))

        # Extract lines from merged ranges
        filtered = []
        for start, end in merged_ranges:
            filtered.extend(lines[start:end])
            filtered.append('---')

        return '\n'.join(filtered)

    def _deduplicate_tool_result(self, tool_name: str, tool_input: dict, result: dict, seen_tools: set) -> dict:
        """Deduplicate tool results to avoid sending same content repeatedly."""
        # Create key for deduplication
        if tool_name == "read_file":
            key = f"read:{tool_input.get('file_path')}:{tool_input.get('start_line', 1)}"
            if key in seen_tools:
                # Already read this file, return short reference
                return {
                    "cached": True,
                    "message": f"File {tool_input.get('file_path')} already read (see earlier output)"
                }
            seen_tools.add(key)

        # Filter large outputs
        if "output" in result and isinstance(result["output"], str):
            if self._estimate_tokens(result["output"]) > 1000:
                result = result.copy()
                result["output"] = self._filter_warnings(result["output"])

        if "content" in result and isinstance(result["content"], str):
            if self._estimate_tokens(result["content"]) > 1000:
                result = result.copy()
                result["content"] = self._filter_warnings(result["content"])

        return result

    def _estimate_messages_tokens(self, messages: list) -> int:
        """Estimate total tokens in messages array."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._estimate_tokens(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        total += self._estimate_tokens(json.dumps(item))
                    elif isinstance(item, str):
                        total += self._estimate_tokens(item)
        return total

    def _truncate_messages(self, messages: list, max_tokens: int = 100000) -> list:
        """Truncate old messages to stay under token limit."""
        current_tokens = self._estimate_messages_tokens(messages)

        if current_tokens <= max_tokens:
            return messages

        # Keep first message (initial request) and recent messages
        if len(messages) <= 3:
            return messages

        # Keep first message and last N messages
        truncated = [messages[0]]

        # Add recent messages from the end
        for msg in reversed(messages[1:]):
            msg_tokens = self._estimate_tokens(json.dumps(msg.get("content", "")))
            if current_tokens - msg_tokens < max_tokens:
                truncated.insert(1, msg)
            else:
                current_tokens -= msg_tokens

        if len(truncated) < len(messages):
            # Add a marker that we truncated
            truncated.insert(1, {
                "role": "user",
                "content": f"[{len(messages) - len(truncated)} earlier messages truncated to save context]"
            })

        return truncated

    def _execute_react_loop(
        self,
        messages: list,
        initial_response,
        file_tools: BuildTools,
        actions: CMakeActions,
        system: str,
    ) -> dict:
        """
        ReAct-style build loop where LLM uses tools iteratively.

        Returns:
            Dict with 'success', 'cmake_flags', 'attempts', 'configure_succeeded', 'build_succeeded'
        """
        # Track state
        max_turns = 15
        configure_succeeded = False
        build_succeeded = False
        final_flags = []
        seen_tools = set()  # For deduplication

        # Process initial response (already has tool_use)
        response = initial_response

        for turn in range(max_turns):
            # OpenAI format returns "stop", Anthropic returns "end_turn"
            if response.stop_reason in ("end_turn", "stop"):
                # LLM finished - extract final flags if available
                if self.verbose:
                    print(f"[ReAct] LLM finished after {turn + 1} turns")
                break

            elif response.stop_reason == "tool_use":
                # Execute tools
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

                        # Execute tool with security validation
                        result = self._execute_tool_safe(
                            file_tools, actions, block.name, block.input
                        )

                        # Apply deduplication and filtering
                        result = self._deduplicate_tool_result(
                            block.name, block.input, result, seen_tools
                        )

                        if self.verbose:
                            print(f"[Tool] {block.name}({json.dumps(block.input)})")
                            if "success" in result:
                                print(f"  Success: {result['success']}")
                            if "error" in result and result["error"]:
                                print(f"  Error: {result['error'][:200]}")
                            if "cached" in result:
                                print(f"  Cached: {result['message']}")

                        # Track configure/build status
                        if block.name == "cmake_configure" and result.get("success"):
                            configure_succeeded = True
                            final_flags = block.input.get("cmake_flags", [])
                        if block.name == "cmake_build" and result.get("success"):
                            build_succeeded = True

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

                if self.log_file:
                    with open(self.log_file, "a") as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"REACT TURN {turn + 1} - TOOL RESULTS\n")
                        f.write(f"{'='*80}\n\n")
                        for tr in tool_results:
                            f.write(f"Tool result for {tr.get('tool_use_id')}:\n")
                            f.write(f"{tr.get('content')}\n\n")

                # Get next response
                if turn < max_turns - 1:  # Don't make another call on last turn
                    # Truncate messages to stay under context limit (100k tokens ~= 128k with system prompt)
                    truncated_messages = self._truncate_messages(messages, max_tokens=100000)

                    if self.verbose and len(truncated_messages) < len(messages):
                        print(f"[Context] Truncated {len(messages) - len(truncated_messages)} messages")
                        print(f"[Context] Estimated tokens: {self._estimate_messages_tokens(truncated_messages)}")

                    # Log the request being sent
                    if self.log_file:
                        with open(self.log_file, "a") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"REACT TURN {turn + 1} - LLM REQUEST\n")
                            f.write(f"{'='*80}\n\n")
                            f.write(f"Messages count: {len(truncated_messages)}\n")
                            f.write(f"Estimated tokens: {self._estimate_messages_tokens(truncated_messages)}\n")
                            f.write(f"System prompt length: {len(system)} chars\n")
                            f.write(f"Tools: {len(ALL_TOOL_DEFINITIONS)} available\n\n")
                            f.write("MESSAGES:\n")
                            for i, msg in enumerate(truncated_messages):
                                f.write(f"[Message {i}] Role: {msg.get('role')}\n")
                                content = msg.get('content', '')
                                if isinstance(content, str):
                                    preview = content[:200] + '...' if len(content) > 200 else content
                                    f.write(f"  Content (preview): {preview}\n")
                                elif isinstance(content, list):
                                    f.write(f"  Content blocks: {len(content)}\n")
                                    for j, block in enumerate(content[:3]):  # Show first 3 blocks
                                        if isinstance(block, dict):
                                            block_type = block.get('type', 'unknown')
                                            f.write(f"    Block {j}: type={block_type}\n")
                            f.write("\n")

                    response = self.llm.complete_with_tools(
                        messages=truncated_messages,
                        tools=ALL_TOOL_DEFINITIONS,
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
                # Unexpected stop reason
                if self.verbose:
                    print(f"[ReAct] Unexpected stop reason: {response.stop_reason}")
                break

        # If we don't have final_flags, use default
        if not final_flags:
            final_flags = self._get_default_config()

        return {
            "success": build_succeeded,
            "cmake_flags": final_flags,
            "attempts": turn + 1,
            "configure_succeeded": configure_succeeded,
            "build_succeeded": build_succeeded,
        }

    def _execute_tool_safe(
        self,
        file_tools: BuildTools,
        actions: CMakeActions,
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
            elif tool_name == "cmake_configure":
                return actions.cmake_configure(
                    cmake_flags=tool_input.get("cmake_flags", [])
                )
            elif tool_name == "cmake_build":
                return actions.cmake_build()
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except ValueError as e:
            # Security violation
            return {"error": f"Security error: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool error: {str(e)}"}

    def _get_default_config(self) -> list[str]:
        """Get default CMake configuration (fallback if LLM fails)."""
        flags = [
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_C_COMPILER=clang-18",
            "-DCMAKE_CXX_COMPILER=clang++-18",
        ]

        if self.enable_lto:
            # Note: Compiler flags with spaces need to be handled carefully
            # The Docker bash -c command will handle the quoting
            if self.generate_ir:
                flags.extend([
                    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
                    "-DCMAKE_C_FLAGS=-flto=full -save-temps=obj",
                    "-DCMAKE_CXX_FLAGS=-flto=full -save-temps=obj",
                ])
            else:
                flags.extend([
                    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
                    "-DCMAKE_C_FLAGS=-flto=full",
                    "-DCMAKE_CXX_FLAGS=-flto=full",
                ])

        if self.prefer_static:
            flags.append("-DBUILD_SHARED_LIBS=OFF")

        return flags

    def _attempt_build(self, project_path: Path, cmake_flags: list[str]) -> str:
        """
        Attempt to build the project with given CMake flags.

        Uses two-phase approach: configure then build.
        Raises BuildError if either phase fails.
        Returns combined log on success.
        """
        # Use custom build_dir if provided, otherwise default to project_path/build
        if self.build_dir:
            build_dir = Path(self.build_dir).resolve()
        else:
            build_dir = project_path / "build"

        # Create CMakeActions instance for two-phase build
        actions = CMakeActions(project_path, build_dir, self.container_image, self.verbose)

        # Phase 1: Configure
        if self.verbose:
            print(f"[Docker] Build directory: {build_dir}")
            print("[Phase 1/2] Running CMake configure...")

        configure_result = actions.cmake_configure(cmake_flags)

        if not configure_result["success"]:
            # Configure failed - raise error with clear message
            error_msg = configure_result.get("error", "Unknown configure error")
            output = configure_result.get("output", "")
            raise BuildError(f"{error_msg}\n{output}")

        configure_output = configure_result["output"]

        # Phase 2: Build
        if self.verbose:
            print("[Phase 2/2] Running ninja build...")

        build_result = actions.cmake_build()

        if not build_result["success"]:
            # Build failed - raise error with clear message
            error_msg = build_result.get("error", "Unknown build error")
            output = build_result.get("output", "")
            raise BuildError(f"{error_msg}\n{output}")

        build_output = build_result["output"]

        # Combine outputs
        return f"=== Configure Output ===\n{configure_output}\n\n=== Build Output ===\n{build_output}"

    def _apply_suggestions(
        self, current_flags: list[str], analysis: dict
    ) -> list[str]:
        """Apply LLM suggestions to the current configuration."""
        # Start with current flags
        flags = current_flags.copy()

        # Add suggested flags
        for suggested_flag in analysis.get("suggested_flags", []):
            # If this flag overrides an existing one (same -D key), remove the old one
            if suggested_flag.startswith("-D"):
                key = suggested_flag.split("=")[0]
                flags = [f for f in flags if not f.startswith(key)]

            flags.append(suggested_flag)

        # Apply compiler flag changes
        for cmake_var, new_value in analysis.get("compiler_flag_changes", {}).items():
            # Remove existing setting
            flags = [f for f in flags if not f.startswith(cmake_var)]
            # Add new setting
            flags.append(f"{cmake_var}={new_value}")

        # Execute install commands (if any)
        install_commands = analysis.get("install_commands", [])
        if install_commands and self.verbose:
            print("[Note] The following packages may need to be installed:")
            for cmd in install_commands:
                print(f"  {cmd}")
            print("[Note] Consider updating the Dockerfile to include these dependencies")

        return flags

    def extract_compile_commands(self, project_path: Path, output_dir: Path | None = None) -> Path:
        """
        Extract compile_commands.json from build directory to output directory.
        Fixes Docker container paths to host paths.

        Args:
            project_path: Path to the project source
            output_dir: Directory to save compile_commands.json (default: build-scripts/<project>/)

        Returns the path to the extracted file.
        """
        # Use custom build_dir if provided, otherwise default to project_path/build
        if self.build_dir:
            build_dir = Path(self.build_dir).resolve()
        else:
            build_dir = project_path / "build"

        compile_commands_src = build_dir / "compile_commands.json"

        # Default output to build-scripts/<project>/compile_commands.json
        if output_dir is None:
            output_dir = Path("build-scripts") / project_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

        compile_commands_dst = output_dir / "compile_commands.json"

        if not compile_commands_src.exists():
            raise FileNotFoundError(
                f"compile_commands.json not found in {build_dir}"
            )

        # Read and fix paths
        import json

        with open(compile_commands_src, "r") as f:
            compile_commands = json.load(f)

        # Replace Docker container paths with host paths
        # /workspace/src -> actual project path
        # /workspace/build -> actual build path
        for entry in compile_commands:
            if "file" in entry:
                entry["file"] = entry["file"].replace("/workspace/src", str(project_path))
                entry["file"] = entry["file"].replace("/workspace/build", str(build_dir))
            if "directory" in entry:
                entry["directory"] = entry["directory"].replace("/workspace/build", str(build_dir))
            if "command" in entry:
                entry["command"] = entry["command"].replace("/workspace/src", str(project_path))
                entry["command"] = entry["command"].replace("/workspace/build", str(build_dir))
            if "arguments" in entry:
                entry["arguments"] = [
                    arg.replace("/workspace/src", str(project_path)).replace("/workspace/build", str(build_dir))
                    for arg in entry["arguments"]
                ]

        # Write fixed version to project root
        with open(compile_commands_dst, "w") as f:
            json.dump(compile_commands, f, indent=2)

        if self.verbose:
            print(f"[Extract] Copied and fixed compile_commands.json to {compile_commands_dst}")
            print(f"[Extract] Fixed Docker paths: /workspace/src -> {project_path}")
            print(f"[Extract] Fixed Docker paths: /workspace/build -> {build_dir}")

        return compile_commands_dst
