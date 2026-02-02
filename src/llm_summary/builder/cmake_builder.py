"""CMake builder with LLM-powered incremental learning."""

import json
import subprocess
from pathlib import Path
from typing import Any

from ..llm.base import LLMBackend
from .error_analyzer import BuildError, ErrorAnalyzer
from .prompts import INITIAL_CONFIG_PROMPT


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
    ):
        self.llm = llm
        self.container_image = container_image
        self.build_dir = build_dir
        self.max_retries = max_retries
        self.enable_lto = enable_lto
        self.prefer_static = prefer_static
        self.generate_ir = generate_ir
        self.verbose = verbose
        self.error_analyzer = ErrorAnalyzer(llm, verbose)

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

        # Read CMakeLists.txt for LLM analysis
        cmakelists_path = project_path / "CMakeLists.txt"
        if not cmakelists_path.exists():
            raise ValueError(f"CMakeLists.txt not found in {project_path}")

        cmakelists_content = cmakelists_path.read_text()

        # Get initial configuration from LLM
        if self.verbose:
            print(f"\n[1/3] Analyzing CMakeLists.txt with LLM...")

        cmake_flags = self._get_initial_config(project_path, cmakelists_content)

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
                    print(f"\n[LLM] Analyzing error to adjust configuration...")

                # Determine if this is a CMake config error or build error
                if "CMake Error" in error_msg or "cmake" in error_msg.lower():
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
        """Get initial CMake configuration using LLM analysis."""
        prompt = INITIAL_CONFIG_PROMPT.format(
            cmakelists_content=cmakelists_content,
            project_path=str(project_path),
        )

        if self.verbose:
            print(f"[LLM] Requesting initial configuration...")
            print(f"[LLM] Prompt length: {len(prompt)} chars")

        response = self.llm.complete(prompt)

        if self.verbose:
            print(f"[LLM] Response length: {len(response)} chars")

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
                    pkg = dep.get("package", "unknown")
                    reason = dep.get("reason", "")
                    print(f"  - {pkg}: {reason}")
                print(f"[Note] Ensure these are installed in the Docker image or build may fail")

            if self.verbose and result.get("potential_issues"):
                print(f"[Potential Issues]")
                for issue in result["potential_issues"]:
                    print(f"  - {issue}")

            return flags
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[ERROR] Failed to parse LLM response: {e}")
                print(f"[ERROR] Response: {response[:500]}...")

            # Fall back to default configuration
            return self._get_default_config()

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

        Raises BuildError if the build fails.
        Returns build log on success.
        """
        # Use custom build_dir if provided, otherwise default to project_path/build
        if self.build_dir:
            build_dir = Path(self.build_dir).resolve()
        else:
            build_dir = project_path / "build"

        build_dir.mkdir(parents=True, exist_ok=True)

        # Construct CMake command with separate volume mounts for source and build
        cmake_cmd = [
            "docker", "run", "--rm",
            "-v", f"{project_path}:/workspace/src",
            "-v", f"{build_dir}:/workspace/build",
            "-w", "/workspace/build",
            self.container_image,
            "bash", "-c",
        ]

        # CMake configure command pointing to source directory
        # Quote flags that contain spaces to prevent shell splitting
        quoted_flags = []
        for flag in cmake_flags:
            if ' ' in flag:
                # Use single quotes to protect the flag from shell splitting
                quoted_flags.append(f"'{flag}'")
            else:
                quoted_flags.append(flag)

        configure_cmd = f"cmake -G Ninja {' '.join(quoted_flags)} /workspace/src"

        # Build command
        build_cmd = "ninja -j$(nproc)"

        # Combined command
        full_cmd = f"{configure_cmd} && {build_cmd}"

        if self.verbose:
            print(f"[Docker] Build directory: {build_dir}")
            print(f"[Docker] Running: {full_cmd}")

        cmake_cmd.append(full_cmd)

        try:
            result = subprocess.run(
                cmake_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode != 0:
                raise BuildError(output)

            return output

        except subprocess.TimeoutExpired:
            raise BuildError("Build timed out after 10 minutes")
        except Exception as e:
            raise BuildError(f"Build failed with exception: {e}")

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
            print(f"[Note] The following packages may need to be installed:")
            for cmd in install_commands:
                print(f"  {cmd}")
            print(f"[Note] Consider updating the Dockerfile to include these dependencies")

        return flags

    def extract_compile_commands(self, project_path: Path) -> Path:
        """
        Extract compile_commands.json from build directory to project root.
        Fixes Docker container paths to host paths.

        Returns the path to the extracted file.
        """
        # Use custom build_dir if provided, otherwise default to project_path/build
        if self.build_dir:
            build_dir = Path(self.build_dir).resolve()
        else:
            build_dir = project_path / "build"

        compile_commands_src = build_dir / "compile_commands.json"
        compile_commands_dst = project_path / "compile_commands.json"

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
