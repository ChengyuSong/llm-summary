"""CMake action tools for the build agent."""

import os
import subprocess
from pathlib import Path
from typing import Any


class CMakeActions:
    """CMake configure and build actions for the agent."""

    def __init__(
        self,
        project_path: Path,
        build_dir: Path | None = None,
        container_image: str = "llm-summary-builder:latest",
        verbose: bool = False,
    ):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir) if build_dir else self.project_path / "build"
        self.container_image = container_image
        self.verbose = verbose

    def cmake_configure(self, cmake_flags: list[str]) -> dict[str, Any]:
        """
        Run CMake configuration step in Docker.

        Args:
            cmake_flags: List of CMake flags (e.g., ["-DCMAKE_BUILD_TYPE=Release"])

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            self.build_dir.mkdir(parents=True, exist_ok=True)

            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            # Build docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
                "-v", f"{self.build_dir}:/workspace/build",
                "-w", "/workspace/build",
                self.container_image,
                "bash", "-c",
            ]

            # Quote flags that contain spaces
            quoted_flags = []
            for flag in cmake_flags:
                if ' ' in flag:
                    quoted_flags.append(f"'{flag}'")
                else:
                    quoted_flags.append(flag)

            cmake_cmd = f"cmake -G Ninja {' '.join(quoted_flags)} /workspace/src"

            if self.verbose:
                print(f"[cmake_configure] Running: {cmake_cmd}")

            docker_cmd.append(cmake_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"CMake configure failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "CMake configure timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"CMake configure failed: {str(e)}",
            }

    def cmake_build(self) -> dict[str, Any]:
        """
        Run ninja build in Docker.

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            if not self.build_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Build directory does not exist. Run cmake_configure first.",
                }

            # Check if CMakeCache exists (indicates successful configure)
            if not (self.build_dir / "CMakeCache.txt").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "CMakeCache.txt not found. Run cmake_configure successfully first.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
                "-v", f"{self.build_dir}:/workspace/build",
                "-w", "/workspace/build",
                self.container_image,
                "bash", "-c",
                "ninja -j$(nproc)",
            ]

            if self.verbose:
                print("[cmake_build] Running: ninja -j$(nproc)")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"Build failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Build timed out after 10 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Build failed: {str(e)}",
            }


# Tool definitions for LLM (Anthropic tool use format)
TOOL_DEFINITIONS = [
    {
        "name": "cmake_configure",
        "description": (
            "Run CMake configuration step. This generates build files but does not compile. "
            "Use this to test different CMake flags before building. The command runs in a Docker "
            "container with the project mounted at /workspace/src and build at /workspace/build."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cmake_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "CMake flags like ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF']. "
                        "Always include: -DCMAKE_EXPORT_COMPILE_COMMANDS=ON, -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON, "
                        "-DCMAKE_C_COMPILER=clang-18, -DCMAKE_CXX_COMPILER=clang++-18, -DBUILD_SHARED_LIBS=OFF, "
                        "-DCMAKE_C_FLAGS='-flto=full -save-temps=obj', -DCMAKE_CXX_FLAGS='-flto=full -save-temps=obj'"
                    ),
                }
            },
            "required": ["cmake_flags"],
        },
    },
    {
        "name": "cmake_build",
        "description": (
            "Run the build step (ninja) after cmake configure has succeeded. "
            "Only call this after a successful cmake_configure. The build runs in a Docker container."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]
