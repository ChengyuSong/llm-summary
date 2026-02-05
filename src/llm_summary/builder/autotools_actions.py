"""Autotools action tools for the build agent."""

import os
import subprocess
from pathlib import Path
from typing import Any

from .constants import (
    DOCKER_WORKSPACE_BUILD,
    DOCKER_WORKSPACE_SRC,
    TIMEOUT_BUILD,
    TIMEOUT_CONFIGURE,
    TIMEOUT_LONG_BUILD,
)


class AutotoolsActions:
    """Autotools configure and build actions for the agent."""

    def __init__(
        self,
        project_path: Path,
        build_dir: Path | None = None,
        container_image: str = "llm-summary-builder:latest",
        unavoidable_asm_path: Path | str | None = None,
        verbose: bool = False,
    ):
        self.project_path = Path(project_path).resolve()
        self.build_dir = Path(build_dir) if build_dir else self.project_path / "build"
        self.container_image = container_image
        self.unavoidable_asm_path = Path(unavoidable_asm_path) if unavoidable_asm_path else None
        self.verbose = verbose

    def _get_default_env_flags(self) -> dict[str, str]:
        """Get default environment variables for autotools builds."""
        return {
            "CC": "clang-18",
            "CXX": "clang++-18",
            "CFLAGS": "-flto=full -save-temps=obj",
            "CXXFLAGS": "-flto=full -save-temps=obj",
            "LDFLAGS": "-flto=full -fuse-ld=lld",
            "LD": "ld.lld-18",
            "AR": "llvm-ar-18",
            "NM": "llvm-nm-18",
            "RANLIB": "llvm-ranlib-18",
        }

    def bootstrap(self, script_path: str = "bootstrap") -> dict[str, Any]:
        """
        Run a bootstrap script to prepare the build system.

        Args:
            script_path: Path to the bootstrap script relative to project root
                        (e.g., "bootstrap", "autogen.sh", "scripts/bootstrap.sh")

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Validate script path is within project directory
            script_full_path = (self.project_path / script_path).resolve()
            if not script_full_path.is_relative_to(self.project_path):
                return {
                    "success": False,
                    "output": "",
                    "error": f"Script path {script_path} is outside project directory",
                }

            if not script_full_path.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Bootstrap script not found: {script_path}",
                }

            # Get relative path from project root for docker execution
            script_rel = script_full_path.relative_to(self.project_path)

            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
                "-w", DOCKER_WORKSPACE_SRC,
                self.container_image,
                "bash", "-c",
                f"chmod +x {script_rel} && ./{script_rel}",
            ]

            if self.verbose:
                print(f"[bootstrap] Running: {script_rel}")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
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
                    "error": f"bootstrap script failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "bootstrap script timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"bootstrap script failed: {str(e)}",
            }

    def autoreconf(self) -> dict[str, Any]:
        """
        Run autoreconf -fi to regenerate configure script.

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Run as host user to avoid root-owned files
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
                "-w", DOCKER_WORKSPACE_SRC,
                self.container_image,
                "bash", "-c",
                "autoreconf -fi",
            ]

            if self.verbose:
                print("[autoreconf] Running: autoreconf -fi")

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
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
                    "error": f"autoreconf failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "autoreconf timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"autoreconf failed: {str(e)}",
            }

    def autotools_configure(
        self,
        configure_flags: list[str],
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run ./configure with flags.

        Args:
            configure_flags: List of configure flags (e.g., ["--disable-shared", "--enable-static"])
            use_build_dir: If True, use out-of-source build in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            if use_build_dir:
                self.build_dir.mkdir(parents=True, exist_ok=True)

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            # Build docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
            ]

            # Add build dir mount if using out-of-source build
            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            # Build environment variables
            env_flags = self._get_default_env_flags()
            env_str = " ".join(f'{k}="{v}"' for k, v in env_flags.items())

            # Quote flags that contain spaces
            quoted_flags = []
            for flag in configure_flags:
                if ' ' in flag:
                    quoted_flags.append(f"'{flag}'")
                else:
                    quoted_flags.append(flag)

            # Determine configure path
            if use_build_dir:
                configure_path = "/workspace/src/configure"
            else:
                configure_path = "./configure"

            configure_cmd = f"{env_str} {configure_path} {' '.join(quoted_flags)}"

            if self.verbose:
                print(f"[autotools_configure] Running: {configure_cmd}")

            docker_cmd.append(configure_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_BUILD,  # 10 minute timeout for configure
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "use_build_dir": use_build_dir,
                }
            else:
                return {
                    "success": False,
                    "output": output,
                    "error": f"configure failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "configure timed out after 10 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"configure failed: {str(e)}",
            }

    def autotools_build(
        self,
        make_target: str = "",
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run bear -- make to build and capture compile commands.

        Args:
            make_target: Optional make target (e.g., "all", empty for default)
            use_build_dir: If True, build in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str), 'assembly_check' (optional)
        """
        try:
            # Determine working directory
            if use_build_dir:
                work_dir = self.build_dir
            else:
                work_dir = self.project_path

            if not work_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Work directory does not exist: {work_dir}. Run autotools_configure first.",
                }

            # Check if Makefile exists (indicates successful configure)
            if not (work_dir / "Makefile").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Makefile not found. Run autotools_configure successfully first.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
            ]

            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            # Build make command with bear
            target_str = make_target if make_target else ""
            make_cmd = f"bear -- make -j$(nproc) {target_str}".strip()

            if self.verbose:
                print(f"[autotools_build] Running: {make_cmd}")

            docker_cmd.append(make_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_LONG_BUILD,  # 20 minute timeout for build
            )

            output = result.stdout + result.stderr

            if result.returncode == 0:
                # Build succeeded - run assembly check
                asm_result = self._check_assembly(use_build_dir)
                return {
                    "success": True,
                    "output": output,
                    "error": "",
                    "assembly_check": asm_result.to_dict() if asm_result else None,
                }
            else:
                # Parallel build failed - retry with -j1 for clearer error output
                if self.verbose:
                    print("[autotools_build] Parallel build failed, retrying with make -j1 for clearer errors")

                docker_cmd_j1 = docker_cmd[:-1]  # Remove the old command
                make_cmd_j1 = f"bear -- make -j1 {target_str}".strip()
                docker_cmd_j1.append(make_cmd_j1)

                result_j1 = subprocess.run(
                    docker_cmd_j1,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_LONG_BUILD,
                )

                output_j1 = result_j1.stdout + result_j1.stderr

                return {
                    "success": False,
                    "output": output_j1,  # Use j1 output - clearer errors
                    "error": f"Build failed with exit code {result_j1.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Build timed out after 20 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Build failed: {str(e)}",
            }

    def autotools_clean(
        self,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run make clean to remove compiled objects.

        Args:
            use_build_dir: If True, clean in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        return self._run_make_target("clean", use_build_dir)

    def autotools_distclean(
        self,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run make distclean to remove all generated files including Makefile.

        Args:
            use_build_dir: If True, clean in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        return self._run_make_target("distclean", use_build_dir)

    def _run_make_target(
        self,
        target: str,
        use_build_dir: bool = True,
    ) -> dict[str, Any]:
        """
        Run a make target (helper for clean/distclean).

        Args:
            target: Make target to run
            use_build_dir: If True, run in /workspace/build

        Returns:
            Dict with 'success' (bool), 'output' (str), 'error' (str)
        """
        try:
            # Determine working directory
            if use_build_dir:
                work_dir = self.build_dir
            else:
                work_dir = self.project_path

            if not work_dir.exists():
                return {
                    "success": False,
                    "output": "",
                    "error": f"Work directory does not exist: {work_dir}",
                }

            # Check if Makefile exists
            if not (work_dir / "Makefile").exists():
                return {
                    "success": False,
                    "output": "",
                    "error": "Makefile not found. Nothing to clean.",
                }

            # Run as host user
            uid = os.getuid()
            gid = os.getgid()

            docker_cmd = [
                "docker", "run", "--rm",
                "-u", f"{uid}:{gid}",
                "-v", f"{self.project_path}:/workspace/src",
            ]

            if use_build_dir:
                docker_cmd.extend([
                    "-v", f"{self.build_dir}:/workspace/build",
                    "-w", DOCKER_WORKSPACE_BUILD,
                ])
            else:
                docker_cmd.extend(["-w", DOCKER_WORKSPACE_SRC])

            docker_cmd.extend([self.container_image, "bash", "-c"])

            make_cmd = f"make {target}"

            if self.verbose:
                print(f"[autotools_{target}] Running: {make_cmd}")

            docker_cmd.append(make_cmd)

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_CONFIGURE,  # 5 minute timeout
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
                    "error": f"make {target} failed with exit code {result.returncode}",
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": f"make {target} timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"make {target} failed: {str(e)}",
            }

    def _check_assembly(self, use_build_dir: bool = True):
        """
        Run assembly verification after successful build.

        Returns:
            AssemblyCheckResult or None if check could not be performed
        """
        try:
            from .assembly_checker import AssemblyChecker

            # compile_commands.json is generated in the working directory
            if use_build_dir:
                compile_commands = self.build_dir / "compile_commands.json"
            else:
                compile_commands = self.project_path / "compile_commands.json"

            if not compile_commands.exists():
                if self.verbose:
                    print("[autotools_build] No compile_commands.json, skipping assembly check")
                return None

            checker = AssemblyChecker(
                compile_commands_path=compile_commands,
                build_dir=self.build_dir if use_build_dir else self.project_path,
                project_path=self.project_path,
                unavoidable_asm_path=self.unavoidable_asm_path,
                verbose=self.verbose,
            )
            return checker.check(scan_ir=True)
        except Exception as e:
            if self.verbose:
                print(f"[autotools_build] Assembly check failed: {e}")
            return None


# Tool definitions for LLM (Anthropic tool use format)
TOOL_DEFINITIONS = [
    {
        "name": "bootstrap",
        "description": (
            "Run a bootstrap script to prepare the build system before configure. "
            "Many autotools projects provide a bootstrap script (bootstrap, autogen.sh, buildconf) "
            "that must be run before ./configure. Use this if the project has such a script. "
            "The script must be in the PROJECT directory (NOT 'build/'). Absolute paths are NOT allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": (
                        "Relative path to bootstrap script in project directory. "
                        "Examples: 'bootstrap', 'autogen.sh', 'buildconf', 'scripts/bootstrap.sh'. "
                        "Do NOT use 'build/' prefix. Default is 'bootstrap'."
                    ),
                    "default": "bootstrap",
                },
            },
        },
    },
    {
        "name": "autoreconf",
        "description": (
            "Run autoreconf -fi to regenerate the configure script from configure.ac. "
            "Use this when the project has configure.ac but no configure script, or when "
            "configure.ac has been modified. This is typically needed for projects cloned "
            "from version control. The command runs in a Docker container."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "autotools_configure",
        "description": (
            "Run ./configure with flags. This generates Makefile but does not compile. "
            "Environment variables CC, CXX, CFLAGS, CXXFLAGS, LD, AR, NM, RANLIB are "
            "automatically set to use clang-18 with LTO. The command runs in a Docker "
            "container with the project mounted at /workspace/src."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "configure_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Configure flags like ['--disable-shared', '--enable-static', '--prefix=/usr']. "
                        "Common flags for minimizing assembly: --disable-asm, --disable-simd, "
                        "--disable-hardware-acceleration."
                    ),
                },
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "If true (default), use out-of-source build in /workspace/build. "
                        "If false, build in-source in /workspace/src. Out-of-source is cleaner "
                        "but some projects require in-source builds."
                    ),
                    "default": True,
                },
            },
            "required": ["configure_flags"],
        },
    },
    {
        "name": "autotools_build",
        "description": (
            "Run 'bear -- make' to build and capture compile commands. Only call this after "
            "a successful autotools_configure. Bear wraps make to generate compile_commands.json. "
            "On success, returns an assembly_check result showing if any assembly code (.s files, "
            "inline asm) was compiled. If assembly is detected, try different configure flags "
            "to avoid it (e.g., --disable-asm, --disable-simd)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make_target": {
                    "type": "string",
                    "description": (
                        "Optional make target (e.g., 'all', 'lib'). Leave empty for default target."
                    ),
                    "default": "",
                },
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from autotools_configure."
                    ),
                    "default": True,
                },
            },
        },
    },
    {
        "name": "autotools_clean",
        "description": (
            "Run 'make clean' to remove compiled object files. Use this before retrying a build "
            "with different flags, or to clean up after a failed build. Does not remove Makefile "
            "or configuration files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from autotools_configure."
                    ),
                    "default": True,
                },
            },
        },
    },
    {
        "name": "autotools_distclean",
        "description": (
            "Run 'make distclean' to remove all generated files including Makefile and "
            "configuration cache. Use this before reconfiguring with completely different flags. "
            "After distclean, you must run autotools_configure again before building."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_build_dir": {
                    "type": "boolean",
                    "description": (
                        "Must match the use_build_dir setting from autotools_configure."
                    ),
                    "default": True,
                },
            },
        },
    },
]
