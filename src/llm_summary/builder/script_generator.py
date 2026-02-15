"""Build script generator for reusable builds."""

import json
import shlex
from datetime import datetime
from pathlib import Path


def _shell_quote(s: str) -> str:
    """Quote a string for safe inclusion in a shell script."""
    return shlex.quote(s)


def _quote_cmake_flag(flag: str) -> str:
    """
    Quote a CMake flag value if it contains spaces.

    Examples:
        -DCMAKE_C_FLAGS=-g -flto  ->  -DCMAKE_C_FLAGS='-g -flto'
        -DBUILD_SHARED_LIBS=OFF   ->  -DBUILD_SHARED_LIBS=OFF  (no change)
    """
    if '=' in flag:
        key, value = flag.split('=', 1)
        # If value contains spaces and isn't already quoted, add quotes
        if ' ' in value and not (value.startswith('"') or value.startswith("'")):
            return f"{key}='{value}'"
    return flag


class ScriptGenerator:
    """Generates reusable build scripts for projects."""

    def __init__(self, scripts_base_dir: Path | None = None):
        self.scripts_base_dir = scripts_base_dir or Path("build-scripts")

    def generate(
        self,
        project_name: str,
        project_path: Path,
        flags: list[str],
        container_image: str = "llm-summary-builder:latest",
        build_system: str = "cmake",
        enable_ir: bool = True,
        use_build_dir: bool = True,
        dependencies: list[str] | None = None,
        build_script: str | None = None,
    ) -> dict[str, Path]:
        """
        Generate a reusable build script for a project.

        Args:
            project_name: Name of the project
            project_path: Path to the project source
            flags: CMake flags or configure flags depending on build system
            container_image: Docker image to use
            build_system: "cmake" or "autotools"
            enable_ir: Whether to generate LLVM IR artifacts
            use_build_dir: For autotools, whether to use out-of-source build
            dependencies: Optional list of apt package names required beyond the base image
            build_script: Validated build script content (for custom build systems)

        Returns a dict with paths to generated files:
        - script: Path to build.sh
        - config: Path to config.json
        - artifacts_dir: Path to artifacts directory
        """
        # Create project directory
        project_dir = self.scripts_base_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create artifacts directory
        artifacts_dir = project_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Generate config.json
        config_path = self._generate_config(
            project_dir,
            project_name,
            str(project_path),
            flags,
            build_system,
            use_build_dir,
            dependencies=dependencies,
            build_script=build_script,
        )

        # Generate build.sh based on build system
        if build_script is not None:
            script_path = self._generate_custom_script(
                project_dir,
                project_name,
                str(project_path),
                build_script,
                container_image,
                enable_ir,
                dependencies=dependencies,
            )
        elif build_system in ("autotools", "configure_make", "make"):
            script_path = self._generate_autotools_script(
                project_dir,
                project_name,
                str(project_path),
                flags,
                container_image,
                enable_ir,
                use_build_dir,
                dependencies=dependencies,
            )
        else:
            script_path = self._generate_cmake_script(
                project_dir,
                project_name,
                str(project_path),
                flags,
                container_image,
                enable_ir,
                dependencies=dependencies,
            )

        return {
            "script": script_path,
            "config": config_path,
            "artifacts_dir": artifacts_dir,
        }

    def _generate_config(
        self,
        project_dir: Path,
        project_name: str,
        project_path: str,
        flags: list[str],
        build_system: str,
        use_build_dir: bool = True,
        dependencies: list[str] | None = None,
        build_script: str | None = None,
    ) -> Path:
        """Generate config.json metadata file."""
        config = {
            "project_name": project_name,
            "project_path": project_path,
            "build_system": build_system,
            "generated_at": datetime.now().isoformat(),
        }

        # Add flags/script with appropriate key based on build system
        if build_script is not None:
            config["build_script"] = build_script
        elif build_system in ("autotools", "configure_make", "make"):
            config["configure_flags"] = flags
            config["use_build_dir"] = use_build_dir
        else:
            config["cmake_flags"] = flags

        if dependencies:
            config["dependencies"] = dependencies

        config_path = project_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2))

        return config_path

    def _generate_cmake_script(
        self,
        project_dir: Path,
        project_name: str,
        project_path: str,
        cmake_flags: list[str],
        container_image: str,
        enable_ir: bool,
        dependencies: list[str] | None = None,
    ) -> Path:
        """Generate build.sh script for CMake projects."""
        # Determine configuration summary
        config_summary = []
        if any("LTO" in flag or "lto" in flag for flag in cmake_flags):
            config_summary.append("LTO enabled")
        if any("SHARED_LIBS=OFF" in flag for flag in cmake_flags):
            config_summary.append("static linking preferred")
        if any("save-temps" in flag for flag in cmake_flags):
            config_summary.append("LLVM IR generation")

        summary = ", ".join(config_summary) if config_summary else "default"

        # Format CMake flags for script (quote values with spaces)
        quoted_flags = [_quote_cmake_flag(flag) for flag in cmake_flags]
        formatted_flags = " \\\n           ".join(quoted_flags)

        # Build dependencies comment block
        deps_comment = ""
        if dependencies:
            deps_lines = "\n".join(f"#   - {d}" for d in dependencies)
            deps_comment = (
                f"\n# Required packages (not in base image):\n"
                f"{deps_lines}\n"
                f"# Install: apt-get install -y {' '.join(dependencies)}\n"
            )

        # Prepare dependency installation command
        deps_install = ""
        if dependencies:
            deps_install = f"apt-get update -qq && apt-get install -y {' '.join(dependencies)} && "

        # Generate script content
        script_content = f'''#!/bin/bash
# Generated build script for {project_name}
# Build system: CMake
# Configuration: {summary}
# Generated by llm-summary build-learn on {datetime.now().strftime("%Y-%m-%d")}
{deps_comment}
set -e

# Determine script directory (where compile_commands.json will be saved)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

PROJECT_PATH="${{1:-{project_path}}}"
ARTIFACTS_DIR="${{2:-$SCRIPT_DIR/artifacts}}"
BUILD_DIR="${{3:-$PROJECT_PATH/build}}"

# ccache support (disable with CCACHE_DISABLE=1)
CCACHE_HOST_DIR="${{CCACHE_DIR:-$HOME/.cache/llm-summary-ccache}}"
CCACHE_ARGS=""
if [ "${{CCACHE_DISABLE:-0}}" != "1" ]; then
    mkdir -p "$CCACHE_HOST_DIR"
    CCACHE_ARGS="-v $CCACHE_HOST_DIR:/ccache -e CCACHE_DIR=/ccache"
fi

# Validate project path
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Create directories
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$BUILD_DIR"

# Run build in Docker container
echo "Building {project_name}..."
docker run --rm \\
  -u $(id -u):$(id -g) \\
  $CCACHE_ARGS \\
  -v "$PROJECT_PATH":/workspace/src \\
  -v "$BUILD_DIR":/workspace/build \\
  -v "$ARTIFACTS_DIR":/artifacts \\
  -w /workspace/build \\
  {container_image} \\
  bash -c "{deps_install}cmake -G Ninja \\
           {formatted_flags} \\
           /workspace/src && \\
           ninja -j\\$(nproc)'''

        # Add IR artifact collection if enabled
        if enable_ir:
            script_content += ''' && \\
           echo 'Collecting LLVM IR artifacts...' && \\
           find . -name '*.bc' -o -name '*.ll' | xargs -I {{}} cp {{}} /artifacts/ 2>/dev/null || true"'''
        else:
            script_content += '''"'''

        # Add post-build steps
        script_content += '''

# Copy compile_commands.json to script directory (build-scripts/<project>/)
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    cp "$BUILD_DIR/compile_commands.json" "$SCRIPT_DIR/"
    echo ""
    echo "Build complete."
    echo "  - compile_commands.json: $SCRIPT_DIR/compile_commands.json"

    # Count IR artifacts if they were generated
    if [ -d "$ARTIFACTS_DIR" ]; then
        IR_COUNT=$(ls -1 "$ARTIFACTS_DIR"/*.bc "$ARTIFACTS_DIR"/*.ll 2>/dev/null | wc -l)
        if [ "$IR_COUNT" -gt 0 ]; then
            echo "  - LLVM IR artifacts: $ARTIFACTS_DIR ($IR_COUNT files)"
        fi
    fi
else
    echo "Warning: compile_commands.json not found in build directory"
    exit 1
fi
'''

        script_path = project_dir / "build.sh"
        script_path.write_text(script_content)
        # NOTE: Intentionally NOT making executable to prevent accidental execution
        # Users must explicitly run: bash build.sh

        return script_path

    def _generate_autotools_script(
        self,
        project_dir: Path,
        project_name: str,
        project_path: str,
        configure_flags: list[str],
        container_image: str,
        enable_ir: bool,
        use_build_dir: bool = True,
        dependencies: list[str] | None = None,
    ) -> Path:
        """Generate build.sh script for autotools projects."""
        # Determine configuration summary
        config_summary = []
        if any("disable-shared" in flag for flag in configure_flags):
            config_summary.append("static linking preferred")
        if any("disable-asm" in flag or "disable-simd" in flag for flag in configure_flags):
            config_summary.append("assembly disabled")

        summary = ", ".join(config_summary) if config_summary else "default"

        # Format configure flags for script
        formatted_flags = " \\\n           ".join(configure_flags) if configure_flags else ""

        # Environment variables for clang-18 with LTO
        env_vars = """CC="clang-18" \\
           CXX="clang++-18" \\
           CFLAGS="-g -flto=full -save-temps=obj" \\
           CXXFLAGS="-g -flto=full -save-temps=obj" \\
           LDFLAGS="-flto=full -fuse-ld=lld" \\
           LD="ld.lld-18" \\
           AR="llvm-ar-18" \\
           NM="llvm-nm-18" \\
           RANLIB="llvm-ranlib-18\""""

        if use_build_dir:
            work_dir = "/workspace/build"
            configure_path = "/workspace/src/configure"
            compile_commands_check = '"$BUILD_DIR/compile_commands.json"'
        else:
            work_dir = "/workspace/src"
            configure_path = "./configure"
            compile_commands_check = '"$PROJECT_PATH/compile_commands.json"'

        # Build dependencies comment block
        deps_comment = ""
        if dependencies:
            deps_lines = "\n".join(f"#   - {d}" for d in dependencies)
            deps_comment = (
                f"\n# Required packages (not in base image):\n"
                f"{deps_lines}\n"
                f"# Install: apt-get install -y {' '.join(dependencies)}\n"
            )

        # Generate script content
        script_content = f'''#!/bin/bash
# Generated build script for {project_name}
# Build system: autotools
# Configuration: {summary}
# Generated by llm-summary build-learn on {datetime.now().strftime("%Y-%m-%d")}
{deps_comment}
set -e

# Determine script directory (where compile_commands.json will be saved)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

PROJECT_PATH="${{1:-{project_path}}}"
ARTIFACTS_DIR="${{2:-$SCRIPT_DIR/artifacts}}"
'''

        if use_build_dir:
            script_content += '''BUILD_DIR="${3:-$PROJECT_PATH/build}"
'''

        script_content += '''
# ccache support (disable with CCACHE_DISABLE=1)
CCACHE_HOST_DIR="${CCACHE_DIR:-$HOME/.cache/llm-summary-ccache}"
CCACHE_ARGS=""
if [ "${CCACHE_DISABLE:-0}" != "1" ]; then
    mkdir -p "$CCACHE_HOST_DIR"
    CCACHE_ARGS="-v $CCACHE_HOST_DIR:/ccache -e CCACHE_DIR=/ccache"
fi

# Validate project path
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Create directories
mkdir -p "$ARTIFACTS_DIR"
'''

        if use_build_dir:
            script_content += '''mkdir -p "$BUILD_DIR"
'''

        # Prepare dependency installation command
        deps_install = ""
        if dependencies:
            deps_install = f"apt-get update -qq && apt-get install -y {' '.join(dependencies)} && "

        script_content += f'''
# Run build in Docker container
echo "Building {project_name}..."
'''

        if use_build_dir:
            script_content += f'''docker run --rm \\
  -u $(id -u):$(id -g) \\
  $CCACHE_ARGS \\
  -v "$PROJECT_PATH":/workspace/src \\
  -v "$BUILD_DIR":/workspace/build \\
  -v "$ARTIFACTS_DIR":/artifacts \\
  -w {work_dir} \\
  {container_image} \\
  bash -c "{deps_install}{env_vars} \\
           {configure_path} {formatted_flags} && \\
           bear -- make -j\\$(nproc)'''
        else:
            script_content += f'''docker run --rm \\
  -u $(id -u):$(id -g) \\
  $CCACHE_ARGS \\
  -v "$PROJECT_PATH":/workspace/src \\
  -v "$ARTIFACTS_DIR":/artifacts \\
  -w {work_dir} \\
  {container_image} \\
  bash -c "{deps_install}{env_vars} \\
           {configure_path} {formatted_flags} && \\
           bear -- make -j\\$(nproc)'''

        # Add IR artifact collection if enabled
        if enable_ir:
            script_content += ''' && \\
           echo 'Collecting LLVM IR artifacts...' && \\
           find . -name '*.bc' -o -name '*.ll' | xargs -I {{}} cp {{}} /artifacts/ 2>/dev/null || true"'''
        else:
            script_content += '''"'''

        # Add post-build steps
        script_content += f'''

# Copy compile_commands.json to script directory (build-scripts/<project>/)
if [ -f {compile_commands_check} ]; then
    cp {compile_commands_check} "$SCRIPT_DIR/"
    echo ""
    echo "Build complete."
    echo "  - compile_commands.json: $SCRIPT_DIR/compile_commands.json"

    # Count IR artifacts if they were generated
    if [ -d "$ARTIFACTS_DIR" ]; then
        IR_COUNT=$(ls -1 "$ARTIFACTS_DIR"/*.bc "$ARTIFACTS_DIR"/*.ll 2>/dev/null | wc -l)
        if [ "$IR_COUNT" -gt 0 ]; then
            echo "  - LLVM IR artifacts: $ARTIFACTS_DIR ($IR_COUNT files)"
        fi
    fi
else
    echo "Warning: compile_commands.json not found"
    exit 1
fi
'''

        script_path = project_dir / "build.sh"
        script_path.write_text(script_content)
        # NOTE: Intentionally NOT making executable to prevent accidental execution
        # Users must explicitly run: bash build.sh

        return script_path

    def _generate_custom_script(
        self,
        project_dir: Path,
        project_name: str,
        project_path: str,
        build_script: str,
        container_image: str,
        enable_ir: bool,
        dependencies: list[str] | None = None,
    ) -> Path:
        """Generate build.sh script for custom build systems (Meson, Bazel, etc.)."""
        # Build dependencies comment block
        deps_comment = ""
        if dependencies:
            deps_lines = "\n".join(f"#   - {d}" for d in dependencies)
            deps_comment = (
                f"\n# Required packages (not in base image):\n"
                f"{deps_lines}\n"
                f"# Install: apt-get install -y {' '.join(dependencies)}\n"
            )

        # Prepare dependency installation command and prepend to build script
        if dependencies:
            deps_install = f"apt-get update -qq && apt-get install -y {' '.join(dependencies)} && "
            build_script_with_deps = deps_install + build_script
        else:
            build_script_with_deps = build_script

        # Escape single quotes in build script for heredoc safety
        # We use a heredoc with a quoted delimiter so no variable expansion happens
        script_content = f'''#!/bin/bash
# Generated build script for {project_name}
# Build system: custom
# Generated by llm-summary build-learn on {datetime.now().strftime("%Y-%m-%d")}
{deps_comment}
set -e

# Determine script directory (where compile_commands.json will be saved)
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

PROJECT_PATH="${{1:-{project_path}}}"
ARTIFACTS_DIR="${{2:-$SCRIPT_DIR/artifacts}}"
BUILD_DIR="${{3:-$PROJECT_PATH/build}}"

# ccache support (disable with CCACHE_DISABLE=1)
CCACHE_HOST_DIR="${{CCACHE_DIR:-$HOME/.cache/llm-summary-ccache}}"
CCACHE_ARGS=""
if [ "${{CCACHE_DISABLE:-0}}" != "1" ]; then
    mkdir -p "$CCACHE_HOST_DIR"
    CCACHE_ARGS="-v $CCACHE_HOST_DIR:/ccache -e CCACHE_DIR=/ccache"
fi

# Validate project path
if [ ! -d "$PROJECT_PATH" ]; then
    echo "Error: Project path does not exist: $PROJECT_PATH"
    exit 1
fi

# Create directories
mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$BUILD_DIR"

# Run build in Docker container
echo "Building {project_name}..."
docker run --rm \\
  -u $(id -u):$(id -g) \\
  $CCACHE_ARGS \\
  -v "$PROJECT_PATH":/workspace/src \\
  -v "$BUILD_DIR":/workspace/build \\
  -w /workspace/build \\
  {container_image} \\
  bash -c {_shell_quote(build_script_with_deps)}'''

        # Add IR artifact collection
        if enable_ir:
            script_content += f'''

# Collect LLVM IR artifacts
echo "Collecting LLVM IR artifacts..."
docker run --rm \\
  -u $(id -u):$(id -g) \\
  -v "$BUILD_DIR":/workspace/build:ro \\
  -v "$ARTIFACTS_DIR":/artifacts \\
  -w /workspace/build \\
  {container_image} \\
  bash -c 'find . -name "*.bc" -o -name "*.ll" | xargs -I {{}} cp {{}} /artifacts/ 2>/dev/null || true'
'''

        # Add post-build steps: search both build dir and project dir for compile_commands.json
        script_content += '''
# Copy compile_commands.json to script directory
FOUND_CC=""
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    FOUND_CC="$BUILD_DIR/compile_commands.json"
elif [ -f "$PROJECT_PATH/compile_commands.json" ]; then
    FOUND_CC="$PROJECT_PATH/compile_commands.json"
else
    # Search subdirectories of build dir
    FOUND_CC=$(find "$BUILD_DIR" -name compile_commands.json -print -quit 2>/dev/null)
fi

if [ -n "$FOUND_CC" ]; then
    cp "$FOUND_CC" "$SCRIPT_DIR/"
    echo ""
    echo "Build complete."
    echo "  - compile_commands.json: $SCRIPT_DIR/compile_commands.json"

    # Count IR artifacts if they were generated
    if [ -d "$ARTIFACTS_DIR" ]; then
        IR_COUNT=$(ls -1 "$ARTIFACTS_DIR"/*.bc "$ARTIFACTS_DIR"/*.ll 2>/dev/null | wc -l)
        if [ "$IR_COUNT" -gt 0 ]; then
            echo "  - LLVM IR artifacts: $ARTIFACTS_DIR ($IR_COUNT files)"
        fi
    fi
else
    echo "Warning: compile_commands.json not found in build or project directory"
    exit 1
fi
'''

        script_path = project_dir / "build.sh"
        script_path.write_text(script_content)
        # NOTE: Intentionally NOT making executable to prevent accidental execution
        # Users must explicitly run: bash build.sh

        return script_path

    def generate_readme(self) -> Path:
        """Generate README.md for the build-scripts directory."""
        readme_content = """# Build Scripts

This directory contains automatically generated build scripts for OSS projects analyzed with llm-summary.

## Structure

Each project has its own subdirectory:

```
build-scripts/
├── README.md               # This file
├── libpng/
│   ├── build.sh            # Reusable build script
│   ├── config.json         # Build configuration metadata
│   ├── compile_commands.json  # Compilation database
│   └── artifacts/          # LLVM IR files (.bc, .ll)
└── ...
```

## Usage

### Running a Build Script

```bash
# Use default project path (from when script was generated)
bash libpng/build.sh

# Or specify custom paths
bash libpng/build.sh /path/to/libpng /path/to/artifacts /path/to/build
```

**Note:** Scripts are intentionally not executable. Use `bash <script>` to run them.

### Script Parameters

1. **PROJECT_PATH** (optional): Path to the project source directory
   - Default: The path used when the script was generated

2. **ARTIFACTS_DIR** (optional): Where to store LLVM IR artifacts
   - Default: `build-scripts/<project>/artifacts/`

3. **BUILD_DIR** (optional): Where to perform the build (supports out-of-source builds)
   - Default: `$PROJECT_PATH/build`

### What the Script Does

1. Validates project path exists
2. Creates artifacts directory
3. Runs CMake configuration in Docker with learned flags
4. Builds the project with Ninja
5. Collects LLVM IR artifacts (`.bc` and `.ll` files)
6. Copies `compile_commands.json` to `build-scripts/<project>/`

## Configuration Files

Each `config.json` contains:
- Project name and path
- Build system type (cmake, autotools, etc.)
- CMake flags learned by the build agent
- Generation timestamp

## Artifacts

LLVM IR artifacts are useful for:
- Static analysis with LLVM tools
- Dynamic analysis and instrumentation
- Understanding optimization behavior
- Cross-module analysis

## Regenerating Scripts

To regenerate a build script (e.g., after project updates):

```bash
llm-summary build-learn \\
  --project-path /path/to/project \\
  --backend claude \\
  --verbose
```

This will update the existing script with newly learned configuration.
"""

        readme_path = self.scripts_base_dir / "README.md"
        self.scripts_base_dir.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(readme_content)

        return readme_path
