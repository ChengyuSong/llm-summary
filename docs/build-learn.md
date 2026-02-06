# Build Agent System (`build-learn`)

This document describes the automated build agent system that learns how to build OSS C/C++ projects in containers and generates reusable build scripts with `compile_commands.json` for static/dynamic analysis.

## Overview

The `build-learn` command uses LLM-powered incremental learning to automatically configure and build CMake and autotools projects with optimal settings for static analysis:

- **Containerized Builds**: Isolated Docker environment with LLVM 18 toolchain
- **LTO Support**: Link-Time Optimization with `-flto=full`
- **IR Generation**: LLVM bitcode (`.bc`) files for advanced analysis
- **Static Linking**: Prefers static libraries over dynamic
- **Incremental Learning**: Analyzes build failures and adapts configuration
- **Runtime Dependency Installation**: Installs missing system packages on-the-fly via derived Docker images
- **Reusable Scripts**: Generates version-controlled build scripts

## Quick Start

### 1. Build the Docker Image

```bash
cd docker/build-env
bash build.sh
```

### 2. Run Build Learning

```bash
# Using Claude via Vertex AI (recommended)
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

llm-summary build-learn \
  --project-path /path/to/project \
  --build-dir /path/to/build \
  --backend claude \
  --verbose
```

### 3. Use Generated Artifacts

```bash
# Run the generated build script for incremental builds
./build-scripts/<project-name>/build.sh

# Analyze with llm-summary
llm-summary extract --path /path/to/project --db analysis.db
```

## Architecture

### ReAct Workflow

The build agent uses a ReAct (Reason + Act) loop where the LLM explores the project, detects the build system, configures, builds, and installs missing dependencies — all iteratively with tool feedback.

```
┌─────────────────┐     ┌────────────────────────────────────────┐
│  Project        │────▶│ LLM Agent with Tools (iterative)       │
│  (CMake/        │     │                                        │
│   Autotools/    │     │ Tools available:                       │
│   Makefile)     │     │ • read_file / list_dir: Explore        │
└─────────────────┘     │ • cmake_configure / cmake_build        │
                        │ • run_configure / make_build           │
                        │ • bootstrap / autoreconf               │
                        │ • install_packages: Install deps       │
                        │ • finish: Signal completion            │
                        │                                        │
                        │ Loop (max 20 turns):                   │
                        │ 1. Explore project structure           │
                        │ 2. Detect build system                 │
                        │ 3. Configure with flags                │
                        │ 4. Build, adjust on errors             │
                        │ 5. Install missing deps if needed      │
                        │ 6. On success: finish with deps list   │
                        └────────────────────────────────────────┘
                                        │
                                        ▼
                        ┌─────────────────────────────────────────┐
                        │ Generated Outputs:                      │
                        │ - compile_commands.json                 │
                        │ - LLVM IR artifacts (.bc)               │
                        │ - build-scripts/<project>/build.sh      │
                        │ - build-scripts/<project>/config.json   │
                        └─────────────────────────────────────────┘
```


## Components

### 1. Unified Builder (`builder/builder.py`)

A single `Builder` class handles CMake, autotools, and plain Makefile projects. The agent explores the project root to detect the build system and uses the appropriate tools.

**ReAct Workflow:**
1. `_get_initial_config_with_tools()`: LLM receives unified system prompt with all tools
2. Enters `_execute_react_loop()` (max 20 turns)
3. LLM explores project with tools (read_file, list_dir) to detect build system
4. Configures and builds using appropriate tools (cmake_configure/cmake_build or run_configure/make_build)
5. Can install missing system dependencies via `install_packages`
6. On successful build: calls `finish(status="success", dependencies=[...])`
7. Returns dict with flags, build status, and validated dependencies

**Features:**
- **Unified build system support**: Single class handles CMake, autotools, and plain Makefile projects
- **Runtime dependency installation**: Agent can install missing packages via `install_packages`
- **Dependency tracking**: Cross-validates agent-reported deps against actually installed packages
- **Sandboxed file access**: All file operations restricted to project directory via `_validate_path()`
- **Docker isolation**: All builds run in containers with LLVM 18, user permissions preserved
- **Path fixing**: Container paths (`/workspace/src`, `/workspace/build`) mapped to host paths via `extract_compile_commands()`

**Key class:** `Builder`

**Constructor:**
```python
Builder(
    llm: LLMBackend,
    container_image: str = "llm-summary-builder:latest",
    build_dir: Path | None = None,
    max_retries: int = 3,
    enable_lto: bool = True,
    prefer_static: bool = True,
    generate_ir: bool = True,
    verbose: bool = False,
    log_file: str | None = None,
)
```

**Key methods:**
- `learn_and_build()`: Main entry point, orchestrates entire workflow
- `_get_initial_config()`: Routes to tool-enabled or simple workflow
- `_get_initial_config_with_tools()`: Unified prompt for tool-enabled backends
- `_get_initial_config_simple()`: Simple JSON prompt for non-tool backends
- `_execute_react_loop()`: ReAct tool execution loop (max 20 turns)
- `_execute_tool_safe()`: Routes tools to appropriate handlers
- `extract_compile_commands()`: Fix Docker paths in compile_commands.json

**Default Environment Variables (auto-injected for autotools/make):**
```bash
CC=clang-18
CXX=clang++-18
CFLAGS="-flto=full -save-temps=obj"
CXXFLAGS="-flto=full -save-temps=obj"
LDFLAGS="-flto=full -fuse-ld=lld"
LD=ld.lld-18
AR=llvm-ar-18
NM=llvm-nm-18
RANLIB=llvm-ranlib-18
```

### 2. Error Analyzer (`builder/error_analyzer.py`)

Uses LLM to diagnose build failures and suggest fixes with optional ReAct mode:

**Error Analysis Modes:**

**ReAct Mode (tool-enabled backends):**
- Uses `analyze_error_with_tools()` method
- Can explore project files to investigate errors
- Available tools: `read_file`, `list_dir` (build directory accessible via `"build/"`)
- Iteratively investigates before suggesting fixes
- Max 10 turns for exploration
- Returns JSON with diagnosis and suggested fixes

**Simple Mode (fallback):**
- Direct LLM prompt with error output
- CMake errors: `analyze_cmake_error()`
- Build errors: `analyze_build_error()`
- Single-turn analysis

**Error Types Handled:**
- **CMake Configuration Errors:** Missing dependencies, invalid flags, incompatible options
- **Build/Compilation Errors:** Compiler flag issues, LTO conflicts, assembly code problems

**Key class:** `ErrorAnalyzer`

### 3. Script Generator (`builder/script_generator.py`)

Generates reusable build scripts in `build-scripts/<project>/`:

**Generated Files:**
- `build.sh`: Executable Docker build script (includes dependency comment block if packages were installed)
- `config.json`: Learned configuration metadata (includes `dependencies` field)
- `artifacts/`: Directory for LLVM IR files
- `README.md`: Documentation

**Key class:** `ScriptGenerator`

### 4. Assembly Checker (`builder/assembly_checker.py`)

Detects assembly code in build artifacts and filters out known unavoidable findings:

**Detection Methods:**
1. **Standalone assembly files**: Scans compile_commands.json for `.s`, `.S`, `.asm` files
2. **Inline assembly in sources**: Pattern matching in C/C++ files for `__asm__`, `asm()`, `__asm__ __volatile__`, etc.
3. **Inline assembly in LLVM IR**: Uses `llvm-dis` to disassemble `.bc` files and searches for `call asm`, `asm sideeffect`, `module asm`

**Path Translation:**
- Automatically translates Docker paths (`/workspace/src`, `/workspace/build`) to host paths
- Enables source file scanning even when compile_commands.json contains container paths

**Unavoidable Assembly Tracking:**
- Loads known unavoidable findings from `build-scripts/<project>/unavoidable_asm.json`
- Filters these from results so agent only sees new/actionable assembly
- Findings identified by stable key: `{type}:{file_path}:{pattern}`
- Prevents agent from repeatedly trying to remove unavoidable inline assembly from dependencies

**Integration:**
- Runs automatically after successful `cmake_build()` or `make_build()`
- Results included in build tool response as `assembly_check` field
- Agent can iteratively try different flags to minimize assembly

**Key class:** `AssemblyChecker`

### 5. Build Agent Tools (`builder/tools.py`, `builder/actions.py`, `builder/tool_definitions.py`)

#### Tool Definitions (`builder/tool_definitions.py`)

Centralizes all tool definitions in Anthropic format:

- **TOOL_DEFINITIONS_READ_ONLY**: File exploration tools only (for error analysis)
- **UNIFIED_TOOL_DEFINITIONS**: File tools + all build action tools + install_packages + finish

Tool definitions are automatically converted to OpenAI format for backends that require it (e.g., llama.cpp).

#### File Exploration Tools (`builder/tools.py`)

**read_file(file_path, max_lines=200, start_line=1)**
- Reads project files (CMakeLists.txt, cmake modules, configs, source files) and build artifacts
- Path must be relative to project root or use `"build/"` prefix for build directory
- Build directory access: `"build/compile_commands.json"` → maps to actual build directory
- `max_lines`: Maximum lines to read (default: 200)
- `start_line`: Line number to start reading from (1-indexed, default: 1)
- Returns dict with: `content`, `path`, `start_line`, `end_line`, `lines_read`, `truncated`
- Security: Blocks absolute paths and path traversal, allows project and build directories only

**list_dir(dir_path=".", pattern=None)**
- Lists files and directories in project or build directory
- `dir_path`: Relative path from project root (default: "."), use `"build"` for build directory
- `pattern`: Optional glob pattern filtering (e.g., "*.cmake", "CMake*")
- Returns dict with: `path`, `files` (with name/path/size), `directories` (with name/path), `total`
- Security: Restricted to project and build directory trees

#### Build Action Tools (`builder/actions.py`)

**CMake Tools:**

**cmake_configure(cmake_flags)**
- Runs CMake configuration step in Docker container
- `cmake_flags`: List of CMake flags (e.g., `["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF"]`)
- Container config: `-G Ninja`, source at `/workspace/src`, build at `/workspace/build`
- Timeout: 5 minutes

**cmake_build()**
- Runs ninja build (`ninja -j$(nproc)`) after successful configure
- Requires CMakeCache.txt to exist (from successful configure)
- Returns dict with: `success`, `output`, `error`, `assembly_check`
- Timeout: 10 minutes

**Autotools/Make Tools:**

**bootstrap(script_path="bootstrap")**
- Runs a bootstrap script (e.g., `bootstrap`, `autogen.sh`, `buildconf`) to prepare the build system
- Security: Validates script path is within project directory
- Timeout: 5 minutes

**autoreconf()**
- Runs `autoreconf -fi` to regenerate configure script from configure.ac
- Timeout: 5 minutes

**run_configure(configure_flags, use_build_dir=True)**
- Runs `./configure` with specified flags
- Auto-injects CC, CXX, CFLAGS, CXXFLAGS, LDFLAGS, LD, AR, NM, RANLIB for clang-18 with LTO
- Timeout: 10 minutes

**make_build(make_target="", use_build_dir=True)**
- Runs `bear -- make -j$(nproc)` to build and capture compile commands
- On failure: Retries with `make -j1` for clearer error output
- Returns dict with: `success`, `output`, `error`, `assembly_check`
- Timeout: 20 minutes

**make_clean(use_build_dir=True)**
- Runs `make clean` to remove compiled object files

**make_distclean(use_build_dir=True)**
- Runs `make distclean` to remove all generated files including Makefile
- After distclean, must run run_configure again

**Package Management:**

**install_packages(packages)**
- Installs system packages (apt) by building a derived Docker image
- `packages`: List of apt package names (e.g., `["zlib1g-dev", "libssl-dev"]`)
- Validates package names against `^[a-zA-Z0-9][a-zA-Z0-9.+\-:]+$` (prevents injection)
- Creates Dockerfile: `FROM <current_image>\nRUN apt-get update && apt-get install -y <packages>`
- Runs `docker build -t llm-summary-builder:ext-<hash> .`
- Updates container image reference for subsequent build commands
- Timeout: 2 minutes

**finish(status, summary, dependencies=None)**
- Signals that the build task is complete
- `status`: Either `"success"` or `"failure"`
- `summary`: Brief description of what was accomplished or why it failed
- `dependencies`: Optional list of apt package names that are actual build dependencies
  - Cross-validated: only packages that were both installed via `install_packages` AND reported here are stored

#### Security Sandboxing

All file tools enforce strict path validation via `BuildTools._validate_path()`:

1. **Reject absolute paths**: Paths like `/etc/passwd` are blocked
2. **Handle build directory paths**: Paths starting with `"build/"` map to the build directory mount
3. **Check containment**: Resolved path must be within allowed directories
4. **Symlink escape detection**: Prevents symlinks pointing outside allowed directories

**Key classes:** `BuildTools` (file exploration), `CMakeActions` (CMake build execution), `AutotoolsActions` (autotools build execution)

### 6. LLM Backends (`llm/`)

Multiple LLM backends are supported with varying capabilities:

**Tool-Enabled Backends (Support ReAct Mode):**
- **Claude** (`llm/claude.py`): Anthropic Claude API with automatic Vertex AI support
  - Auto-detects: uses Vertex AI if GCP project env vars are set, otherwise direct API
  - Default model: `claude-sonnet-4-20250514`
  - Direct API: Requires `ANTHROPIC_API_KEY`
  - Vertex AI: Requires `GOOGLE_CLOUD_PROJECT` (or `VERTEX_AI_PROJECT`, `ANTHROPIC_VERTEX_PROJECT_ID`, `CLOUD_ML_PROJECT_ID`)
  - Debug: Set `CLAUDE_DEBUG=1` for full API responses
- **Gemini** (`llm/gemini.py`): Google Gemini via Vertex AI
  - Default model: `gemini-2.5-flash-preview-05-20`
  - Environment variables: `VERTEX_AI_PROJECT`, `GOOGLE_CLOUD_PROJECT`, `CLOUD_ML_PROJECT_ID`
  - Supports tool calling with Anthropic-compatible response format
- **llama.cpp** (`llm/llamacpp.py`): Local model server with tool calling
  - OpenAI-compatible /v1/chat/completions endpoint
  - Supports thinking mode control
  - Tool format conversion: Anthropic → OpenAI

**Other Backends:**
- **Ollama** (`llm/ollama.py`): Local Ollama server
  - Default model: `qwen3-coder:30b`
- **OpenAI** (`llm/openai.py`): OpenAI API
  - Default model: `gpt-4`

## Usage

### CLI Arguments

```bash
llm-summary build-learn [OPTIONS]
```

**Required:**
- `--project-path PATH`: Path to the project source directory

**Optional:**
- `--build-dir PATH`: Custom build directory (default: `<project-path>/build`)
- `--backend {claude|openai|ollama|llamacpp|gemini}`: LLM backend (default: `claude`)
- `--model NAME`: Model name (default varies by backend)
- `--max-retries N`: Maximum build attempts (default: `3`)
- `--container-image NAME`: Docker image (default: `llm-summary-builder:latest`)
- `--enable-lto / --no-lto`: Enable LLVM LTO (default: `true`)
- `--prefer-static / --no-static`: Prefer static linking (default: `true`)
- `--generate-ir / --no-ir`: Generate LLVM IR (default: `true`)
- `--db PATH`: Database file (default: `summaries.db`)
- `--verbose, -v`: Show detailed output
- `--log-llm PATH`: Log all LLM interactions (prompts, responses, tool calls)

### Examples

**Basic usage with Claude (auto-detects Vertex AI or direct API):**
```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
llm-summary build-learn --project-path /data/csong/opensource/libpng
```

**Custom build directory:**
```bash
llm-summary build-learn \
  --project-path /data/csong/opensource/libpng \
  --build-dir /data/csong/build-artifacts/libpng \
  --verbose
```

**Using Gemini:**
```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
llm-summary build-learn \
  --project-path /path/to/project \
  --backend gemini \
  --verbose
```

**Using llama.cpp (local model with tool support):**
```bash
llm-summary build-learn \
  --project-path /path/to/project \
  --backend llamacpp \
  --verbose
```

**Using Ollama (local LLM, simple mode):**
```bash
llm-summary build-learn \
  --project-path /path/to/project \
  --backend ollama \
  --model qwen3-coder:30b
```

**Disable IR generation (faster builds):**
```bash
llm-summary build-learn \
  --project-path /path/to/project \
  --no-ir
```

### Running Generated Build Scripts

```bash
# Use default paths
./build-scripts/libpng/build.sh

# Custom paths
./build-scripts/libpng/build.sh /custom/source/path /custom/artifacts/path
```

## How ReAct Mode Works

### Example: Building a Project with Missing Dependencies

The LLM receives the project name (e.g., "libpng") upfront. If it recognizes the project, it can leverage its knowledge of typical build requirements.

**Turn 1 - Exploration:**
```
LLM: Let me explore the project structure
Tool: list_dir(".")
Result: [CMakeLists.txt, src/, cmake/, ...]
```

**Turn 2 - Read build file:**
```
LLM: Let me read the CMakeLists.txt
Tool: read_file("CMakeLists.txt")
Result: [CMake content showing find_package(ZLIB), custom modules]
```

**Turn 3 - Try initial configure:**
```
LLM: I'll try this configuration
Tool: cmake_configure([
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DCMAKE_C_COMPILER=clang-18",
    "-DBUILD_SHARED_LIBS=OFF"
])
Result: Error: Could not find zlib.h
```

**Turn 4 - Install missing dependency:**
```
LLM: zlib is missing, let me install it
Tool: install_packages(["zlib1g-dev"])
Result: Success! Built derived image llm-summary-builder:ext-a1b2c3d4e5f6
```

**Turn 5 - Retry configure:**
```
LLM: Now let me reconfigure with zlib available
Tool: cmake_configure([...same flags...])
Result: Success! Configuration complete
```

**Turn 6 - Build:**
```
LLM: Building...
Tool: cmake_build()
Result: Success! Build complete, no assembly detected
```

**Turn 7 - Finish:**
```
LLM: Build succeeded
Tool: finish(status="success", summary="Built with cmake...", dependencies=["zlib1g-dev"])
```

### Dependency Tracking

When the agent installs packages and finishes successfully:

1. **Installed set**: The ReAct loop tracks which packages were successfully installed via `install_packages`
2. **Agent reports**: The `finish` tool includes a `dependencies` field where the agent reports which packages are actual dependencies
3. **Cross-validation**: Only packages appearing in BOTH sets are stored — this filters out hallucinated deps
4. **Storage**: Dependencies are written to `config.json` and the database `configuration_json`

## Docker Build Environment

**Image:** `llm-summary-builder:latest`

**Installed Tools:**
- LLVM 18 toolchain (clang-18, clang++, lld-18)
- libc++-18-dev, libc++abi-18-dev
- CMake 3.28+, Ninja, Make
- Autotools (autoconf, automake, libtool)
- Bear (for compile_commands.json generation with make)
- Git
- Common dependencies (zlib, libssl, libpng, libjpeg)

**Environment Variables:**
- `CC=clang-18`
- `CXX=clang++-18`
- `AR=llvm-ar-18`
- `LD=ld.lld-18`
- `NM=llvm-nm-18`
- `RANLIB=llvm-ranlib-18`

**Volume Mounts:**
- `/workspace/src`: Project source (read-only) - Maps to host project directory
- `/workspace/build`: Build directory (read-write) - Maps to host build directory (can be separate from project)
- `/artifacts`: LLVM IR output (read-write)

**Note on Build Directory Access:**
- The build directory is mounted separately and can be on a different host path than the project
- Agent tools can access build artifacts using `"build/"` prefix (e.g., `"build/compile_commands.json"`)

## LLM Prompts (`builder/builder.py`)

### System Prompt (Tool-Enabled Backends)

Defined in `builder.py:_get_initial_config_with_tools()`.

The agent receives a system prompt instructing it to:

1. **Explore** the project root to determine the build system
2. **Choose** appropriate tools (cmake_configure/cmake_build or run_configure/make_build)
3. **Install** missing dependencies via `install_packages` when needed
4. **Finish** with status and dependency list

**Build Requirements:**
- Generate compile_commands.json
- Use Clang 18
- Enable LLVM LTO
- Prefer static linking
- Disable SIMD/hardware optimizations to minimize assembly code
- Generate LLVM IR with `-save-temps=obj`

**Assembly Verification:**
- After each successful build, assembly check runs automatically
- Agent can iterate on flags to minimize assembly

**Package Installation:**
- Agent uses `install_packages` when build fails due to missing headers/libraries
- Reports installed dependencies via `finish(dependencies=[...])`

### Error Analysis Prompts

Defined in `prompts.py` and `error_analyzer.py`. Used in the retry loop when the initial ReAct build fails.

**ReAct Mode (tool-enabled backends):**
- Can use read_file and list_dir to investigate errors
- Max 10 turns for exploration
- Returns JSON with diagnosis and fixes


## Database Schema

### `build_configs` Table

```sql
CREATE TABLE build_configs (
    project_path TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    build_system TEXT NOT NULL,       -- 'cmake', 'autotools', 'make'
    configuration_json TEXT,          -- JSON with flags, options, and dependencies
    script_path TEXT,                 -- Path to generated build script
    artifacts_dir TEXT,               -- Path to LLVM IR artifacts
    compile_commands_path TEXT,       -- Path to compile_commands.json
    llm_backend TEXT,                 -- LLM backend used
    llm_model TEXT,                   -- Model used
    build_attempts INTEGER,           -- Number of attempts before success
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_built_at TIMESTAMP
);
```

## Output Files

### 1. compile_commands.json

**Location:** `<project-path>/compile_commands.json`

**Format:** Standard JSON compilation database
- All Docker paths replaced with host paths
- Ready for libclang analysis

### 2. Build Script

**Location:** `build-scripts/<project>/build.sh`

**Features:**
- Parameterized (accepts custom paths)
- Executable (chmod +x)
- Self-documenting (comments with config summary)
- Lists required packages if dependencies were installed
- Idempotent (can be run multiple times)

### 3. Configuration Metadata

**Location:** `build-scripts/<project>/config.json`

**Contains:**
- Project name and path
- Build system type
- Learned flags (cmake_flags or configure_flags)
- Generation timestamp
- `dependencies`: List of apt packages required beyond the base image (if any)

**Example:**
```json
{
  "project_name": "libpng",
  "project_path": "/data/csong/opensource/libpng",
  "build_system": "cmake",
  "generated_at": "2026-02-05T21:36:45.956961",
  "cmake_flags": [
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
    "-DCMAKE_C_COMPILER=clang-18",
    "-DCMAKE_CXX_COMPILER=clang++-18",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DCMAKE_C_FLAGS=-flto=full -save-temps=obj",
    "-DCMAKE_CXX_FLAGS=-flto=full -save-temps=obj",
    "-DPNG_HARDWARE_OPTIMIZATIONS=OFF"
  ],
  "dependencies": ["zlib1g-dev"]
}
```

### 4. LLVM IR Artifacts

**Location:** `build-scripts/<project>/artifacts/*.bc`

**Types:**
- `.bc`: LLVM bitcode (binary)
- `.ll`: LLVM IR (text, human-readable)

**Generated by:** `-save-temps=obj` compiler flag

## Debugging and Logging

### Log File Output

The `--log-llm` option enables comprehensive logging of all LLM interactions:

**What's logged:**
- System prompts for each phase (initial config, error analysis)
- User messages with full context
- Tool definitions (JSON format)
- LLM responses including stop_reason
- Tool calls with parameters and results
- Each turn in ReAct loops

**Example:**
```bash
llm-summary build-learn \
  --project-path /path/to/project \
  --log-llm build-debug.log \
  --verbose
```

The log file is append-mode, so multiple runs accumulate in the same file with clear separators.

## Troubleshooting

### Build Fails: "Unknown argument -save-temps=obj"

**Cause:** Compiler flags not properly quoted for shell

**Solution:** Already handled by flag quoting in `builder.py`

### Build Fails: Missing Dependencies

The agent will automatically attempt to install missing dependencies via `install_packages`. If this fails (e.g., package not found in apt), you can:

1. Check the error output for the correct package name
2. Add packages to `docker/build-env/Dockerfile` and rebuild the base image

### Authentication Errors

**Claude via Vertex AI:**
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

**Claude via direct API:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### Paths in compile_commands.json Wrong

**Cause:** Docker container paths not replaced

**Solution:** Path fixing is automatic in `extract_compile_commands()`. If issues persist, check:
- Source mounted at `/workspace/src`
- Build mounted at `/workspace/build`

### LLM Response Parsing Errors

**Cause:** LLM returns markdown-wrapped JSON (` ```json ... ``` `)

**Solution:** Already handled by markdown stripping in `error_analyzer.py` and `builder.py`

## TODOs / Future Work

### Phase 2: Non-CMake Support
- [x] Autotools support (`./configure` with Bear)
- [ ] Meson support (`meson setup`)
- [ ] Plain Make support (intercept with Bear)
- [ ] Custom build script detection

### Phase 3: Advanced Configuration
- [x] ReAct-style iterative building with tools
- [x] Separate cmake_configure and cmake_build steps
- [x] Path sandboxing and security validation
- [x] Project exploration with read_file/list_dir
- [x] ReAct error analysis with file exploration tools
- [x] llama.cpp backend with tool calling support
- [x] Assembly code detection and avoidance
- [x] Runtime dependency installation via `install_packages`
- [x] Dependency tracking and cross-validation
- [ ] Fuzzing harness detection
- [ ] Enhanced ReAct features
  - [ ] Memory across sessions (save exploration results)
  - [ ] Parallel configure attempts (test multiple flag combinations)

### Phase 4: Dependency Management
- [x] Runtime apt package installation
- [ ] Parse CMake `find_package()` calls
- [ ] Attempt git clone of dependencies
- [ ] Recursively build dependencies with same settings
- [ ] Create dependency graph
- [ ] Support vcpkg/Conan package managers

### Phase 5: Caching and Optimization
- [ ] ccache integration for faster rebuilds
- [ ] Docker layer caching
- [ ] Incremental rebuilds (only changed files)
- [ ] Build artifact reuse across projects

### Phase 6: Multi-Project Batch Processing
- [ ] Read project list from `gpr_projects.json`
- [ ] Parallel learning for multiple projects
- [ ] Aggregate statistics (success rate, common issues)
- [ ] Shared dependency pool

## Known Limitations

1. **CMake and Autotools Only**: Meson and plain Make projects not yet supported
2. **No Fuzzing**: Fuzzing harness detection not implemented
3. **Assembly from Dependencies**: While the agent can minimize assembly in the project itself through flags, inline assembly from third-party dependencies (e.g., BoringSSL crypto functions) cannot be eliminated. These are tracked as unavoidable.
4. **English Only**: LLM prompts assume English build file comments
5. **Single Architecture**: Builds for host architecture only (x86_64)

## References

- [LLVM Link-Time Optimization](https://llvm.org/docs/LinkTimeOptimization.html)
- [CMake Compile Commands](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html)
- [Clang Compilation Flags](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Anthropic Claude API](https://docs.anthropic.com/en/api/)
