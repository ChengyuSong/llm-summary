# Build Agent System (`build-learn`)

This document describes the automated build agent system that learns how to build OSS C/C++ projects in containers and generates reusable build scripts with `compile_commands.json` for static/dynamic analysis.

## Overview

The `build-learn` command uses LLM-powered incremental learning to automatically configure and build CMake projects with optimal settings for static analysis:

- **Containerized Builds**: Isolated Docker environment with LLVM 18 toolchain
- **LTO Support**: Link-Time Optimization with `-flto=full`
- **IR Generation**: LLVM bitcode (`.bc`) files for advanced analysis
- **Static Linking**: Prefers static libraries over dynamic
- **Incremental Learning**: Analyzes build failures and adapts configuration
- **Reusable Scripts**: Generates version-controlled build scripts

## Quick Start

### 1. Build the Docker Image

```bash
cd docker/build-env
./build.sh
```

### 2. Run Build Learning

```bash
# Using GCP Vertex AI (recommended)
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"

llm-summary build-learn \
  --project-path /path/to/project \
  --build-dir /path/to/build \
  --backend vertex \
  --model claude-haiku-4-5@20251001 \
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

### Hybrid Workflow: Simple vs ReAct Mode

The build agent automatically chooses between two modes based on project complexity:

**Simple Mode (for straightforward projects):**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CMakeLists.txt │────▶│ LLM Analysis    │────▶│ JSON Config     │
│  (Project)      │     │ (1 turn)        │     │ (cmake_flags)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Docker Build    │
                                               │ (configure +    │
                                               │  build)         │
                                               └─────────────────┘
```

**ReAct Mode (for complex projects):**
```
┌─────────────────┐     ┌────────────────────────────────────────┐
│  CMakeLists.txt │────▶│ LLM Agent with Tools (iterative)       │
│  (Project)      │     │                                        │
└─────────────────┘     │ Tools available:                       │
                        │ • read_file: Read project files        │
                        │ • list_dir: Explore structure          │
                        │ • cmake_configure: Test config         │
                        │ • cmake_build: Run build               │
                        │                                        │
                        │ Loop (max 15 turns):                   │
                        │ 1. Explore project                     │
                        │ 2. Try cmake configure                 │
                        │ 3. See results, adjust                 │
                        │ 4. Try build                           │
                        │ 5. On success: done                    │
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

**Mode Selection:**
- LLM decides automatically based on project complexity
- Simple projects (standard CMakeLists.txt) → Simple mode (1 LLM turn)
- Complex projects (unusual build system, many includes) → ReAct mode (iterative)

## Components

### 1. Build System Detector (`builder/detector.py`)

Identifies the build system used by a project:
- CMake (CMakeLists.txt)
- Meson (meson.build)
- Autotools (configure.ac)
- Make (Makefile)

**Current support:** CMake only (Phase 1)

### 2. CMake Builder (`builder/cmake_builder.py`)

**Hybrid Workflow:**

**Simple Mode (for standard projects):**
1. LLM reads CMakeLists.txt
2. Returns JSON with cmake_flags immediately
3. Build executes with retry loop (max 3 attempts)
4. On failure: error analysis and flag adjustment

**ReAct Mode (for complex projects):**
1. LLM explores project with tools (read_file, list_dir)
2. Iteratively tries cmake_configure with different flags
3. Sees configure output, adjusts based on errors
4. Once configure succeeds, tries cmake_build
5. Adjusts and retries build if needed
6. Returns flags on successful build

**Features:**
- **Hybrid mode selection**: LLM chooses simple or ReAct automatically
- **Sandboxed file access**: All file operations restricted to project directory
- **Separate configure/build steps**: ReAct mode can iterate on configure before building
- **Docker isolation**: All builds run in containers with LLVM 18
- **Path fixing**: Container paths automatically mapped to host paths
- **Tool-based exploration**: Can read cmake modules, explore directories
- **Security**: Path traversal and absolute paths blocked

**Key class:** `CMakeBuilder`

### 3. Error Analyzer (`builder/error_analyzer.py`)

Uses LLM to diagnose build failures and suggest fixes:

**CMake Configuration Errors:**
- Missing dependencies
- Invalid flags
- Incompatible options

**Build/Compilation Errors:**
- Compiler flag issues
- LTO conflicts
- Assembly code problems

**Key class:** `ErrorAnalyzer`

### 4. Script Generator (`builder/script_generator.py`)

Generates reusable build scripts in `build-scripts/<project>/`:

**Generated Files:**
- `build.sh`: Executable Docker build script
- `config.json`: Learned configuration metadata
- `artifacts/`: Directory for LLVM IR files
- `README.md`: Documentation

**Key class:** `ScriptGenerator`

### 5. Build Agent Tools (`builder/tools.py`, `builder/actions.py`)

The LLM agent has access to four tools for exploring and building projects:

#### File Exploration Tools (`builder/tools.py`)

**read_file(file_path, max_lines=200)**
- Reads project files (CMakeLists.txt, cmake modules, configs)
- Path must be relative to project root
- Returns file content or error message
- Security: Blocks absolute paths and path traversal

**list_dir(dir_path=".", pattern=None)**
- Lists files and directories in project
- Optional glob pattern filtering (e.g., "*.cmake")
- Returns files, directories, and metadata
- Security: Restricted to project directory tree

#### Build Action Tools (`builder/actions.py`)

**cmake_configure(cmake_flags)**
- Runs CMake configuration step only in Docker
- Tests flags without triggering full build
- Returns success status and output
- Allows iterative flag refinement

**cmake_build()**
- Runs ninja build after successful configure
- Requires CMakeCache.txt to exist
- Returns build output and status
- Separate from configure for targeted debugging

#### Security Sandboxing

All file tools enforce strict path validation:
- **Absolute paths blocked**: `/etc/passwd` → Error
- **Path traversal blocked**: `../../../etc/passwd` → Error
- **Symlinks validated**: Links outside project → Error
- **Resolution check**: Resolved path must be within project root

Example security validation:
```python
# Valid (within project)
read_file("CMakeLists.txt")              ✓
read_file("cmake/FindZLIB.cmake")        ✓
list_dir("src")                          ✓

# Invalid (blocked by security)
read_file("/etc/passwd")                 ✗ Absolute path
read_file("../../../etc/passwd")         ✗ Path traversal
list_dir("../../")                       ✗ Escapes project
```

**Key classes:** `BuildTools`, `CMakeActions`

### 6. LLM Backends (`llm/vertex.py`)

**Vertex AI Backend:**
- Uses Anthropic Claude via GCP Vertex AI
- Default model: `claude-haiku-4-5@20251001`
- Environment variables checked (in order):
  1. `VERTEX_AI_PROJECT`
  2. `ANTHROPIC_VERTEX_PROJECT_ID`
  3. `GOOGLE_CLOUD_PROJECT`
  4. `CLOUD_ML_PROJECT_ID`

**Debug mode:** Set `VERTEX_DEBUG=1` to see full API responses

## Usage

### CLI Arguments

```bash
llm-summary build-learn [OPTIONS]
```

**Required:**
- `--project-path PATH`: Path to the project source directory

**Optional:**
- `--build-dir PATH`: Custom build directory (default: `<project-path>/build`)
- `--backend {claude|openai|ollama|vertex}`: LLM backend (default: `vertex`)
- `--model NAME`: Model name (default varies by backend)
- `--max-retries N`: Maximum build attempts (default: `3`)
- `--container-image NAME`: Docker image (default: `llm-summary-builder:latest`)
- `--enable-lto / --no-lto`: Enable LLVM LTO (default: `true`)
- `--prefer-static / --no-static`: Prefer static linking (default: `true`)
- `--generate-ir / --no-ir`: Generate LLVM IR (default: `true`)
- `--db PATH`: Database file (default: `summaries.db`)
- `--verbose, -v`: Show detailed output

### Examples

**Basic usage with Vertex AI:**
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

**Using Ollama (local LLM):**
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

### Example: Building a Complex Project

The LLM receives the project name (e.g., "libpng") upfront. If it recognizes the project, it can leverage its knowledge of typical build requirements.

**Turn 1 - Exploration:**
```
LLM: I see this is libpng. Let me read the CMakeLists.txt to confirm
Tool: read_file("CMakeLists.txt")
Result: [CMake content showing find_package(ZLIB), custom modules]
```

**Turn 2 - Discover modules:**
```
LLM: I see it includes custom cmake modules. Let me explore
Tool: list_dir("cmake", pattern="*.cmake")
Result: FindZLIB.cmake, CompilerFlags.cmake, Options.cmake
```

**Turn 3 - Read critical module:**
```
LLM: Let me check the Options.cmake file
Tool: read_file("cmake/Options.cmake")
Result: [Options like ENABLE_SIMD, BUILD_TESTS, etc.]
```

**Turn 4 - Try initial configure:**
```
LLM: Based on analysis, I'll try this configuration
Tool: cmake_configure([
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DCMAKE_C_COMPILER=clang-18",
    "-DENABLE_SIMD=OFF",
    "-DBUILD_SHARED_LIBS=OFF"
])
Result: Success! Configuration complete
```

**Turn 5 - Build:**
```
LLM: Configuration succeeded, now building
Tool: cmake_build()
Result: Success! Build complete in 25.3 seconds
```

### Simple Mode vs ReAct: When Each is Used

**Simple Mode** (LLM returns JSON immediately):
- Standard CMakeLists.txt structure
- Common dependencies (zlib, openssl)
- No unusual cmake modules
- Standard project layout
- Example: libpng, zlib, simple projects

**ReAct Mode** (LLM uses tools iteratively):
- Complex cmake module includes
- Custom find_package() scripts
- Conditional compilation based on options
- Multi-target builds
- Deep directory structure
- Example: LLVM, Qt, complex projects

### Efficiency Comparison

**libpng (Simple Mode):**
- LLM turns: 1
- Time: ~30 seconds total
- Token usage: ~12K tokens

**Complex Project (ReAct Mode):**
- LLM turns: 5-10
- Time: ~45-60 seconds
- Token usage: ~30K tokens
- **But**: Higher success rate on first attempt (fewer retries)

## Docker Build Environment

**Image:** `llm-summary-builder:latest`

**Installed Tools:**
- LLVM 18 toolchain (clang-18, clang++, lld-18)
- libc++-18-dev, libc++abi-18-dev
- CMake 3.28+, Ninja, Make
- Git, ccache
- Common dependencies (zlib, libssl, libpng, libjpeg)

**Environment Variables:**
- `CC=clang-18`
- `CXX=clang++-18`
- `AR=llvm-ar-18`
- `LD=ld.lld-18`
- `NM=llvm-nm-18`
- `RANLIB=llvm-ranlib-18`

**Volume Mounts:**
- `/workspace/src`: Project source (read-only)
- `/workspace/build`: Build directory (read-write)
- `/artifacts`: LLVM IR output (read-write)

## LLM Prompts

### Hybrid System Prompt (Tool-Enabled Backends)

For backends supporting tool use (Vertex AI, Anthropic API), the agent receives a hybrid system prompt offering both modes:

**Option 1 - Simple Mode:**
- Return JSON configuration immediately for straightforward projects
- Format: `{"cmake_flags": [...], "reasoning": "...", "dependencies": [...]}`

**Option 2 - ReAct Mode:**
- Use tools iteratively for complex projects
- Available tools: read_file, list_dir, cmake_configure, cmake_build
- Explore, configure, build with feedback loop

**Requirements (both modes):**
- CMAKE_EXPORT_COMPILE_COMMANDS=ON
- CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON (LTO)
- CMAKE_C_COMPILER=clang-18, CMAKE_CXX_COMPILER=clang++-18
- BUILD_SHARED_LIBS=OFF (static linking)
- Disable SIMD/hardware optimizations
- CMAKE_C_FLAGS=-flto=full -save-temps=obj (IR generation)

**Mode Selection:**
- LLM decides based on project complexity
- No user configuration needed

### Initial Configuration Prompt (Fallback for Non-Tool Backends)

For backends without tool support, uses simpler prompt:

Analyzes CMakeLists.txt and generates:
- CMake configuration flags
- System dependencies (apt packages)
- Potential issues
- Reasoning for flag choices

**Goals:**
- Generate `compile_commands.json`
- Enable LLVM LTO
- Prefer static linking
- Use Clang 18
- Minimize assembly code
- Generate LLVM IR

### Error Analysis Prompt

On build failure in retry loop, analyzes:
- Error output
- Current configuration
- CMakeLists.txt context

**Suggests:**
- Flag changes
- Missing packages
- Compatibility fixes

**Note:** ReAct mode may not reach this prompt if it builds successfully in exploration phase

## Database Schema

### `build_configs` Table

```sql
CREATE TABLE build_configs (
    project_path TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    build_system TEXT NOT NULL,       -- 'cmake', 'autotools', 'make'
    configuration_json TEXT,          -- JSON with flags and options
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

**Example entry:**
```json
{
  "directory": "/data/csong/build-artifacts/libpng",
  "command": "clang-18 -DPNG_INTEL_SSE_OPT=1 -I/data/csong/opensource/libpng ...",
  "file": "/data/csong/opensource/libpng/png.c",
  "output": "CMakeFiles/png_static.dir/png.c.o"
}
```

### 2. Build Script

**Location:** `build-scripts/<project>/build.sh`

**Features:**
- Parameterized (accepts custom paths)
- Executable (chmod +x)
- Self-documenting (comments with config summary)
- Idempotent (can be run multiple times)

### 3. Configuration Metadata

**Location:** `build-scripts/<project>/config.json`

**Contains:**
- Project name and path
- Build system type
- Learned CMake flags
- Generation timestamp

### 4. LLVM IR Artifacts

**Location:** `build-scripts/<project>/artifacts/*.bc`

**Types:**
- `.bc`: LLVM bitcode (binary)
- `.ll`: LLVM IR (text, human-readable)

**Generated by:** `-save-temps=obj` compiler flag

## Troubleshooting

### Build Fails: "Unknown argument -save-temps=obj"

**Cause:** Compiler flags not properly quoted for shell

**Solution:** Already handled by flag quoting in `cmake_builder.py`

### Build Fails: Missing Dependencies

**Solution:**
1. Check LLM-identified dependencies in output
2. Add packages to `docker/build-env/Dockerfile`
3. Rebuild Docker image: `cd docker/build-env && ./build.sh`

### Vertex AI Authentication Errors

**Solution:**
```bash
# Authenticate with GCP
gcloud auth application-default login

# Set project
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### Paths in compile_commands.json Wrong

**Cause:** Docker container paths not replaced

**Solution:** Path fixing is automatic in `extract_compile_commands()`. If issues persist, check:
- Source mounted at `/workspace/src`
- Build mounted at `/workspace/build`

### LLM Response Parsing Errors

**Cause:** LLM returns markdown-wrapped JSON (` ```json ... ``` `)

**Solution:** Already handled by markdown stripping in `error_analyzer.py` and `cmake_builder.py`

## TODOs / Future Work

### Phase 2: Non-CMake Support
- [ ] Autotools support (`./configure`)
- [ ] Meson support (`meson setup`)
- [ ] Plain Make support (intercept with Bear)
- [ ] Custom build script detection

### Phase 3: Advanced Configuration
- [x] **Completed:** ReAct-style iterative building with tools
- [x] **Completed:** Separate cmake_configure and cmake_build steps
- [x] **Completed:** Path sandboxing and security validation
- [x] **Completed:** Project exploration with read_file/list_dir
- [ ] Assembly code avoidance strategies
  - [ ] Detect assembly files in project
  - [ ] Add compiler flags to minimize asm usage
  - [ ] Patch build scripts if needed
- [ ] Fuzzing harness detection
  - [ ] Scan for `fuzzing/` or `fuzz/` directories
  - [ ] Build with AFL/libFuzzer flags
- [ ] Enhanced ReAct features
  - [ ] Memory across sessions (save exploration results)
  - [ ] Parallel configure attempts (test multiple flag combinations)
  - [ ] Automatic dependency detection from cmake modules

### Phase 4: Dependency Management
- [ ] Parse CMake `find_package()` calls
- [ ] Attempt git clone of dependencies
- [ ] Recursively build dependencies with same settings
- [ ] Create dependency graph
- [ ] Support vcpkg/Conan package managers

### Phase 5: Caching and Optimization
- [ ] ccache integration for faster rebuilds
- [ ] Docker layer caching
- [ ] Parallel builds with optimal `-j` value
- [ ] Incremental rebuilds (only changed files)
- [ ] Build artifact reuse across projects

### Phase 6: Multi-Project Batch Processing
- [ ] Read project list from `gpr_projects.json`
- [ ] Parallel learning for multiple projects
- [ ] Aggregate statistics (success rate, common issues)
- [ ] Shared dependency pool

### Improvements
- [ ] Better error messages for common failures
- [ ] Progress bars for long builds
- [ ] Estimated time remaining
- [ ] Build log streaming (instead of waiting for completion)
- [ ] Support for cross-compilation
- [ ] Custom compiler selection (GCC, other LLVM versions)
- [ ] IR optimization level selection
- [ ] Generate `.ll` (text IR) in addition to `.bc`

### Testing
- [ ] Unit tests for each component
- [ ] Integration tests with real OSS projects
- [ ] Test suite for error analyzer prompts
- [ ] Regression tests for path fixing
- [ ] Performance benchmarks

### Documentation
- [ ] Video walkthrough
- [ ] Example projects repository
- [ ] Troubleshooting guide expansion
- [ ] Best practices for Docker image customization

## Known Limitations

1. **CMake Only**: Only CMake projects supported (Autotools/Meson planned for Phase 2)
2. **No Fuzzing**: Fuzzing harness detection not implemented
3. **No Dependency Fetch**: Manual dependency installation required (dependencies must be in Docker image)
4. **Basic Assembly Handling**: No advanced asm avoidance strategies beyond disabling SIMD
5. **English Only**: LLM prompts assume English CMakeLists.txt comments
6. **Single Architecture**: Builds for host architecture only (x86_64)
7. **Tool-Use Backend Required**: ReAct mode requires Vertex AI or Anthropic API (Ollama/OpenAI fall back to simple mode)

## New Capabilities (Recently Added)

1. **Hybrid Simple/ReAct Workflow**: Automatically chooses optimal mode based on complexity
2. **Sandboxed File Access**: Security validation prevents path traversal attacks
3. **Iterative Configure/Build**: Can refine configuration before building
4. **Project Exploration**: Can read cmake modules and explore directory structure
5. **Separate Build Steps**: cmake_configure and cmake_build can be run independently
6. **Tool-Based Debugging**: LLM can see configure output and adjust flags iteratively
7. **Project Name Hints**: LLM receives project name (e.g., "libpng", "zlib") to leverage its knowledge of common projects

## Performance Characteristics

**Typical Build Times (libpng example):**
- Initial analysis: ~5 seconds (LLM)
- CMake configuration: ~2 seconds
- Ninja build: ~20 seconds
- Path fixing: <1 second
- Total: ~30 seconds

**LLM Token Usage (libpng, Claude Haiku 4.5):**
- Input: ~47K chars (~11K tokens)
- Output: ~3K chars (~750 tokens)
- Cost: ~$0.001 per build (Vertex AI pricing)

**Disk Usage:**
- Docker image: ~2GB
- Build artifacts: ~50MB (libpng)
- LLVM IR files: ~2MB (18 .bc files for libpng)

## References

- [LLVM Link-Time Optimization](https://llvm.org/docs/LinkTimeOptimization.html)
- [CMake Compile Commands](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html)
- [Clang Compilation Flags](https://clang.llvm.org/docs/ClangCommandLineReference.html)
- [Anthropic Vertex AI](https://docs.anthropic.com/en/api/claude-on-vertex-ai)
