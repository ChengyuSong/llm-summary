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

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CMakeLists.txt │────▶│ LLM Analysis    │────▶│ Initial Config  │
│  (Project)      │     │ (Vertex AI)     │     │ (CMake flags)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ compile_        │◀────│ Docker Build    │◀────│ Retry Loop      │
│ commands.json   │     │ (LLVM 18)       │     │ (max 3 attempts)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                        │
         │                      │                        │ (on failure)
         │                      ▼                        ▼
         │              ┌─────────────────┐     ┌─────────────────┐
         │              │ LLVM IR         │     │ Error Analysis  │
         │              │ Artifacts (.bc) │     │ (LLM)           │
         │              └─────────────────┘     └─────────────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ Generated Outputs:                                               │
│ - build-scripts/<project>/build.sh                              │
│ - build-scripts/<project>/config.json                           │
│ - build-scripts/<project>/artifacts/*.bc                        │
│ - <project>/compile_commands.json (with fixed paths)            │
│ - Database: build_configs table                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Build System Detector (`builder/detector.py`)

Identifies the build system used by a project:
- CMake (CMakeLists.txt)
- Meson (meson.build)
- Autotools (configure.ac)
- Make (Makefile)

**Current support:** CMake only (Phase 1)

### 2. CMake Builder (`builder/cmake_builder.py`)

**Incremental Learning Loop:**
1. Analyze CMakeLists.txt with LLM
2. Generate initial CMake configuration
3. Attempt build in Docker container
4. If failure: analyze error with LLM, adjust config, retry
5. On success: extract compile_commands.json

**Features:**
- Custom build directory support
- Docker path fixing (container → host)
- Compiler flag quoting for shell safety
- Up to 3 build attempts with learning

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

### 5. LLM Backends (`llm/vertex.py`)

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

### Initial Configuration Prompt

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

On build failure, analyzes:
- Error output
- Current configuration
- CMakeLists.txt context

**Suggests:**
- Flag changes
- Missing packages
- Compatibility fixes

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
- [ ] Assembly code avoidance strategies
  - [ ] Detect assembly files in project
  - [ ] Add compiler flags to minimize asm usage
  - [ ] Patch build scripts if needed
- [ ] Fuzzing harness detection
  - [ ] Scan for `fuzzing/` or `fuzz/` directories
  - [ ] Build with AFL/libFuzzer flags
- [ ] LLM-driven build debugging
  - [ ] Multi-turn conversations for complex errors
  - [ ] Code patch suggestions (avoid if possible)

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

## Known Limitations (Phase 1)

1. **CMake Only**: Only CMake projects supported
2. **No Autotools**: configure/automake not yet supported
3. **No Fuzzing**: Fuzzing harness detection not implemented
4. **No Dependency Fetch**: Manual dependency installation required
5. **Basic Assembly Handling**: No advanced asm avoidance strategies
6. **English Only**: LLM prompts assume English CMakeLists.txt comments
7. **Single Architecture**: Builds for host architecture only (x86_64)

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
