"""LLM prompts for build configuration and error analysis."""

# Note: INITIAL_CONFIG_PROMPT is kept for backwards compatibility with non-tool-enabled backends
# The main hybrid workflow uses the system prompt in cmake_builder.py instead
INITIAL_CONFIG_PROMPT = """Analyze this CMakeLists.txt and suggest initial CMake configuration flags.

Project: {project_name}

If you recognize this project, use your knowledge of its typical build requirements and dependencies.

Goals (mandatory):
- Generate compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Minimize assembly code usage:
  - Disable hardware-specific optimizations (SIMD/SSE/AVX/NEON)
  - Turn OFF any project-specific flags for architecture-specific code
  - Prefer portable C/C++ code over assembly or intrinsics

Preferences (use by default, but fall back if the project doesn't support them):
- Prefer Clang 18 (CMAKE_C_COMPILER=clang-18, CMAKE_CXX_COMPILER=clang++-18). Fall back to gcc if needed.
- Prefer LLVM LTO (CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON). Disable if it causes build failures.
- Prefer static libraries only (BUILD_SHARED_LIBS=OFF). Allow shared libs if required by the project.
- Prefer LLVM IR generation with -flto=full and -save-temps=obj (only applicable with clang + LTO)

CMakeLists.txt:
{cmakelists_content}

Output format (JSON):
{{
  "cmake_flags": ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ...],
  "dependencies": [
    {{"package": "zlib1g-dev", "reason": "Required for PNG compression"}}
  ],
  "reasoning": "Explanation of why these flags were chosen",
  "potential_issues": ["List of possible problems"]
}}

Important:
- Return ONLY valid JSON, no additional text or markdown formatting
- Include all necessary flags for our goals
- List any system dependencies (apt packages) the project needs
- Consider project-specific options (e.g., TESTS, EXAMPLES, SHARED_LIBS flags)
- Ensure flags are compatible with Clang 18 and LTO when possible; suggest gcc/no-LTO fallback if not
"""

ERROR_ANALYSIS_PROMPT = """A CMake build failed with the following error. Suggest fixes.

Current configuration:
{current_flags}

Error output:
{error_output}

CMakeLists.txt excerpt (if relevant):
{cmakelists_excerpt}

Note: The build runs in a Docker container where:
- Source files are mounted at: /workspace/src
- Build directory is at: /workspace/build
- Error messages will reference these container paths
- Dependencies CANNOT be installed at runtime - the Docker image must be updated

Analyze the error and suggest:
1. What went wrong
2. Specific flag changes to fix it (if possible with available tools)
3. Whether this requires missing dependencies (if so, list them for the user)

Output format (JSON):
{{
  "diagnosis": "Brief description of the problem",
  "suggested_flags": ["-DFLAG=VALUE", ...],
  "missing_dependencies": ["package-name", ...],
  "confidence": "high|medium|low"
}}

Important:
- Return ONLY valid JSON, no additional text
- Be specific about which flags to add, remove, or change
- If dependencies are missing, list them in "missing_dependencies" - these CANNOT be installed, only reported
- We prefer Clang 18 and LTO but these are soft preferences — suggest falling back to gcc or disabling LTO if they cause the failure
"""

BUILD_FAILURE_PROMPT = """A build (ninja/make) failed after successful CMake configuration.

Current CMake configuration:
{current_flags}

Build error output:
{error_output}

Analyze the error and suggest:
1. What went wrong during compilation
2. Which CMake flags need to be adjusted
3. Whether source code patches are needed (avoid if possible)

Output format (JSON):
{{
  "diagnosis": "Brief description of the compilation problem",
  "suggested_flags": ["-DFLAG=VALUE", ...],
  "compiler_flag_changes": {{"-DCMAKE_C_FLAGS": "-flto=full ...", "-DCMAKE_CXX_FLAGS": "..."}},
  "confidence": "high|medium|low",
  "notes": "Additional context or warnings"
}}

Important:
- Return ONLY valid JSON, no additional text
- Focus on configuration changes, not source code modifications
- Consider LTO and Clang 18 compatibility issues. If LTO or Clang 18 is causing the failure, suggest falling back (disable LTO, switch to gcc)
- If the issue is with assembly code, suggest flags to disable or work around it
"""
