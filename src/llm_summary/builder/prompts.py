"""LLM prompts for build configuration and error analysis."""

INITIAL_CONFIG_PROMPT = """Analyze this CMakeLists.txt and suggest initial CMake configuration flags.

Goals:
- Generate compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON)
- Enable LLVM LTO (CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
- Prefer static linking (BUILD_SHARED_LIBS=OFF)
- Use Clang 18 (CMAKE_C_COMPILER=clang-18, CMAKE_CXX_COMPILER=clang++-18)
- Minimize assembly code usage where possible
- Enable LLVM IR generation with -flto=full and -save-temps=obj

CMakeLists.txt:
{cmakelists_content}

Additional context:
- Project path: {project_path}
- Build system detected: CMake

Output format (JSON):
{{
  "cmake_flags": ["-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", ...],
  "dependencies": [
    {{"package": "zlib1g-dev", "reason": "Required for PNG compression"}},
    {{"package": "libssl-dev", "reason": "Optional, for crypto support"}}
  ],
  "reasoning": "Explanation of why these flags were chosen",
  "potential_issues": ["List of possible problems"]
}}

Important:
- Return ONLY valid JSON, no additional text or markdown formatting
- Include all necessary flags for our goals
- List any system dependencies (apt packages) the project needs
- Consider project-specific options (e.g., TESTS, EXAMPLES, SHARED_LIBS flags)
- Ensure flags are compatible with Clang 18 and LTO
"""

ERROR_ANALYSIS_PROMPT = """A CMake build failed with the following error. Suggest fixes.

Current configuration:
{current_flags}

Error output:
{error_output}

CMakeLists.txt excerpt (if relevant):
{cmakelists_excerpt}

Project path: {project_path}

Analyze the error and suggest:
1. What went wrong
2. Specific flag changes or commands to fix it
3. Whether this is a dependency issue (if so, which package)

Output format (JSON):
{{
  "diagnosis": "Brief description of the problem",
  "suggested_flags": ["-DFLAG=VALUE", ...],
  "install_commands": ["apt install package", ...],
  "confidence": "high|medium|low"
}}

Important:
- Return ONLY valid JSON, no additional text
- Be specific about which flags to add, remove, or change
- If dependencies are missing, provide exact apt install commands
- Consider that we're using Clang 18 in a Docker container
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
- Consider LTO and Clang 18 compatibility issues
- If the issue is with assembly code, suggest flags to disable or work around it
"""
