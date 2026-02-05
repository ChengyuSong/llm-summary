"""Centralized tool definitions for the build agent."""

from .actions import TOOL_DEFINITIONS as CMAKE_ACTION_TOOLS
from .autotools_actions import TOOL_DEFINITIONS as AUTOTOOLS_ACTION_TOOLS
from .tools import TOOL_DEFINITIONS as FILE_TOOLS

# Read-only tools for error analysis (no cmake_configure/cmake_build)
TOOL_DEFINITIONS_READ_ONLY = FILE_TOOLS

# CMake tool definitions (for CMakeBuilder)
CMAKE_TOOL_DEFINITIONS = FILE_TOOLS + CMAKE_ACTION_TOOLS

# Autotools tool definitions (for AutotoolsBuilder)
AUTOTOOLS_TOOL_DEFINITIONS = FILE_TOOLS + AUTOTOOLS_ACTION_TOOLS

# Combine all tool definitions for the LLM (initial config phase)
# Note: ALL_TOOL_DEFINITIONS is kept for backwards compatibility (CMake)
ALL_TOOL_DEFINITIONS = CMAKE_TOOL_DEFINITIONS
