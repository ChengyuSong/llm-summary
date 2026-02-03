"""Centralized tool definitions for the build agent."""

from .actions import TOOL_DEFINITIONS as ACTION_TOOLS
from .tools import TOOL_DEFINITIONS as FILE_TOOLS

# Read-only tools for error analysis (no cmake_configure/cmake_build)
TOOL_DEFINITIONS_READ_ONLY = FILE_TOOLS

# Combine all tool definitions for the LLM (initial config phase)
ALL_TOOL_DEFINITIONS = FILE_TOOLS + ACTION_TOOLS
