"""Centralized tool definitions for the build agent."""

from .actions import TOOL_DEFINITIONS as ACTION_TOOLS
from .tools import TOOL_DEFINITIONS as FILE_TOOLS

# Combine all tool definitions for the LLM
ALL_TOOL_DEFINITIONS = FILE_TOOLS + ACTION_TOOLS
