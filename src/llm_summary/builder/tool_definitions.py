"""Centralized tool definitions for the build agent."""

from .actions import CMAKE_TOOL_DEFINITIONS as CMAKE_ACTION_TOOLS
from .actions import CONFIGURE_MAKE_TOOL_DEFINITIONS as CONFIGURE_MAKE_ACTION_TOOLS
from .actions import FINISH_TOOL_DEFINITION
from .actions import INSTALL_PACKAGES_TOOL_DEFINITION
from .tools import TOOL_DEFINITIONS as FILE_TOOLS

# Read-only tools for error analysis (no build tools)
TOOL_DEFINITIONS_READ_ONLY = FILE_TOOLS

# Unified tool definitions: file tools + all build tools + install_packages + finish
UNIFIED_TOOL_DEFINITIONS = (
    FILE_TOOLS
    + CMAKE_ACTION_TOOLS
    + CONFIGURE_MAKE_ACTION_TOOLS
    + [INSTALL_PACKAGES_TOOL_DEFINITION, FINISH_TOOL_DEFINITION]
)
