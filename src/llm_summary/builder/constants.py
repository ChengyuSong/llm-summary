"""Shared constants for the builder module."""

# Docker mount paths
DOCKER_WORKSPACE_SRC = "/workspace/src"
DOCKER_WORKSPACE_BUILD = "/workspace/build"

# Timeouts (seconds)
TIMEOUT_CONFIGURE = 300
TIMEOUT_BUILD = 600
TIMEOUT_LONG_BUILD = 1200
TIMEOUT_RUN_COMMAND = 600
TIMEOUT_INSTALL = 120

# ReAct loop parameters
MAX_TURNS = 20
MAX_TURNS_ERROR_ANALYSIS = 10
TURNS_LOW_WARNING = 5  # Warn model when this many turns remain
TURNS_EXTENSION = 10  # Extra turns granted per request_more_turns call
MAX_TURN_EXTENSIONS = 3  # Maximum number of extensions allowed

# Token limits
MAX_CONTEXT_TOKENS = 100000
LARGE_OUTPUT_TOKEN_THRESHOLD = 1000
CHARS_PER_TOKEN = 4
