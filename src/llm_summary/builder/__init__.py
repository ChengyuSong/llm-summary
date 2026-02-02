"""Build system agent for OSS project analysis."""

from .detector import BuildSystem, detect_build_system

__all__ = ["BuildSystem", "detect_build_system"]
