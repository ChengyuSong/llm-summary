"""Indirect call analysis components."""

from .callsites import IndirectCallsiteFinder
from .resolver import IndirectCallResolver
from .scanner import AddressTakenScanner

__all__ = ["AddressTakenScanner", "IndirectCallsiteFinder", "IndirectCallResolver"]
