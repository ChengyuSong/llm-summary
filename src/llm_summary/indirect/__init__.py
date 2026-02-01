"""Indirect call analysis components."""

from .callsites import IndirectCallsiteFinder
from .flow_summarizer import FlowSummarizer
from .resolver import IndirectCallResolver
from .scanner import AddressTakenScanner

__all__ = [
    "AddressTakenScanner",
    "FlowSummarizer",
    "IndirectCallsiteFinder",
    "IndirectCallResolver",
]
