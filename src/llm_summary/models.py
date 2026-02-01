"""Data models for the LLM summary system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AllocationType(str, Enum):
    """Type of memory allocation."""

    HEAP = "heap"
    STACK = "stack"
    STATIC = "static"
    UNKNOWN = "unknown"


@dataclass
class Function:
    """Represents an extracted C/C++ function."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    source: str
    signature: str
    id: int | None = None
    source_hash: str | None = None

    def __hash__(self) -> int:
        return hash((self.name, self.signature, self.file_path))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Function):
            return False
        return (
            self.name == other.name
            and self.signature == other.signature
            and self.file_path == other.file_path
        )


@dataclass
class Allocation:
    """Represents a single memory allocation within a function."""

    alloc_type: AllocationType
    source: str  # e.g., "malloc", "calloc", "new"
    size_expr: str | None = None  # e.g., "n + 1", "sizeof(struct foo)"
    size_params: list[str] = field(default_factory=list)  # Parameters used in size
    returned: bool = False  # Is the allocation returned?
    stored_to: str | None = None  # Field/variable where stored
    may_be_null: bool = True  # Can the allocation fail?

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.alloc_type.value,
            "source": self.source,
            "size_expr": self.size_expr,
            "size_params": self.size_params,
            "returned": self.returned,
            "stored_to": self.stored_to,
            "may_be_null": self.may_be_null,
        }


@dataclass
class ParameterInfo:
    """Information about a function parameter's role in allocation."""

    role: str  # e.g., "size_indicator", "buffer", "count"
    used_in_allocation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "used_in_allocation": self.used_in_allocation,
        }


@dataclass
class AllocationSummary:
    """Complete allocation summary for a function."""

    function_name: str
    allocations: list[Allocation] = field(default_factory=list)
    parameters: dict[str, ParameterInfo] = field(default_factory=dict)
    description: str = ""  # Human-readable summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "allocations": [a.to_dict() for a in self.allocations],
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "description": self.description,
        }


@dataclass
class IndirectCallsite:
    """Represents an indirect call site (function pointer call)."""

    caller_function_id: int
    file_path: str
    line_number: int
    callee_expr: str  # e.g., "ctx->handler", "callbacks[i]"
    signature: str
    context_snippet: str
    id: int | None = None


@dataclass
class AddressTakenFunction:
    """A function whose address is taken somewhere."""

    function_id: int
    signature: str
    id: int | None = None


@dataclass
class AddressFlow:
    """Tracks where a function's address flows to."""

    function_id: int
    flow_target: str  # e.g., "struct task.callback", "global_handler"
    file_path: str | None = None
    line_number: int | None = None
    context_snippet: str | None = None
    id: int | None = None


@dataclass
class IndirectCallTarget:
    """A resolved target for an indirect call."""

    callsite_id: int
    target_function_id: int
    confidence: str  # "high", "medium", "low"
    llm_reasoning: str = ""


@dataclass
class FlowDestination:
    """Describes where a function pointer flows to."""

    dest_type: str  # "struct_field", "global_var", "parameter", "array"
    name: str  # e.g., "handler_t.on_event", "g_handlers", "register_handler[0]"
    confidence: str  # "high", "medium", "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.dest_type,
            "name": self.name,
            "confidence": self.confidence,
        }


@dataclass
class AddressFlowSummary:
    """LLM-generated summary of where a function's address flows."""

    function_id: int
    flow_destinations: list[FlowDestination] = field(default_factory=list)
    semantic_role: str = ""  # LLM's interpretation of callback purpose
    likely_callers: list[str] = field(default_factory=list)  # Likely caller function names
    model_used: str = ""
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "flow_destinations": [fd.to_dict() for fd in self.flow_destinations],
            "semantic_role": self.semantic_role,
            "likely_callers": self.likely_callers,
            "model_used": self.model_used,
        }


@dataclass
class CallEdge:
    """Represents a call edge in the call graph."""

    caller_id: int
    callee_id: int
    is_indirect: bool = False
    file_path: str | None = None
    line: int | None = None
    column: int | None = None

    def callsite_str(self) -> str:
        """Return callsite as 'file:line:column' string."""
        if self.file_path and self.line:
            if self.column:
                return f"{self.file_path}:{self.line}:{self.column}"
            return f"{self.file_path}:{self.line}"
        return ""
