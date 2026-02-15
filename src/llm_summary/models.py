"""Data models for the LLM summary system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TargetType(str, Enum):
    """Type of indirect call target."""

    ADDRESS_TAKEN = "address_taken"
    VIRTUAL_METHOD = "virtual_method"
    CONSTRUCTOR_ATTR = "constructor_attr"
    DESTRUCTOR_ATTR = "destructor_attr"
    SECTION_PLACED = "section_placed"
    IFUNC = "ifunc"
    WEAK_SYMBOL = "weak_symbol"


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
    canonical_signature: str | None = None
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
class BufferSizePair:
    """A paired (buffer, size) relationship for bounds checking."""

    buffer: str  # e.g., "buf", "data", "ptr->data"
    size: str  # e.g., "len", "buf_len", "ptr->length"
    kind: str  # "param_pair", "struct_field", "flexible_array"
    relationship: str  # e.g., "byte count", "element count", "max capacity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "size": self.size,
            "kind": self.kind,
            "relationship": self.relationship,
        }


@dataclass
class FreeOp:
    """Represents a single free/deallocation within a function."""

    target: str  # what gets freed: "ptr", "info_ptr->palette", "row_buf"
    target_kind: str  # "parameter", "field", "local", "return_value"
    deallocator: str  # "free", "png_free", "g_free", custom
    conditional: bool  # True if free is inside an if/error path
    nulled_after: bool  # True if pointer is set to NULL after free

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_kind": self.target_kind,
            "deallocator": self.deallocator,
            "conditional": self.conditional,
            "nulled_after": self.nulled_after,
        }


@dataclass
class FreeSummary:
    """Complete free/deallocation summary for a function."""

    function_name: str
    frees: list[FreeOp] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "frees": [f.to_dict() for f in self.frees],
            "description": self.description,
        }


@dataclass
class InitOp:
    """A single guaranteed initialization performed by a function (caller-visible only)."""

    target: str  # "*out", "ctx->data", "return value"
    target_kind: str  # "parameter", "field", "return_value"
    initializer: str  # "memset", "assignment", "calloc", "callee:init_struct"
    byte_count: str | None = None  # "n", "sizeof(T)", "full", or None if unknown

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target": self.target,
            "target_kind": self.target_kind,
            "initializer": self.initializer,
        }
        if self.byte_count is not None:
            result["byte_count"] = self.byte_count
        return result


@dataclass
class InitSummary:
    """Complete initialization summary for a function."""

    function_name: str
    inits: list[InitOp] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "inits": [i.to_dict() for i in self.inits],
            "description": self.description,
        }


@dataclass
class MemsafeContract:
    """A single safety pre-condition required for memory-safe execution."""

    target: str  # "ptr", "buf", "ctx->data", "s"
    contract_kind: str  # "not_null", "not_freed", "initialized", "buffer_size"
    description: str  # "buf must point to at least n bytes"
    size_expr: str | None = None  # buffer_size only: "n", "sizeof(T)", "strlen(src)+1"
    relationship: str | None = None  # buffer_size only: "byte_count", "element_count"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "target": self.target,
            "contract_kind": self.contract_kind,
            "description": self.description,
        }
        if self.size_expr is not None:
            result["size_expr"] = self.size_expr
        if self.relationship is not None:
            result["relationship"] = self.relationship
        return result


@dataclass
class MemsafeSummary:
    """Complete safety contract summary for a function (pre-conditions)."""

    function_name: str
    contracts: list[MemsafeContract] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "contracts": [c.to_dict() for c in self.contracts],
            "description": self.description,
        }


@dataclass
class SafetyIssue:
    """A potential memory safety issue found during verification."""

    location: str  # "line 42" or "call to memcpy at line 42"
    issue_kind: str  # "null_deref", "buffer_overflow", "use_after_free",
    # "double_free", "uninitialized_use"
    description: str  # human-readable explanation
    severity: str  # "high", "medium", "low"
    callee: str | None = None  # callee whose contract was violated, None if foo's own op
    contract_kind: str | None = None  # which contract kind was violated

    def to_dict(self) -> dict[str, Any]:
        result = {
            "location": self.location,
            "issue_kind": self.issue_kind,
            "description": self.description,
            "severity": self.severity,
        }
        if self.callee is not None:
            result["callee"] = self.callee
        if self.contract_kind is not None:
            result["contract_kind"] = self.contract_kind
        return result


@dataclass
class VerificationSummary:
    """Complete verification result for a function."""

    function_name: str
    simplified_contracts: list[MemsafeContract] = field(default_factory=list)
    issues: list[SafetyIssue] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function_name,
            "simplified_contracts": [c.to_dict() for c in self.simplified_contracts],
            "issues": [i.to_dict() for i in self.issues],
            "description": self.description,
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
    buffer_size_pairs: list[BufferSizePair] = field(default_factory=list)
    description: str = ""  # Human-readable summary

    def to_dict(self) -> dict[str, Any]:
        result = {
            "function": self.function_name,
            "allocations": [a.to_dict() for a in self.allocations],
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "description": self.description,
        }
        if self.buffer_size_pairs:
            result["buffer_size_pairs"] = [p.to_dict() for p in self.buffer_size_pairs]
        return result


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
    target_type: str = "address_taken"
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
    access_path: str = ""  # Full dereference chain, e.g., "ctx->events->handler"
    root_type: str = ""  # "arg", "global", or "return"
    root_name: str = ""  # e.g., parameter name "ctx" or global "g_context"
    file_path: str = ""  # Source file where flow occurs
    line_number: int = 0  # Line number where flow occurs

    def to_dict(self) -> dict[str, Any]:
        result = {
            "type": self.dest_type,
            "name": self.name,
            "confidence": self.confidence,
        }
        if self.access_path:
            result["access_path"] = self.access_path
        if self.root_type:
            result["root_type"] = self.root_type
        if self.root_name:
            result["root_name"] = self.root_name
        if self.file_path:
            result["file_path"] = self.file_path
        if self.line_number:
            result["line_number"] = self.line_number
        return result


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


class AssemblyType(str, Enum):
    """Type of assembly code detected."""

    STANDALONE_FILE = "standalone_file"
    INLINE_SOURCE = "inline_source"
    INLINE_LLVM_IR = "inline_llvm_ir"


@dataclass
class ContainerSummary:
    """Summary of a container/collection function detected via heuristic + LLM."""

    function_id: int
    container_arg: int  # 0-based index of container object param
    store_args: list[int] = field(default_factory=list)  # indices of value params stored INTO container
    load_return: bool = False  # return value loaded FROM container
    container_type: str = "other"  # hash_table, linked_list, tree, map, cache, queue, set, array, other
    confidence: str = "medium"  # high, medium, low
    heuristic_score: int = 0  # pre-filter score
    heuristic_signals: list[str] = field(default_factory=list)  # which signals triggered
    model_used: str = ""
    id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "container_arg": self.container_arg,
            "store_args": self.store_args,
            "load_return": self.load_return,
            "container_type": self.container_type,
            "confidence": self.confidence,
            "heuristic_score": self.heuristic_score,
            "heuristic_signals": self.heuristic_signals,
            "model_used": self.model_used,
        }


@dataclass
class AssemblyFinding:
    """Single assembly detection with location info."""

    asm_type: AssemblyType
    file_path: str
    line_number: int | None = None
    snippet: str | None = None
    pattern_matched: str | None = None

    def stable_key(self) -> str:
        """
        Generate a stable key for identifying this finding across builds.

        Uses file path + pattern (not line number, which can change).
        """
        return f"{self.asm_type.value}:{self.file_path}:{self.pattern_matched or ''}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "type": self.asm_type.value,
            "file_path": self.file_path,
        }
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.snippet:
            result["snippet"] = self.snippet
        if self.pattern_matched:
            result["pattern_matched"] = self.pattern_matched
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssemblyFinding":
        """Create from dictionary representation."""
        return cls(
            asm_type=AssemblyType(data["type"]),
            file_path=data["file_path"],
            line_number=data.get("line_number"),
            snippet=data.get("snippet"),
            pattern_matched=data.get("pattern_matched"),
        )


@dataclass
class AssemblyCheckResult:
    """Complete assembly verification result."""

    has_assembly: bool
    standalone_asm_files: list[AssemblyFinding] = field(default_factory=list)
    inline_asm_sources: list[AssemblyFinding] = field(default_factory=list)
    inline_asm_ir: list[AssemblyFinding] = field(default_factory=list)
    # Known unavoidable findings (filtered from above lists)
    known_unavoidable: list[AssemblyFinding] = field(default_factory=list)
    # Truncation flags (True if checker stopped early due to too many findings)
    standalone_truncated: bool = False
    inline_sources_truncated: bool = False
    inline_ir_truncated: bool = False

    @property
    def has_new_assembly(self) -> bool:
        """True if there are new (not known unavoidable) assembly findings."""
        return bool(self.standalone_asm_files or self.inline_asm_sources or self.inline_asm_ir)

    def summary(self) -> str:
        """Return human-readable summary of assembly findings."""
        if not self.has_assembly:
            return "No assembly code detected."

        parts = []
        if self.standalone_asm_files:
            files = [f.file_path for f in self.standalone_asm_files]
            count_str = f"{len(self.standalone_asm_files)}+" if self.standalone_truncated else str(len(self.standalone_asm_files))
            parts.append(f"{count_str} standalone .s/.S/.asm file(s): {', '.join(files[:3])}")
            if len(files) > 3:
                parts[-1] += f" (+{len(files) - 3} more)"
        if self.inline_asm_sources:
            count_str = f"{len(self.inline_asm_sources)}+" if self.inline_sources_truncated else str(len(self.inline_asm_sources))
            parts.append(f"{count_str} C/C++ file(s) with inline asm")
        if self.inline_asm_ir:
            count_str = f"{len(self.inline_asm_ir)}+" if self.inline_ir_truncated else str(len(self.inline_asm_ir))
            parts.append(f"{count_str} LLVM IR file(s) with inline asm")

        summary = "Assembly detected: " + "; ".join(parts) if parts else "No new assembly"
        if self.known_unavoidable:
            summary += f" ({len(self.known_unavoidable)} known unavoidable filtered)"
        if self.standalone_truncated or self.inline_sources_truncated or self.inline_ir_truncated:
            summary += " (stopped checking after limit, more may exist)"
        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "has_assembly": self.has_assembly,
            "has_new_assembly": self.has_new_assembly,
            "standalone_asm_files": [f.to_dict() for f in self.standalone_asm_files],
            "inline_asm_sources": [f.to_dict() for f in self.inline_asm_sources],
            "inline_asm_ir": [f.to_dict() for f in self.inline_asm_ir],
        }
        if self.known_unavoidable:
            result["known_unavoidable_count"] = len(self.known_unavoidable)
        if self.standalone_truncated:
            result["standalone_truncated"] = True
            result["standalone_note"] = "Stopped checking after limit, more files may exist"
        if self.inline_sources_truncated:
            result["inline_sources_truncated"] = True
            result["inline_sources_note"] = "Stopped checking after limit, more files may exist"
        if self.inline_ir_truncated:
            result["inline_ir_truncated"] = True
            result["inline_ir_note"] = "Stopped checking after limit, more files may exist"
        return result
