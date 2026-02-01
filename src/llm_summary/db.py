"""SQLite database for storing functions, summaries, and call graph."""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import (
    AddressFlow,
    AddressTakenFunction,
    Allocation,
    AllocationSummary,
    AllocationType,
    CallEdge,
    Function,
    IndirectCallsite,
    IndirectCallTarget,
    ParameterInfo,
)

SCHEMA = """
-- Functions table
CREATE TABLE IF NOT EXISTS functions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    signature TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    source TEXT,
    source_hash TEXT,
    UNIQUE(name, signature, file_path)
);

-- Allocation summaries table
CREATE TABLE IF NOT EXISTS allocation_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    summary_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_used TEXT,
    UNIQUE(function_id)
);

-- Call graph edges (with callsite info)
CREATE TABLE IF NOT EXISTS call_edges (
    id INTEGER PRIMARY KEY,
    caller_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    callee_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    is_indirect INTEGER DEFAULT 0,
    file_path TEXT,
    line INTEGER,
    column INTEGER
);

-- Address-taken functions
CREATE TABLE IF NOT EXISTS address_taken_functions (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    signature TEXT NOT NULL,
    UNIQUE(function_id)
);

-- Where function addresses flow to
CREATE TABLE IF NOT EXISTS address_flows (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    flow_target TEXT NOT NULL,
    file_path TEXT,
    line_number INTEGER,
    context_snippet TEXT
);

-- Indirect call sites
CREATE TABLE IF NOT EXISTS indirect_callsites (
    id INTEGER PRIMARY KEY,
    caller_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    line_number INTEGER NOT NULL,
    callee_expr TEXT NOT NULL,
    signature TEXT NOT NULL,
    context_snippet TEXT
);

-- Resolved indirect call targets
CREATE TABLE IF NOT EXISTS indirect_call_targets (
    callsite_id INTEGER REFERENCES indirect_callsites(id) ON DELETE CASCADE,
    target_function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    confidence TEXT,
    llm_reasoning TEXT,
    PRIMARY KEY(callsite_id, target_function_id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_functions_name ON functions(name);
CREATE INDEX IF NOT EXISTS idx_functions_file ON functions(file_path);
CREATE INDEX IF NOT EXISTS idx_call_edges_caller ON call_edges(caller_id);
CREATE INDEX IF NOT EXISTS idx_call_edges_callee ON call_edges(callee_id);
CREATE INDEX IF NOT EXISTS idx_address_flows_function ON address_flows(function_id);
CREATE INDEX IF NOT EXISTS idx_indirect_callsites_caller ON indirect_callsites(caller_function_id);
"""


def compute_source_hash(source: str) -> str:
    """Compute a hash of the source code for change detection."""
    return hashlib.sha256(source.encode()).hexdigest()[:16]


class SummaryDB:
    """SQLite database interface for the summary system."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    # ========== Function Operations ==========

    def insert_function(self, func: Function) -> int:
        """Insert a function and return its ID."""
        source_hash = compute_source_hash(func.source) if func.source else None
        cursor = self.conn.execute(
            """
            INSERT OR REPLACE INTO functions
            (name, signature, file_path, line_start, line_end, source, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                func.name,
                func.signature,
                func.file_path,
                func.line_start,
                func.line_end,
                func.source,
                source_hash,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_functions_batch(self, functions: list[Function]) -> dict[Function, int]:
        """Batch insert functions and return mapping to IDs."""
        result = {}
        for func in functions:
            func_id = self.insert_function(func)
            func.id = func_id
            result[func] = func_id
        return result

    def get_function(self, func_id: int) -> Function | None:
        """Get a function by ID."""
        row = self.conn.execute(
            "SELECT * FROM functions WHERE id = ?", (func_id,)
        ).fetchone()
        if row:
            return self._row_to_function(row)
        return None

    def get_function_by_name(
        self, name: str, signature: str | None = None
    ) -> list[Function]:
        """Get functions by name, optionally filtering by signature."""
        if signature:
            rows = self.conn.execute(
                "SELECT * FROM functions WHERE name = ? AND signature = ?",
                (name, signature),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM functions WHERE name = ?", (name,)
            ).fetchall()
        return [self._row_to_function(row) for row in rows]

    def get_functions_by_file(self, file_path: str) -> list[Function]:
        """Get all functions in a file."""
        rows = self.conn.execute(
            "SELECT * FROM functions WHERE file_path = ?", (file_path,)
        ).fetchall()
        return [self._row_to_function(row) for row in rows]

    def get_all_functions(self) -> list[Function]:
        """Get all functions in the database."""
        rows = self.conn.execute("SELECT * FROM functions").fetchall()
        return [self._row_to_function(row) for row in rows]

    def _row_to_function(self, row: sqlite3.Row) -> Function:
        """Convert a database row to a Function object."""
        return Function(
            id=row["id"],
            name=row["name"],
            signature=row["signature"],
            file_path=row["file_path"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            source=row["source"],
            source_hash=row["source_hash"],
        )

    # ========== Summary Operations ==========

    def get_summary(
        self, name: str, signature: str | None = None
    ) -> AllocationSummary | None:
        """Get allocation summary for a function."""
        if signature:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM allocation_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ? AND f.signature = ?
                """,
                (name, signature),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT s.summary_json FROM allocation_summaries s
                JOIN functions f ON s.function_id = f.id
                WHERE f.name = ?
                """,
                (name,),
            ).fetchone()

        if row:
            return self._json_to_summary(row["summary_json"])
        return None

    def get_summary_by_function_id(self, func_id: int) -> AllocationSummary | None:
        """Get allocation summary for a function by ID."""
        row = self.conn.execute(
            "SELECT summary_json FROM allocation_summaries WHERE function_id = ?",
            (func_id,),
        ).fetchone()
        if row:
            return self._json_to_summary(row["summary_json"])
        return None

    def get_summaries_by_file(self, file_path: str) -> list[AllocationSummary]:
        """Get all summaries for functions in a file."""
        rows = self.conn.execute(
            """
            SELECT s.summary_json FROM allocation_summaries s
            JOIN functions f ON s.function_id = f.id
            WHERE f.file_path = ?
            """,
            (file_path,),
        ).fetchall()
        return [self._json_to_summary(row["summary_json"]) for row in rows]

    def upsert_summary(
        self, func: Function, summary: AllocationSummary, model_used: str = ""
    ) -> None:
        """Insert or update an allocation summary."""
        if func.id is None:
            raise ValueError("Function must have an ID")

        summary_json = json.dumps(summary.to_dict())
        self.conn.execute(
            """
            INSERT INTO allocation_summaries (function_id, summary_json, model_used)
            VALUES (?, ?, ?)
            ON CONFLICT(function_id) DO UPDATE SET
                summary_json = excluded.summary_json,
                updated_at = CURRENT_TIMESTAMP,
                model_used = excluded.model_used
            """,
            (func.id, summary_json, model_used),
        )
        self.conn.commit()

    def _json_to_summary(self, json_str: str) -> AllocationSummary:
        """Convert JSON string to AllocationSummary."""
        data = json.loads(json_str)
        allocations = [
            Allocation(
                alloc_type=AllocationType(a.get("type", "unknown")),
                source=a.get("source", ""),
                size_expr=a.get("size_expr"),
                size_params=a.get("size_params", []),
                returned=a.get("returned", False),
                stored_to=a.get("stored_to"),
                may_be_null=a.get("may_be_null", True),
            )
            for a in data.get("allocations", [])
        ]
        parameters = {
            k: ParameterInfo(
                role=v.get("role", ""),
                used_in_allocation=v.get("used_in_allocation", False),
            )
            for k, v in data.get("parameters", {}).items()
        }
        return AllocationSummary(
            function_name=data.get("function", ""),
            allocations=allocations,
            parameters=parameters,
            description=data.get("description", ""),
        )

    def needs_update(self, func: Function) -> bool:
        """Check if a function's summary needs updating based on source hash."""
        current_hash = compute_source_hash(func.source) if func.source else None
        row = self.conn.execute(
            """
            SELECT f.source_hash, s.id FROM functions f
            LEFT JOIN allocation_summaries s ON f.id = s.function_id
            WHERE f.name = ? AND f.signature = ? AND f.file_path = ?
            """,
            (func.name, func.signature, func.file_path),
        ).fetchone()

        if not row:
            return True  # Function not in DB
        if row["id"] is None:
            return True  # No summary exists
        if row["source_hash"] != current_hash:
            return True  # Source has changed

        return False

    # ========== Call Graph Operations ==========

    def add_call_edge(self, edge: CallEdge) -> int:
        """Add a call edge between two functions. Returns the edge ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO call_edges (caller_id, callee_id, is_indirect, file_path, line, column)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                edge.caller_id,
                edge.callee_id,
                1 if edge.is_indirect else 0,
                edge.file_path,
                edge.line,
                edge.column,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def add_call_edges_batch(self, edges: list[CallEdge]) -> None:
        """Batch insert call edges."""
        self.conn.executemany(
            """
            INSERT INTO call_edges (caller_id, callee_id, is_indirect, file_path, line, column)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.caller_id,
                    e.callee_id,
                    1 if e.is_indirect else 0,
                    e.file_path,
                    e.line,
                    e.column,
                )
                for e in edges
            ],
        )
        self.conn.commit()

    def get_callees(self, func_id: int) -> list[int]:
        """Get IDs of functions called by the given function."""
        rows = self.conn.execute(
            "SELECT DISTINCT callee_id FROM call_edges WHERE caller_id = ?", (func_id,)
        ).fetchall()
        return [row["callee_id"] for row in rows]

    def get_callers(self, func_id: int) -> list[int]:
        """Get IDs of functions that call the given function."""
        rows = self.conn.execute(
            "SELECT DISTINCT caller_id FROM call_edges WHERE callee_id = ?", (func_id,)
        ).fetchall()
        return [row["caller_id"] for row in rows]

    def get_all_call_edges(self) -> list[CallEdge]:
        """Get all call edges."""
        rows = self.conn.execute("SELECT * FROM call_edges").fetchall()
        return [self._row_to_call_edge(row) for row in rows]

    def get_call_edges_by_caller(self, caller_id: int) -> list[CallEdge]:
        """Get all call edges from a specific caller."""
        rows = self.conn.execute(
            "SELECT * FROM call_edges WHERE caller_id = ?", (caller_id,)
        ).fetchall()
        return [self._row_to_call_edge(row) for row in rows]

    def _row_to_call_edge(self, row: sqlite3.Row) -> CallEdge:
        """Convert a database row to a CallEdge object."""
        return CallEdge(
            caller_id=row["caller_id"],
            callee_id=row["callee_id"],
            is_indirect=bool(row["is_indirect"]),
            file_path=row["file_path"],
            line=row["line"],
            column=row["column"],
        )

    def invalidate_and_cascade(self, func_id: int) -> list[int]:
        """Invalidate a function's summary and all callers. Returns invalidated IDs."""
        invalidated = []
        to_process = [func_id]

        while to_process:
            current = to_process.pop()
            if current in invalidated:
                continue

            self.conn.execute(
                "DELETE FROM allocation_summaries WHERE function_id = ?", (current,)
            )
            invalidated.append(current)

            callers = self.get_callers(current)
            to_process.extend(callers)

        self.conn.commit()
        return invalidated

    # ========== Indirect Call Operations ==========

    def add_address_taken_function(self, atf: AddressTakenFunction) -> int:
        """Add an address-taken function."""
        cursor = self.conn.execute(
            """
            INSERT OR REPLACE INTO address_taken_functions (function_id, signature)
            VALUES (?, ?)
            """,
            (atf.function_id, atf.signature),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_address_taken_functions(
        self, signature: str | None = None
    ) -> list[AddressTakenFunction]:
        """Get address-taken functions, optionally filtered by signature."""
        if signature:
            rows = self.conn.execute(
                "SELECT * FROM address_taken_functions WHERE signature = ?",
                (signature,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM address_taken_functions"
            ).fetchall()

        return [
            AddressTakenFunction(
                id=row["id"],
                function_id=row["function_id"],
                signature=row["signature"],
            )
            for row in rows
        ]

    def add_address_flow(self, flow: AddressFlow) -> int:
        """Add an address flow record."""
        cursor = self.conn.execute(
            """
            INSERT INTO address_flows
            (function_id, flow_target, file_path, line_number, context_snippet)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                flow.function_id,
                flow.flow_target,
                flow.file_path,
                flow.line_number,
                flow.context_snippet,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_address_flows(self, function_id: int) -> list[AddressFlow]:
        """Get address flows for a function."""
        rows = self.conn.execute(
            "SELECT * FROM address_flows WHERE function_id = ?", (function_id,)
        ).fetchall()
        return [
            AddressFlow(
                id=row["id"],
                function_id=row["function_id"],
                flow_target=row["flow_target"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                context_snippet=row["context_snippet"],
            )
            for row in rows
        ]

    def add_indirect_callsite(self, callsite: IndirectCallsite) -> int:
        """Add an indirect call site."""
        cursor = self.conn.execute(
            """
            INSERT INTO indirect_callsites
            (caller_function_id, file_path, line_number, callee_expr, signature, context_snippet)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                callsite.caller_function_id,
                callsite.file_path,
                callsite.line_number,
                callsite.callee_expr,
                callsite.signature,
                callsite.context_snippet,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_indirect_callsites(
        self, caller_function_id: int | None = None
    ) -> list[IndirectCallsite]:
        """Get indirect call sites, optionally filtered by caller."""
        if caller_function_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM indirect_callsites WHERE caller_function_id = ?",
                (caller_function_id,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM indirect_callsites").fetchall()

        return [
            IndirectCallsite(
                id=row["id"],
                caller_function_id=row["caller_function_id"],
                file_path=row["file_path"],
                line_number=row["line_number"],
                callee_expr=row["callee_expr"],
                signature=row["signature"],
                context_snippet=row["context_snippet"],
            )
            for row in rows
        ]

    def add_indirect_call_target(self, target: IndirectCallTarget) -> None:
        """Add a resolved indirect call target."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO indirect_call_targets
            (callsite_id, target_function_id, confidence, llm_reasoning)
            VALUES (?, ?, ?, ?)
            """,
            (
                target.callsite_id,
                target.target_function_id,
                target.confidence,
                target.llm_reasoning,
            ),
        )
        self.conn.commit()

    def get_indirect_call_targets(self, callsite_id: int) -> list[IndirectCallTarget]:
        """Get resolved targets for an indirect call site."""
        rows = self.conn.execute(
            "SELECT * FROM indirect_call_targets WHERE callsite_id = ?",
            (callsite_id,),
        ).fetchall()
        return [
            IndirectCallTarget(
                callsite_id=row["callsite_id"],
                target_function_id=row["target_function_id"],
                confidence=row["confidence"],
                llm_reasoning=row["llm_reasoning"],
            )
            for row in rows
        ]

    # ========== Utility Operations ==========

    def clear_all(self) -> None:
        """Clear all data from the database."""
        tables = [
            "indirect_call_targets",
            "indirect_callsites",
            "address_flows",
            "address_taken_functions",
            "call_edges",
            "allocation_summaries",
            "functions",
        ]
        for table in tables:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        stats = {}
        tables = [
            "functions",
            "allocation_summaries",
            "call_edges",
            "address_taken_functions",
            "address_flows",
            "indirect_callsites",
            "indirect_call_targets",
        ]
        for table in tables:
            row = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            stats[table] = row["cnt"]
        return stats
