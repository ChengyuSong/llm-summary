"""Container/collection function detection via heuristic pre-filter + LLM confirmation."""

import json
import re

from .db import SummaryDB
from .llm.base import LLMBackend
from .models import ContainerSummary, Function

# Container-related keywords in function names
CONTAINER_KEYWORDS = {
    "hash", "map", "list", "queue", "tree", "cache", "table", "set",
    "btree", "rbtree", "heap", "pool", "trie", "bucket", "ring",
    "dict", "slist", "dlist", "deque", "stack", "avl", "skiplist",
    "lru", "fifo", "lifo",
}

# Action keywords suggesting container operations
ACTION_KEYWORDS = {
    "insert", "find", "lookup", "add", "remove", "get", "put",
    "push", "pop", "enqueue", "dequeue", "search", "store",
    "retrieve", "fetch", "delete", "erase", "append", "prepend",
    "emplace", "contains", "has",
}

CONTAINER_PROMPT = """You are analyzing a C/C++ function to determine if it is a container/collection \
operation (hash table, linked list, tree, map, cache, queue, etc.) that stores \
or retrieves pointer values.

## Function Source

```c
{source_with_lines}
```

Function: `{name}`
Signature: `{signature}`
Parameters (0-indexed): {param_list}

## Detected Evidence

The following patterns were detected in this function:
{evidence_section}

## What is NOT a container function

- Struct initializers that set fields once during setup (not dynamic add/remove)
- Simple setters like set_callback(ctx, fn) that store a single value, not a collection
- Callback registration functions (one slot, not a dynamic collection)
- Functions that just read struct fields without collection semantics
- Memory allocators (malloc wrappers, pool allocators that return raw memory)
- String manipulation functions (even if they use linked structures internally)

## Task

Based on the source code and evidence above, is this function a container \
operation that stores or retrieves pointer values in a dynamic collection?

Respond in JSON:
```json
{{
  "is_container": true,
  "name": "function name",
  "container_arg": 0,
  "store_args": [2],
  "load_return": false,
  "container_type": "hash_table",
  "confidence": "high",
  "reasoning": "brief explanation"
}}
```

If this is NOT a container function, return:
```json
{{"is_container": false, "reasoning": "brief explanation why not"}}
```
"""


class ContainerDetector:
    """
    Detects container/collection functions using heuristic pre-filter + LLM confirmation.

    Phase 1: Score each function based on source code patterns (no LLM).
    Phase 2: For candidates above threshold, use LLM to confirm and extract details.
    """

    def __init__(
        self,
        db: SummaryDB,
        llm: LLMBackend | None = None,
        verbose: bool = False,
        log_file: str | None = None,
        min_score: int = 5,
    ):
        self.db = db
        self.llm = llm
        self.verbose = verbose
        self.log_file = log_file
        self.min_score = min_score
        self._stats = {
            "functions_scanned": 0,
            "candidates": 0,
            "llm_calls": 0,
            "containers_found": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return self._stats.copy()

    def detect_all(self, force: bool = False) -> dict[int, ContainerSummary]:
        """
        Run full detection: heuristic pre-filter + LLM confirmation.

        Args:
            force: If True, re-analyze even if summary exists

        Returns:
            Mapping of function_id to ContainerSummary
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        if self.verbose:
            print(f"Scanning {len(functions)} functions for container patterns...")

        # Phase 1: Heuristic pre-filter
        candidates = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self._heuristic_score(func)
            if score >= self.min_score:
                candidates.append((func, score, signals))

        self._stats["candidates"] = len(candidates)

        if self.verbose:
            print(f"Found {len(candidates)} candidates (score >= {self.min_score})")

        # Phase 2: LLM confirmation
        results: dict[int, ContainerSummary] = {}

        for func, score, signals in candidates:
            # Check cache
            if not force and self.db.has_container_summary(func.id):
                existing = self.db.get_container_summary(func.id)
                if existing:
                    results[func.id] = existing
                    self._stats["cache_hits"] += 1
                    if self.verbose:
                        print(f"  Cached: {func.name}")
                    continue

            if self.llm is None:
                # No LLM - store based on heuristic only
                summary = ContainerSummary(
                    function_id=func.id,
                    container_arg=0,
                    store_args=[],
                    load_return=False,
                    container_type="other",
                    confidence="low",
                    heuristic_score=score,
                    heuristic_signals=[s.split(": ", 1)[0] if ": " in s else s for s in signals],
                    model_used="heuristic_only",
                )
                self.db.add_container_summary(summary)
                results[func.id] = summary
                self._stats["containers_found"] += 1
                continue

            # LLM confirmation
            summary = self._analyze_function(func, score, signals)
            if summary:
                self.db.add_container_summary(summary)
                results[func.id] = summary
                self._stats["containers_found"] += 1

        return results

    def heuristic_only(self) -> list[tuple[Function, int, list[str]]]:
        """
        Run heuristic scoring only (no LLM calls).

        Returns:
            List of (function, score, signals) tuples for candidates above threshold.
        """
        functions = self.db.get_all_functions()
        self._stats["functions_scanned"] = len(functions)

        candidates = []
        for func in functions:
            if not func.source:
                continue
            score, signals = self._heuristic_score(func)
            if score >= self.min_score:
                candidates.append((func, score, signals))

        self._stats["candidates"] = len(candidates)
        return candidates

    def _heuristic_score(self, func: Function) -> tuple[int, list[str]]:
        """
        Score a function based on container-related patterns.

        Returns:
            (score, evidence_signals) where each signal is a descriptive string.
        """
        score = 0
        signals: list[str] = []
        name_lower = func.name.lower()
        source = func.source or ""

        # --- Name-based signals ---

        # Container keyword in name
        for kw in CONTAINER_KEYWORDS:
            if kw in name_lower:
                score += 3
                signals.append(f"container_keyword: name contains '{kw}'")
                break  # Only count once

        # Action keyword in name
        for kw in ACTION_KEYWORDS:
            # Match as word boundary: _insert, insert_, insert (whole name part)
            if re.search(rf'(?:^|_){kw}(?:_|$)', name_lower):
                score += 2
                signals.append(f"action_keyword: name contains '{kw}'")
                break

        # --- Signature-based signals ---
        sig = func.signature or ""

        # void* parameter (generic value storage)
        void_star_params = re.findall(r'void\s*\*(?!\*)\s*\w*', sig)
        if void_star_params:
            score += 2
            signals.append(f"void_ptr_param: {void_star_params[0].strip()}")

        # void* return type
        if re.match(r'\s*void\s*\*(?!\*)', sig):
            score += 2
            signals.append("void_ptr_return: returns void*")

        # void** parameter (output param)
        void_dstar_params = re.findall(r'void\s*\*\*\s*\w*', sig)
        if void_dstar_params:
            score += 2
            signals.append(f"void_dblptr_param: {void_dstar_params[0].strip()}")

        # --- Body-based signals ---
        if not source:
            return score, signals

        lines = source.split('\n')

        # Parse parameter names from signature for tracking stores
        param_names = self._extract_param_names(sig)

        # Param pointer stored to struct field: node->data = value;
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Match patterns like: ptr->field = param; or ptr->field = (type*)param;
            m = re.match(r'(\w+(?:->|\.)(?:\w+(?:->|\.))*\w+)\s*=\s*(.+?)\s*;', stripped)
            if m:
                lhs = m.group(1)
                rhs = m.group(2).strip()
                # Check if RHS references a parameter
                rhs_clean = re.sub(r'\([^)]*\)\s*', '', rhs)  # remove casts
                for pname in param_names:
                    if pname and re.search(rf'\b{re.escape(pname)}\b', rhs_clean):
                        score += 4
                        signals.append(f"param_stored_to_field: Line {i}: `{stripped}` stores param '{pname}' into struct field")
                        break
                else:
                    continue
                break  # Only count this signal once

        # Return of struct field (pointer)
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            m = re.match(r'return\s+(\w+(?:->|\.)(?:\w+(?:->|\.))*\w+)\s*;', stripped)
            if m:
                score += 4
                signals.append(f"return_struct_field: Line {i}: `{stripped}` returns struct field")
                break

        # next/prev pointer manipulation
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.search(r'\b\w+(?:->|\.)(?:next|prev|left|right|child|parent|sibling|flink|blink)\b', stripped):
                score += 3
                signals.append(f"linked_ptr_manipulation: Line {i}: `{stripped.strip()[:60]}`")
                break

        # Hash/bucket computation
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if re.search(r'%\s*\w*(?:size|cap|len|count|num|buckets|slots)\b', stripped) or \
               re.search(r'\b(?:hash|bucket)\s*[=(]', stripped, re.IGNORECASE):
                score += 2
                signals.append(f"hash_computation: Line {i}: `{stripped.strip()[:60]}`")
                break

        # void* cast in body
        if re.search(r'\(\s*void\s*\*\s*\)', source):
            score += 1
            signals.append("void_cast: body contains (void*) cast")

        # Key comparison call
        if re.search(r'\b(?:strcmp|memcmp|strncmp|wcscmp)\s*\(', source):
            score += 1
            signals.append("key_comparison: body calls strcmp/memcmp")

        return score, signals

    def _extract_param_names(self, signature: str) -> list[str]:
        """Extract parameter names from a C function signature."""
        # Find content between parentheses
        m = re.search(r'\(([^)]*)\)', signature)
        if not m:
            return []

        params_str = m.group(1).strip()
        if not params_str or params_str == 'void':
            return []

        names = []
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            # Remove array brackets
            param = re.sub(r'\[.*?\]', '', param)
            # Get the last word (parameter name), ignoring pointer stars
            tokens = param.replace('*', ' ').split()
            if tokens:
                name = tokens[-1]
                # Filter out type-only params (e.g., "void" or "int")
                if name not in ('void', 'int', 'char', 'long', 'short', 'unsigned',
                                'signed', 'float', 'double', 'const', 'struct',
                                'enum', 'union', 'size_t', 'uint8_t', 'uint16_t',
                                'uint32_t', 'uint64_t', 'int8_t', 'int16_t',
                                'int32_t', 'int64_t', 'bool', '...'):
                    names.append(name)
                else:
                    names.append('')
            else:
                names.append('')

        return names

    def _build_prompt(self, func: Function, signals: list[str]) -> str:
        """Build the evidence-grounded LLM prompt for a function."""
        # Add line numbers to source
        lines = (func.source or "").split('\n')
        numbered = []
        for i, line in enumerate(lines, func.line_start or 1):
            numbered.append(f"{i:4d} | {line}")
        source_with_lines = '\n'.join(numbered)

        # Build parameter list
        param_names = self._extract_param_names(func.signature or "")
        sig = func.signature or ""
        m = re.search(r'\(([^)]*)\)', sig)
        param_list_str = ""
        if m:
            params_raw = m.group(1).strip()
            if params_raw and params_raw != 'void':
                parts = [p.strip() for p in params_raw.split(',')]
                param_entries = []
                for idx, part in enumerate(parts):
                    pname = param_names[idx] if idx < len(param_names) else "?"
                    param_entries.append(f"  [{idx}] {part}")
                param_list_str = '\n'.join(param_entries)

        if not param_list_str:
            param_list_str = "(none)"

        # Build evidence section
        evidence_lines = []
        for s in signals:
            evidence_lines.append(f"- {s}")
        evidence_section = '\n'.join(evidence_lines) if evidence_lines else "(no specific evidence)"

        return CONTAINER_PROMPT.format(
            source_with_lines=source_with_lines,
            name=func.name,
            signature=func.signature or "(unknown)",
            param_list=param_list_str,
            evidence_section=evidence_section,
        )

    def _analyze_function(
        self, func: Function, score: int, signals: list[str]
    ) -> ContainerSummary | None:
        """
        Analyze a single candidate function using the LLM.

        Returns ContainerSummary if confirmed as container, None otherwise.
        """
        prompt = self._build_prompt(func, signals)

        try:
            if self.verbose:
                print(f"  Analyzing: {func.name} (score={score})")

            response = self.llm.complete(prompt)
            self._stats["llm_calls"] += 1

            if self.log_file:
                self._log_interaction(func.name, prompt, response)

            return self._parse_response(response, func, score, signals)

        except Exception as e:
            self._stats["errors"] += 1
            if self.verbose:
                print(f"  Error analyzing {func.name}: {e}")
            return None

    def _parse_response(
        self, response: str, func: Function, score: int, signals: list[str]
    ) -> ContainerSummary | None:
        """Parse LLM response. Returns None if is_container=false."""
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                if self.verbose:
                    print(f"  Failed to extract JSON from response for {func.name}")
                return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"  JSON parse error for {func.name}: {e}")
            return None

        if not data.get("is_container", False):
            if self.verbose:
                reason = data.get("reasoning", "no reason given")
                print(f"  Not a container: {func.name} - {reason}")
            return None

        container_arg = data.get("container_arg", 0)
        store_args = data.get("store_args", [])
        if not isinstance(store_args, list):
            store_args = []

        return ContainerSummary(
            function_id=func.id,
            container_arg=container_arg,
            store_args=store_args,
            load_return=bool(data.get("load_return", False)),
            container_type=data.get("container_type", "other"),
            confidence=data.get("confidence", "medium"),
            heuristic_score=score,
            heuristic_signals=[s.split(": ", 1)[0] if ": " in s else s for s in signals],
            model_used=self.llm.model if self.llm else "",
        )

    def _log_interaction(self, func_name: str, prompt: str, response: str) -> None:
        """Log LLM interaction to file."""
        import datetime

        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n{'='*80}\n")
            f.write(f"[CONTAINER DETECTION] Function: {func_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.llm.model if self.llm else 'N/A'}\n")
            f.write(f"{'-'*80}\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write(f"\n{'-'*80}\n")
            f.write("RESPONSE:\n")
            f.write(response)
            f.write(f"\n{'='*80}\n\n")
