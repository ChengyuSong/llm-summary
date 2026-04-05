"""Import KAMain call graph JSON into the summary database."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .db import SummaryDB
from .docker_paths import strip_docker_prefix as _strip_docker_prefix
from .models import CallEdge
from .stdlib import get_stdlib_attributes


@dataclass
class ImportStats:
    """Statistics from a call graph import."""

    functions_in_json: int = 0
    functions_matched: int = 0
    functions_matched_by_name: int = 0
    functions_matched_by_file: int = 0
    functions_matched_by_suffix: int = 0
    functions_matched_by_demangle: int = 0
    stubs_created: int = 0
    edges_imported: int = 0
    direct_edges: int = 0
    indirect_edges: int = 0
    edges_skipped_self: int = 0
    unresolved_functions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Functions in JSON: {self.functions_in_json}",
            f"  Matched to DB: {self.functions_matched}",
            f"    By name: {self.functions_matched_by_name}",
            f"    By file+name: {self.functions_matched_by_file}",
            f"    By suffix+name: {self.functions_matched_by_suffix}",
            f"    By demangled name: {self.functions_matched_by_demangle}",
            f"  Stubs created: {self.stubs_created}",
            f"Edges imported: {self.edges_imported}",
            f"  Direct: {self.direct_edges}",
            f"  Indirect: {self.indirect_edges}",
        ]
        if self.edges_skipped_self:
            lines.append(f"  Self-edges skipped: {self.edges_skipped_self}")
        if self.unresolved_functions:
            lines.append(f"Unresolved (no stub): {len(self.unresolved_functions)}")
        return "\n".join(lines)


_LLVM_INTRINSIC_MAP = {
    "llvm.memcpy": "memcpy",
    "llvm.memmove": "memmove",
    "llvm.memset": "memset",
}


def _normalize_callee_name(name: str) -> str:
    """Normalize LLVM intrinsic names to their C stdlib equivalents.

    KAMain emits intrinsics like 'llvm.memcpy.p0.p0.i64' — strip the
    type suffixes and map to the canonical C name so that stdlib summaries
    (keyed on 'memcpy' etc.) are matched correctly.
    """
    for prefix, canonical in _LLVM_INTRINSIC_MAP.items():
        if name == prefix or name.startswith(prefix + "."):
            return canonical
    return name


def _node_to_name_part(node: Any) -> str | None:
    """Convert a single AST node to a name string, or None to skip."""
    kind: str = node.kind
    if kind == "name":
        val: str = node.value
        return None if val == "_GLOBAL__N_1" else val
    if kind == "oper":
        return f"operator{node.value}"
    if kind == "oper_cast":
        return "operator cast"
    if kind == "abi":
        # abi wraps a name or oper — unwrap, skip the tag itself
        return _node_to_name_part(node.value)
    return None


def _extract_name_parts(name_node: Any) -> list[str]:
    """Walk an itanium_demangler name AST node and collect qualified name parts.

    Skips _GLOBAL__N_1 (anonymous namespace), template args, and ABI tags.
    Unwraps cv_qual (const/volatile methods).
    """
    node = name_node
    # Unwrap cv_qual
    while node.kind == "cv_qual":
        node = node.value
    parts: list[str] = []
    if node.kind == "qual_name":
        prev_name = ""
        for part in node.value:
            if part.kind == "ctor":
                parts.append(prev_name)
            elif part.kind == "dtor":
                parts.append(f"~{prev_name}")
            elif part.kind == "tpl_args":
                pass
            else:
                name = _node_to_name_part(part)
                if name is not None:
                    prev_name = name
                    parts.append(name)
    else:
        name = _node_to_name_part(node)
        if name is not None:
            parts.append(name)
    return parts


def _demangle_structured(name: str) -> str | None:
    """Demangle using itanium_demangler AST — returns qualified name without
    template args, ABI tags, anonymous namespace prefixes, or return types."""
    try:
        from itanium_demangler import parse  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        node = parse(name)
    except Exception:
        return None
    if node is None:
        return None
    # Extract the name node from function or top-level
    name_node = node.name if node.kind == "func" else node
    # Unwrap cv_qual (const/volatile methods)
    if name_node.kind == "cv_qual":
        name_node = name_node.value
    parts: list[str] = _extract_name_parts(name_node)
    if not parts:
        return None
    return "::".join(parts)


def _demangle(name: str) -> str | None:
    """Demangle a C++ symbol name via llvm-cxxfilt or c++filt subprocess."""
    if not name.startswith("_Z"):
        return None
    for tool in ("llvm-cxxfilt", "c++filt"):
        try:
            result = subprocess.run(
                [tool, name],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                demangled = result.stdout.strip()
                if demangled != name:
                    return demangled
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            pass
    return None


def _strip_return_type(qualified: str) -> str:
    """Strip leading return type from a demangled C++ qualified name.

    c++filt/llvm-cxxfilt sometimes includes the return type:
    'unsigned long libunwind::getSparcWCookie' -> 'libunwind::getSparcWCookie'
    'std::__1::pair<char*, char*> std::__1::__copy_trivial::operator()' ->
        'std::__1::__copy_trivial::operator()'

    Finds the last space at template depth 0 where the text after it
    contains '::' (i.e. looks like a qualified name, not a parameter).
    """
    # Find last space at depth 0 where remainder contains '::'
    depth = 0
    last_space = -1
    for i, ch in enumerate(qualified):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == " " and depth == 0:
            # Check if remainder looks like a qualified name
            rest = qualified[i + 1:]
            if "::" in rest:
                last_space = i
    if last_space == -1:
        return qualified
    return qualified[last_space + 1:]


_ANON_NS_PREFIX = "(anonymous namespace)::"
_ABI_TAG_RE = re.compile(r"\[abi:[^\]]*\]")


def _strip_anon_ns(name: str) -> str:
    """Strip '(anonymous namespace)::' prefixes from a demangled name.

    libclang omits anonymous namespace qualifiers, but c++filt includes them.
    """
    while name.startswith(_ANON_NS_PREFIX):
        name = name[len(_ANON_NS_PREFIX):]
    return name


def _strip_abi_tags(name: str) -> str:
    """Strip [abi:...] tags from a demangled name.

    c++filt emits e.g. 'std::__1::all_of[abi:ne180100]' but libclang
    stores just 'std::__1::all_of'.
    """
    return _ABI_TAG_RE.sub("", name)


def _extract_base_name(demangled: str) -> str:
    """Extract the base function name from a demangled C++ name.

    E.g., 'ns::Class::method(int, char*)' -> 'method'
    """
    # Strip everything from '(' onwards (parameters)
    paren_idx = demangled.find("(")
    if paren_idx != -1:
        demangled = demangled[:paren_idx]
    # Take the last component after '::'
    parts = demangled.split("::")
    return parts[-1]


def _strip_template_params(name: str) -> str:
    """Strip template parameters from a demangled C++ name.

    E.g., 'ns::Cls<A, B>::method(int)' -> 'ns::Cls::method(int)'
    Handles nested templates like 'A<B<C>>::foo' -> 'A::foo'.
    """
    result: list[str] = []
    depth = 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            result.append(ch)
    return "".join(result)



def _parse_ka_function_key(key: str) -> tuple[str | None, str]:
    """Parse a KAMain function key into (file_path, func_name).

    Internal: '/path/to/file.c:func_name' -> ('/path/to/file.c', 'func_name')
              'src/aio/aio.c:cleanup'     -> ('src/aio/aio.c', 'cleanup')
    External: 'func_name' -> (None, 'func_name')

    Handles edge case: C++ mangled names in internal functions like
    '/path/file.cc:_ZN3FooC2Ev' where ':' also appears in the key prefix.
    """
    if ":" in key:
        file_path, name = key.rsplit(":", 1)
        # Internal if the file part contains '/' (absolute or relative path)
        # or has a source file extension
        if "/" in file_path or "." in file_path:
            return file_path, name
    # External function: plain name
    return None, key


class CallGraphImporter:
    """Import KAMain call graph JSON into the summary database."""

    def __init__(self, db: SummaryDB, verbose: bool = False):
        self.db = db
        self.verbose = verbose
        # Cache: KAMain func key -> DB function ID
        self._id_cache: dict[str, int | None] = {}
        # Pre-build DB lookup caches
        self._db_funcs_by_name: dict[str, list[int]] = {}
        self._db_funcs_by_name_file: dict[tuple[str, str], int] = {}
        # id -> (file_path, line_start, line_end) for line-based disambiguation
        self._db_func_loc: dict[int, tuple[str, int, int]] = {}
        self._build_db_caches()

    def _build_db_caches(self) -> None:
        """Pre-load DB functions into lookup caches."""
        all_funcs = self.db.get_all_functions()
        for f in all_funcs:
            if f.id is None:
                continue
            self._db_funcs_by_name.setdefault(f.name, []).append(f.id)
            self._db_funcs_by_name_file[(f.name, f.file_path)] = f.id
            self._db_func_loc[f.id] = (f.file_path, f.line_start, f.line_end)

    def _match_demangled(
        self, qualified: str, ka_file: str,
        ka_line_start: int, ka_line_end: int = 0,
    ) -> int | None:
        """Try to match a demangled qualified name against DB functions."""
        ids = self._db_funcs_by_name.get(qualified, [])
        if len(ids) == 1:
            return ids[0]
        if len(ids) > 1:
            return self._pick_by_location(ids, ka_file, ka_line_start, ka_line_end)
        return None

    def _pick_by_location(
        self, ids: list[int], ka_file: str, ka_line_start: int,
        ka_line_end: int = 0,
    ) -> int | None:
        """Disambiguate multiple name-matched candidates using file + line.

        Returns the DB function ID whose file suffix matches and whose
        line range overlaps with (or is closest to) the KAMain range,
        or None if no candidate matches by file.
        """
        if not ka_file or not ka_line_start:
            return None
        if not ka_line_end:
            ka_line_end = ka_line_start
        ka_suffix = _file_suffix(ka_file, components=2)
        candidates: list[tuple[int, int]] = []  # (id, distance)
        for fid in ids:
            loc = self._db_func_loc.get(fid)
            if loc is None:
                continue
            db_file, db_start, db_end = loc
            # Check file suffix match
            if not db_file.endswith(ka_suffix):
                continue
            # Check range overlap: [ka_start, ka_end] ∩ [db_start, db_end]
            if ka_line_start <= db_end and ka_line_end >= db_start:
                # Overlapping — compute overlap size (larger = better, use
                # negative distance so sort picks it first)
                overlap = min(ka_line_end, db_end) - max(ka_line_start, db_start)
                candidates.append((fid, -overlap))
            else:
                dist = min(
                    abs(ka_line_start - db_end), abs(ka_line_end - db_start),
                )
                candidates.append((fid, dist))
        if not candidates:
            return None
        # Pick the closest (prefer overlapping, i.e. negative distance)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def import_json(
        self, json_path: Path, clear_existing: bool = False
    ) -> ImportStats:
        """Import KAMain callgraph JSON into the DB."""
        with open(json_path) as f:
            data = json.load(f)

        functions = data.get("functions", {})
        metadata = data.get("metadata", {})
        stats = ImportStats(functions_in_json=len(functions))

        if self.verbose:
            print(f"KAMain metadata: {json.dumps(metadata)}")

        if clear_existing:
            deleted = self.db.clear_call_edges()
            stubs_deleted = self.db.clear_call_graph_stubs()
            if self.verbose:
                print(f"Cleared {deleted} existing call edges, {stubs_deleted} stubs")
            # Rebuild caches after deleting stubs
            if stubs_deleted:
                self._db_funcs_by_name.clear()
                self._db_funcs_by_name_file.clear()
                self._db_func_loc.clear()
                self._build_db_caches()

        # Phase 1: Resolve all function keys to DB IDs
        for ka_key, ka_info in functions.items():
            self._resolve_func_id(ka_key, ka_info, stats)

        # Phase 2: Import call edges
        edges_batch: list[CallEdge] = []
        # Track indirect edges per caller for callsite back-fill
        indirect_targets: dict[int, list[str]] = {}  # caller_id -> [callee_name]

        for ka_key, ka_info in functions.items():
            caller_id = self._id_cache.get(ka_key)
            if caller_id is None:
                continue

            caller_file = ka_info.get("file", "")

            for callee_entry in ka_info.get("callees", []):
                callee_name = _normalize_callee_name(callee_entry.get("callee", ""))
                call_type = callee_entry.get("call_type", "direct")
                line = callee_entry.get("line")

                # Resolve callee - could be a full key or just a name
                callee_id = self._resolve_callee(callee_name, functions, stats)
                if callee_id is None:
                    continue

                if caller_id == callee_id:
                    stats.edges_skipped_self += 1
                    continue

                is_indirect = call_type == "indirect"
                edge = CallEdge(
                    caller_id=caller_id,
                    callee_id=callee_id,
                    is_indirect=is_indirect,
                    file_path=caller_file,
                    line=line,
                )
                edges_batch.append(edge)

                if is_indirect:
                    stats.indirect_edges += 1
                    # Extract short callee name for callsite back-fill
                    short = _parse_ka_function_key(callee_name)[1]
                    indirect_targets.setdefault(caller_id, []).append(short)
                else:
                    stats.direct_edges += 1

        # Batch insert
        if edges_batch:
            self.db.add_call_edges_batch(edges_batch)
        stats.edges_imported = len(edges_batch)

        # Phase 3: Back-fill __indirect__ callsites with resolved callee names
        if indirect_targets:
            self._backfill_indirect_callsites(indirect_targets)

        return stats

    def _backfill_indirect_callsites(
        self, indirect_targets: dict[int, list[str]],
    ) -> None:
        """Replace ``__indirect__`` callsite placeholders with resolved names.

        The extractor records indirect calls as ``callee='__indirect__'``.
        After KAMain resolves them, this method patches in the real callee
        names so the verify pass can annotate them with contracts.
        """
        for caller_id, callee_names in indirect_targets.items():
            funcs = [
                f for f in self.db.get_all_functions() if f.id == caller_id
            ]
            if not funcs:
                continue
            func = funcs[0]
            if not func.callsites:
                continue

            # Find __indirect__ entries and assign resolved names in order
            callee_iter = iter(callee_names)
            updated = False
            for cs in func.callsites:
                if cs.get("callee") == "__indirect__":
                    resolved = next(callee_iter, None)
                    if resolved is not None:
                        cs["callee"] = resolved
                        cs.pop("is_indirect", None)
                        cs["was_indirect"] = True
                        updated = True

            if updated:
                self.db.update_callsites(caller_id, func.callsites)

    def _resolve_callee(
        self, callee_ref: str, all_functions: dict, stats: ImportStats
    ) -> int | None:
        """Resolve a callee reference from a callees entry.

        The callee field can be:
        - A full key matching a functions dict key (e.g., '/path/file.c:func')
        - Just a function name (e.g., 'malloc')
        """
        # Check if it's already cached (as a full key)
        if callee_ref in self._id_cache:
            return self._id_cache[callee_ref]

        # Check if it matches a key in the functions dict
        if callee_ref in all_functions:
            return self._id_cache.get(callee_ref)

        # It might be a name reference - try to find the full key
        # Look for keys ending with :callee_ref
        for key in all_functions:
            _, name = _parse_ka_function_key(key)
            if name == callee_ref and key in self._id_cache:
                return self._id_cache[key]

        # Not in KAMain functions dict - try direct DB lookup
        if callee_ref not in self._id_cache:
            self._resolve_func_id(callee_ref, {}, stats)
        return self._id_cache.get(callee_ref)

    def _resolve_func_id(
        self, ka_key: str, ka_info: dict, stats: ImportStats
    ) -> int | None:
        """Resolve a KAMain function key to a DB function ID."""
        if ka_key in self._id_cache:
            return self._id_cache[ka_key]

        file_path, raw_name = _parse_ka_function_key(ka_key)
        if file_path:
            file_path = _strip_docker_prefix(file_path)
        name = _normalize_callee_name(raw_name)
        ka_file = _strip_docker_prefix(ka_info.get("file", file_path or ""))
        ka_linkage = ka_info.get("linkage", "external")
        ka_line_start = ka_info.get("line_start", 0)
        ka_line_end = ka_info.get("line_end", 0)

        func_id = None

        # Strategy 1: Match by name + file (internal functions)
        if file_path and name:
            # Exact file path match
            key = (name, file_path)
            if key in self._db_funcs_by_name_file:
                func_id = self._db_funcs_by_name_file[key]
                stats.functions_matched_by_file += 1
            else:
                # Try suffix matching (build dirs differ)
                func = self.db.find_function_by_name_and_file_suffix(
                    name, _file_suffix(file_path)
                )
                if func:
                    func_id = func.id
                    stats.functions_matched_by_suffix += 1

        # Strategy 2: Match by name only (external functions / fallback)
        if func_id is None and name:
            ids = self._db_funcs_by_name.get(name, [])
            if len(ids) == 1:
                func_id = ids[0]
                stats.functions_matched_by_name += 1
            elif len(ids) > 1:
                func_id = self._pick_by_location(ids, ka_file, ka_line_start, ka_line_end)
                if func_id is not None:
                    stats.functions_matched_by_name += 1
            if func_id is None and name.startswith("_Z"):
                # Try structured demangling first (AST-based, no string hacks)
                qualified = _demangle_structured(name)
                if qualified:
                    func_id = self._match_demangled(
                        qualified, ka_file, ka_line_start, ka_line_end,
                    )
                # Fall back to c++filt string demangling
                if func_id is None:
                    demangled = _demangle(name)
                    if demangled:
                        paren_idx = demangled.find("(")
                        q = demangled[:paren_idx] if paren_idx != -1 else demangled
                        q = _strip_return_type(q)
                        q = _strip_anon_ns(q)
                        q = _strip_abi_tags(q)
                        if q != qualified:  # avoid re-trying same name
                            func_id = self._match_demangled(
                                q, ka_file, ka_line_start, ka_line_end,
                            )
                        if func_id is None:
                            stripped = _strip_template_params(q)
                            if stripped != q and stripped != qualified:
                                func_id = self._match_demangled(
                                    stripped, ka_file, ka_line_start, ka_line_end,
                                )
                        if func_id is None:
                            base = _extract_base_name(demangled)
                            func_id = self._match_demangled(
                                base, ka_file, ka_line_start, ka_line_end,
                            )
                if func_id is not None:
                    stats.functions_matched_by_demangle += 1

        if func_id is not None:
            stats.functions_matched += 1
        else:
            # Create stub
            func_id = self.db.insert_function_stub(
                name=name,
                file_path=ka_file,
                line_start=ka_line_start,
                line_end=ka_line_end,
                linkage=ka_linkage,
                attributes=get_stdlib_attributes(name),
            )
            stats.stubs_created += 1
            # Update caches
            self._db_funcs_by_name.setdefault(name, []).append(func_id)
            if ka_file:
                self._db_funcs_by_name_file[(name, ka_file)] = func_id

            if self.verbose:
                print(f"  Stub: {ka_key} -> id={func_id}")

        self._id_cache[ka_key] = func_id
        return func_id


def _file_suffix(path: str, components: int = 3) -> str:
    """Get the last N path components as a suffix for matching.

    Strips Docker container path prefixes first so that paths like
    '/workspace/src/dh.c' match host-path DB entries ending in 'dh.c'.

    E.g., '/magma/targets/libpng/repo/png.c' -> 'repo/png.c' (components=2)
    """
    path = _strip_docker_prefix(path)
    parts = Path(path).parts
    if len(parts) <= components:
        return path
    return str(Path(*parts[-components:]))
