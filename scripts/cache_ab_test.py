#!/usr/bin/env python3
"""Cache hypothesis A/B test for the contract pipeline.

Per `docs/design-llm-first.md` Step 6: validate whether prompt caching
(mode 2) and cached multi-turn Q&A (mode 3) save tokens / wall-time over the
baseline (mode 1, no cache) before committing to the multi-round driver.

Three modes per (function, property):
  Mode 1 — uncached single-turn. One LLM call per property; no cache_control.
  Mode 2 — cached single-turn. One LLM call per property; system + source-prep
           cached. Three properties of one function share the source-prep cache.
  Mode 3 — cached multi-turn. Three LLM calls per property (enumerate → derive
           requires → emit JSON). System + source-prep cached; conversation
           grows per property. Three properties of one function share the
           system+source-prep cache.

Inputs come from a pre-built functions.db (per `llm-summary scan`). The
script walks the requested functions in topological (callee-first) order so
that by the time a caller is summarised, every callee has a Mode-1 summary
to inline as `// >>>` callee-contract hints.

Usage:
    source ~/project/llm-summary/venv/bin/activate
    python scripts/cache_ab_test.py \\
        --db func-scans/sv-benchmarks/ntdrivers-simplified/floppy_simpl3.cil-2/functions.db \\
        --backend claude --model claude-haiku-4-5@20251001 \\
        --functions KeSetEvent,FloppyPnpComplete \\
        --output cache_ab_results.json --summary-out cache_ab_summary.md -v
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Reuse prototype helpers — these are stable, importable functions.
sys.path.insert(0, str(Path(__file__).parent))
from contract_pipeline import (  # noqa: E402
    MEMLEAK_PROMPT,
    MEMSAFE_PROMPT,
    OVERFLOW_PROMPT,
    PROPERTY_SCHEMA,
    SYSTEM_PROMPT,
    FunctionSummary,
    _STDLIB_CONTRACTS,
    _build_callee_block,
    _build_db_edges,
    _data_model_note,
    _inline_callee_contracts,
    _is_nontrivial,
    _ordered_callee_names,
    _topo_order,
)

from llm_summary.builder.json_utils import extract_json  # noqa: E402
from llm_summary.db import SummaryDB  # noqa: E402
from llm_summary.models import Function  # noqa: E402

log = logging.getLogger("cache_ab_test")

PROPERTIES = ["memsafe", "memleak", "overflow"]

PROPERTY_PROMPT = {
    "memsafe": MEMSAFE_PROMPT,
    "memleak": MEMLEAK_PROMPT,
    "overflow": OVERFLOW_PROMPT,
}

# Multi-turn question library for Mode 3. These are intentionally short —
# the bulky context (annotated source + callee block) lives in the cached
# system message; user turns just steer the model through enumerate →
# derive → emit. Production Q&A would refine these per `design-llm-first.md`
# §Phase 3.
MULTI_TURN_QUESTIONS: dict[str, list[str]] = {
    "memsafe": [
        # T1: enumerate
        "List every memsafe-relevant site in this function: pointer "
        "dereferences, buffer indexings, calls whose `requires[memsafe]` is "
        "non-trivial. One per line: `<expr> @ <line> when <cond>`. "
        "Plain text, no JSON.",
        # T2: derive
        "For each site above, what must the CALLER establish so the site "
        "cannot trigger UB? One per line: `requires[memsafe]: <C-expr>`. "
        "Restate verbatim any callee `requires[memsafe]` you propagated. "
        "Plain text, no JSON.",
        # T3: emit
        "Now emit the final memsafe summary as JSON. Schema: "
        "`{requires: [...], ensures: [...], modifies: [...], "
        "notes: '...', noreturn: bool}`. Use the requires you listed; "
        "add ensures (what does this function establish for callers?) and "
        "modifies (stack/heap regions written). Output JSON only.",
    ],
    "memleak": [
        "List every memleak-relevant operation: heap acquisitions "
        "(malloc/realloc/calloc/equivalent), heap releases (free/equivalent), "
        "calls whose `ensures[memleak]: acquired` or non-trivial "
        "`requires[memleak]` apply. One per line: "
        "`acquire/release <res> @ <line> when <cond>`. "
        "Plain text, no JSON.",
        "For each acquisition, is it released on every path out of this "
        "function? If not, what must the caller release? One per line: "
        "`requires[memleak]: <fact>`. Plain text, no JSON.",
        "Now emit the final memleak summary as JSON. Schema as before. "
        "Output JSON only.",
    ],
    "overflow": [
        "List every overflow-relevant op: signed arithmetic (+ - * unary-), "
        "division/modulo, shifts, calls whose `requires[overflow]` is "
        "non-trivial. One per line: `<op> @ <line> when <cond>`. "
        "Plain text, no JSON.",
        "For each op, what input range avoids UB? One per line: "
        "`requires[overflow]: <C-expr or x: [lo, hi]>`. Plain text, no JSON.",
        "Now emit the final overflow summary as JSON. Schema as before. "
        "Output JSON only.",
    ],
}


# ── Backends (raw, so we control cache_control / cache_prompt placement) ────


@dataclass
class CallStat:
    """Per-LLM-call metrics."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    wall_time_s: float = 0.0
    content: str = ""


class ClaudeRaw:
    """Direct Anthropic SDK wrapper. Lets us place cache_control freely
    and supports both single-turn and multi-turn calls."""

    def __init__(self, model: str, max_tokens: int = 16384):
        self.model = model
        self.max_tokens = max_tokens
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        project_id = (
            os.environ.get("VERTEX_AI_PROJECT")
            or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUD_ML_PROJECT_ID")
        )
        location = os.environ.get("VERTEX_AI_LOCATION", "global")
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("anthropic package required") from e
        if api_key:
            self.client: Any = anthropic.Anthropic(api_key=api_key)
        elif project_id:
            from anthropic import AnthropicVertex
            self.client = AnthropicVertex(project_id=project_id, region=location)
        else:
            raise ValueError(
                "Set ANTHROPIC_API_KEY or GOOGLE_CLOUD_PROJECT for Vertex AI."
            )

    def call(
        self,
        system_blocks: list[dict],
        messages: list[dict],
    ) -> CallStat:
        """One Anthropic API call. system_blocks and messages are passed
        through as-is (caller controls cache_control)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_blocks,
            "messages": messages,
        }
        t0 = time.monotonic()
        resp = self.client.messages.create(**kwargs)
        wall = time.monotonic() - t0
        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text
        u = resp.usage
        return CallStat(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            wall_time_s=wall,
            content=content,
        )


class LlamaCppRaw:
    """Direct llama.cpp /v1/chat/completions wrapper.
    `cache_prompt: true` is set on every call; the server reuses KV across
    requests when prefixes match."""

    def __init__(self, model: str | None, host: str, port: int, timeout: float = 600.0):
        self.model = model or "default"
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def call(
        self,
        system_blocks: list[dict],
        messages: list[dict],
    ) -> CallStat:
        """One llama.cpp call. Flattens system_blocks (which may carry
        cache_control on Anthropic; ignored here) into a single system
        message in OpenAI format."""
        sys_text = "\n".join(blk.get("text", "") for blk in system_blocks)
        # Flatten message content blocks into OpenAI-shape strings.
        oa_messages: list[dict] = []
        if sys_text:
            oa_messages.append({"role": "system", "content": sys_text})
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                text = "".join(
                    blk.get("text", "")
                    for blk in content
                    if isinstance(blk, dict)
                )
            else:
                text = content or ""
            oa_messages.append({"role": m["role"], "content": text})

        payload = {
            "model": self.model,
            "messages": oa_messages,
            "temperature": 0.5,
            "stream": False,
            "cache_prompt": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        wall = time.monotonic() - t0
        content = ""
        if "choices" in result and result["choices"]:
            content = result["choices"][0]["message"].get("content", "") or ""
        usage = result.get("usage", {}) or {}
        # llama.cpp upstream sometimes reports cached prefix length as
        # `prompt_tokens_details.cached_tokens`. Best-effort extraction.
        cache_read = 0
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cache_read = int(details.get("cached_tokens", 0) or 0)
        return CallStat(
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cache_read_tokens=cache_read,
            cache_creation_tokens=0,
            wall_time_s=wall,
            content=content,
        )


# ── Source-prep ─────────────────────────────────────────────────────────────


def build_source_prep(
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    prop: str,
) -> str:
    """The bulky shared block: annotated source + callee block for one
    (function, property). Same across all 3 turns of mode 3 and across
    properties when only the prop-filter changes."""
    callee_names = _ordered_callee_names(func, edges, summaries)
    callee_block = _build_callee_block(func, summaries, prop, callee_names)
    source_inlined = _inline_callee_contracts(func, summaries, edges, prop)
    # Keep the format aligned with what PROPERTY_PROMPT[prop] would inline,
    # so the model sees an equivalent context across modes.
    return (
        f"{callee_block}\n\n"
        f"=== SOURCE ===\n{source_inlined}\n"
    )


def build_full_prompt(
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    prop: str,
    data_model_note: str,
) -> str:
    """Mode 1 / Mode 2 single-turn prompt: the full per-property prompt
    with source and callee block formatted in. Mirrors what
    contract_pipeline._summarize_one builds."""
    callee_names = _ordered_callee_names(func, edges, summaries)
    callee_block = _build_callee_block(func, summaries, prop, callee_names)
    source_inlined = _inline_callee_contracts(func, summaries, edges, prop)
    fmt: dict[str, Any] = dict(
        name=func.name,
        callee_block=callee_block,
        source=source_inlined,
    )
    if prop == "overflow":
        fmt["data_model_note"] = data_model_note
    return PROPERTY_PROMPT[prop].format(**fmt)


# ── Per-mode driver ─────────────────────────────────────────────────────────


@dataclass
class PropertyResult:
    """Outcome of summarising one property in one mode."""
    requires: list[str] = field(default_factory=list)
    ensures: list[str] = field(default_factory=list)
    modifies: list[str] = field(default_factory=list)
    notes: str = ""
    noreturn: bool = False
    calls: list[CallStat] = field(default_factory=list)
    parse_error: str | None = None

    @property
    def in_tok(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def out_tok(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def cache_read(self) -> int:
        return sum(c.cache_read_tokens for c in self.calls)

    @property
    def cache_create(self) -> int:
        return sum(c.cache_creation_tokens for c in self.calls)

    @property
    def wall(self) -> float:
        return sum(c.wall_time_s for c in self.calls)


def _parse_json_or_record(stat: CallStat, result: PropertyResult) -> None:
    """Pull requires/ensures/modifies/notes/noreturn from stat.content
    (assumed JSON) into result. Records parse_error if invalid."""
    try:
        data = extract_json(stat.content)
    except (json.JSONDecodeError, ValueError) as e:
        result.parse_error = f"{type(e).__name__}: {e}"
        return
    result.requires = list(data.get("requires") or [])
    result.ensures = list(data.get("ensures") or [])
    result.modifies = list(data.get("modifies") or [])
    result.notes = str(data.get("notes") or "")
    result.noreturn = bool(data.get("noreturn", False))


def run_mode_1(
    backend: ClaudeRaw | LlamaCppRaw,
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    data_model_note: str,
) -> dict[str, PropertyResult]:
    """Baseline: 3 separate calls, no cache markers."""
    results: dict[str, PropertyResult] = {}
    for prop in PROPERTIES:
        prompt = build_full_prompt(func, summaries, edges, prop, data_model_note)
        # Append schema instruction (mirrors complete_with_metadata's wrapping).
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(PROPERTY_SCHEMA, indent=2)}\n```\n"
            f"Output only the JSON object, no markdown fences or explanatory text."
        )
        sys_blocks = [{"type": "text", "text": SYSTEM_PROMPT}]
        messages = [{"role": "user", "content": full_prompt}]
        stat = backend.call(sys_blocks, messages)
        result = PropertyResult(calls=[stat])
        _parse_json_or_record(stat, result)
        results[prop] = result
    return results


def run_mode_2(
    backend: ClaudeRaw | LlamaCppRaw,
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    data_model_note: str,
) -> dict[str, PropertyResult]:
    """Cached single-turn. SYSTEM_PROMPT cached cross-call.
    Per-property: source-prep block cached, question is the tail."""
    results: dict[str, PropertyResult] = {}
    sys_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    for prop in PROPERTIES:
        source_prep = build_source_prep(func, summaries, edges, prop)
        # Property-specific question tail: lift just the tail of
        # PROPERTY_PROMPT[prop] (the "Output JSON …" + guidance) by replacing
        # the source/callee_block placeholders with markers that point back
        # to the cached prep.
        # Simpler approach: send the full prompt as one user message, but
        # split into two content blocks so source-prep is cached.
        question_tail = (
            f"# {prop.upper()} pass for `{func.name}`\n\n"
            f"Use the source and callee block above. "
            f"Answer for {prop} ONLY (skip other properties' concerns).\n\n"
            f"Output JSON in this schema:\n"
            f"```json\n{json.dumps(PROPERTY_SCHEMA, indent=2)}\n```\n"
            f"Output only the JSON object."
        )
        if prop == "overflow":
            question_tail = (
                f"## Data model\n{data_model_note}\n\n" + question_tail
            )
        user_blocks = [
            {
                "type": "text",
                "text": source_prep,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": question_tail},
        ]
        messages = [{"role": "user", "content": user_blocks}]
        stat = backend.call(sys_blocks, messages)
        result = PropertyResult(calls=[stat])
        _parse_json_or_record(stat, result)
        results[prop] = result
    return results


def run_mode_3(
    backend: ClaudeRaw | LlamaCppRaw,
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    data_model_note: str,
) -> dict[str, PropertyResult]:
    """Cached multi-turn. Per property: 3 turns (enumerate → derive → emit).
    SYSTEM_PROMPT + per-property source-prep cached. Each property is its
    own conversation (independent dialog over the cached prefix)."""
    results: dict[str, PropertyResult] = {}
    sys_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    for prop in PROPERTIES:
        source_prep = build_source_prep(func, summaries, edges, prop)
        if prop == "overflow":
            source_prep = f"## Data model\n{data_model_note}\n\n" + source_prep
        result = PropertyResult()
        # Build conversation incrementally: each turn appends the prior
        # assistant reply + a new user turn. Cache_control on system + on
        # the first user block (source-prep) keeps the prefix cached across
        # the 3 turns of this property.
        questions = MULTI_TURN_QUESTIONS[prop]
        # Turn 1: source-prep (cached) + first question
        user_t1: list[dict] = [
            {
                "type": "text",
                "text": source_prep,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": questions[0]},
        ]
        messages: list[dict] = [{"role": "user", "content": user_t1}]
        stat1 = backend.call(sys_blocks, messages)
        result.calls.append(stat1)
        # Turn 2: prior assistant reply + question 2
        messages.append({"role": "assistant", "content": stat1.content})
        messages.append({"role": "user", "content": questions[1]})
        stat2 = backend.call(sys_blocks, messages)
        result.calls.append(stat2)
        # Turn 3: prior assistant reply + final emit-JSON question
        messages.append({"role": "assistant", "content": stat2.content})
        # Append schema explicitly to T3 so the model emits it.
        t3 = (
            f"{questions[2]}\n\n"
            f"Schema:\n```json\n{json.dumps(PROPERTY_SCHEMA, indent=2)}\n```"
        )
        messages.append({"role": "user", "content": t3})
        stat3 = backend.call(sys_blocks, messages)
        result.calls.append(stat3)
        _parse_json_or_record(stat3, result)
        results[prop] = result
    return results


# ── Driver ──────────────────────────────────────────────────────────────────


@dataclass
class FunctionRecord:
    """All measurements for one function across all modes."""
    function: str
    src_lines: int
    n_callsites: int
    mode_results: dict[str, dict[str, PropertyResult]] = field(default_factory=dict)


def _normalize(pred: str) -> str:
    """Lowercase + collapse whitespace; for jaccard set comparison."""
    return re.sub(r"\s+", " ", pred.strip().lower())


def jaccard(a: list[str], b: list[str]) -> float:
    sa = {_normalize(x) for x in a if _normalize(x)}
    sb = {_normalize(x) for x in b if _normalize(x)}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def to_function_summary_from_mode1(
    func: Function, mode1: dict[str, PropertyResult]
) -> FunctionSummary:
    """Convert mode-1 results into a FunctionSummary so downstream callers
    in topo order can inline this function's contracts."""
    props_with_content = [
        p for p, r in mode1.items()
        if r.requires or r.ensures or r.modifies or r.notes
    ]
    s = FunctionSummary(
        function=func.name,
        properties=props_with_content,
    )
    for p, r in mode1.items():
        if r.requires:
            s.requires[p] = r.requires
        if r.ensures:
            s.ensures[p] = r.ensures
        if r.modifies:
            s.modifies[p] = r.modifies
        if r.notes:
            s.notes[p] = r.notes
        if r.noreturn:
            s.noreturn = True
    if func.attributes and "noreturn" in (func.attributes or "").lower():
        s.noreturn = True
    return s


def run_function(
    backend: ClaudeRaw | LlamaCppRaw,
    func: Function,
    summaries: dict[str, FunctionSummary],
    edges: dict[str, set[str]],
    data_model_note: str,
    modes: list[int],
) -> FunctionRecord:
    rec = FunctionRecord(
        function=func.name,
        src_lines=len((func.source or "").splitlines()),
        n_callsites=len(func.callsites or []),
    )
    if 1 in modes:
        log.info("  [%s] mode 1 (uncached single-turn)", func.name)
        rec.mode_results["1"] = run_mode_1(
            backend, func, summaries, edges, data_model_note
        )
    if 2 in modes:
        log.info("  [%s] mode 2 (cached single-turn)", func.name)
        rec.mode_results["2"] = run_mode_2(
            backend, func, summaries, edges, data_model_note
        )
    if 3 in modes:
        log.info("  [%s] mode 3 (cached multi-turn)", func.name)
        rec.mode_results["3"] = run_mode_3(
            backend, func, summaries, edges, data_model_note
        )
    return rec


# ── Reporting ───────────────────────────────────────────────────────────────


def write_results_json(records: list[FunctionRecord], path: Path) -> None:
    """Dump raw per-call data."""
    payload = []
    for r in records:
        item: dict[str, Any] = {
            "function": r.function,
            "src_lines": r.src_lines,
            "n_callsites": r.n_callsites,
            "modes": {},
        }
        for mode_id, prop_results in r.mode_results.items():
            mode_block: dict[str, Any] = {}
            for prop, pr in prop_results.items():
                mode_block[prop] = {
                    "requires": pr.requires,
                    "ensures": pr.ensures,
                    "modifies": pr.modifies,
                    "notes": pr.notes,
                    "noreturn": pr.noreturn,
                    "parse_error": pr.parse_error,
                    "calls": [asdict(c) for c in pr.calls],
                    "totals": {
                        "in_tok": pr.in_tok,
                        "out_tok": pr.out_tok,
                        "cache_read": pr.cache_read,
                        "cache_create": pr.cache_create,
                        "wall_s": pr.wall,
                    },
                }
            item["modes"][mode_id] = mode_block
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2))


def _mode_totals(mode_results: dict[str, PropertyResult]) -> dict[str, float]:
    return {
        "in_tok": sum(r.in_tok for r in mode_results.values()),
        "out_tok": sum(r.out_tok for r in mode_results.values()),
        "cache_read": sum(r.cache_read for r in mode_results.values()),
        "cache_create": sum(r.cache_create for r in mode_results.values()),
        "wall_s": sum(r.wall for r in mode_results.values()),
        "n_calls": sum(len(r.calls) for r in mode_results.values()),
    }


def write_summary_md(
    records: list[FunctionRecord],
    backend_name: str,
    model: str,
    path: Path,
) -> None:
    """Aggregate table + decision per design-doc Step 6 rule."""
    lines: list[str] = []
    lines.append(f"# Cache hypothesis A/B summary\n")
    lines.append(f"- Backend: `{backend_name}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Functions tested: {len(records)}")
    lines.append("")

    # Per-function table.
    lines.append("## Per-function totals\n")
    header = (
        "| function | src/cs | mode | calls | in_tok | out_tok | "
        "cache_read | cache_create | wall (s) |"
    )
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for r in records:
        for mode_id in sorted(r.mode_results.keys()):
            t = _mode_totals(r.mode_results[mode_id])
            lines.append(
                f"| {r.function} | {r.src_lines}/{r.n_callsites} | "
                f"{mode_id} | {int(t['n_calls'])} | {int(t['in_tok'])} | "
                f"{int(t['out_tok'])} | {int(t['cache_read'])} | "
                f"{int(t['cache_create'])} | {t['wall_s']:.2f} |"
            )
    lines.append("")

    # Equivalence table (mode 2/3 vs mode 1).
    lines.append("## Equivalence vs mode 1 (jaccard on requires/ensures)\n")
    eq_header = (
        "| function | property | "
        "j_req(2v1) | j_ens(2v1) | j_req(3v1) | j_ens(3v1) |"
    )
    eq_sep = "|---|---|---:|---:|---:|---:|"
    lines.append(eq_header)
    lines.append(eq_sep)
    for r in records:
        m1 = r.mode_results.get("1", {})
        m2 = r.mode_results.get("2", {})
        m3 = r.mode_results.get("3", {})
        for prop in PROPERTIES:
            r1 = m1.get(prop)
            r2 = m2.get(prop)
            r3 = m3.get(prop)
            j2_req = jaccard(r2.requires, r1.requires) if r1 and r2 else float("nan")
            j2_ens = jaccard(r2.ensures, r1.ensures) if r1 and r2 else float("nan")
            j3_req = jaccard(r3.requires, r1.requires) if r1 and r3 else float("nan")
            j3_ens = jaccard(r3.ensures, r1.ensures) if r1 and r3 else float("nan")

            def _f(x: float) -> str:
                return "—" if x != x else f"{x:.2f}"  # NaN → —

            lines.append(
                f"| {r.function} | {prop} | "
                f"{_f(j2_req)} | {_f(j2_ens)} | {_f(j3_req)} | {_f(j3_ens)} |"
            )
    lines.append("")

    # Decision summary (per Step 6 of plan / design doc).
    lines.append("## Decision\n")
    lines.append(
        "Decision rule (from plan Step 6):\n"
        "- If mode 3 saves >25% input tokens AND maintains summary quality "
        "(jaccard ≥0.8 on requires/ensures vs mode 1) → build full multi-turn "
        "Q&A driver (plan Step 7).\n"
        "- Else if mode 2 alone gives most of the savings → ship mode 2 as "
        "ContractPass default; skip Step 7.\n"
        "- Else → ship mode 1 (current prototype behavior).\n"
    )

    # Aggregate across all functions.
    agg: dict[str, dict[str, float]] = {}
    for mode_id in ("1", "2", "3"):
        rows = [
            _mode_totals(r.mode_results[mode_id])
            for r in records
            if mode_id in r.mode_results
        ]
        if not rows:
            continue
        agg[mode_id] = {
            k: sum(row[k] for row in rows) for k in rows[0]
        }
    if agg:
        lines.append("### Aggregate across all functions\n")
        lines.append("| mode | calls | in_tok | out_tok | cache_read | wall (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for mode_id in sorted(agg.keys()):
            t = agg[mode_id]
            lines.append(
                f"| {mode_id} | {int(t['n_calls'])} | {int(t['in_tok'])} | "
                f"{int(t['out_tok'])} | {int(t['cache_read'])} | "
                f"{t['wall_s']:.2f} |"
            )
        # Compute %savings using Anthropic billing model:
        # billed_input = uncached_in + 0.1*cache_read + 1.25*cache_create.
        # cache_create costs MORE than uncached input (it's a one-time write
        # premium); cache_read is the savings driver on subsequent hits.
        def _billed(t: dict) -> float:
            return (t["in_tok"]
                    + 0.10 * t.get("cache_read", 0)
                    + 1.25 * t.get("cache_create", 0))

        if "1" in agg:
            base_billed = _billed(agg["1"])
            lines.append(
                f"\n- mode 1 billed input tokens (baseline): "
                f"**{base_billed:.0f}**"
            )
            for mode_id in ("2", "3"):
                if mode_id in agg:
                    eff = _billed(agg[mode_id])
                    pct = (100 * (base_billed - eff) / base_billed
                           if base_billed else 0.0)
                    lines.append(
                        f"- mode {mode_id} billed input tokens: "
                        f"**{eff:.0f}** "
                        f"({pct:+.1f}% vs mode 1)"
                    )
            # Wall-time comparison.
            base_wall = agg["1"]["wall_s"]
            lines.append(
                f"\n- mode 1 wall time (baseline): **{base_wall:.2f}s**"
            )
            for mode_id in ("2", "3"):
                if mode_id in agg:
                    w = agg[mode_id]["wall_s"]
                    pct_w = (100 * (base_wall - w) / base_wall
                             if base_wall else 0.0)
                    lines.append(
                        f"- mode {mode_id} wall time: **{w:.2f}s** "
                        f"({pct_w:+.1f}% vs mode 1)"
                    )

    path.write_text("\n".join(lines))


# ── Main ────────────────────────────────────────────────────────────────────


DEFAULT_FUNCTIONS = [
    "KeSetEvent",
    "FloppyPnpComplete",
    "stubMoreProcessingRequired",
    "FlQueueIrpToThread",
    "FloppyPnp",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path,
                    help="Path to functions.db")
    ap.add_argument(
        "--functions", default=",".join(DEFAULT_FUNCTIONS),
        help="Comma-separated function names to test (in addition to their "
             "callees, which are pulled in for topo order).",
    )
    ap.add_argument("--backend", default="claude", choices=["claude", "llamacpp"])
    ap.add_argument("--model", default="claude-haiku-4-5@20251001")
    ap.add_argument("--llm-host", default="localhost")
    ap.add_argument("--llm-port", default=8080, type=int)
    ap.add_argument("--data-model", default="ILP32",
                    help="ILP32 (sv-comp default) or LP64.")
    ap.add_argument(
        "--modes", default="1,2,3",
        help="Comma-separated subset of {1,2,3}.",
    )
    ap.add_argument("--output", default=Path("cache_ab_results.json"), type=Path)
    ap.add_argument("--summary-out", default=Path("cache_ab_summary.md"),
                    type=Path)
    ap.add_argument("--max-src-lines", default=400, type=int,
                    help="Skip functions with src lines > this. 0 disables.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    modes = [int(m) for m in args.modes.split(",") if m.strip()]
    target_names = [n.strip() for n in args.functions.split(",") if n.strip()]

    db = SummaryDB(str(args.db))
    all_funcs = db.get_all_functions()
    by_name = {f.name: f for f in all_funcs}

    missing = [n for n in target_names if n not in by_name]
    if missing:
        log.error("Functions not in DB: %s", missing)
        return 1

    # Build edges across the whole DB so callee discovery works.
    edges = _build_db_edges(db, all_funcs)

    # Closure: pull in all callees reachable from target_names so topo order
    # produces sensible callee-first ordering.
    in_set: set[str] = set()
    stack = list(target_names)
    while stack:
        n = stack.pop()
        if n in in_set:
            continue
        in_set.add(n)
        for c in edges.get(n, set()):
            if c not in in_set and c in by_name:
                stack.append(c)

    closure_funcs = [by_name[n] for n in in_set]
    ordered = _topo_order(closure_funcs, edges)

    # Filter by max-src-lines and explicit-target. Callees outside target_names
    # still get summarised in mode 1 (so caller has hints to inline).
    target_set = set(target_names)

    if args.backend == "claude":
        backend: ClaudeRaw | LlamaCppRaw = ClaudeRaw(args.model)
    else:
        backend = LlamaCppRaw(args.model, args.llm_host, args.llm_port)

    summaries: dict[str, FunctionSummary] = dict(_STDLIB_CONTRACTS)
    records: list[FunctionRecord] = []
    data_model_note = _data_model_note(args.data_model)

    for func in ordered:
        is_target = func.name in target_set
        n_lines = len((func.source or "").splitlines())
        if args.max_src_lines and n_lines > args.max_src_lines:
            log.info(
                "Skipping %s (%d lines > --max-src-lines %d)",
                func.name, n_lines, args.max_src_lines,
            )
            continue
        if is_target:
            log.info(
                "=== TARGET %s (%d lines, %d callsites) ===",
                func.name, n_lines, len(func.callsites or []),
            )
            rec = run_function(
                backend, func, summaries, edges, data_model_note, modes,
            )
            records.append(rec)
            # Use mode-1 result (or mode-2 if mode-1 absent) for downstream
            # callers' callee_block / source-prep.
            chosen_mode = "1" if "1" in rec.mode_results else next(iter(rec.mode_results))
            summaries[func.name] = to_function_summary_from_mode1(
                func, rec.mode_results[chosen_mode]
            )
        else:
            # Non-target callee: still need a summary for caller inlining.
            # Use mode 1 only (cheapest) and don't record metrics.
            log.info(
                "callee-only %s (%d lines): summarising for caller inlining",
                func.name, n_lines,
            )
            mode1 = run_mode_1(backend, func, summaries, edges, data_model_note)
            summaries[func.name] = to_function_summary_from_mode1(func, mode1)

    write_results_json(records, args.output)
    write_summary_md(records, args.backend, args.model, args.summary_out)
    log.info("Wrote %s", args.output)
    log.info("Wrote %s", args.summary_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
