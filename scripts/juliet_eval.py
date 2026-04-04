#!/usr/bin/env python3
"""Evaluate llm-summary verify pass on SV-COMP Juliet benchmarks.

Phases:
  0  scan          Extract functions from .i files
  1  callgraph     Compile to .bc, run KAMain, import call graph
  2  summarize     Bottom-up passes: alloc, free, init, memsafe
  3  verify        Verification pass
  4  evaluate      Collect issues, score against ground truth

Usage:
    source ~/project/llm-summary/venv/bin/activate
    python scripts/juliet_eval.py \
        --benchmarks /data/csong/opensource/sv-benchmarks/c/Juliet_Test \
        --cwe CWE415 \
        --variant 01 \
        --backend claude \
        --output juliet_eval_results.json \
        -v

    # Re-run only verify + evaluate (skip scan/callgraph/summarize):
    python scripts/juliet_eval.py \
        --benchmarks ... --cwe CWE415 --variant 01 \
        --backend claude --from-phase 3 -v
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_summary.callgraph_import import CallGraphImporter
from llm_summary.cli import _load_compile_commands
from llm_summary.db import SummaryDB
from llm_summary.driver import (
    AllocationPass,
    BottomUpDriver,
    FreePass,
    InitPass,
    MemsafePass,
    SummaryPass,
    VerificationPass,
)
from llm_summary.extractor import FunctionExtractor
from llm_summary.free_summarizer import FreeSummarizer
from llm_summary.init_summarizer import InitSummarizer
from llm_summary.llm import create_backend
from llm_summary.memsafe_summarizer import MemsafeSummarizer
from llm_summary.summarizer import AllocationSummarizer
from llm_summary.verification_summarizer import VerificationSummarizer

log = logging.getLogger("juliet_eval")

DEFAULT_KAMAIN = "/home/csong/project/kanalyzer/release/lib/KAMain"
DEFAULT_FUNC_SCANS = "func-scans/sv-benchmarks"

# ── SV-COMP property -> our issue_kind mapping ─────────────────────────────
SUBPROP_TO_KINDS: dict[str, set[str]] = {
    "valid-free": {"double_free", "use_after_free", "invalid_free"},
    "valid-deref": {"null_deref", "buffer_overflow", "use_after_free"},
    "valid-memtrack": {"memory_leak"},
    "no-overflow": {"integer_overflow"},
}

# Boilerplate function prefixes shared across all Juliet .i files
BOILERPLATE_PREFIXES = (
    "print", "ldv_", "decode", "global", "internal_start",
    "stdThread", "assume_abort", "reach_error",
)


@dataclass
class TaskResult:
    """Result for one Juliet task (.yml file)."""

    yml_file: str
    i_file: str
    cwe: str
    variant: str  # "bad" or "good"
    expected_safe: bool  # True = no bug, False = has bug
    subproperty: str
    issues_found: list[dict[str, Any]] = field(default_factory=list)
    predicted_safe: bool | None = None
    correct: bool | None = None
    classification: str = ""  # TP, TN, FP, FN
    error: str | None = None
    elapsed_s: float = 0.0
    llm_calls: int = 0


def parse_yml(yml_path: Path) -> dict[str, Any]:
    """Parse a Juliet .yml task file."""
    with open(yml_path) as f:
        data = yaml.safe_load(f)

    i_file = data["input_files"]
    props = data.get("properties", [])

    expected_verdict = True
    subproperty = ""
    for prop in props:
        if "expected_verdict" in prop:
            expected_verdict = prop["expected_verdict"]
            subproperty = prop.get("subproperty", "")
            if not subproperty:
                # Derive from property_file, e.g., "no-overflow.prp"
                pf = prop.get("property_file", "")
                base = Path(pf).stem  # "no-overflow"
                if base in SUBPROP_TO_KINDS:
                    subproperty = base
            break

    return {
        "i_file": i_file,
        "expected_verdict": expected_verdict,
        "subproperty": subproperty,
    }


def find_target_functions(db: SummaryDB, variant: str) -> list[str]:
    """Find CWE-specific functions to verify.

    For _bad files: the CWE*_bad named function.
    For _good files: static helpers like goodG2B, goodB2G.
    """
    funcs = db.get_all_functions()
    targets = []
    for f in funcs:
        name = f.name
        if name == "main":
            continue
        if any(name.startswith(p) for p in BOILERPLATE_PREFIXES):
            continue
        # Skip empty stubs (bad1-bad9, good1-good9)
        if re.match(r"^(bad|good)\d+$", name):
            continue
        if variant == "bad":
            if name.endswith("_bad"):
                targets.append(name)
        else:
            # Static helpers (goodG2B, goodB2G, etc.) and wrapper
            if name.startswith("good") or name.endswith("_good"):
                targets.append(name)

    return targets


def task_dir_name(yml_stem: str) -> str:
    """Derive a directory name from the .yml stem.

    E.g. 'CWE415_..._char_44_bad' -> same string, used as subdir name.
    The full stem is unique per task so we use it directly.
    """
    return yml_stem


# ── Phase 0: scan ───────────────────────────────────────────────────────────

def phase_scan(
    i_path: Path,
    work_dir: Path,
    db: SummaryDB,
) -> int:
    """Extract functions from .i file into DB. Returns function count."""
    cc_json = [{
        "directory": str(i_path.parent),
        "file": str(i_path),
        "command": f"clang -c {i_path.name}",
    }]
    cc_path = work_dir / "compile_commands.json"
    cc_path.write_text(json.dumps(cc_json))

    compile_commands, _ = _load_compile_commands(str(cc_path), None)

    extractor = FunctionExtractor(
        compile_commands=compile_commands,
        enable_preprocessing=False,
    )
    functions = extractor.extract_from_file(i_path)
    db.insert_functions_batch(functions)
    return len(functions)


# ── Phase 1: callgraph ─────────────────────────────────────────────────────

def phase_callgraph(
    i_path: Path,
    work_dir: Path,
    db: SummaryDB,
    kamain_bin: str,
    verbose: bool,
) -> int:
    """Compile .i to .bc, run KAMain, import call graph. Returns edge count."""
    bc_path = work_dir / "input.bc"
    cg_json = work_dir / "callgraph.json"

    compile_cmd = [
        "clang-18", "-emit-llvm", "-c", "-Wno-everything",
        str(i_path), "-o", str(bc_path),
    ]
    result = subprocess.run(
        compile_cmd, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"clang -emit-llvm failed: {result.stderr}")

    kamain_cmd = [
        kamain_bin, str(bc_path),
        "--callgraph-json", str(cg_json),
        "--verbose", "0",
    ]
    result = subprocess.run(
        kamain_cmd, capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"KAMain failed: {result.stderr}")

    importer = CallGraphImporter(db, verbose=verbose)
    stats = importer.import_json(cg_json, clear_existing=True)
    log.debug(
        "  KAMain: %d edges (%d direct, %d indirect)",
        stats.edges_imported, stats.direct_edges, stats.indirect_edges,
    )
    return int(stats.edges_imported)


# ── Phase 2: summarize ──────────────────────────────────────────────────────

def phase_summarize(
    db: SummaryDB,
    backend: str,
    model: str | None,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
) -> int:
    """Run alloc/free/init/memsafe passes. Returns LLM call count."""
    llm = create_backend(backend, model=model)
    cache_mode = "source" if backend == "claude" else "none"

    alloc_s = AllocationSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    free_s = FreeSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    init_s = InitSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )
    memsafe_s = MemsafeSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )

    passes: list[SummaryPass] = [
        AllocationPass(alloc_s, db, llm.model),
        FreePass(free_s, db, llm.model),
        InitPass(init_s, db, llm.model),
        MemsafePass(memsafe_s, db, llm.model),
    ]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(passes, force=force, target_ids=reachable_ids)

    return (
        int(alloc_s.stats.get("llm_calls", 0))
        + int(free_s.stats.get("llm_calls", 0))
        + int(init_s.stats.get("llm_calls", 0))
        + int(memsafe_s.stats.get("llm_calls", 0))
    )


# ── Phase 3: verify ─────────────────────────────────────────────────────────

def phase_verify(
    db: SummaryDB,
    backend: str,
    model: str | None,
    verbose: bool,
    log_llm: str | None,
    force: bool,
    reachable_ids: set[int] | None = None,
) -> int:
    """Run verification pass. Returns LLM call count."""
    llm = create_backend(backend, model=model)
    cache_mode = "source" if backend == "claude" else "none"

    verify_s = VerificationSummarizer(
        db, llm, verbose=verbose, cache_mode=cache_mode, log_file=log_llm,
    )

    v_passes: list[SummaryPass] = [VerificationPass(verify_s, db, llm.model)]

    driver = BottomUpDriver(db, verbose=verbose)
    driver.run(v_passes, force=force, target_ids=reachable_ids)

    return int(verify_s.stats.get("llm_calls", 0))


# ── Phase 4: evaluate ───────────────────────────────────────────────────────

def phase_evaluate(
    db: SummaryDB,
    variant: str,
) -> list[dict[str, Any]]:
    """Collect issues from target functions. Returns issues list."""
    targets = find_target_functions(db, variant)
    issues: list[dict[str, Any]] = []
    for tname in targets:
        for f in db.get_function_by_name(tname):
            assert f.id is not None
            vsm = db.get_verification_summary_by_function_id(f.id)
            if vsm and vsm.issues:
                for issue in vsm.issues:
                    issues.append(issue.to_dict())
    return issues


# ── Task runner ──────────────────────────────────────────────────────────────

def run_one_task(
    i_path: Path,
    variant: str,
    work_dir: Path,
    backend: str,
    model: str | None,
    verbose: bool,
    kamain_bin: str = DEFAULT_KAMAIN,
    log_llm: str | None = None,
    from_phase: int = 0,
    to_phase: int = 4,
    force: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Run pipeline phases on one .i file using persistent work_dir.

    Returns (issues_list, llm_call_count).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "functions.db"
    db = SummaryDB(str(db_path))
    total_llm = 0

    try:
        # Phase 0: scan
        if from_phase <= 0:
            n_funcs = phase_scan(i_path, work_dir, db)
            log.debug("  Extracted %d functions", n_funcs)

        # Phase 1: callgraph
        if from_phase <= 1:
            n_edges = phase_callgraph(
                i_path, work_dir, db, kamain_bin, verbose,
            )
            log.debug("  Call edges: %d", n_edges)

        # Compute reachable functions from main
        reachable_ids: set[int] | None = None
        main_funcs = db.get_function_by_name("main")
        if main_funcs:
            driver = BottomUpDriver(db, verbose=False)
            graph, _ = driver.build_graph()
            main_id = main_funcs[0].id
            assert main_id is not None
            reachable_ids = driver.compute_reachable({main_id}, graph)
            log.debug("  Reachable from main: %d functions", len(reachable_ids))

        # Phase 2: summarize
        if from_phase <= 2 and to_phase >= 2:
            total_llm += phase_summarize(
                db, backend, model, verbose, log_llm,
                force=force or from_phase <= 0,
                reachable_ids=reachable_ids,
            )

        # Phase 3: verify
        if from_phase <= 3 and to_phase >= 3:
            total_llm += phase_verify(
                db, backend, model, verbose, log_llm,
                force=force or from_phase <= 0,
                reachable_ids=reachable_ids,
            )

        if to_phase < 4:
            db.close()
            return [], total_llm

        # Find targets (needed for phase 4: evaluate)
        targets = find_target_functions(db, variant)
        if not targets:
            log.warning("  No target functions found")
            db.close()
            return [], 0
        log.debug("  Targets: %s", targets)

        # Phase 4: evaluate
        issues = phase_evaluate(db, variant)
        db.close()
        return issues, total_llm

    except Exception:
        db.close()
        raise


def collect_tasks_from_set_file(
    set_file: Path,
    benchmarks_base: Path,
    cwe_filter: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Collect tasks from a .set file (glob patterns relative to benchmarks_base)."""
    seen: set[Path] = set()
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for line in set_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for yml_path in sorted(benchmarks_base.glob(line)):
            if yml_path in seen:
                continue
            seen.add(yml_path)
            name = yml_path.stem
            if cwe_filter:
                if not any(name.startswith(c) for c in cwe_filter.split(",")):
                    continue
            try:
                info = parse_yml(yml_path)
            except Exception as e:
                log.warning("Failed to parse %s: %s", yml_path, e)
                continue
            tasks.append((yml_path, info))
    return tasks


def collect_tasks(
    benchmarks_dir: Path,
    cwe_filter: str | None,
    variant_filter: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Collect all matching .yml tasks from the Juliet_Test directory."""
    tasks: list[tuple[Path, dict[str, Any]]] = []
    for yml_path in sorted(benchmarks_dir.glob("*.yml")):
        name = yml_path.stem

        if cwe_filter:
            if not any(name.startswith(c) for c in cwe_filter.split(",")):
                continue

        if variant_filter:
            if (f"_{variant_filter}_bad" not in name
                    and f"_{variant_filter}_good" not in name):
                continue

        try:
            info = parse_yml(yml_path)
        except Exception as e:
            log.warning("Failed to parse %s: %s", yml_path, e)
            continue

        tasks.append((yml_path, info))

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate llm-summary on Juliet benchmarks",
    )
    parser.add_argument(
        "--benchmarks", required=True,
        help="Path to Juliet_Test directory",
    )
    parser.add_argument(
        "--cwe", default=None,
        help="CWE filter (e.g., CWE415 or CWE415,CWE476)",
    )
    parser.add_argument(
        "--variant", default=None,
        help="Variant number filter (e.g., 01)",
    )
    parser.add_argument("--backend", default="claude", help="LLM backend")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument(
        "--output", "-o", default="juliet_eval_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max tasks to run (0=all)",
    )
    parser.add_argument(
        "--kamain-bin", default=DEFAULT_KAMAIN,
        help="Path to KAMain binary",
    )
    parser.add_argument(
        "--func-scans-dir", default=DEFAULT_FUNC_SCANS,
        help="Persistent work directory (default: %(default)s)",
    )
    parser.add_argument(
        "--set-file", default=None,
        help="SV-COMP .set file with glob patterns (e.g., Juliet.set)",
    )
    parser.add_argument(
        "--from-phase", type=int, default=0,
        help="Start from phase N (0=scan, 1=callgraph, 2=summarize, "
             "3=verify, 4=evaluate-only)",
    )
    parser.add_argument(
        "--to-phase", type=int, default=4,
        help="Stop after phase N (0=scan, 1=callgraph, 2=summarize, "
             "3=verify, 4=evaluate)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-run even if cached",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--llm-log-dir", default=None,
        help="Directory for per-target LLM interaction logs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.verbose:
        log.setLevel(logging.DEBUG)

    benchmarks_dir = Path(args.benchmarks)
    if not benchmarks_dir.exists():
        log.error("Benchmarks directory not found: %s", benchmarks_dir)
        sys.exit(1)

    func_scans = Path(args.func_scans_dir)

    if args.set_file:
        set_path = Path(args.set_file)
        if not set_path.exists():
            log.error("Set file not found: %s", set_path)
            sys.exit(1)
        tasks = collect_tasks_from_set_file(
            set_path, benchmarks_dir.parent, args.cwe,
        )
    else:
        tasks = collect_tasks(benchmarks_dir, args.cwe, args.variant)
    log.info("Collected %d tasks", len(tasks))

    if args.limit > 0:
        tasks = tasks[: args.limit]
        log.info("Limited to %d tasks", len(tasks))

    if args.from_phase > 0:
        log.info("Starting from phase %d", args.from_phase)

    results: list[TaskResult] = []
    tp = tn = fp = fn = errors = 0
    total_funcs = 0
    total_reachable = 0

    for idx, (yml_path, info) in enumerate(tasks):
        i_file = info["i_file"]
        i_path = benchmarks_dir / i_file
        expected_safe: bool = info["expected_verdict"]
        subprop: str = info["subproperty"]

        if "_bad" in yml_path.stem:
            variant = "bad"
        elif "_good" in yml_path.stem:
            variant = "good"
        else:
            log.warning("Cannot determine variant for %s", yml_path.name)
            continue

        stem = yml_path.stem
        cwe = stem.split("---")[0] if "---" in stem else stem.split("_")[0]
        work_dir = func_scans / task_dir_name(stem)

        result = TaskResult(
            yml_file=yml_path.name,
            i_file=i_file,
            cwe=cwe,
            variant=variant,
            expected_safe=expected_safe,
            subproperty=subprop,
        )

        log.info(
            "[%d/%d] %s (expected: %s)", idx + 1, len(tasks),
            stem, "safe" if expected_safe else "UNSAFE",
        )

        t0 = time.time()
        log_llm: str | None = None
        if args.llm_log_dir:
            log_dir = Path(args.llm_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_llm = str(log_dir / f"{stem}.log")

        try:
            issues, llm_calls = run_one_task(
                i_path, variant, work_dir,
                args.backend, args.model,
                args.verbose, args.kamain_bin, log_llm,
                from_phase=args.from_phase, to_phase=args.to_phase,
                force=args.force,
            )
            result.elapsed_s = time.time() - t0
            result.llm_calls = llm_calls
            result.issues_found = issues

            # Collect scan stats from DB
            db_path = work_dir / "functions.db"
            if db_path.exists():
                sdb = SummaryDB(str(db_path))
                all_funcs = sdb.get_all_functions()
                total_funcs += len(all_funcs)
                main_funcs = sdb.get_function_by_name("main")
                if main_funcs:
                    drv = BottomUpDriver(sdb, verbose=False)
                    graph, _ = drv.build_graph()
                    mid = main_funcs[0].id
                    assert mid is not None
                    reachable = drv.compute_reachable({mid}, graph)
                    total_reachable += len(reachable)
                sdb.close()

            if args.to_phase >= 4:
                # Filter issues by subproperty if applicable
                if subprop and subprop in SUBPROP_TO_KINDS:
                    relevant_kinds = SUBPROP_TO_KINDS[subprop]
                    relevant_issues = [
                        i for i in issues
                        if i.get("issue_kind", "") in relevant_kinds
                    ]
                else:
                    relevant_issues = issues

                result.predicted_safe = len(relevant_issues) == 0
                result.correct = result.predicted_safe == expected_safe

                if expected_safe and result.predicted_safe:
                    result.classification = "TN"
                    tn += 1
                elif expected_safe and not result.predicted_safe:
                    result.classification = "FP"
                    fp += 1
                elif not expected_safe and not result.predicted_safe:
                    result.classification = "TP"
                    tp += 1
                else:
                    result.classification = "FN"
                    fn += 1

                log.info(
                    "  -> %s (%d issues, %.1fs, %d LLM calls)",
                    result.classification, len(relevant_issues),
                    result.elapsed_s, llm_calls,
                )

        except Exception as e:
            result.elapsed_s = time.time() - t0
            result.error = str(e)
            errors += 1
            log.error("  -> ERROR: %s", e)

        results.append(result)

    # -- Summary --
    total = len(results)
    log.info("")

    if args.to_phase < 4:
        log.info("=== Scan Stats (phase 0-%d) ===", args.to_phase)
        log.info("Tasks: %d  Total functions: %d  Reachable: %d",
                 total, total_funcs, total_reachable)
        log.info("Errors: %d", errors)
    else:
        correct = tp + tn
        accuracy = correct / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0
        )

        summary: dict[str, Any] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "errors": errors,
        }

        log.info("=== Results ===")
        log.info(
            "Total: %d  Correct: %d  Accuracy: %.1f%%",
            total, correct, accuracy * 100,
        )
        log.info(
            "TP: %d  TN: %d  FP: %d  FN: %d  Errors: %d",
            tp, tn, fp, fn, errors,
        )
        log.info(
            "Precision: %.3f  Recall: %.3f  F1: %.3f",
            precision, recall, f1,
        )

    output: dict[str, Any] = {
        "summary": {
            "total_tasks": total,
            "total_functions": total_funcs,
            "total_reachable": total_reachable,
            "errors": errors,
        } if args.to_phase < 4 else summary,
        "config": {
            "backend": args.backend,
            "model": args.model,
            "cwe": args.cwe,
            "variant": getattr(args, "variant", None),
            "to_phase": args.to_phase,
        },
        "results": [asdict(r) for r in results],
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
