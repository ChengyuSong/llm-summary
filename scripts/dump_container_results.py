#!/usr/bin/env python3
"""Dump container detection results from func-scans DBs to per-project JSON files.

Usage:
    python scripts/dump_container_results.py -o results/llamacpp
    python scripts/dump_container_results.py -o results/gemini --filter lib
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_summary.db import SummaryDB

FUNC_SCANS_DIR = Path(__file__).resolve().parent.parent / "func-scans"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dump container detection results to per-project JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True,
        help="Output directory (one JSON file per project)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only include projects matching this substring",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projects = sorted(
        d for d in FUNC_SCANS_DIR.iterdir()
        if d.is_dir() and (d / "functions.db").exists()
    )

    if args.filter:
        projects = [p for p in projects if args.filter in p.name]

    total_entries = 0
    projects_written = 0

    for project_dir in projects:
        name = project_dir.name
        db = SummaryDB(project_dir / "functions.db")

        try:
            summaries = db.get_all_container_summaries()
            # Skip heuristic-only entries (not LLM-confirmed)
            summaries = [s for s in summaries if s.model_used != "heuristic_only"]
            if not summaries:
                continue

            functions = {f.id: f for f in db.get_all_functions()}

            entries = []
            for cs in summaries:
                func = functions.get(cs.function_id)
                entries.append({
                    "function": func.name if func else f"id:{cs.function_id}",
                    "file": func.file_path if func else "",
                    "signature": func.signature if func else "",
                    "container_type": cs.container_type,
                    "container_arg": cs.container_arg,
                    "store_args": cs.store_args,
                    "load_return": cs.load_return,
                    "confidence": cs.confidence,
                    "heuristic_score": cs.heuristic_score,
                    "model_used": cs.model_used,
                })

            out_path = out_dir / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(entries, f, indent=2)

            total_entries += len(entries)
            projects_written += 1
            print(f"  {name}: {len(entries)} containers -> {out_path}")

        finally:
            db.close()

    print(f"\nDumped {total_entries} containers from {projects_written} projects to {out_dir}/")


if __name__ == "__main__":
    main()
