"""Command-line interface for LLM-based allocation summary analysis."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .db import SummaryDB
from .extractor import FunctionExtractor
from .callgraph import CallGraphBuilder
from .indirect import AddressTakenScanner, IndirectCallsiteFinder, IndirectCallResolver
from .llm import create_backend
from .ordering import ProcessingOrderer
from .summarizer import AllocationSummarizer
from .stdlib import get_all_stdlib_summaries

console = Console()


@click.group()
@click.version_option()
def main():
    """LLM-based memory allocation summary analysis for C/C++ code."""
    pass


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama"]), default="claude")
@click.option("--model", default=None, help="Model name to use")
@click.option("--recursive/--no-recursive", default=True, help="Scan directories recursively")
@click.option("--include-headers/--no-include-headers", default=False, help="Include header files")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis of all functions")
def analyze(path, db_path, backend, model, recursive, include_headers, verbose, force):
    """Analyze C/C++ source files and generate allocation summaries."""
    path = Path(path).resolve()

    # Determine extensions to process
    extensions = [".c", ".cpp", ".cc", ".cxx"]
    if include_headers:
        extensions.extend([".h", ".hpp", ".hxx"])

    # Collect files
    if path.is_file():
        files = [path]
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in extensions]

    if not files:
        console.print("[red]No source files found[/red]")
        return

    console.print(f"Found {len(files)} source file(s)")

    # Initialize database
    db = SummaryDB(db_path)

    try:
        # Phase 1: Extract functions
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting functions...", total=None)

            extractor = FunctionExtractor()
            all_functions = []

            for f in files:
                try:
                    functions = extractor.extract_from_file(f)
                    all_functions.extend(functions)
                    if verbose:
                        progress.console.print(f"  {f.name}: {len(functions)} functions")
                except Exception as e:
                    progress.console.print(f"  [yellow]Warning: {f.name}: {e}[/yellow]")

            progress.update(task, completed=True)

        console.print(f"Extracted {len(all_functions)} functions")

        # Store functions
        func_ids = db.insert_functions_batch(all_functions)

        # Phase 2: Build call graph
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building call graph...", total=None)

            cg_builder = CallGraphBuilder(db)
            edges = cg_builder.build_from_files(files)

            progress.update(task, completed=True)

        console.print(f"Found {len(edges)} call edges")

        # Phase 3: Indirect call analysis (optional, with LLM)
        # This is expensive, so we skip it unless explicitly enabled
        # TODO: Add --analyze-indirect flag

        # Phase 4: Generate summaries
        llm = create_backend(backend, model=model)
        summarizer = AllocationSummarizer(db, llm, verbose=verbose)

        console.print(f"Using {backend} backend ({llm.model})")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating summaries...", total=None)

            summaries = summarizer.summarize_all(force=force)

            progress.update(task, completed=True)

        stats = summarizer.stats
        console.print(f"\nSummary generation complete:")
        console.print(f"  Functions processed: {stats['functions_processed']}")
        console.print(f"  LLM calls: {stats['llm_calls']}")
        console.print(f"  Cache hits: {stats['cache_hits']}")
        if stats["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats['errors']}[/yellow]")

        # Count allocating functions
        allocating = sum(1 for s in summaries.values() if s.allocations)
        console.print(f"\nFunctions with allocations: {allocating}")

    finally:
        db.close()


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--recursive/--no-recursive", default=True)
def extract(path, db_path, recursive):
    """Extract functions and build call graph (no LLM)."""
    path = Path(path).resolve()
    extensions = [".c", ".cpp", ".cc", ".cxx"]

    if path.is_file():
        files = [path]
    else:
        if recursive:
            files = [f for f in path.rglob("*") if f.suffix.lower() in extensions]
        else:
            files = [f for f in path.glob("*") if f.suffix.lower() in extensions]

    if not files:
        console.print("[red]No source files found[/red]")
        return

    db = SummaryDB(db_path)

    try:
        extractor = FunctionExtractor()
        all_functions = []

        for f in files:
            try:
                functions = extractor.extract_from_file(f)
                all_functions.extend(functions)
                console.print(f"  {f.name}: {len(functions)} functions")
            except Exception as e:
                console.print(f"  [yellow]Warning: {f.name}: {e}[/yellow]")

        func_ids = db.insert_functions_batch(all_functions)
        console.print(f"\nExtracted {len(all_functions)} functions")

        # Build call graph
        cg_builder = CallGraphBuilder(db)
        edges = cg_builder.build_from_files(files)
        console.print(f"Found {len(edges)} call edges")

        # Show stats
        stats = db.get_stats()
        console.print("\nDatabase statistics:")
        for key, value in stats.items():
            console.print(f"  {key}: {value}")

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--name", help="Filter by function name")
@click.option("--file", "file_path", help="Filter by file path")
@click.option("--allocating-only", is_flag=True, help="Only show allocating functions")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show(db_path, name, file_path, allocating_only, fmt):
    """Show stored summaries."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()

        if name:
            functions = [f for f in functions if name in f.name]

        if file_path:
            functions = [f for f in functions if file_path in f.file_path]

        if fmt == "json":
            output = []
            for func in functions:
                summary = db.get_summary_by_function_id(func.id)
                if summary:
                    if allocating_only and not summary.allocations:
                        continue
                    output.append({
                        "function": func.name,
                        "file": func.file_path,
                        "line": func.line_start,
                        "summary": summary.to_dict(),
                    })
            console.print(json.dumps(output, indent=2))

        else:
            table = Table(title="Allocation Summaries")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Allocations", style="green")
            table.add_column("Description")

            for func in functions:
                summary = db.get_summary_by_function_id(func.id)
                if summary:
                    if allocating_only and not summary.allocations:
                        continue

                    alloc_str = ", ".join(a.source for a in summary.allocations) or "-"
                    table.add_row(
                        func.name,
                        Path(func.file_path).name,
                        alloc_str,
                        summary.description[:50] + "..." if len(summary.description) > 50 else summary.description,
                    )

            console.print(table)

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
def stats(db_path):
    """Show database statistics."""
    db = SummaryDB(db_path)

    try:
        stats = db.get_stats()

        table = Table(title="Database Statistics")
        table.add_column("Table", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for key, value in stats.items():
            table.add_row(key, str(value))

        console.print(table)

        # Call graph stats
        edges = db.get_all_call_edges()
        if edges:
            graph = {}
            for edge in edges:
                if edge.caller_id not in graph:
                    graph[edge.caller_id] = []
                graph[edge.caller_id].append(edge.callee_id)

            orderer = ProcessingOrderer(graph)
            cg_stats = orderer.get_stats()

            console.print("\nCall Graph Statistics:")
            console.print(f"  Nodes: {cg_stats['nodes']}")
            console.print(f"  Edges: {cg_stats['edges']}")
            console.print(f"  SCCs: {cg_stats['sccs']}")
            console.print(f"  Recursive SCCs: {cg_stats['recursive_sccs']}")
            console.print(f"  Largest SCC: {cg_stats['largest_scc']}")

    finally:
        db.close()


@main.command()
@click.argument("name")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--signature", help="Function signature for disambiguation")
def lookup(name, db_path, signature):
    """Look up summary for a specific function."""
    db = SummaryDB(db_path)

    try:
        summary = db.get_summary(name, signature)

        if summary:
            console.print(json.dumps(summary.to_dict(), indent=2))
        else:
            console.print(f"[yellow]No summary found for {name}[/yellow]")

            # Check if function exists
            functions = db.get_function_by_name(name)
            if functions:
                console.print("Function exists but has no summary. Run 'analyze' to generate.")
            else:
                console.print("Function not found in database.")

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
def clear(db_path):
    """Clear all data from the database."""
    if not click.confirm("This will delete all data. Continue?"):
        return

    db = SummaryDB(db_path)
    try:
        db.clear_all()
        console.print("Database cleared.")
    finally:
        db.close()


@main.command("init-stdlib")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
def init_stdlib(db_path):
    """Initialize database with standard library summaries."""
    db = SummaryDB(db_path)

    try:
        stdlib = get_all_stdlib_summaries()
        console.print(f"Adding {len(stdlib)} standard library summaries...")

        for name, summary in stdlib.items():
            # Create a pseudo-function for stdlib
            from .models import Function
            func = Function(
                name=name,
                file_path="<stdlib>",
                line_start=0,
                line_end=0,
                source="",
                signature=f"{name}(...)",
            )
            func.id = db.insert_function(func)
            db.upsert_summary(func, summary, model_used="builtin")

        console.print("Done.")

    finally:
        db.close()


@main.command()
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
def export(db_path, output):
    """Export all summaries to JSON."""
    db = SummaryDB(db_path)

    try:
        functions = db.get_all_functions()
        output_data = []

        for func in functions:
            summary = db.get_summary_by_function_id(func.id)
            if summary:
                output_data.append({
                    "name": func.name,
                    "signature": func.signature,
                    "file": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "summary": summary.to_dict(),
                })

        json_str = json.dumps(output_data, indent=2)

        if output:
            with open(output, "w") as f:
                f.write(json_str)
            console.print(f"Exported {len(output_data)} summaries to {output}")
        else:
            console.print(json_str)

    finally:
        db.close()


@main.command("callgraph")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--output", "-o", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--format", "fmt",
    type=click.Choice(["tuples", "csv", "json"]),
    default="tuples",
    help="Output format"
)
@click.option("--no-header", is_flag=True, help="Omit header row (for csv/tuples)")
def callgraph(db_path, output, fmt, no_header):
    """Export call graph as (caller, callsite, callee) tuples."""
    db = SummaryDB(db_path)

    try:
        edges = db.get_all_call_edges()
        functions = {f.id: f for f in db.get_all_functions()}

        lines = []

        if fmt == "json":
            output_data = []
            for edge in edges:
                caller = functions.get(edge.caller_id)
                callee = functions.get(edge.callee_id)
                if caller and callee:
                    output_data.append({
                        "caller": caller.name,
                        "callsite": {
                            "file": edge.file_path,
                            "line": edge.line,
                            "column": edge.column,
                        },
                        "callee": callee.name,
                        "is_indirect": edge.is_indirect,
                    })
            lines.append(json.dumps(output_data, indent=2))

        else:
            # tuples or csv format
            separator = "," if fmt == "csv" else "\t"

            if not no_header:
                lines.append(separator.join(["caller", "callsite", "callee"]))

            for edge in edges:
                caller = functions.get(edge.caller_id)
                callee = functions.get(edge.callee_id)
                if caller and callee:
                    callsite = edge.callsite_str() or f"{edge.file_path}:{edge.line}"
                    lines.append(separator.join([caller.name, callsite, callee.name]))

        result = "\n".join(lines)

        if output:
            with open(output, "w") as f:
                f.write(result)
                f.write("\n")
            console.print(f"Exported {len(edges)} call edges to {output}")
        else:
            console.print(result)

    finally:
        db.close()


if __name__ == "__main__":
    main()
