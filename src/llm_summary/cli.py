"""Command-line interface for LLM-based allocation summary analysis."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .compile_commands import CompileCommandsDB
from .db import SummaryDB
from .extractor import FunctionExtractor
from .callgraph import CallGraphBuilder
from .indirect import (
    AddressTakenScanner,
    FlowSummarizer,
    IndirectCallsiteFinder,
    IndirectCallResolver,
)
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
@click.argument("path_arg", type=click.Path(exists=True), required=False, default=None)
@click.option("--path", "path_opt", type=click.Path(exists=True), default=None, help="Path to analyze")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--recursive/--no-recursive", default=True, help="Scan directories recursively")
@click.option("--include-headers/--no-include-headers", default=False, help="Include header files")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis of all functions")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
def analyze(path_arg, path_opt, db_path, backend, model, llm_host, llm_port, disable_thinking, recursive, include_headers, verbose, force, log_llm):
    """Analyze C/C++ source files and generate allocation summaries."""
    # Accept path as either positional argument or --path option
    path = path_opt or path_arg
    if not path:
        raise click.UsageError("PATH is required (provide as argument or use --path)")

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
        # Build backend kwargs for local LLM servers
        backend_kwargs = {}
        if backend == "llamacpp":
            backend_kwargs["host"] = llm_host
            backend_kwargs["port"] = llm_port if llm_port is not None else 8080
        elif backend == "ollama":
            if llm_port is None:
                llm_port = 11434
            backend_kwargs["base_url"] = f"http://{llm_host}:{llm_port}"

        # Apply thinking mode control to backends that support it
        if disable_thinking:
            backend_kwargs["enable_thinking"] = False

        llm = create_backend(backend, model=model, **backend_kwargs)
        summarizer = AllocationSummarizer(db, llm, verbose=verbose, log_file=log_llm)

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
@click.argument("path_arg", type=click.Path(exists=True), required=False, default=None)
@click.option("--path", "path_opt", type=click.Path(exists=True), default=None, help="Path to extract from")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), default=None,
              help="Path to compile_commands.json for proper macro/include handling")
@click.option("--recursive/--no-recursive", default=True)
def extract(path_arg, path_opt, db_path, compile_commands_path, recursive):
    """Extract functions and build call graph (no LLM)."""
    # Accept path as either positional argument or --path option
    path = path_opt or path_arg
    if not path:
        raise click.UsageError("PATH is required (provide as argument or use --path)")

    path = Path(path).resolve()

    # Load compile_commands.json if provided
    compile_commands = None
    if compile_commands_path:
        try:
            compile_commands = CompileCommandsDB(compile_commands_path)
            console.print(f"Loaded compile_commands.json ({len(compile_commands)} entries)")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load compile_commands.json: {e}[/yellow]")

    # Determine file extensions based on whether compile_commands is available
    # Without compile_commands, include headers to get more complete coverage
    if compile_commands:
        extensions = [".c", ".cpp", ".cc", ".cxx"]
    else:
        extensions = [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"]
        console.print("[dim]No compile_commands.json - including header files[/dim]")

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

    db = SummaryDB(db_path)

    try:
        extractor = FunctionExtractor(compile_commands=compile_commands)
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
        cg_builder = CallGraphBuilder(db, compile_commands=compile_commands)
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


@main.command("indirect-analyze")
@click.argument("path_arg", type=click.Path(exists=True), required=False, default=None)
@click.option("--path", "path_opt", type=click.Path(exists=True), default=None, help="Path to analyze")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), default=None,
              help="Path to compile_commands.json for proper macro/include handling")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--recursive/--no-recursive", default=True, help="Scan directories recursively")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis (ignore cache)")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--pass1-only", is_flag=True, help="Only run Pass 1 (flow summarization)")
@click.option("--pass2-only", is_flag=True, help="Only run Pass 2 (resolution), requires Pass 1 already done")
def indirect_analyze(
    path_arg, path_opt, db_path, compile_commands_path, backend, model, llm_host, llm_port, disable_thinking,
    recursive, verbose, force, log_llm, pass1_only, pass2_only
):
    """
    Analyze indirect calls using LLM-based two-pass approach.

    Pass 1: Summarize where address-taken function pointers flow.
    Pass 2: Resolve indirect callsites using flow summaries.

    Example:
        llm-summary indirect-analyze --path src/ --db out.db --compile-commands compile_commands.json
    """
    # Accept path as either positional argument or --path option
    path = path_opt or path_arg
    if not path:
        raise click.UsageError("PATH is required (provide as argument or use --path)")

    path = Path(path).resolve()

    # Load compile_commands.json if provided
    compile_commands = None
    if compile_commands_path:
        try:
            compile_commands = CompileCommandsDB(compile_commands_path)
            console.print(f"Loaded compile_commands.json ({len(compile_commands)} entries)")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load compile_commands.json: {e}[/yellow]")

    # Determine file extensions based on whether compile_commands is available
    # Without compile_commands, include headers to get more complete coverage
    if compile_commands:
        extensions = [".c", ".cpp", ".cc", ".cxx"]
    else:
        extensions = [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"]
        console.print("[dim]No compile_commands.json - including header files[/dim]")

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
        # Check if we need to extract functions first
        existing_funcs = db.get_all_functions()
        if not existing_funcs:
            console.print("No functions in database. Running extraction first...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Extracting functions...", total=None)

                extractor = FunctionExtractor(compile_commands=compile_commands)
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

            db.insert_functions_batch(all_functions)
            console.print(f"Extracted {len(all_functions)} functions")

        # Scan for address-taken functions and indirect callsites
        # Skip scanning if --pass2-only (use existing data from DB)
        if pass2_only:
            atf_count = len(db.get_address_taken_functions())
            callsite_count = len(db.get_indirect_callsites())
            console.print(f"Using existing data: {atf_count} address-taken functions, {callsite_count} indirect callsites")

            if atf_count == 0:
                console.print("[red]No address-taken functions in database. Run without --pass2-only first.[/red]")
                return

            if callsite_count == 0:
                console.print("[red]No indirect callsites in database. Run without --pass2-only first.[/red]")
                return
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Scanning for address-taken functions...", total=None)

                scanner = AddressTakenScanner(db, compile_commands=compile_commands)
                scanner.scan_files(files)

                progress.update(task, completed=True)

            atf_count = len(db.get_address_taken_functions())
            console.print(f"Found {atf_count} address-taken functions")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Finding indirect call sites...", total=None)

                finder = IndirectCallsiteFinder(db, compile_commands=compile_commands)
                callsites = finder.find_in_files(files)

                progress.update(task, completed=True)

            console.print(f"Found {len(callsites)} indirect call sites")

            if atf_count == 0:
                console.print("[yellow]No address-taken functions found. Nothing to analyze.[/yellow]")
                return

            if len(callsites) == 0:
                console.print("[yellow]No indirect call sites found. Nothing to resolve.[/yellow]")
                return

        # Initialize LLM backend
        # Build backend kwargs for local LLM servers
        backend_kwargs = {}
        if backend == "llamacpp":
            backend_kwargs["host"] = llm_host
            backend_kwargs["port"] = llm_port if llm_port is not None else 8080
        elif backend == "ollama":
            if llm_port is None:
                llm_port = 11434
            backend_kwargs["base_url"] = f"http://{llm_host}:{llm_port}"

        # Apply thinking mode control to backends that support it
        if disable_thinking:
            backend_kwargs["enable_thinking"] = False

        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        # Pass 1: Flow summarization
        if not pass2_only:
            console.print("\n[bold]Pass 1: Flow Summarization[/bold]")

            flow_summarizer = FlowSummarizer(db, llm, verbose=verbose, log_file=log_llm)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Summarizing function pointer flows...", total=None)

                flow_summaries = flow_summarizer.summarize_all(force=force)

                progress.update(task, completed=True)

            stats1 = flow_summarizer.stats
            console.print(f"Pass 1 complete:")
            console.print(f"  Functions processed: {stats1['functions_processed']}")
            console.print(f"  LLM calls: {stats1['llm_calls']}")
            console.print(f"  Cache hits: {stats1['cache_hits']}")
            if stats1["errors"] > 0:
                console.print(f"  [yellow]Errors: {stats1['errors']}[/yellow]")

        if pass1_only:
            console.print("\n[green]Pass 1 complete. Use --pass2-only to run resolution.[/green]")
            return

        # Pass 2: Indirect call resolution
        console.print("\n[bold]Pass 2: Indirect Call Resolution[/bold]")

        resolver = IndirectCallResolver(db, llm, verbose=verbose, log_file=log_llm)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Resolving indirect calls...", total=None)

            resolutions = resolver.resolve_all_callsites(force=force)

            progress.update(task, completed=True)

        stats2 = resolver.stats
        console.print(f"Pass 2 complete:")
        console.print(f"  Callsites processed: {stats2['callsites_processed']}")
        console.print(f"  LLM calls: {stats2['llm_calls']}")
        console.print(f"  Cache hits: {stats2['cache_hits']}")
        if stats2["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats2['errors']}[/yellow]")

        # Summary
        total_targets = sum(len(targets) for targets in resolutions.values())
        high_conf = sum(
            1 for targets in resolutions.values()
            for t in targets if t.confidence == "high"
        )
        console.print(f"\n[green]Analysis complete![/green]")
        console.print(f"  Total resolved targets: {total_targets}")
        console.print(f"  High confidence matches: {high_conf}")

        # Show database stats
        stats = db.get_stats()
        console.print("\nDatabase statistics:")
        console.print(f"  Address-taken functions: {stats['address_taken_functions']}")
        console.print(f"  Address flow summaries: {stats['address_flow_summaries']}")
        console.print(f"  Indirect callsites: {stats['indirect_callsites']}")
        console.print(f"  Resolved targets: {stats['indirect_call_targets']}")

    finally:
        db.close()


@main.command("show-indirect")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_indirect(db_path, fmt):
    """Show indirect call resolution results."""
    db = SummaryDB(db_path)

    try:
        callsites = db.get_indirect_callsites()
        functions = {f.id: f for f in db.get_all_functions()}

        if fmt == "json":
            output = []
            for cs in callsites:
                targets = db.get_indirect_call_targets(cs.id)
                caller = functions.get(cs.caller_function_id)

                target_list = []
                for t in targets:
                    target_func = functions.get(t.target_function_id)
                    target_list.append({
                        "function": target_func.name if target_func else f"ID:{t.target_function_id}",
                        "confidence": t.confidence,
                        "reasoning": t.llm_reasoning,
                    })

                output.append({
                    "callsite": {
                        "caller": caller.name if caller else "unknown",
                        "expression": cs.callee_expr,
                        "file": cs.file_path,
                        "line": cs.line_number,
                        "signature": cs.signature,
                    },
                    "targets": target_list,
                })

            console.print(json.dumps(output, indent=2))

        else:
            table = Table(title="Indirect Call Resolutions")
            table.add_column("Caller", style="cyan")
            table.add_column("Expression", style="yellow")
            table.add_column("Location", style="dim")
            table.add_column("Targets", style="green")
            table.add_column("Confidence", style="magenta")

            for cs in callsites:
                targets = db.get_indirect_call_targets(cs.id)
                caller = functions.get(cs.caller_function_id)

                if not targets:
                    table.add_row(
                        caller.name if caller else "?",
                        cs.callee_expr,
                        f"{Path(cs.file_path).name}:{cs.line_number}",
                        "[dim]unresolved[/dim]",
                        "-",
                    )
                else:
                    for i, t in enumerate(targets):
                        target_func = functions.get(t.target_function_id)
                        table.add_row(
                            caller.name if caller and i == 0 else "",
                            cs.callee_expr if i == 0 else "",
                            f"{Path(cs.file_path).name}:{cs.line_number}" if i == 0 else "",
                            target_func.name if target_func else f"ID:{t.target_function_id}",
                            t.confidence,
                        )

            console.print(table)

    finally:
        db.close()


@main.command("container-analyze")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="ollama")
@click.option("--model", default=None, help="Model name to use")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--force", "-f", is_flag=True, help="Force re-analysis (ignore cache)")
@click.option("--min-score", default=5, type=int, help="Minimum heuristic score for LLM confirmation (default: 5)")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--heuristic-only", is_flag=True, help="Only run heuristic scoring, skip LLM")
@click.option("--project-name", default=None, help="Project name (default: inferred from DB path)")
def container_analyze(
    db_path, backend, model, llm_host, llm_port, disable_thinking,
    verbose, force, min_score, log_llm, heuristic_only, project_name
):
    """Detect container/collection functions (hash tables, lists, trees, etc.).

    Uses a heuristic pre-filter to find candidates, then confirms with LLM.
    Results are stored in the database for downstream analysis.

    Example:
        llm-summary container-analyze --db summaries.db --backend ollama --model qwen3 -v
        llm-summary container-analyze --db summaries.db --heuristic-only -v
    """
    from .container import ContainerDetector

    # Infer project name from DB path: func-scans/<project>/functions.db -> <project>
    if not project_name:
        db_dir = Path(db_path).resolve().parent
        project_name = db_dir.name if db_dir.name != "." else Path(db_path).stem

    db = SummaryDB(db_path)

    try:
        # Check for functions in DB
        functions = db.get_all_functions()
        if not functions:
            console.print("[red]No functions in database. Run 'extract' or 'scan' first.[/red]")
            return

        console.print(f"Database: {db_path} ({len(functions)} functions, project: {project_name})")

        if heuristic_only:
            # Heuristic-only mode: no LLM
            detector = ContainerDetector(
                db, llm=None, verbose=verbose, min_score=min_score
            )
            candidates = detector.heuristic_only()

            if not candidates:
                console.print("[yellow]No candidates found above threshold.[/yellow]")
                return

            # Display results as a table
            table = Table(title=f"Container Candidates (score >= {min_score})")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Signals")

            # Sort by score descending
            candidates.sort(key=lambda x: x[1], reverse=True)

            for func, score, signals in candidates:
                signal_strs = []
                for s in signals:
                    # Shorten signal display
                    tag = s.split(": ", 1)[0] if ": " in s else s
                    signal_strs.append(tag)
                table.add_row(
                    str(score),
                    func.name,
                    Path(func.file_path).name,
                    ", ".join(signal_strs),
                )

            console.print(table)
            console.print(f"\nTotal candidates: {len(candidates)}")

            stats = detector.stats
            console.print(f"Functions scanned: {stats['functions_scanned']}")
            return

        # Full mode: heuristic + LLM
        backend_kwargs = {}
        if backend == "llamacpp":
            backend_kwargs["host"] = llm_host
            backend_kwargs["port"] = llm_port if llm_port is not None else 8080
        elif backend == "ollama":
            if llm_port is None:
                llm_port = 11434
            backend_kwargs["base_url"] = f"http://{llm_host}:{llm_port}"

        if disable_thinking:
            backend_kwargs["enable_thinking"] = False

        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using {backend} backend ({llm.model})")

        detector = ContainerDetector(
            db, llm=llm, verbose=verbose, log_file=log_llm,
            min_score=min_score, project_name=project_name,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Detecting container functions...", total=None)
            results = detector.detect_all(force=force)
            progress.update(task, completed=True)

        stats = detector.stats
        console.print(f"\nContainer detection complete:")
        console.print(f"  Functions scanned: {stats['functions_scanned']}")
        console.print(f"  Candidates (score >= {min_score}): {stats['candidates']}")
        console.print(f"  LLM calls: {stats['llm_calls']}")
        console.print(f"  Cache hits: {stats['cache_hits']}")
        console.print(f"  Containers found: {stats['containers_found']}")
        if stats["input_tokens"] > 0 or stats["output_tokens"] > 0:
            total_tok = stats['input_tokens'] + stats['output_tokens']
            console.print(f"  Tokens: {total_tok:,} ({stats['input_tokens']:,} in + {stats['output_tokens']:,} out)")
        if stats["errors"] > 0:
            console.print(f"  [yellow]Errors: {stats['errors']}[/yellow]")

    finally:
        db.close()


@main.command("show-containers")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def show_containers(db_path, fmt):
    """Show detected container/collection functions."""
    db = SummaryDB(db_path)

    try:
        summaries = db.get_all_container_summaries()
        functions = {f.id: f for f in db.get_all_functions()}

        if not summaries:
            console.print("[yellow]No container summaries found. Run 'container-analyze' first.[/yellow]")
            return

        if fmt == "json":
            output = []
            for cs in summaries:
                func = functions.get(cs.function_id)
                entry = cs.to_dict()
                if func:
                    entry["function_name"] = func.name
                    entry["file_path"] = func.file_path
                    entry["signature"] = func.signature
                output.append(entry)
            console.print(json.dumps(output, indent=2))

        else:
            table = Table(title="Container Functions")
            table.add_column("Function", style="cyan")
            table.add_column("File", style="dim")
            table.add_column("Type", style="green")
            table.add_column("Container", justify="center")
            table.add_column("Store", justify="center")
            table.add_column("Load", justify="center")
            table.add_column("Conf.", style="magenta")
            table.add_column("Score", justify="right")

            for cs in summaries:
                func = functions.get(cs.function_id)
                table.add_row(
                    func.name if func else f"ID:{cs.function_id}",
                    Path(func.file_path).name if func else "?",
                    cs.container_type,
                    f"arg[{cs.container_arg}]",
                    ",".join(str(a) for a in cs.store_args) if cs.store_args else "-",
                    "ret" if cs.load_return else "-",
                    cs.confidence,
                    str(cs.heuristic_score),
                )

            console.print(table)
            console.print(f"\nTotal: {len(summaries)} container functions")

    finally:
        db.close()


@main.command()
@click.option("--compile-commands", "compile_commands_path", type=click.Path(exists=True), required=True,
              help="Path to compile_commands.json")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def scan(compile_commands_path, db_path, verbose):
    """Extract functions, scan indirect call targets, and find callsites (no LLM).

    This command runs the pre-LLM scanner phases:
    1. Extract all functions from source files
    2. Scan for indirect call targets (address-taken, virtual, attributes)
    3. Find indirect callsites

    Example:
        llm-summary scan --compile-commands build/compile_commands.json --db out.db
    """
    from collections import Counter

    from rich.progress import BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn

    from .models import TargetType

    # Load compile_commands.json
    try:
        compile_commands = CompileCommandsDB(compile_commands_path)
    except Exception as e:
        console.print(f"[red]Failed to load compile_commands.json: {e}[/red]")
        return

    # Filter to C/C++ source files
    c_extensions = {".c", ".cpp", ".cc", ".cxx", ".c++"}
    all_files = compile_commands.get_all_files()
    source_files = [
        f for f in all_files
        if Path(f).suffix.lower() in c_extensions
    ]

    if not source_files:
        console.print("[red]No C/C++ source files found in compile_commands.json[/red]")
        return

    console.print(f"Loaded compile_commands.json: {len(all_files)} entries, {len(source_files)} C/C++ source files")

    db = SummaryDB(db_path)

    try:
        # Phase 1: Extract functions
        console.print("\n[bold]Phase 1: Extracting functions[/bold]")

        extractor = FunctionExtractor(compile_commands=compile_commands)
        all_functions = []
        extract_errors = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting...", total=len(source_files))

            for f in source_files:
                try:
                    functions = extractor.extract_from_file(f)
                    all_functions.extend(functions)
                    if verbose:
                        progress.console.print(f"  {Path(f).name}: {len(functions)} functions")
                except Exception as e:
                    extract_errors += 1
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
                progress.advance(task)

        db.insert_functions_batch(all_functions)
        console.print(f"  Functions: {len(all_functions)}")
        if extract_errors:
            console.print(f"  [yellow]Errors: {extract_errors}[/yellow]")

        # Phase 2: Scan for indirect call targets
        console.print("\n[bold]Phase 2: Scanning indirect call targets[/bold]")

        scanner = AddressTakenScanner(db, compile_commands=compile_commands)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning...", total=len(source_files))

            for f in source_files:
                try:
                    scanner.scan_files([f])
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
                progress.advance(task)

        # Count targets by type
        atfs = db.get_address_taken_functions()
        type_counts: Counter[str] = Counter()
        for atf in atfs:
            type_counts[atf.target_type] += 1

        console.print(f"  Total targets: {len(atfs)}")
        for tt in TargetType:
            count = type_counts.get(tt.value, 0)
            if count > 0:
                console.print(f"    {tt.value}: {count}")

        # Phase 3: Find indirect callsites
        console.print("\n[bold]Phase 3: Finding indirect callsites[/bold]")

        finder = IndirectCallsiteFinder(db, compile_commands=compile_commands)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Finding callsites...", total=len(source_files))

            all_callsites = []
            for f in source_files:
                try:
                    callsites = finder.find_in_files([f])
                    all_callsites.extend(callsites)
                except Exception as e:
                    if verbose:
                        progress.console.print(f"  [yellow]{Path(f).name}: {e}[/yellow]")
                progress.advance(task)

        console.print(f"  Indirect callsites: {len(all_callsites)}")

        # Summary
        console.print(f"\n[bold]Summary[/bold]")
        console.print(f"  Source files: {len(source_files)}")
        console.print(f"  Functions: {len(all_functions)}")
        console.print(f"  Indirect call targets: {len(atfs)}")
        console.print(f"  Indirect callsites: {len(all_callsites)}")
        console.print(f"  Database: {db_path}")

    finally:
        db.close()


@main.command("build-learn")
@click.option("--project-path", type=click.Path(exists=True), required=True, help="Path to the project to build")
@click.option("--build-dir", type=click.Path(), default=None, help="Custom build directory (default: <project-path>/build)")
@click.option("--backend", type=click.Choice(["claude", "openai", "ollama", "llamacpp", "gemini"]), default="claude", help="LLM backend for incremental learning")
@click.option("--model", default=None, help="Model name (default depends on backend)")
@click.option("--llm-host", default="localhost", help="Hostname for local LLM backends (llamacpp, ollama)")
@click.option("--llm-port", default=None, type=int, help="Port for local LLM backends (llamacpp: 8080, ollama: 11434)")
@click.option("--disable-thinking", is_flag=True, help="Disable thinking/reasoning mode for llamacpp (useful for structured output)")
@click.option("--max-retries", default=3, help="Maximum build attempts")
@click.option("--container-image", default="llm-summary-builder:latest", help="Docker image to use")
@click.option("--enable-lto/--no-lto", default=True, help="Enable LLVM LTO")
@click.option("--prefer-static/--no-static", default=True, help="Prefer static linking")
@click.option("--generate-ir/--no-ir", default=True, help="Generate and save LLVM IR artifacts")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--log-llm", type=click.Path(), default=None, help="Log all LLM prompts and responses to file")
@click.option("--ccache-dir", type=click.Path(), default="~/.cache/llm-summary-ccache",
              help="Host ccache directory (default: ~/.cache/llm-summary-ccache)")
@click.option("--no-ccache", is_flag=True, help="Disable ccache")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def build_learn(
    project_path,
    build_dir,
    backend,
    model,
    llm_host,
    llm_port,
    disable_thinking,
    max_retries,
    container_image,
    enable_lto,
    prefer_static,
    generate_ir,
    db_path,
    log_llm,
    ccache_dir,
    no_ccache,
    verbose,
):
    """Learn how to build a project and generate reusable build script."""
    from .builder import Builder
    from .builder.script_generator import ScriptGenerator

    project_path = Path(project_path).resolve()

    console.print(f"\n[bold]Build Agent System[/bold]")
    console.print(f"Project: {project_path}")
    if build_dir:
        console.print(f"Build directory: {build_dir}")
    else:
        console.print(f"Build directory: {project_path}/build")
    console.print(f"Backend: {backend}")
    if model:
        console.print(f"Model: {model}")
    console.print()

    # Initialize LLM backend
    console.print(f"\n[bold]Initializing LLM backend...[/bold]")
    try:
        # Build backend kwargs for local LLM servers
        backend_kwargs = {}
        if backend == "llamacpp":
            backend_kwargs["host"] = llm_host
            backend_kwargs["port"] = llm_port if llm_port is not None else 8080
        elif backend == "ollama":
            if llm_port is None:
                llm_port = 11434
            backend_kwargs["base_url"] = f"http://{llm_host}:{llm_port}"

        # Apply thinking mode control to backends that support it
        if disable_thinking:
            backend_kwargs["enable_thinking"] = False

        llm = create_backend(backend, model=model, **backend_kwargs)
        console.print(f"Using model: [bold]{llm.model}[/bold]")
    except Exception as e:
        console.print(f"[red]Error initializing LLM backend: {e}[/red]")
        sys.exit(1)

    # Resolve ccache directory
    ccache_path = None if no_ccache else Path(ccache_dir).expanduser()

    # Initialize unified builder
    builder = Builder(
        llm=llm,
        container_image=container_image,
        build_dir=Path(build_dir) if build_dir else None,
        build_scripts_dir=Path("build-scripts"),
        max_retries=max_retries,
        enable_lto=enable_lto,
        prefer_static=prefer_static,
        generate_ir=generate_ir,
        verbose=verbose,
        log_file=log_llm,
        ccache_dir=ccache_path,
    )

    # Learn and build
    console.print(f"\n[bold]Learning build configuration...[/bold]")
    try:
        result = builder.learn_and_build(project_path)
    except Exception as e:
        console.print(f"[red]Error during build learning: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    if not result["success"]:
        console.print(f"\n[red]Build failed after {result['attempts']} attempts[/red]")
        console.print(f"\n[bold]Error messages:[/bold]")
        for i, error in enumerate(result["error_messages"], 1):
            console.print(f"\n[yellow]Attempt {i}:[/yellow]")
            console.print(error[:500] + "..." if len(error) > 500 else error)
        sys.exit(1)

    console.print(f"\n[green]Build successful after {result['attempts']} attempts![/green]")

    # Generate build script
    console.print(f"\n[bold]Generating reusable build script...[/bold]")
    project_name = project_path.name
    generator = ScriptGenerator()

    flags = result.get("flags", [])
    build_system_used = result.get("build_system_used", "unknown")
    use_build_dir = result.get("use_build_dir", True)
    dependencies = result.get("dependencies", []) or []
    build_script = result.get("build_script")

    try:
        paths = generator.generate(
            project_name=project_name,
            project_path=project_path,
            flags=flags,
            container_image=container_image,
            build_system=build_system_used,
            enable_ir=generate_ir,
            use_build_dir=use_build_dir,
            dependencies=dependencies,
            build_script=build_script,
        )

        console.print(f"Script: [bold]{paths['script']}[/bold]")
        console.print(f"Config: {paths['config']}")
        console.print(f"Artifacts: {paths['artifacts_dir']}")

        # Extract compile_commands.json to build-scripts/<project>/
        console.print(f"\n[bold]Extracting compile_commands.json...[/bold]")
        try:
            project_dir = paths['script'].parent
            compile_commands_path = builder.extract_compile_commands(
                project_path, output_dir=project_dir, use_build_dir=use_build_dir
            )
            console.print(f"Extracted to: {compile_commands_path}")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to extract compile_commands.json: {e}[/yellow]")
            compile_commands_path = None

        # Generate README if it doesn't exist
        readme_path = generator.scripts_base_dir / "README.md"
        if not readme_path.exists():
            generator.generate_readme()
            console.print(f"README: {readme_path}")

    except Exception as e:
        console.print(f"[red]Error generating build script: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

    # Store in database
    console.print(f"\n[bold]Storing build configuration in database...[/bold]")
    db = SummaryDB(db_path)
    try:
        config_dict = {
            "flags": flags,
            "use_build_dir": use_build_dir,
        }
        if dependencies:
            config_dict["dependencies"] = dependencies

        db.add_build_config(
            project_path=str(project_path),
            project_name=project_name,
            build_system=build_system_used,
            configuration=config_dict,
            script_path=str(paths["script"]),
            artifacts_dir=str(paths["artifacts_dir"]),
            compile_commands_path=str(compile_commands_path) if compile_commands_path else None,
            llm_backend=backend,
            llm_model=llm.model,
            build_attempts=result["attempts"],
        )
        console.print(f"Database: {db_path}")
    finally:
        db.close()

    # Summary
    console.print(f"\n[bold green]✓ Build learning complete![/bold green]")
    console.print(f"\nNext steps:")
    console.print(f"1. Run the build script:")
    console.print(f"   [cyan]{paths['script']}[/cyan]")
    console.print(f"\n2. Analyze with llm-summary:")
    console.print(f"   [cyan]llm-summary extract --path {project_path} --db {db_path}[/cyan]")

    if generate_ir:
        console.print(f"\n3. LLVM IR artifacts will be in:")
        console.print(f"   [cyan]{paths['artifacts_dir']}[/cyan]")


@main.command("generate-kanalyzer-script")
@click.option("--project", required=True, help="Project name (looks up build-scripts/<project>/)")
@click.option("--artifacts-dir", default=None, help="Artifacts directory (default: build-scripts/<project>/artifacts)")
@click.option("--output-json", required=True, help="Output path for KAMain JSON call graph")
@click.option("--kamain-bin", default="/home/csong/project/kanalyzer/release/lib/KAMain",
              help="Path to KAMain binary")
@click.option("--allocator-file", default=None, help="Path to allocator candidates JSON")
@click.option("--container-file", default=None, help="Path to container functions JSON")
@click.option("--verbose-level", default=1, type=int, help="KAMain verbose level (default: 1)")
@click.option("--output", "-o", default=None, help="Output script path (default: stdout)")
def generate_kanalyzer_script(project, artifacts_dir, output_json, kamain_bin, allocator_file, container_file, verbose_level, output):
    """Generate a shell script to run KAMain on a project's .bc files."""
    scripts_dir = Path("build-scripts") / project

    if not scripts_dir.exists():
        console.print(f"[red]Project directory not found: {scripts_dir}[/red]")
        return

    if artifacts_dir is None:
        artifacts_dir = str(scripts_dir / "artifacts")

    lines = [
        "#!/bin/bash",
        f"# KAMain call graph analysis for {project}",
        f"# Generated by llm-summary generate-kanalyzer-script",
        "",
        "set -e",
        "",
        f'ARTIFACTS_DIR="{artifacts_dir}"',
        f'OUTPUT_JSON="{output_json}"',
        f'KAMAIN="{kamain_bin}"',
        "",
        'if [ ! -x "$KAMAIN" ]; then',
        '    echo "Error: KAMain not found at $KAMAIN"',
        '    exit 1',
        'fi',
        "",
        '# Find all .bc files',
        'BC_FILES=$(find "$ARTIFACTS_DIR" -name "*.bc" -type f)',
        "",
        'if [ -z "$BC_FILES" ]; then',
        '    echo "Error: No .bc files found in $ARTIFACTS_DIR"',
        '    exit 1',
        'fi',
        "",
        'BC_COUNT=$(echo "$BC_FILES" | wc -l)',
        'echo "Found $BC_COUNT .bc files in $ARTIFACTS_DIR"',
        "",
        '# Run KAMain',
        '"$KAMAIN" \\',
        '    $BC_FILES \\',
        f'    --callgraph-json "$OUTPUT_JSON" \\',
    ]

    if allocator_file:
        lines.append(f'    --allocator-file "{allocator_file}" \\')
    if container_file:
        lines.append(f'    --container-file "{container_file}" \\')

    lines.append(f'    --verbose {verbose_level}')
    lines.append("")
    lines.append('echo "Call graph written to $OUTPUT_JSON"')
    lines.append("")

    script_content = "\n".join(lines)

    if output:
        output_path = Path(output)
        output_path.write_text(script_content)
        output_path.chmod(0o755)
        console.print(f"Script written to: {output_path}")
    else:
        console.print(script_content)


@main.command("import-callgraph")
@click.option("--json", "json_path", required=True, type=click.Path(exists=True),
              help="Path to KAMain callgraph JSON")
@click.option("--db", "db_path", default="summaries.db", help="Database file path")
@click.option("--clear-edges", is_flag=True, help="Clear existing call_edges before import")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def import_callgraph(json_path, db_path, clear_edges, verbose):
    """Import a KAMain call graph JSON into the database.

    Parses the call graph, matches functions to existing DB entries,
    creates stubs for unmatched functions, and populates call_edges.

    Example:
        llm-summary import-callgraph --json /tmp/libpng_cg.json --db analysis.db --clear-edges -v
    """
    from .callgraph_import import CallGraphImporter

    json_path = Path(json_path)
    db = SummaryDB(db_path)

    try:
        # Show before stats
        before_stats = db.get_stats()
        console.print(f"Database: {db_path}")
        console.print(f"  Functions before: {before_stats['functions']}")
        console.print(f"  Call edges before: {before_stats['call_edges']}")

        importer = CallGraphImporter(db, verbose=verbose)
        stats = importer.import_json(json_path, clear_existing=clear_edges)

        console.print(f"\n[bold]Import complete[/bold]")
        console.print(stats.summary())

        # Show after stats
        after_stats = db.get_stats()
        console.print(f"\nDatabase after:")
        console.print(f"  Functions: {after_stats['functions']}")
        console.print(f"  Call edges: {after_stats['call_edges']}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
