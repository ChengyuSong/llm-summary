# LLM-Based Memory Allocation Summary Analysis

A compositional, bottom-up analysis tool that generates memory allocation summaries for C/C++ functions using Large Language Models.

## Features

- **Function Extraction**: Uses libclang to accurately parse C/C++ source files
- **Call Graph Analysis**: Builds call graphs including indirect call resolution
- **Bottom-Up Analysis**: Processes functions in topological order (callees before callers)
- **LLM Integration**: Supports Claude, OpenAI, and local models via Ollama
- **Incremental Updates**: Tracks source changes and re-analyzes only affected functions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-summary.git
cd llm-summary

# Install with pip
pip install -e .

# For development
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- libclang (for C/C++ parsing)
- One of: Anthropic API key, OpenAI API key, or local Ollama

Install libclang on Ubuntu/Debian:
```bash
# Ubuntu 24.04
sudo apt-get install libclang-18-dev

# Ubuntu 22.04
sudo apt-get install libclang-14-dev
```

On macOS:
```bash
brew install llvm
```

## Quick Start

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"

# Analyze a C/C++ project
llm-summary analyze /path/to/project

# View results
llm-summary show --allocating-only

# Look up a specific function
llm-summary lookup create_buffer
```

## Commands

### `analyze`
Analyze C/C++ source files and generate allocation summaries.

```bash
llm-summary analyze <path> [options]

Options:
  --db PATH          Database file (default: summaries.db)
  --backend TYPE     LLM backend: claude, openai, ollama
  --model NAME       Specific model to use
  --recursive        Scan directories recursively (default: true)
  --include-headers  Include header files
  --verbose          Verbose output
  --force            Force re-analysis of all functions
```

### `extract`
Extract functions and build call graph without LLM (for testing).

```bash
llm-summary extract <path> [options]
```

### `show`
Display stored summaries.

```bash
llm-summary show [options]

Options:
  --name TEXT        Filter by function name
  --file PATH        Filter by file path
  --allocating-only  Only show functions with allocations
  --format TYPE      Output format: table, json
```

### `lookup`
Look up a specific function's summary.

```bash
llm-summary lookup <function-name> [options]
```

### `stats`
Show database statistics.

```bash
llm-summary stats
```

### `init-stdlib`
Initialize database with standard library summaries.

```bash
llm-summary init-stdlib
```

### `export`
Export all summaries to JSON.

```bash
llm-summary export -o summaries.json
```

### `callgraph`
Export call graph as (caller, callsite, callee) tuples.

```bash
llm-summary callgraph [options]

Options:
  --db PATH                   Database file (default: summaries.db)
  -o, --output PATH           Output file (default: stdout)
  --format [tuples|csv|json]  Output format
  --no-header                 Omit header row (for csv/tuples)
```

## Output Format

For each function, the tool generates a summary like:

```json
{
  "function": "create_buffer",
  "allocations": [
    {
      "type": "heap",
      "source": "malloc",
      "size_expr": "n + 1",
      "size_params": ["n"],
      "returned": true,
      "stored_to": null,
      "may_be_null": true
    }
  ],
  "parameters": {
    "n": {
      "role": "size_indicator",
      "used_in_allocation": true
    }
  },
  "description": "Allocates n+1 bytes of heap memory for a null-terminated buffer."
}
```

## Using Different Models

### Claude (Anthropic)
```bash
export ANTHROPIC_API_KEY="your-key"

# Default model (claude-sonnet-4-20250514)
llm-summary analyze /path/to/project --backend claude

# Specific models
llm-summary analyze /path/to/project --backend claude --model claude-sonnet-4-20250514
llm-summary analyze /path/to/project --backend claude --model claude-opus-4-20250514
```

### OpenAI
```bash
export OPENAI_API_KEY="your-key"

# Default model (gpt-4o)
llm-summary analyze /path/to/project --backend openai

# Specific models
llm-summary analyze /path/to/project --backend openai --model gpt-4o
llm-summary analyze /path/to/project --backend openai --model gpt-4-turbo
```

### Ollama (Local Models)
```bash
# Start ollama server first: ollama serve

# Default model (llama3.1)
llm-summary analyze /path/to/project --backend ollama

# Specific models
llm-summary analyze /path/to/project --backend ollama --model llama3.1:70b
llm-summary analyze /path/to/project --backend ollama --model codellama:34b
llm-summary analyze /path/to/project --backend ollama --model deepseek-coder:33b
```

## Comparing with LLVM IR Call Graph

Extract and export call graph for comparison with LLVM-based analysis:

```bash
# 1. Extract call graph with llm-summary
llm-summary extract /path/to/project --db project.db
llm-summary callgraph --db project.db --format csv -o llm_callgraph.csv

# 2. Generate LLVM call graph
clang -S -emit-llvm -o project.ll /path/to/project/*.c
opt -passes=print-callgraph project.ll 2> llvm_callgraph.txt

# 3. Compare results
# llm-summary format: caller,callsite,callee
# LLVM format needs parsing from the opt output
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Source Files   │────▶│ Function        │────▶│ Call Graph      │
│  (.c/.cpp/.h)   │     │ Extractor       │     │ Builder         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Summary         │◀────│ LLM Summary     │◀────│ Topological     │
│ Database        │     │ Generator       │     │ Ordering (SCCs) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=llm_summary

# Type checking
mypy src/llm_summary

# Linting
ruff check src/
```

## Project Structure

```
llm-summary/
├── src/llm_summary/
│   ├── __init__.py
│   ├── models.py          # Data models
│   ├── db.py              # SQLite database
│   ├── extractor.py       # Function extraction (libclang)
│   ├── callgraph.py       # Call graph construction
│   ├── ordering.py        # SCC and topological ordering
│   ├── summarizer.py      # LLM-based summary generation
│   ├── stdlib.py          # Standard library summaries
│   ├── cli.py             # Command-line interface
│   ├── indirect/
│   │   ├── scanner.py     # Address-taken function scanner
│   │   ├── callsites.py   # Indirect callsite finder
│   │   └── resolver.py    # LLM-based indirect call resolution
│   └── llm/
│       ├── base.py        # Abstract LLM interface
│       ├── claude.py      # Anthropic Claude backend
│       ├── openai.py      # OpenAI GPT backend
│       └── ollama.py      # Local model backend
├── prompts/               # LLM prompt templates
├── tests/                 # Unit tests
└── pyproject.toml
```

## License

MIT License
