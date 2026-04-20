# Claude Code Development Notes

## Style

**IMPORTANT**: when updating documents under `docs`, use brief commit message.

## Pre-commit Checks

A pre-commit hook runs **ruff**, **mypy**, and **pytest** on every commit. All three must pass or the commit is rejected.

**IMPORTANT**: All new code must pass these checks. Add type annotations to all functions, fix lint issues before committing. Do not use `--no-verify` to bypass.

**IMPORTANT**: Do not dismiss lint/type/test failures as "pre-existing" or "unrelated". If a check fails in any file you touched (or in the broader run), fix it. Never report a task as done while any error remains.

## Development Environment

### Virtual Environment
- **Location**: `~/project/llm-summary/venv`
- **Installation**: The `llm-summary` package is installed in the venv in development mode (`pip install -e .`)

**IMPORTANT**: Always activate the venv before running tests or the CLI:
```bash
source ~/project/llm-summary/venv/bin/activate
```

To verify the venv is activated:
```bash
which python  # Should show: /home/csong/project/llm-summary/venv/bin/python
which llm-summary  # Should show: /home/csong/project/llm-summary/venv/bin/llm-summary
```

### Testing the Build Agent

**Before testing, ALWAYS activate the venv:**
```bash
source ~/project/llm-summary/venv/bin/activate
```

Test the build-learn feature with libpng:
```bash
# Set GCP project for Vertex AI
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Test with Claude Sonnet 4.5 (recommended for ReAct mode)
llm-summary build-learn \
  --project-path /data/csong/opensource/libpng \
  --build-dir /data/csong/build-artifacts/libpng \
  --backend claude \
  --model claude-sonnet-4-5@20250929 \
  --verbose
```

### Testing with Ollama

A script `run_ollama_analysis.sh` is available to run the analysis with Ollama in Docker:
- **Ollama Models Volume**: `/docker/ollama-models` (requires root/docker access)
- **Model**: `qwen3-coder:30b` (default, 18GB)
- **Container**: Named `ollama`, exposes port 11434

Usage:
```bash
./run_ollama_analysis.sh --path tests/fixtures/simple.c --db test_simple.db --verbose
```

### Project Structure
- Source code: `src/llm_summary/`
- Test fixtures: `tests/fixtures/` (simple.c, recursive.c, callbacks.c)
- LLM backends: Claude, OpenAI, Ollama (local)

## Assembly Detection Feature

The build agent automatically detects assembly code after successful builds and can iteratively explore CMake flags to minimize it.

### How It Works

1. **Automatic Detection**: After `cmake_build()` succeeds, `AssemblyChecker` scans for:
   - Standalone .s/.S/.asm files (from compile_commands.json)
   - Inline assembly in C/C++ sources (`__asm__`, `asm()`, etc.)
   - Inline assembly in LLVM IR (.bc files, using llvm-dis)

2. **Unavoidable Filtering**: Known unavoidable findings are saved to `build-scripts/<project>/unavoidable_asm.json` and filtered from future results

3. **ReAct Loop Integration**: Agent sees `assembly_check` result and can try different CMake flags (e.g., `-DDISABLE_ASM=ON`, `-DENABLE_SIMD=OFF`)

### Testing the Assembly Checker

```bash
source ~/project/llm-summary/venv/bin/activate
python -m pytest tests/test_assembly_checker.py -v
```

### Manual Assembly Check

```python
from pathlib import Path
from llm_summary.builder.assembly_checker import AssemblyChecker

checker = AssemblyChecker(
    compile_commands_path=Path("/path/to/compile_commands.json"),
    build_dir=Path("/path/to/build"),
    project_path=Path("/path/to/project"),
    unavoidable_asm_path=Path("build-scripts/project/unavoidable_asm.json"),
    verbose=True,
)

result = checker.check(scan_ir=True)
print(f"has_new_assembly: {result.has_new_assembly}")
print(f"known_unavoidable: {len(result.known_unavoidable)}")

# Save findings as unavoidable
if result.has_new_assembly:
    all_findings = result.standalone_asm_files + result.inline_asm_sources + result.inline_asm_ir
    checker.save_unavoidable(all_findings)
```

### Example Results

**Before (Tink project):**
- 131 standalone .S files (BoringSSL crypto assembly)
- 6 inline asm sources
- 19 inline asm in IR

**After agent iteration:**
- 0 standalone files (agent disabled via CMake flags)
- 6 inline asm sources (dependencies, saved as unavoidable)
- 20 inline asm in IR (dependencies, saved as unavoidable)
