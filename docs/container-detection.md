# Container Function Detection

Detects container/collection functions (hash tables, linked lists, trees, maps, caches, queues) that store or retrieve pointer values. Results feed into downstream LLVM IR-based points-to analysis to simplify data-flow graphs.

## Approach

Two-phase: fast heuristic pre-filter followed by per-function LLM confirmation.

```
┌─────────────────────┐
│   functions table    │
│   (from DB)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Phase 1: Heuristic  │  No LLM, scores each function
│ Pre-Filter          │  based on source code patterns
└──────────┬──────────┘
           │ candidates (score >= threshold)
           ▼
┌─────────────────────┐
│ Phase 2: LLM        │  One call per candidate
│ Confirmation        │  with evidence-grounded prompt
└──────────┬──────────┘
           │ confirmed containers
           ▼
┌─────────────────────┐
│ container_summaries │
│ table (in DB)       │
└─────────────────────┘
```

## Heuristic Scoring

Each function is scored against 11 signals. Only candidates above `--min-score` (default 5) go to the LLM.

| Signal | Score | Example |
|--------|-------|---------|
| Container keyword in name | +3 | `hash_insert`, `list_add` |
| Action keyword in name | +2 | `_insert`, `_find`, `_push` |
| `void *` parameter | +2 | generic value storage |
| `void *` return type | +2 | generic value retrieval |
| `void **` parameter | +2 | output param for retrieval |
| Param stored to struct field | +4 | `node->data = value;` |
| Return of struct field | +4 | `return entry->value;` |
| next/prev pointer manipulation | +3 | `node->next = new_node;` |
| Hash/bucket computation | +2 | `idx = hash % size;` |
| void* cast in body | +1 | `(void *)ptr` |
| Key comparison call | +1 | `strcmp(key, entry->key)` |

## LLM Prompt Design

The prompt is evidence-grounded: it includes the specific source lines that triggered each heuristic signal. The LLM confirms or denies a pre-filtered candidate rather than discovering containers from scratch.

Negative anchoring is included — the prompt lists what is NOT a container (struct initializers, simple setters, callback registration) to prevent over-classification.

The LLM returns a JSON response with:
- `is_container`: boolean
- `container_arg`: 0-based index of the container object parameter
- `store_args`: indices of value parameters stored into the container
- `load_return`: whether the return value is loaded from the container
- `container_type`: hash_table, linked_list, tree, map, cache, queue, etc.
- `confidence`: high/medium/low

## CLI Usage

```bash
# Heuristic-only (no LLM, fast)
llm-summary container-analyze --db summaries.db --heuristic-only -v

# Full analysis with LLM
llm-summary container-analyze --db summaries.db \
  --backend llamacpp --llm-host 192.168.1.11 --llm-port 8001 -v

# View results
llm-summary show-containers --db summaries.db
llm-summary show-containers --db summaries.db --format json
```

## Batch Processing

`scripts/batch_container_detect.py` iterates over `func-scans/<project>/functions.db`:

```bash
# Heuristic-only across all projects
python scripts/batch_container_detect.py --heuristic-only -v

# With LLM, parallel
python scripts/batch_container_detect.py \
  --backend llamacpp --llm-host 192.168.1.11 --llm-port 8001 -j4 -v

# Filter by tier or project name
python scripts/batch_container_detect.py --tier 1 --backend llamacpp ...
python scripts/batch_container_detect.py --filter libpng --heuristic-only
```

## Database Schema

Results are stored in `container_summaries`:

```sql
CREATE TABLE container_summaries (
    id INTEGER PRIMARY KEY,
    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    container_arg INTEGER NOT NULL,
    store_args_json TEXT NOT NULL,
    load_return INTEGER NOT NULL DEFAULT 0,
    container_type TEXT NOT NULL,
    confidence TEXT NOT NULL,
    heuristic_score INTEGER,
    heuristic_signals_json TEXT,
    model_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(function_id)
);
```

## Source Files

- `src/llm_summary/container.py` — `ContainerDetector` class
- `src/llm_summary/models.py` — `ContainerSummary` dataclass
- `src/llm_summary/db.py` — DB table and methods
- `src/llm_summary/cli.py` — `container-analyze` and `show-containers` commands
- `scripts/batch_container_detect.py` — batch processing script
