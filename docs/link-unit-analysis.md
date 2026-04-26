# Link-Unit-Aware Analysis

## Problem

Projects often build multiple link units — libraries and executables — from a
single source tree. The current pipeline combines all bitcode files into one
KAMain invocation, which causes:

1. **Symbol collisions**: multiple `main` functions (one per executable)
   create ambiguous stubs in the DB.
2. **Polluted call graphs**: executable-specific code (test drivers, CLI
   tools) appears in the library analysis, adding noise.
3. **Wasted work**: library functions are re-analyzed for every project that
   depends on them (e.g. zlib summaries recomputed when analyzing libpng).
4. **Path mismatches**: container build paths vs host paths for the same
   function cause duplicate orphaned stubs.

## Design

### Key Principles

- **One DB per link unit.** Each library or executable gets its own
  `functions.db`. Cross-target references use imported summary stubs.
- **Dependency-ordered analysis.** Libraries are analyzed before the
  executables and libraries that depend on them.
- **Cross-project reuse.** A project's library summaries (e.g. zlib) can be
  imported as dependency stubs when analyzing a downstream project (e.g.
  libpng), without re-running the LLM.
- **KAMain summary export/import.** KAMain will support exporting reusable
  analysis results (alias info, call edges, points-to summaries) so that
  dependents don't need the full bitcode of their deps — only the exported
  summary.

### Directory Layout

```
func-scans/
  zlib/
    link_units.json                 # discovered link structure (output paths written back by batch_call_graph_gen)
    allocator_candidates.json       # project-level allocator hints
    zlibstatic/
      functions.db                  # library analysis results
      callgraph.json                # KAMain call graph output
      zlibstatic.cflcg              # compressed CFL constraint graph (Phase 1)
      zlibstatic.vsnap              # serialized V-relation snapshot (Phase 2)
    zlib_static_example/
      functions.db                  # executable analysis
      callgraph.json
      zlib_static_example.cflcg
      zlib_static_example.vsnap
  libpng/
    link_units.json
    libpng16/
      functions.db                  # depends on zlib/zlibstatic
      callgraph.json
      libpng16.cflcg
      libpng16.vsnap
    pngtest/
      functions.db                  # depends on libpng16
      callgraph.json
```

### link_units.json Format

Produced by the link-unit discovery agent. Describes all link units, their
bitcode files, and intra-project dependency relationships.

Top-level key is `"link_units"` (array). `bc_files` are **absolute host
paths**. Same-project deps are listed under `"link_deps"` (by output
filename). The `"objects"` field lists the `.o` members used to derive the
`bc_files`.

The analysis output fields (`db_path`, `callgraph_json`, `cflcg`,
`vsnapshot`) are **written back by `batch_call_graph_gen.py`** after
processing. They use absolute paths and may be `null` if the step did not
produce that artifact.

```json
{
  "project": "zlib",
  "build_system": "cmake",
  "build_dir": "/data/csong/build-artifacts/zlib",
  "link_units": [
    {
      "name": "zlibstatic",
      "type": "static_library",
      "output": "libz.a",
      "objects": [
        "CMakeFiles/zlibstatic.dir/adler32.c.o",
        "CMakeFiles/zlibstatic.dir/compress.c.o",
        "CMakeFiles/zlibstatic.dir/deflate.c.o"
      ],
      "bc_files": [
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/adler32.bc",
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/compress.bc",
        "/data/csong/build-artifacts/zlib/CMakeFiles/zlibstatic.dir/deflate.bc"
      ],
      "link_deps": [],
      "db_path": "func-scans/zlib/zlibstatic/functions.db",
      "callgraph_json": "func-scans/zlib/zlibstatic/callgraph.json",
      "cflcg": "func-scans/zlib/zlibstatic/zlibstatic.cflcg",
      "vsnapshot": "func-scans/zlib/zlibstatic/zlibstatic.vsnap"
    },
    {
      "name": "zlib_static_example",
      "type": "executable",
      "output": "test/zlib_static_example",
      "objects": [
        "test/CMakeFiles/zlib_static_example.dir/example.c.o"
      ],
      "bc_files": [
        "/data/csong/build-artifacts/zlib/test/CMakeFiles/zlib_static_example.dir/example.bc"
      ],
      "link_deps": ["libz.a"],
      "db_path": "func-scans/zlib/zlib_static_example/functions.db",
      "callgraph_json": "func-scans/zlib/zlib_static_example/callgraph.json",
      "cflcg": "func-scans/zlib/zlib_static_example/zlib_static_example.cflcg",
      "vsnapshot": "func-scans/zlib/zlib_static_example/zlib_static_example.vsnap"
    }
  ]
}
```

Cross-project dependencies (e.g. libpng → zlib) are **not yet recorded** in
`link_units.json`. They are tracked separately in the top-level
`project_deps.json` registry (see Batch Processing).

Source-set relations (Phase 1.5) populate two additional optional fields:
`alias_of: <name>` for equal source sets, and `imported_from: [<name>]` +
`imported_files: [<abs paths>]` when this unit is a strict superset of the
named one.

## Pipeline

### Phase 0: Build (existing)

`build-learn` agent produces `build.sh`, compiles with LTO +
`-save-temps=obj`, generates `compile_commands.json`.

**Note:** `compile_commands.json` is generated inside the Docker build
container and retains container-internal paths (`/workspace/src/`,
`/workspace/build/`). Downstream tools that consume it on the host must
remap these to the real host paths (see Phase 2).

### Phase 1: Discover Link Units (new)

A ReAct agent explores the build artifacts directory to identify all link
units and their dependency relationships.

**Input:** build directory after Phase 0.

**Agent skills (deterministic tools):**

| Skill | Description |
|-------|-------------|
| `detect_cmake_targets` | Parse `build.ninja` or `CMakeFiles/<t>.dir/link.txt` for link rules. Classify targets as library/executable from CMake target type. |
| `detect_autotools_targets` | Parse `Makefile` for `_LTLIBRARIES`, `_PROGRAMS`, `_LDADD`. Inspect `.la` libtool archives. |
| `detect_meson_targets` | Parse `build.ninja` meson-generated link rules. |
| `inspect_archive` | `ar t lib.a` → list members, map `.o` → `.bc` via `-save-temps` naming. |
| `inspect_elf` | `readelf -d` for dynamic deps; `nm --defined-only` for symbols. |
| `match_bc_files` | Given a list of `.o` files, find corresponding `.bc` files from `-save-temps=obj`. |

**Output:** `link_units.json` written to `func-scans/<project>/`.

The agent uses build-system knowledge to call the right skills. For unknown
build systems, it falls back to artifact inspection (archives, ELF headers,
`nm` for `main`-defining translation units).

### Phase 1.5: Source-Set Relation Detection

After discovery, `batch_scan_targets.py` and `batch_summarize.py` run
`detect_source_set_relations` (`link_units/pipeline.py`) to find units that
share source files — common when CMake builds static + shared variants of
the same library, or when one library is a strict superset of another
(libjpeg-turbo: `libturbojpeg` is `libjpeg` + ~28 extra files). Two
relationships are recorded back into `link_units.json`:

- **`alias_of: <other>`** — the unit's source set equals another unit's
  (and neither has `link_deps`). Pipeline steps skip aliased units; they
  share the canonical unit's DB. Canonical pick: `static_library` >
  `shared_library` > `executable`, ties by name.
- **`imported_from: [<smaller>]` + `imported_files: [...]`** — the unit's
  source set is a strict superset of another's. The smaller unit is
  scanned/summarized first; this unit copies its functions and (at
  summarize time) its summaries via `SummaryDB.import_unit_data`, then
  scans/summarizes only the residual files. The largest proper subset is
  picked so import chains compound (A ⊂ B ⊂ C → C imports from B).

Both relations are derived from the resolved per-unit source-file sets, so
they handle the static/shared CMake case where `.bc`/`.o` paths differ even
though sources match (the older `detect_bc_alias_relations` cannot).
Detection is idempotent and clears stale fields when source sets change.

### Phase 2: Scan Functions (existing, scoped per target)

`llm-summary scan` runs per link unit, producing per-target `functions.db`.

```bash
llm-summary scan \
  --compile-commands /data/csong/build-artifacts/zlib/compile_commands.json \
  --link-units func-scans/zlib/link_units.json \
  --target zlibstatic \
  --project-path /data/csong/opensource/zlib \
  --db func-scans/zlib/zlibstatic/functions.db
```

- `--link-units` / `--target`: restrict extraction to source files whose
  corresponding `.bc` files appear in the named target's `bc_files` list.
  Without these options, all source files in `compile_commands.json` are
  scanned (original behaviour).
- `--project-path`: host path to the project source root. Required when
  `compile_commands.json` uses Docker container paths (`/workspace/src/`).
  Maps `/workspace/src` → `project-path` and `/workspace/build` → `build_dir`
  from `link_units.json`.

### Phase 3: Call Graph (compositional CFL)

Implemented in `batch_call_graph_gen.py --compositional`. Processes targets
in topological order (deps before dependents). Two KAMain phases per target:

#### Phase 1 — Compress constraint graph

Run KAMain on the target's own bc_files to produce a compressed CFL
constraint graph (`.cflcg`):

```bash
KAMain --bc-list zlibstatic_bc.txt \
       --cfl-compressed-output func-scans/zlib/zlibstatic/zlibstatic.cflcg \
       --allocator-file func-scans/zlib/allocator_candidates.json
```

#### Phase 2 — Compositional solve + call graph

Compose the target's constraint graph with those of its transitive deps to
produce the call graph. All bc_files (target + deps) are passed so KAMain
can resolve cross-boundary calls:

```bash
KAMain --bc-list <target+dep bc_files> \
       --cfl-compositional \
       --cfl-compressed-input func-scans/zlib/zlibstatic/zlibstatic.cflcg \
       --cfl-compressed-input func-scans/zlib/zlib_static_example/zlib_static_example.cflcg \
       --callgraph-json func-scans/zlib/zlib_static_example/callgraph.json \
       --v-snapshot   func-scans/zlib/zlib_static_example/zlib_static_example.vsnap \
       --allocator-file func-scans/zlib/allocator_candidates.json
```

The `.vsnap` (V-relation snapshot) records value aliasing for downstream
reuse without re-running CFL solving.

After both phases, `batch_call_graph_gen.py` writes `db_path`,
`callgraph_json`, `cflcg`, and `vsnapshot` back into `link_units.json`.

### Phase 4: Import Call Graph (existing, per target)

```bash
llm-summary import-callgraph \
  --json func-scans/zlib/zlibstatic/callgraph.json \
  --db func-scans/zlib/zlibstatic/functions.db \
  --clear-edges
```

### Phase 5: Init Stdlib + Import Dep Summaries (modified)

Before summarization, populate the target DB with:

1. **Stdlib summaries** (via `init-stdlib`, using the global cache).
2. **Dependency summaries** — import from dep DB(s) so the bottom-up driver
   treats dep functions as pre-summarized leaves.

```bash
# Stdlib (existing)
llm-summary init-stdlib \
  --db func-scans/libpng/libpng16/functions.db \
  --backend gemini

# Dep summaries (new)
llm-summary import-dep-summaries \
  --db func-scans/libpng/libpng16/functions.db \
  --dep-db func-scans/zlib/zlibstatic/functions.db
```

`import-dep-summaries` copies function stubs and their code-contract
summaries from the dep DB into the target DB, tagged with
`model_used="dep:<project>/<target>"`.

The bottom-up driver's cache check will find these and skip them during
summarization.

### Phase 6: Summarize (existing, incremental)

```bash
llm-summary summarize \
  --db func-scans/libpng/libpng16/functions.db \
  --backend gemini
```

The driver processes functions in topological order, running the
code-contract pass per function. Functions with existing summaries (from
stdlib, deps, or prior runs) are cache hits — only new target-specific
functions get LLM calls.

## Incremental Analysis Flow

### Example: libpng depends on zlib

```
Step 1: Analyze zlib (no deps)
  discover-link-units  → zlib/link_units.json
  scan --link-units --target zlibstatic
                       → zlib/zlibstatic/functions.db
  batch_call_graph_gen --compositional
    Phase 1            → zlib/zlibstatic/zlibstatic.cflcg
    Phase 2            → zlib/zlibstatic/callgraph.json
                          zlib/zlibstatic/zlibstatic.vsnap
    import-callgraph   → call_edges in functions.db
    (writes db_path, callgraph_json, cflcg, vsnapshot → link_units.json)
  batch_code_contract
    import-dep-summaries (no intra-project deps for zlibstatic)
    init-stdlib          → stdlib stubs populated
    summarize            → all zlibstatic functions summarized

Step 2: Analyze libpng (depends on zlib)
  discover-link-units  → libpng/link_units.json
  scan --link-units --target libpng16
                       → libpng/libpng16/functions.db
  batch_call_graph_gen --compositional
    Phase 1            → libpng/libpng16/libpng16.cflcg
    Phase 2 (+ zlib cflcg)
                       → libpng/libpng16/callgraph.json
                          libpng/libpng16/libpng16.vsnap
    import-callgraph   → call_edges
  batch_code_contract
    import-dep + import-dep-summaries --dep-db zlib/zlibstatic/functions.db
                         → zlib function contracts copied in
    init-stdlib          → stdlib stubs
    summarize            → only libpng-specific functions need LLM calls
                           zlib functions → cache hit, skipped
```

### Reanalysis

When zlib is updated and re-summarized:
1. Re-run the zlib pipeline (Phases 2–6).
2. For downstream projects (libpng), re-run `import-dep-summaries` to
   refresh the imported stubs, then re-run `summarize --force` on functions
   whose dep summaries changed.

## KAMain Artifacts Per Link Unit

| File | Flag | Description |
|------|------|-------------|
| `<target>.cflcg` | `--cfl-compressed-output` | Compressed CFL constraint graph. Phase 1 output; input to Phase 2 of dependents. |
| `callgraph.json` | `--callgraph-json` | Call graph edges. Imported into `functions.db` via `import-callgraph`. |
| `<target>.vsnap` | `--v-snapshot` | Serialized V-relation (value aliasing). For downstream reuse without re-running CFL solving. |

**Future:** `--export-summary` / `--import-summary` flags for opaque
function-level summaries (alias info, call edges) without passing dep
bitcode. Not yet implemented; current workaround passes dep bc_files in
Phase 2.

## Batch Processing

`batch_call_graph_gen.py` and `batch_summarize.py` implement link-unit
awareness:

- `batch_call_graph_gen.py --compositional`: reads `link_units.json`,
  toposorts targets, runs two-phase KAMain per target, writes `db_path`,
  `callgraph_json`, `cflcg`, `vsnapshot` back into `link_units.json`.
- `batch_summarize.py`: auto-detects `link_units.json`; processes targets
  in topo order, runs `import-dep-summaries` from intra-project dep DBs
  before each target's summarization passes.

A `project_deps.json` registry at the top level records cross-project
dependencies (used for inter-project `import-dep-summaries`):

```json
{
  "libpng": {"deps": ["zlib"]},
  "libtiff": {"deps": ["zlib", "libjpeg-turbo"]},
  "libavif": {"deps": ["libjpeg-turbo", "libpng", "zlib"]}
}
```

## Migration

The transition from one-DB-per-project to one-DB-per-link-unit is
incremental:

1. Projects without `link_units.json` continue to work as today (all
   bc_files → one KAMain run → one functions.db).
2. When `link_units.json` is present, the pipeline switches to per-target
   mode.
3. Existing `func-scans/<project>/functions.db` can be kept as-is or
   migrated by re-running the pipeline with link-unit awareness.
