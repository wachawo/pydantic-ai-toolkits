# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `FilesystemToolset`: `move`, `copy_file`, `copy_dir`, `delete_dir`,
  and `grep` tools. All honour the sandbox root and the existing
  `read_only` / `max_bytes` limits. `copy_dir` and `move` refuse to
  place a destination inside its source, and `delete_dir` refuses to
  remove the sandbox root.

### Changed
- **Breaking**: renamed PyPI distribution from `pydantic-ai-toolkits` to
  `pydantic-ai-toolbox`, the import module from `pydantic_ai_toolkits`
  to `pydantic_ai_toolbox`, and every public class from `*Toolkit` to
  `*Toolset` (`BaseToolset`, `FilesystemToolset`, `SQLToolset`,
  `PandasToolset`, `MemoryToolset`, `RAGToolset`). The internal
  subpackage `pydantic_ai_toolkits/toolkits/` is now
  `pydantic_ai_toolbox/toolsets/`. The `@tool` decorator name is
  unchanged. Class names now match the `Toolset` concept from
  `pydantic_ai.toolsets`, so passing them reads naturally as
  `toolsets=[FilesystemToolset(...), ...]`. Migration: replace
  `from pydantic_ai_toolkits import FilesystemToolkit` with
  `from pydantic_ai_toolbox import FilesystemToolset` (and the
  analogous lines for the other toolsets) and reinstall under the new
  distribution name.

## [0.0.2] - 2026-05-18

### Added
- Initial repository layout mirroring `pydantic-ai`: top-level package
  directory, `toolsets/` subpackage (analogous to
  `pydantic_ai/toolsets/` and `pydantic_ai/common_tools/`), `tests/` flat
  at the top, `examples/` next to it, mkdocs site under `docs/`,
  `Makefile`, `.pre-commit-config.yaml`, `py.typed`, `AGENTS.md`,
  `LICENSE`, `requirements.txt`.
- `BaseToolset` and `@tool` decorator on top of
  `pydantic_ai.toolsets.FunctionToolset`.
- `FilesystemToolset`: stdlib-only sandboxed FS ops
  (list/read/write/append/delete/mkdir/stat/glob).
- `SQLToolset`: SQLAlchemy-backed list/describe/query/execute with a
  read-only guard for single-statement reads.
- `PandasToolset`: in-memory dataframe registry plus
  head/describe/schema/query/aggregate/value_counts.
- `MemoryToolset`: local conversation/scratchpad memory with chat
  history, summary, and buffer-window helpers. Stdlib-only.
- `RAGToolset`: local retrieval-augmented generation — recursive
  character text splitter and an in-memory vector store with cosine
  similarity search. Depends only on numpy.
- Optional extras `[sql]`, `[pandas]`, `[rag]`, `[all]`. `[all]` is
  self-referential (`pydantic-ai-toolbox[sql,pandas,memory,rag]`) so
  new extras propagate automatically. Each toolset module imports its
  third-party library lazily; toolset modules do not import each other.
- `black`, `ruff`, and `mypy` enforced through `.pre-commit-config.yaml`.
  `pyright` is still available via `make pyright` for opt-in strict
  checks, but is not in the default gate (its `python` discovery
  doesn't reliably find venv-installed extras without project-specific
  `pythonPath` configuration).
- `pytest-cov` wired in via `[tool.coverage.run/report]` in
  `pyproject.toml` (`fail_under = 80`, `branch = true`). `pytest --cov`
  picks up every option automatically; plain `pytest` works in any
  env without the plugin. Current coverage 90.07%.
- Test suite for `FilesystemToolset`, `PandasToolset`, `SQLToolset`,
  and the public lazy-attribute surface (`tests/toolsets/test_*.py`,
  `tests/test_public_api.py`). 144 tests, 0 skips.
- `pyarrow>=15.0` added to the `[pandas]` extra so
  `PandasToolset.load_parquet` works out of the box; mirrored in
  `requirements.txt`.
- `requirements-dev.txt` mirroring `[dependency-groups] dev/lint/docs`
  for environments that pin from a flat list (pytest, pytest-cov,
  anyio, mypy, pyright, ruff, black, mkdocs, mkdocs-material, build).
- `.github/workflows/ci.yml`: lint (ruff + black), typecheck
  (mypy + pyright), test matrix on Python 3.10–3.13 with coverage gate,
  and `mkdocs build --strict`.

### Changed
- Removed every `_`-prefixed identifier (functions, methods, instance
  attributes, classes) across the package. Internal helpers are now
  named without the leading underscore: e.g. `MemoryStore`, `Namespace`,
  `VectorIndex`, `RecursiveCharacterTextSplitter`, `atomic_write_json`,
  `cosine_scores`, `normalize_matrix`, `split_text_recursively`. Tool
  registration in `BaseToolset` still filters dunders so this does not
  expose helpers as agent tools.
- `MemoryToolset` constructor keyword `_now` renamed to `now_fn`.
- `VectorIndex` no longer exposes a lazy-consolidating `vectors`
  property; callers (including `save()`) invoke `consolidate()`
  explicitly before reading `vectors` directly.
- `Makefile`: split `typecheck` into `mypy`/`pyright` sub-targets and
  collapsed into one umbrella target.

### Removed
- `HttpToolset` — superseded for the "fetch a page" case by
  `pydantic_ai.common_tools.web_fetch.web_fetch_tool`. For arbitrary
  HTTP APIs, use a dedicated client tool sized to the API in question
  or an MCP server.
