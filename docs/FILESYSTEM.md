# FilesystemToolset

[README](https://github.com/wachawo/pydantic-ai-toolkits/blob/main/README.md)

Sandboxed file ops rooted at a single directory. No third-party
dependency beyond the standard library.

[Filesystem](https://github.com/wachawo/pydantic-ai-toolkits/blob/main/examples/filesystem_example.py) — Example Create / Read / Append / Delete files

```bash
pip install pydantic-ai-toolbox
```

```python
from pydantic_ai_toolbox import FilesystemToolset

fs = FilesystemToolset(
    root="./workspace",
    read_only=False,
    max_bytes=1_000_000,
    max_glob_results=500,
)
```

All tool arguments are paths **relative to `root`**. Absolute paths and
any `..` segments that escape `root` are rejected with `ValueError`.
With `read_only=True` (default), mutating tools raise `PermissionError`.

## Tools

| Tool          | Signature                                              | Notes                              |
|---------------|--------------------------------------------------------|------------------------------------|
| `list_dir`    | `(path: str = ".") -> list[str]`                       | dirs suffixed with `/`             |
| `read_file`   | `(path: str) -> str`                                   | rejects files over `max_bytes`     |
| `write_file`  | `(path: str, content: str, overwrite: bool = True)`    | creates parents                    |
| `append_file` | `(path: str, content: str) -> bool`                    | creates parents                    |
| `delete_file` | `(path: str) -> bool`                                  | refuses directories                |
| `make_dir`    | `(path: str) -> bool`                                  | `mkdir -p`                         |
| `stat`        | `(path: str) -> dict`                                  | kind, size, mtime ISO              |
| `glob`        | `(pattern: str = "**/*", include_dirs: bool = False)`  | capped at `max_glob_results`       |
| `move`        | `(src: str, dst: str, overwrite: bool = False) -> bool` | refuses root / self-subtree moves  |
| `copy_file`   | `(src: str, dst: str, overwrite: bool = False) -> bool` | rejects files over `max_bytes`     |
| `copy_dir`    | `(src: str, dst: str, overwrite: bool = False) -> bool` | `dirs_exist_ok=overwrite`          |
| `delete_dir`  | `(path: str, recursive: bool = False) -> bool`         | `recursive=False` needs empty dir  |
| `grep`        | `(pattern, path=".", include="**/*", case_insensitive=False, fixed=False, max_matches=200)` | regex over file contents; skips files over `max_bytes` |

## Working example

Seven-turn agent flow: create → append → copy → grep → move → delete_dir → list.
Full script: `examples/filesystem_example.py`.

```python
import tempfile
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_toolbox import FilesystemToolset

with tempfile.TemporaryDirectory() as tmp:
    fs = FilesystemToolset(root=tmp, read_only=False)

    agent = Agent(
        model=OpenAIChatModel(
            "qwen3:8b",
            provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
        ),
        toolsets=[fs],
        system_prompt=(
            "/no_think\n"
            "You operate inside a sandboxed workspace. Tools: `write_file`, "
            "`append_file`, `read_file`, `list_dir`, `copy_file(src, dst)`, "
            "`move(src, dst)`, `grep(pattern, path='.', include='**/*')`, "
            "`delete_file`, `delete_dir(path, recursive=True)`. All paths are "
            "relative to the sandbox root — never use absolute paths or `..`."
        ),
    )

    agent.run_sync('Create a file named "notes.txt" with the text "hello".')
    agent.run_sync('Append a new line "second line" to notes.txt.')
    agent.run_sync('Copy notes.txt to a new file called "backup.txt".')
    agent.run_sync('Search for the word "second" across every file in the sandbox.')
    agent.run_sync('Move backup.txt into a new directory called "archive" (final path archive/backup.txt).')
    agent.run_sync('Delete the archive directory and everything inside it.')
    agent.run_sync("List every file in the sandbox root.")          # ['notes.txt']
```

The `/no_think` directive switches qwen3 out of its default chain-of-thought
mode — without it each turn generates 500-2000 reasoning tokens before any
tool call.

## Direct (no-agent) flow

If you just want the toolset's contract verified, the same flow without an
LLM is in `tests/test_example_flows.py::TestFilesystemFlow`:

```python
fs = FilesystemToolset(root=tmp_path, read_only=False)
fs.write_file("notes.txt", "hello")
fs.append_file("notes.txt", "\nsecond line\n")
fs.copy_file("notes.txt", "backup.txt")
hits = fs.grep("second")
assert sorted(h["path"] for h in hits) == ["backup.txt", "notes.txt"]
fs.move("backup.txt", "archive/backup.txt")
fs.delete_dir("archive", recursive=True)
assert fs.list_dir(".") == ["notes.txt"]
```
