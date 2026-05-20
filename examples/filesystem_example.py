#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemToolset example: create, modify, copy, grep, move, cleanup.

A temporary directory becomes the sandbox root. The agent is asked, in
sequence, to:

  1. Create `notes.txt` with one short line.
  2. Append a second line.
  3. Copy `notes.txt` to `backup.txt`.
  4. Grep for the word `second` (matches both files).
  5. Move `backup.txt` into a fresh `archive/` directory.
  6. Delete `archive/` recursively.
  7. List the sandbox contents (only `notes.txt` remains).

The toolset rejects any path that would escape the sandbox, so the
agent literally cannot touch anything outside `root`.

Prereqs:
- ollama running locally
- `ollama pull qwen3:latest`
- `pip install pydantic-ai-toolbox`  (base install — stdlib only)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

from pydantic_ai import Agent

from pydantic_ai_toolbox import FilesystemToolset

LOGGING: dict[str, Any] = {
    "format": "%(asctime)s.%(msecs)03d [%(levelname)s]: (%(name)s) %(message)s",
    "level": logging.INFO,
    "datefmt": "%Y-%m-%d %H:%M:%S",
}
logging.basicConfig(**LOGGING)
logger = logging.getLogger(__name__)

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    pass  # python-dotenv is optional

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest")


def main() -> None:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    logging.info(f"Building agent with Ollama model {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
    model = OpenAIChatModel(
        OLLAMA_MODEL,
        provider=OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="ollama"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(
            model=model,
            toolsets=[FilesystemToolset(root=tmp, read_only=False)],
            system_prompt=(
                "/no_think\n"
                "You operate inside a sandboxed workspace. The sandbox root is "
                "addressed as '.' (a single dot). Tools: `write_file(path, content)` "
                "creates or overwrites; `append_file(path, content)` appends; "
                "`read_file(path)` returns text; `list_dir(path='.')` lists entries; "
                "`copy_file(src, dst)` copies one file (parent dirs created); "
                "`move(src, dst)` renames or relocates (parent dirs created); "
                "`grep(pattern, path='.', include='**/*')` searches file contents "
                "and returns a list of `{path, line, text}`; "
                "`delete_file(path)` removes one file; "
                "`delete_dir(path, recursive=True)` removes a directory tree. "
                "All paths are relative to the sandbox root — never use absolute "
                "paths like '/' and never use '..'. "
                "REPORT EVERY TOOL RESULT VERBATIM. Never invent file or "
                "directory names. If `list_dir` returns an empty list, "
                "literally say 'the directory is empty'."
            ),
        )

        turn1 = agent.run_sync('Create a file named "notes.txt" with the text "hello".')
        logger.info(f"Turn 1 (create):     {turn1.output}")

        turn2 = agent.run_sync('Append a new line "second line" to notes.txt.')
        logger.info(f"Turn 2 (append):     {turn2.output}")

        turn3 = agent.run_sync('Copy notes.txt to a new file called "backup.txt".')
        logger.info(f"Turn 3 (copy):       {turn3.output}")

        turn4 = agent.run_sync('Search for the word "second" across every file in the sandbox.')
        logger.info(f"Turn 4 (grep):       {turn4.output}")

        turn5 = agent.run_sync('Move backup.txt into a new directory called "archive" (final path archive/backup.txt).')
        logger.info(f"Turn 5 (move):       {turn5.output}")

        turn6 = agent.run_sync("Delete the archive directory and everything inside it.")
        logger.info(f"Turn 6 (delete_dir): {turn6.output}")

        turn7 = agent.run_sync("List every file at path '.'.")
        logger.info(f"Turn 7 (list):       {turn7.output}")


if __name__ == "__main__":
    main()
