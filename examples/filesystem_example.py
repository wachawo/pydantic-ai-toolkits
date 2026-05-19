#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemToolset example: create, list, modify, read, delete, list.

A temporary directory becomes the sandbox root. The agent is asked, in
sequence, to:

  1. Create a file `notes.txt` containing one short line.
  2. List the sandbox contents (should show notes.txt).
  3. Append a second line to it.
  4. Read the file back and report its contents.
  5. Delete the file.
  6. List the sandbox contents again (should be empty).

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
                "addressed as '.' (a single dot). To create or overwrite a "
                "file call `write_file(path, content)`. To extend an existing "
                "file call `append_file`. To read use `read_file`. To list "
                "entries call `list_dir(path='.')`. To remove call `delete_file`. "
                "All paths are relative to the sandbox root — never use absolute "
                "paths like '/' and never use '..'. "
                "REPORT EVERY TOOL RESULT VERBATIM. Never invent file or "
                "directory names. If `list_dir` returns an empty list, "
                "literally say 'the directory is empty'."
            ),
        )

        turn1 = agent.run_sync('Create a file named "notes.txt" with the text "hello".')
        logger.info(f"Turn 1 (create):     {turn1.output}")

        turn2 = agent.run_sync("List every file at path '.'.")
        logger.info(f"Turn 2 (list-1):     {turn2.output}")

        turn3 = agent.run_sync('Append a new line "second line" to notes.txt.')
        logger.info(f"Turn 3 (modify):     {turn3.output}")

        turn4 = agent.run_sync("Read notes.txt and tell me what's in it.")
        logger.info(f"Turn 4 (read):       {turn4.output}")

        turn5 = agent.run_sync("Delete notes.txt.")
        logger.info(f"Turn 5 (delete):     {turn5.output}")

        turn6 = agent.run_sync("List every file at path '.' again.")
        logger.info(f"Turn 6 (list-2):     {turn6.output}")


if __name__ == "__main__":
    main()
