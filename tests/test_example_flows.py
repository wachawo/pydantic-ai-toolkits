#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end toolset flows that mirror the runnable scripts in `examples/`.

These exercise the same tool sequences an agent would issue, but call the
toolset methods directly so the tests stay fast and deterministic — no
LLM, no network. If one of these fails, the matching example will fail
against any model.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest


class TestFilesystemFlow:
    """Mirrors examples/filesystem_example.py: create / append / copy / grep / move / delete_dir / list."""

    def test_full_flow(self, tmp_path: Path) -> None:
        from pydantic_ai_toolbox import FilesystemToolset

        fs = FilesystemToolset(root=tmp_path, read_only=False)

        # 1. create
        fs.write_file("notes.txt", "hello")
        # 2. append
        fs.append_file("notes.txt", "\nsecond line\n")
        assert fs.read_file("notes.txt") == "hello\nsecond line\n"
        # 3. copy
        fs.copy_file("notes.txt", "backup.txt")
        assert fs.read_file("backup.txt") == "hello\nsecond line\n"
        # 4. grep — both files contain "second"
        hits = fs.grep("second")
        assert sorted(h["path"] for h in hits) == ["backup.txt", "notes.txt"]
        # 5. move backup.txt into a fresh archive/ subdirectory
        fs.move("backup.txt", "archive/backup.txt")
        assert "archive/" in fs.list_dir(".")
        # 6. delete archive/ recursively
        fs.delete_dir("archive", recursive=True)
        # 7. list — only notes.txt remains
        assert fs.list_dir(".") == ["notes.txt"]


class TestSQLFlow:
    """Mirrors examples/sql_example.py: INSERT / UPDATE / SELECT against sqlite."""

    def test_full_flow(self, tmp_path: Path) -> None:
        pytest.importorskip("sqlalchemy")
        from sqlalchemy import create_engine, text

        from pydantic_ai_toolbox import SQLToolset

        db_path = tmp_path / "demo.db"
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER)"))
        engine.dispose()

        sql = SQLToolset(dsn=f"sqlite:///{db_path}", read_only=False)
        # INSERT
        sql.execute("INSERT INTO users (name, age) VALUES (:n, :a)", params={"n": "Alex", "a": 30})
        # UPDATE
        sql.execute("UPDATE users SET age = :a WHERE name = :n", params={"a": 31, "n": "Alex"})
        # SELECT
        rows = sql.query("SELECT id, name, age FROM users ORDER BY id")
        assert rows == [{"id": 1, "name": "Alex", "age": 31}]


class TestPandasFlow:
    """Mirrors examples/pandas_example.py: load_csv then query rows > threshold."""

    def test_count_rows_above_threshold(self, tmp_path: Path) -> None:
        pytest.importorskip("pandas")
        from pydantic_ai_toolbox import PandasToolset

        csv_path = tmp_path / "sales.csv"
        csv_path.write_text(
            "country,price,qty\n" "US,10,1\n" "US,25,2\n" "DE,30,3\n" "FR,40,4\n" "DE,15,5\n" "US,50,2\n" "FR,18,1\n",
            encoding="utf-8",
        )

        pd_kit = PandasToolset()
        pd_kit.load_csv("sales", str(csv_path))
        rows = pd_kit.query("sales", "price > 20", limit=1000)
        assert len(rows) == 4
        prices = {r["price"] for r in rows}
        assert prices == {25, 30, 40, 50}


class TestMemoryFlow:
    """Mirrors examples/memory_example.py: set_fact then retrieve via get_fact."""

    def test_set_and_recall(self) -> None:
        from pydantic_ai_toolbox import MemoryToolset

        mem = MemoryToolset()
        # Turn 1: agent would set the fact
        mem.set_fact("user_name", "Alex")
        # Turn 2: unrelated work (no memory needed)
        # Turn 3: agent recalls
        assert mem.get_fact("user_name") == "Alex"
        assert mem.list_facts() == {"user_name": "Alex"}


class TestRAGFlow:
    """Mirrors examples/rag_example.py: index a contradiction-of-priors, search returns it."""

    def test_search_returns_indexed_fact(self) -> None:
        pytest.importorskip("numpy")
        from pydantic_ai_toolbox import RAGToolset

        def stub_embedder(texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                digest = hashlib.sha256(t.encode("utf-8")).digest()[:32]
                out.append([(b - 128) / 128.0 for b in digest])
            return out

        rag = RAGToolset(embedder=stub_embedder, chunk_size=200, chunk_overlap=20)
        rag.add_text("The sky is green.", doc_id="d-sky")

        hits = rag.search("What color is the sky?", k=1)
        assert len(hits) == 1
        assert "green" in hits[0]["text"].lower()


class TestExampleSmoke:
    """Each example module should at least import cleanly."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "examples.filesystem_example",
            "examples.memory_example",
            "examples.quickstart",
        ],
    )
    def test_imports_no_extras(self, module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # Block any accidental network at import time.
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        importlib.import_module(module_name)

    def test_pandas_example_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("pandas")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        importlib.import_module("examples.pandas_example")

    def test_sql_example_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("sqlalchemy")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        importlib.import_module("examples.sql_example")

    def test_rag_example_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("numpy")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        importlib.import_module("examples.rag_example")
