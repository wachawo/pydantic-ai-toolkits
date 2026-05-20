#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for FilesystemToolset."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai_toolbox.toolsets.filesystem import FilesystemToolset


class TestConstructorValidation:
    def test_root_must_exist(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FilesystemToolset(root=tmp_path / "missing")

    def test_root_must_be_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "not-a-dir"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            FilesystemToolset(root=f)

    def test_defaults_to_read_only(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        assert tk.read_only is True


class TestListDir:
    def test_list_root(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        entries = tk.list_dir(".")
        assert "hello.txt" in entries
        assert "nested/" in entries

    def test_list_nested(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        entries = tk.list_dir("nested")
        assert "data.csv" in entries

    def test_list_non_dir_raises(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(NotADirectoryError):
            tk.list_dir("hello.txt")


class TestReadFile:
    def test_read_existing(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        assert tk.read_file("hello.txt").startswith("hello world")

    def test_read_missing_raises(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(FileNotFoundError):
            tk.read_file("absent.txt")

    def test_read_too_large(self, tmp_workspace: Path) -> None:
        big = tmp_workspace / "big.txt"
        big.write_text("x" * 200)
        tk = FilesystemToolset(root=tmp_workspace, max_bytes=100)
        with pytest.raises(ValueError, match="too large"):
            tk.read_file("big.txt")


class TestWriteOps:
    def test_write_requires_writable(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=True)
        with pytest.raises(PermissionError):
            tk.write_file("new.txt", "content")

    def test_write_creates_parents(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        assert tk.write_file("a/b/c.txt", "hi") is True
        assert (tmp_workspace / "a" / "b" / "c.txt").read_text() == "hi"

    def test_write_refuses_overwrite_when_disabled(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.write_file("x.txt", "first")
        with pytest.raises(FileExistsError):
            tk.write_file("x.txt", "second", overwrite=False)

    def test_append_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.write_file("log.txt", "line1\n")
        tk.append_file("log.txt", "line2\n")
        assert (tmp_workspace / "log.txt").read_text() == "line1\nline2\n"

    def test_delete_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        target = tmp_workspace / "doomed.txt"
        target.write_text("bye")
        tk.delete_file("doomed.txt")
        assert not target.exists()

    def test_delete_refuses_directory(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(IsADirectoryError):
            tk.delete_file("nested")

    def test_make_dir(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.make_dir("brand/new")
        assert (tmp_workspace / "brand" / "new").is_dir()


class TestPathEscape:
    def test_rejects_parent_escape(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(ValueError, match="escapes sandbox"):
            tk.read_file("../outside.txt")

    def test_rejects_absolute_path_outside(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(ValueError, match="escapes sandbox"):
            tk.read_file("/etc/passwd")


class TestStat:
    def test_stat_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        meta = tk.stat("hello.txt")
        assert meta["kind"] == "file"
        assert meta["size"] > 0
        assert "T" in meta["mtime"]

    def test_stat_dir(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        assert tk.stat("nested")["kind"] == "dir"

    def test_stat_missing(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(FileNotFoundError):
            tk.stat("absent")


class TestGlob:
    def test_glob_files_only(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        results = tk.glob("**/*")
        assert "hello.txt" in results
        assert any(r.endswith("data.csv") for r in results)
        assert "nested" not in results  # dirs excluded by default

    def test_glob_include_dirs(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        results = tk.glob("**/*", include_dirs=True)
        assert "nested" in results

    def test_glob_respects_limit(self, tmp_workspace: Path) -> None:
        for i in range(20):
            (tmp_workspace / f"f{i}.txt").write_text("x")
        tk = FilesystemToolset(root=tmp_workspace, max_glob_results=5)
        assert len(tk.glob("*.txt")) == 5


class TestMove:
    def test_requires_writable(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(PermissionError):
            tk.move("hello.txt", "renamed.txt")

    def test_move_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.move("hello.txt", "renamed.txt")
        assert not (tmp_workspace / "hello.txt").exists()
        assert (tmp_workspace / "renamed.txt").read_text().startswith("hello world")

    def test_move_creates_parent_dirs(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.move("hello.txt", "deep/nest/renamed.txt")
        assert (tmp_workspace / "deep" / "nest" / "renamed.txt").is_file()

    def test_move_missing_source(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileNotFoundError):
            tk.move("absent.txt", "anywhere.txt")

    def test_move_refuses_overwrite_by_default(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "other.txt").write_text("there")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileExistsError):
            tk.move("hello.txt", "other.txt")

    def test_move_overwrite_replaces_file(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "other.txt").write_text("there")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.move("hello.txt", "other.txt", overwrite=True)
        assert (tmp_workspace / "other.txt").read_text().startswith("hello world")

    def test_move_overwrite_replaces_dir(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "from").mkdir()
        (tmp_workspace / "from" / "a.txt").write_text("a")
        (tmp_workspace / "to").mkdir()
        (tmp_workspace / "to" / "stale.txt").write_text("stale")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.move("from", "to", overwrite=True)
        assert (tmp_workspace / "to" / "a.txt").read_text() == "a"
        assert not (tmp_workspace / "to" / "stale.txt").exists()

    def test_move_rejects_same_path(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="same"):
            tk.move("hello.txt", "hello.txt")

    def test_move_refuses_root(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="sandbox root"):
            tk.move(".", "elsewhere")

    def test_move_refuses_dir_into_own_subtree(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="inside source"):
            tk.move("nested", "nested/child")

    def test_move_rejects_escape(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="escapes sandbox"):
            tk.move("hello.txt", "../escape.txt")


class TestCopyFile:
    def test_requires_writable(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(PermissionError):
            tk.copy_file("hello.txt", "copy.txt")

    def test_copy_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.copy_file("hello.txt", "deep/copy.txt")
        assert (tmp_workspace / "hello.txt").exists()
        assert (tmp_workspace / "deep" / "copy.txt").read_text().startswith("hello world")

    def test_copy_file_rejects_dir_source(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileNotFoundError):
            tk.copy_file("nested", "copy")

    def test_copy_file_rejects_dir_target(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(IsADirectoryError):
            tk.copy_file("hello.txt", "nested")

    def test_copy_file_rejects_existing_without_overwrite(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "there.txt").write_text("old")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileExistsError):
            tk.copy_file("hello.txt", "there.txt")

    def test_copy_file_overwrite(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "there.txt").write_text("old")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.copy_file("hello.txt", "there.txt", overwrite=True)
        assert (tmp_workspace / "there.txt").read_text().startswith("hello world")

    def test_copy_file_rejects_too_large(self, tmp_workspace: Path) -> None:
        big = tmp_workspace / "big.txt"
        big.write_text("x" * 200)
        tk = FilesystemToolset(root=tmp_workspace, read_only=False, max_bytes=100)
        with pytest.raises(ValueError, match="too large"):
            tk.copy_file("big.txt", "big-copy.txt")


class TestCopyDir:
    def test_copy_dir(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.copy_dir("nested", "nested-copy")
        assert (tmp_workspace / "nested-copy" / "data.csv").read_text().startswith("a,b\n")
        assert (tmp_workspace / "nested" / "data.csv").exists()  # source untouched

    def test_copy_dir_rejects_file_source(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(NotADirectoryError):
            tk.copy_dir("hello.txt", "x")

    def test_copy_dir_refuses_root(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="sandbox root"):
            tk.copy_dir(".", "elsewhere")

    def test_copy_dir_refuses_into_own_subtree(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="inside source"):
            tk.copy_dir("nested", "nested/child")

    def test_copy_dir_refuses_existing_without_overwrite(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "nested-copy").mkdir()
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileExistsError):
            tk.copy_dir("nested", "nested-copy")

    def test_copy_dir_overwrite_merges(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "nested-copy").mkdir()
        (tmp_workspace / "nested-copy" / "extra.txt").write_text("kept")
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.copy_dir("nested", "nested-copy", overwrite=True)
        # both pre-existing and new files coexist
        assert (tmp_workspace / "nested-copy" / "extra.txt").read_text() == "kept"
        assert (tmp_workspace / "nested-copy" / "data.csv").exists()


class TestDeleteDir:
    def test_requires_writable(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(PermissionError):
            tk.delete_dir("nested")

    def test_delete_empty_dir(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "empty").mkdir()
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.delete_dir("empty")
        assert not (tmp_workspace / "empty").exists()

    def test_delete_non_empty_without_recursive(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(OSError):
            tk.delete_dir("nested")

    def test_delete_recursive(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        tk.delete_dir("nested", recursive=True)
        assert not (tmp_workspace / "nested").exists()

    def test_delete_refuses_root(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(ValueError, match="sandbox root"):
            tk.delete_dir(".", recursive=True)

    def test_delete_refuses_file(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(NotADirectoryError):
            tk.delete_dir("hello.txt")

    def test_delete_missing_path(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace, read_only=False)
        with pytest.raises(FileNotFoundError):
            tk.delete_dir("absent")


class TestGrep:
    def test_finds_matches(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "log.txt").write_text("alpha\nERROR: bad\nbeta\nERROR: worse\n")
        tk = FilesystemToolset(root=tmp_workspace)
        hits = tk.grep(r"ERROR")
        assert [(h["path"], h["line"], h["text"]) for h in hits] == [
            ("log.txt", 2, "ERROR: bad"),
            ("log.txt", 4, "ERROR: worse"),
        ]

    def test_case_insensitive(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "log.txt").write_text("ERROR\nerror\n")
        tk = FilesystemToolset(root=tmp_workspace)
        hits = tk.grep("error", case_insensitive=True)
        assert len(hits) == 2

    def test_fixed_pattern(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "code.txt").write_text("a.b\nadb\n")
        tk = FilesystemToolset(root=tmp_workspace)
        # `.` regex matches both lines; fixed escapes it to a literal
        assert [h["line"] for h in tk.grep("a.b", path="code.txt")] == [1, 2]
        assert [h["line"] for h in tk.grep("a.b", path="code.txt", fixed=True)] == [1]

    def test_include_filter(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "a.txt").write_text("match\n")
        (tmp_workspace / "b.md").write_text("match\n")
        tk = FilesystemToolset(root=tmp_workspace)
        hits = tk.grep("match", include="*.md")
        assert [h["path"] for h in hits] == ["b.md"]

    def test_max_matches_cap(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "log.txt").write_text("hit\n" * 50)
        tk = FilesystemToolset(root=tmp_workspace)
        assert len(tk.grep("hit", max_matches=10)) == 10

    def test_skips_files_over_max_bytes(self, tmp_workspace: Path) -> None:
        big = tmp_workspace / "big.txt"
        big.write_text("needle\n" + "x" * 1000)
        small = tmp_workspace / "small.txt"
        small.write_text("needle\n")
        tk = FilesystemToolset(root=tmp_workspace, max_bytes=100)
        hits = tk.grep("needle")
        assert [h["path"] for h in hits] == ["small.txt"]

    def test_single_file_path(self, tmp_workspace: Path) -> None:
        (tmp_workspace / "log.txt").write_text("alpha\nbeta\n")
        tk = FilesystemToolset(root=tmp_workspace)
        hits = tk.grep("beta", path="log.txt")
        assert hits == [{"path": "log.txt", "line": 2, "text": "beta"}]

    def test_missing_path_raises(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(FileNotFoundError):
            tk.grep("x", path="absent")

    def test_rejects_escape(self, tmp_workspace: Path) -> None:
        tk = FilesystemToolset(root=tmp_workspace)
        with pytest.raises(ValueError, match="escapes sandbox"):
            tk.grep("x", path="../outside")
