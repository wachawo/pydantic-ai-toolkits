#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local RAG toolset: recursive text splitter + numpy vector index."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from ..base import BaseToolset, tool

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MAX_RESULTS = 20
DEFAULT_MAX_FILE_BYTES = 10_000_000

DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
DOC_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


class Embedder(Protocol):
    """Callable that maps a batch of texts to a batch of equal-length vectors."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class Document:
    """A piece of text with optional metadata and id."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": copy.deepcopy(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        if "text" not in data:
            raise ValueError("Document.from_dict: missing 'text' field")
        return cls(
            text=data["text"],
            metadata=dict(data.get("metadata") or {}),
            id=data.get("id"),
        )


def split_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
    """Split `text` by `separator` regex; optionally re-attach the separator (prefix style)."""
    if separator:
        if keep_separator:
            parts = re.split(f"({separator})", text)
            splits = [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
            if len(parts) % 2 == 0:
                splits += parts[-1:]
            splits = [parts[0], *splits]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s]


class RecursiveCharacterTextSplitter:
    """Recursive character splitter using a prioritised separator list.

    Tries each separator in order; falls back to the next when the current
    leaves a chunk that still exceeds `chunk_size`. Always uses `len`.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than " f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators) if separators else list(DEFAULT_SEPARATORS)
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> list[str]:
        return self.split_recursive(text, self.separators)

    def split_recursive(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for i, candidate in enumerate(separators):
            pattern = re.escape(candidate)
            if not candidate:
                separator = candidate
                break
            if re.search(pattern, text):
                separator = candidate
                new_separators = separators[i + 1 :]
                break

        sep_pattern = re.escape(separator)
        splits = split_with_regex(text, sep_pattern, self.keep_separator)

        good_splits: list[str] = []
        merge_sep = "" if self.keep_separator else separator
        for piece in splits:
            if len(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    final_chunks.extend(self.merge_splits(good_splits, merge_sep))
                    good_splits = []
                if not new_separators:
                    final_chunks.append(piece)
                else:
                    final_chunks.extend(self.split_recursive(piece, new_separators))
        if good_splits:
            final_chunks.extend(self.merge_splits(good_splits, merge_sep))
        return final_chunks

    def join_chunks(self, docs: deque[str] | list[str], separator: str) -> str | None:
        text = separator.join(docs).strip()
        return text or None

    def merge_splits(self, splits: list[str], separator: str) -> list[str]:
        sep_len = len(separator)
        docs: list[str] = []
        current: deque[str] = deque()
        total = 0
        for piece in splits:
            piece_len = len(piece)
            extra = sep_len if current else 0
            if total + piece_len + extra > self.chunk_size:
                if total > self.chunk_size:
                    logger.warning(f"Created a chunk of size {total} which exceeds chunk_size {self.chunk_size}")
                if current:
                    joined = self.join_chunks(current, separator)
                    if joined is not None:
                        docs.append(joined)
                    while total > self.chunk_overlap or (
                        total + piece_len + (sep_len if current else 0) > self.chunk_size and total > 0
                    ):
                        head = current.popleft()
                        total -= len(head) + (sep_len if current else 0)
                        if not current:
                            break
            current.append(piece)
            total += piece_len + (sep_len if len(current) > 1 else 0)
        joined = self.join_chunks(current, separator)
        if joined is not None:
            docs.append(joined)
        return docs


def split_text_recursively(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Module-level wrapper around the recursive splitter (test seam)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def normalize_matrix(np_mod: Any, matrix: Any) -> Any:
    """Row-wise L2 normalisation; rows with zero norm stay zero."""
    norms = np_mod.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np_mod.where(norms == 0, 1.0, norms)
    return matrix / safe


def cosine_scores(np_mod: Any, matrix: Any, query: Any) -> Any:
    """Cosine similarity between a (n, d) matrix and a (d,) query vector."""
    if matrix.size == 0:
        return np_mod.zeros((0,), dtype=np_mod.float64)
    mat_norm = np_mod.linalg.norm(matrix, axis=1)
    q_norm = float(np_mod.linalg.norm(query))
    if q_norm == 0.0:
        return np_mod.zeros((matrix.shape[0],), dtype=np_mod.float64)
    with np_mod.errstate(divide="ignore", invalid="ignore"):
        scores = matrix @ query / (mat_norm * q_norm)
    scores = np_mod.where(np_mod.isnan(scores) | np_mod.isinf(scores), 0.0, scores)
    return scores


class VectorIndex:
    """In-memory parallel-arrays vector index keyed by chunk id.

    ``add`` defers concatenation: each call appends a sub-matrix to
    ``pending`` so a tight ``add_text`` loop stays O(total rows) instead of
    O(N**2) from repeated full-matrix copies. Callers must invoke
    ``consolidate()`` before reading ``vectors``.
    """

    def __init__(self, np_mod: Any) -> None:
        self.np = np_mod
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metas: list[dict[str, Any]] = []
        self.vectors: Any = None
        self.pending: list[Any] = []
        self.doc_index: dict[str, list[int]] = {}
        self.dim: int | None = None

    def consolidate(self) -> None:
        """Fold any buffered vectors into ``self.vectors`` via a single vstack."""
        if not self.pending:
            return
        parts: list[Any] = []
        if self.vectors is not None and self.vectors.size > 0:
            parts.append(self.vectors)
        parts.extend(self.pending)
        self.vectors = self.np.vstack(parts)
        self.pending = []

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metas: list[dict[str, Any]],
        vectors: Any,
        doc_id: str,
    ) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if self.dim is None:
            self.dim = int(vectors.shape[1])
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: got {vectors.shape[1]}, expected {self.dim}")
        start = len(self.ids)
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metas.extend(metas)
        self.pending.append(vectors.astype(self.np.float64, copy=True))
        bucket = self.doc_index.setdefault(doc_id, [])
        bucket.extend(range(start, start + len(ids)))

    def matches_filter(self, meta: dict[str, Any], filt: dict[str, Any]) -> bool:
        return all(meta.get(k) == v for k, v in filt.items())

    def search(
        self,
        query_vec: Any,
        k: int,
        filt: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        if len(self.ids) == 0 or k <= 0:
            return []
        self.consolidate()
        scores = cosine_scores(self.np, self.vectors, query_vec)
        candidates: list[tuple[int, float]] = []
        for i in range(len(self.ids)):
            if filt and not self.matches_filter(self.metas[i], filt):
                continue
            candidates.append((i, float(scores[i])))
        candidates.sort(key=lambda t: (-t[1], t[0]))
        return candidates[:k]

    def delete_doc(self, doc_id: str) -> int:
        rows = self.doc_index.get(doc_id)
        if not rows:
            return 0
        self.consolidate()
        drop = set(rows)
        n = len(rows)
        keep_ids: list[str] = []
        keep_texts: list[str] = []
        keep_metas: list[dict[str, Any]] = []
        keep_rows: list[int] = []
        for i, cid in enumerate(self.ids):
            if i in drop:
                continue
            keep_rows.append(i)
            keep_ids.append(cid)
            keep_texts.append(self.texts[i])
            keep_metas.append(self.metas[i])
        self.ids = keep_ids
        self.texts = keep_texts
        self.metas = keep_metas
        if self.vectors is not None:
            if keep_rows:
                self.vectors = self.vectors[keep_rows]
            else:
                self.vectors = self.np.zeros((0, self.dim or 0), dtype=self.np.float64)
        del self.doc_index[doc_id]
        old_to_new = {old: new for new, old in enumerate(keep_rows)}
        rebuilt: dict[str, list[int]] = {}
        for did, idx_list in self.doc_index.items():
            rebuilt[did] = [old_to_new[i] for i in idx_list if i in old_to_new]
        self.doc_index = rebuilt
        return n

    def clear(self) -> int:
        n = len(self.ids)
        self.ids = []
        self.texts = []
        self.metas = []
        self.doc_index = {}
        self.pending = []
        if self.vectors is not None:
            self.vectors = self.np.zeros((0, self.dim or 0), dtype=self.np.float64)
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": list(self.ids),
            "texts": list(self.texts),
            "metas": [dict(m) for m in self.metas],
            "doc_index": {k: list(v) for k, v in self.doc_index.items()},
            "dim": self.dim,
        }

    def from_dict(self, payload: dict[str, Any], vectors: Any) -> None:
        self.ids = list(payload.get("ids") or [])
        self.texts = list(payload.get("texts") or [])
        self.metas = [dict(m) for m in (payload.get("metas") or [])]
        self.doc_index = {k: list(v) for k, v in (payload.get("doc_index") or {}).items()}
        self.dim = payload.get("dim")
        self.pending = []
        if vectors is None or vectors.size == 0:
            self.vectors = self.np.zeros((0, self.dim or 0), dtype=self.np.float64)
        else:
            self.vectors = vectors.astype(self.np.float64, copy=True)
            if self.dim is None:
                self.dim = int(self.vectors.shape[1])


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write_bytes(path, encoded)


def save_npz(np_mod: Any, base_path: Path, vectors: Any) -> None:
    """Save the vector matrix atomically as `<base>.npz`."""
    target = base_path.parent / (base_path.name + ".npz")
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            arr = vectors if vectors is not None else np_mod.zeros((0, 0), dtype=np_mod.float64)
            np_mod.savez(fh, vectors=arr)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()


def load_npz(np_mod: Any, base_path: Path) -> Any:
    target = base_path.parent / (base_path.name + ".npz")
    with np_mod.load(target) as data:
        return np_mod.array(data["vectors"])


class RAGToolset(BaseToolset):
    """Retrieval-augmented generation toolset: split, embed, store, search.

    Text is chunked with a recursive character splitter, embedded via a
    user-supplied callable, and stored in an in-memory numpy matrix indexed
    by chunk id. Cosine similarity is the only distance in v1. State can
    be persisted atomically to `<path>.npz` + `<path>.json`.
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        storage_path: str | os.PathLike[str] | None = None,
        distance: Literal["cosine"] = "cosine",
        max_results: int = DEFAULT_MAX_RESULTS,
        namespace: str = "default",
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("RAGToolset requires numpy. Install via `pip install pydantic-ai-toolbox[rag]`.") from exc

        if not callable(embedder):
            raise ValueError("embedder must be callable")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than " f"chunk_size ({chunk_size})"
            )
        if distance != "cosine":
            raise ValueError(f"distance must be 'cosine' in v1, got {distance!r}")
        if max_results <= 0:
            raise ValueError(f"max_results must be > 0, got {max_results}")
        if not NAMESPACE_RE.match(namespace):
            raise ValueError(f"Invalid namespace: {namespace!r}")

        path: Path | None = None
        if storage_path is not None:
            path = Path(storage_path).expanduser().resolve()
            if not path.parent.exists():
                raise FileNotFoundError(f"Parent directory does not exist: {path.parent}")

        self.np = np
        self.embedder: Embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_path = path
        self.distance = distance
        self.max_results = max_results
        self.namespace = namespace
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.index = VectorIndex(np)
        super().__init__()
        logger.info(
            f"RAGToolset ready: namespace={namespace} chunk_size={chunk_size} "
            f"chunk_overlap={chunk_overlap} max_results={max_results} "
            f"storage_path={path}"
        )

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently held in the index."""
        return len(self.index.ids)

    def validate_doc_id(self, doc_id: str | None) -> str:
        if doc_id is None:
            return uuid.uuid4().hex
        if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        return doc_id

    def validate_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict")
        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata must be JSON-serialisable: {exc}") from exc
        return dict(metadata)

    def embed_batch(self, texts: list[str]) -> Any:
        vectors = self.embedder(texts)
        if vectors is None:
            raise ValueError("embedder returned None")
        arr = self.np.array(vectors, dtype=self.np.float64)
        if arr.ndim != 2:
            raise ValueError(f"embedder must return a 2-D matrix, got shape {arr.shape}")
        if arr.shape[0] != len(texts):
            raise ValueError(f"embedder returned {arr.shape[0]} vectors for {len(texts)} texts")
        return arr

    def ingest_chunks(self, chunks: list[str], metadata: dict[str, Any], doc_id: str) -> list[str]:
        if not chunks:
            return []
        vectors = self.embed_batch(chunks)
        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        for i in range(len(chunks)):
            chunk_id = f"{doc_id}:{i:06d}"
            ids.append(chunk_id)
            meta = dict(metadata)
            meta.setdefault("doc_id", doc_id)
            meta["chunk_index"] = i
            metas.append(meta)
        self.index.add(ids, chunks, metas, vectors, doc_id)
        return ids

    @tool
    def add_text(
        self,
        text: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> list[str]:
        """Chunk `text`, embed each chunk, and store. Returns the new chunk ids."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        did = self.validate_doc_id(doc_id)
        meta = self.validate_metadata(metadata)
        chunks = self.splitter.split_text(text)
        ids = self.ingest_chunks(chunks, meta, did)
        logger.info(f"add_text doc_id={did} chunks={len(ids)}")
        return ids

    @tool
    def add_file(
        self,
        path: str,
        metadata: dict | None = None,
        doc_id: str | None = None,
        encoding: str = "utf-8",
    ) -> list[str]:
        """Read a text file, chunk it, embed, and store. Refuses files over `DEFAULT_MAX_FILE_BYTES`."""
        target = Path(path).expanduser()
        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        size = target.stat().st_size
        if size > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"File too large: {size} bytes > limit {DEFAULT_MAX_FILE_BYTES}")
        text = target.read_text(encoding=encoding)
        meta = self.validate_metadata(metadata)
        meta.setdefault("source", str(target))
        did = self.validate_doc_id(doc_id)
        chunks = self.splitter.split_text(text)
        ids = self.ingest_chunks(chunks, meta, did)
        logger.info(f"add_file path={target} doc_id={did} chunks={len(ids)}")
        return ids

    @tool
    def add_documents(self, documents: list[dict]) -> list[str]:
        """Ingest a batch of `{text, metadata?, id?}` records. Returns every assigned chunk id."""
        if not isinstance(documents, list):
            raise ValueError("documents must be a list of dicts")
        all_ids: list[str] = []
        for i, raw in enumerate(documents):
            if not isinstance(raw, dict):
                raise ValueError(f"documents[{i}] must be a dict")
            if "text" not in raw or not isinstance(raw["text"], str) or not raw["text"].strip():
                raise ValueError(f"documents[{i}] missing non-empty 'text'")
            did = self.validate_doc_id(raw.get("id"))
            meta = self.validate_metadata(raw.get("metadata"))
            chunks = self.splitter.split_text(raw["text"])
            all_ids.extend(self.ingest_chunks(chunks, meta, did))
        logger.info(f"add_documents docs={len(documents)} chunks={len(all_ids)}")
        return all_ids

    @tool
    def search(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
    ) -> list[dict]:
        """Cosine search; returns `[{id, text, score, metadata}, ...]`, highest score first."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k is None:
            k = self.max_results
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if filter is not None and not isinstance(filter, dict):
            raise ValueError("filter must be a dict")
        capped = min(k, self.max_results)
        if capped == 0:
            return []
        query_vec = self.embed_batch([query])[0]
        hits = self.index.search(query_vec, capped, filter)
        out: list[dict] = []
        for idx, score in hits:
            out.append(
                {
                    "id": self.index.ids[idx],
                    "text": self.index.texts[idx],
                    "score": float(score),
                    "metadata": copy.deepcopy(self.index.metas[idx]),
                }
            )
        return out

    @tool
    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self.chunk_count

    @tool
    def list_documents(self, limit: int | None = None, offset: int = 0) -> list[dict]:
        """List unique documents with chunk counts: `[{doc_id, chunks}, ...]`."""
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        items = [{"doc_id": did, "chunks": len(rows)} for did, rows in self.index.doc_index.items()]
        items.sort(key=lambda d: str(d["doc_id"]))
        sliced = items[offset:]
        if limit is not None:
            sliced = sliced[:limit]
        return sliced

    @tool
    def delete_document(self, doc_id: str) -> int:
        """Drop every chunk belonging to `doc_id`. Returns the number removed (0 if absent)."""
        if not isinstance(doc_id, str) or not DOC_ID_RE.match(doc_id):
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        n = self.index.delete_doc(doc_id)
        if n:
            logger.info(f"delete_document doc_id={doc_id} chunks={n}")
        return n

    @tool
    def clear(self) -> int:
        """Drop every chunk in the index. Returns the count cleared."""
        n = self.index.clear()
        logger.info(f"clear cleared={n}")
        return n

    @tool
    def save(self, path: str | None = None) -> str:
        """Persist the index atomically to `<path>.npz` + `<path>.json`. Returns the base path."""
        base = Path(path).expanduser().resolve() if path is not None else self.storage_path
        if base is None:
            raise ValueError("save() requires a path (none configured on the toolset)")
        if not base.parent.exists():
            raise FileNotFoundError(f"Parent directory does not exist: {base.parent}")
        self.index.consolidate()
        payload = {
            "version": SCHEMA_VERSION,
            "namespace": self.namespace,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "distance": self.distance,
            "index": self.index.to_dict(),
        }
        save_npz(self.np, base, self.index.vectors)
        atomic_write_json(base.with_suffix(base.suffix + ".json"), payload)
        logger.info(f"save base={base} chunks={self.chunk_count}")
        return str(base)

    @tool
    def load(self, path: str | None = None) -> int:
        """Load a previously-saved index. Overwrites in-memory state; returns chunk count."""
        base = Path(path).expanduser().resolve() if path is not None else self.storage_path
        if base is None:
            raise ValueError("load() requires a path (none configured on the toolset)")
        npz_path = base.parent / (base.name + ".npz")
        json_path = base.with_suffix(base.suffix + ".json")
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing vector file: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing index sidecar: {json_path}")
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted index sidecar at {json_path}: {exc}") from exc
        version = payload.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RAG snapshot version {version!r} at {json_path} " f"(expected {SCHEMA_VERSION})"
            )
        index_payload = payload.get("index") or {}
        vectors = load_npz(self.np, base)
        fresh = VectorIndex(self.np)
        fresh.from_dict(index_payload, vectors)
        self.index = fresh
        logger.info(f"load base={base} chunks={self.chunk_count}")
        return self.chunk_count


def main() -> None:
    pass


if __name__ == "__main__":
    main()
