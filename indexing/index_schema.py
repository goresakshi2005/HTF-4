from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

INDEX_VERSION = "1.0.0"
DEFAULT_INDEX_DIR_NAME = ".code_index"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EXTENSIONS = {
    ".kt",
    ".java",
    ".xml",
    ".py",
    ".txt",
    ".md",
    ".yml",
    ".yaml",
}
IGNORED_DIRS = {
    ".git",
    ".gradle",
    ".idea",
    ".kotlin",
    "build",
    "out",
    "__pycache__",
}
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 2


@dataclass(slots=True)
class FileRecord:
    path: str
    rel_path: str
    language: str
    file_hash: str
    mtime: float
    size_bytes: int
    text: str


@dataclass(slots=True)
class SymbolRecord:
    symbol_id: str
    file_rel_path: str
    language: str
    symbol_type: str
    name: str
    start_line: int
    end_line: int
    signature: str
    parent_symbol: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    file_rel_path: str
    language: str
    symbol_name: str | None
    symbol_type: str | None
    start_line: int
    end_line: int
    text: str
    token_count: int
    file_hash: str


def ensure_index_paths(target_folder: Path, index_dir: Path | None = None) -> dict[str, Path]:
    target_folder = target_folder.resolve()
    resolved_index_dir = (index_dir or (target_folder / DEFAULT_INDEX_DIR_NAME)).resolve()
    resolved_index_dir.mkdir(parents=True, exist_ok=True)
    vector_dir = resolved_index_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    return {
        "target_folder": target_folder,
        "index_dir": resolved_index_dir,
        "db_path": resolved_index_dir / "index.db",
        "config_path": resolved_index_dir / "config.json",
        "vector_dir": vector_dir,
        "faiss_path": vector_dir / "chunks.faiss",
        "chunks_jsonl_path": vector_dir / "chunks.jsonl",
    }


def file_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8", errors="ignore")).hexdigest()


def split_identifier(token: str) -> list[str]:
    pieces = [token]
    pieces.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token))
    snake_split = re.split(r"[_\W]+", token)
    pieces.extend(snake_split)
    normalized = []
    for part in pieces:
        lowered = part.strip().lower()
        if lowered and len(lowered) >= 2:
            normalized.append(lowered)
    return normalized


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.extend(split_identifier(token))
    return tokens


def detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".kt":
        return "kotlin"
    if suffix == ".java":
        return "java"
    if suffix == ".xml":
        return "xml"
    return "text"


def chunk_lines(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    start_idx = max(start_line - 1, 0)
    end_idx = min(end_line, len(lines))
    return "\n".join(lines[start_idx:end_idx]).strip()


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS files (
            rel_path TEXT PRIMARY KEY,
            abs_path TEXT NOT NULL,
            language TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            size_bytes INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS symbols (
            symbol_id TEXT PRIMARY KEY,
            file_rel_path TEXT NOT NULL,
            language TEXT NOT NULL,
            symbol_type TEXT NOT NULL,
            name TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            signature TEXT NOT NULL,
            parent_symbol TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_rel_path);
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            file_rel_path TEXT NOT NULL,
            language TEXT NOT NULL,
            symbol_name TEXT,
            symbol_type TEXT,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            file_hash TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_rel_path);

        CREATE TABLE IF NOT EXISTS lexical_postings (
            token TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            tf REAL NOT NULL,
            PRIMARY KEY(token, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_lexical_token ON lexical_postings(token);

        CREATE TABLE IF NOT EXISTS imports (
            file_rel_path TEXT NOT NULL,
            imported_symbol TEXT NOT NULL,
            PRIMARY KEY(file_rel_path, imported_symbol)
        );
        CREATE INDEX IF NOT EXISTS idx_import_symbol ON imports(imported_symbol);
        """
    )
    conn.commit()


def write_config(config_path: Path, config: dict[str, Any]) -> None:
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def serialize_dataclass_rows(rows: Iterable[Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False))
            f.write("\n")
