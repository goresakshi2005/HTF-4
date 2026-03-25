from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from index_schema import (
    DEFAULT_EMBEDDING_MODEL,
    IGNORED_DIRS,
    INDEX_VERSION,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS,
    ChunkRecord,
    FileRecord,
    SymbolRecord,
    create_schema,
    detect_language,
    ensure_index_paths,
    file_sha256,
    read_config,
    stable_id,
    tokenize,
    write_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local hybrid code index.")
    parser.add_argument("--folder", required=True, help="Target source folder to index.")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Optional output index directory. Defaults to <folder>/.code_index.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild index.")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformer model name.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    return parser.parse_args()


def should_skip_dir(dirname: str) -> bool:
    return dirname in IGNORED_DIRS or dirname.startswith(".")


def collect_files(target_folder: Path) -> list[Path]:
    file_paths: list[Path] = []
    for root, dirs, files in os.walk(target_folder):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
            except OSError:
                continue
            file_paths.append(file_path)
    return sorted(file_paths)


def read_file_record(path: Path, root_folder: Path) -> FileRecord | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    rel_path = str(path.relative_to(root_folder)).replace("\\", "/")
    language = detect_language(path)
    return FileRecord(
        path=str(path.resolve()),
        rel_path=rel_path,
        language=language,
        file_hash=file_sha256(text),
        mtime=stat.st_mtime,
        size_bytes=stat.st_size,
        text=text,
    )


def try_build_tree_sitter_parsers() -> dict[str, object]:
    parsers: dict[str, object] = {}
    try:
        import tree_sitter_language_pack as tslp  # type: ignore
        from tree_sitter import Parser  # type: ignore
    except Exception:
        return parsers

    for lang_name, language_key in (("kotlin", "kotlin"), ("java", "java")):
        try:
            parser = Parser(tslp.get_language(language_key))
            parsers[lang_name] = parser
        except Exception:
            continue
    return parsers


def extract_kotlin_or_java_symbols_regex(file_rec: FileRecord) -> tuple[list[SymbolRecord], list[str]]:
    symbols: list[SymbolRecord] = []
    imports: list[str] = []
    lines = file_rec.text.splitlines()
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("import "):
            imports.append(stripped.removeprefix("import ").strip())

        match = re.match(r"(?:public|private|internal|protected)?\s*(class|interface|object)\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
        if match:
            symbol_type = match.group(1)
            name = match.group(2)
            symbols.append(
                SymbolRecord(
                    symbol_id=stable_id(file_rec.rel_path, symbol_type, name, str(idx)),
                    file_rel_path=file_rec.rel_path,
                    language=file_rec.language,
                    symbol_type=symbol_type,
                    name=name,
                    start_line=idx,
                    end_line=min(idx + 30, len(lines)),
                    signature=stripped,
                )
            )
            continue

        fn_match = re.match(r"(?:suspend\s+)?fun\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
        if fn_match:
            name = fn_match.group(1)
            symbols.append(
                SymbolRecord(
                    symbol_id=stable_id(file_rec.rel_path, "function", name, str(idx)),
                    file_rel_path=file_rec.rel_path,
                    language=file_rec.language,
                    symbol_type="function",
                    name=name,
                    start_line=idx,
                    end_line=min(idx + 40, len(lines)),
                    signature=stripped,
                )
            )
            continue

        java_fn = re.match(
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?[A-Za-z0-9_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{?",
            stripped,
        )
        if java_fn and " class " not in stripped:
            name = java_fn.group(1)
            symbols.append(
                SymbolRecord(
                    symbol_id=stable_id(file_rec.rel_path, "method", name, str(idx)),
                    file_rel_path=file_rec.rel_path,
                    language=file_rec.language,
                    symbol_type="method",
                    name=name,
                    start_line=idx,
                    end_line=min(idx + 40, len(lines)),
                    signature=stripped,
                )
            )
    return symbols, imports


def extract_xml_symbols(file_rec: FileRecord) -> tuple[list[SymbolRecord], list[str]]:
    symbols: list[SymbolRecord] = []
    imports: list[str] = []
    lines = file_rec.text.splitlines()
    try:
        root = ET.fromstring(file_rec.text)
    except ET.ParseError:
        return symbols, imports

    line_index_lookup: dict[str, int] = {}
    for idx, line in enumerate(lines, start=1):
        key = line.strip()
        if key:
            line_index_lookup.setdefault(key[:100], idx)

    for elem in root.iter():
        tag = str(elem.tag).split("}")[-1]
        raw_id = elem.attrib.get("{http://schemas.android.com/apk/res/android}id") or elem.attrib.get("android:id")
        raw_name = elem.attrib.get("name")
        symbol_name = None
        if raw_id:
            symbol_name = raw_id.split("/")[-1]
        elif raw_name:
            symbol_name = raw_name
        elif tag:
            symbol_name = tag
        if not symbol_name:
            continue

        preview = f"<{tag}"
        line_no = line_index_lookup.get(preview, 1)
        symbols.append(
            SymbolRecord(
                symbol_id=stable_id(file_rec.rel_path, "xml", symbol_name, str(line_no)),
                file_rel_path=file_rec.rel_path,
                language=file_rec.language,
                symbol_type="xml_node",
                name=symbol_name,
                start_line=line_no,
                end_line=min(line_no + 10, len(lines)),
                signature=f"{tag} id={raw_id or ''} name={raw_name or ''}".strip(),
            )
        )

    for key in ("name", "route", "action"):
        if key in file_rec.text:
            imports.append(key)
    return symbols, imports


def extract_symbols_and_imports(file_rec: FileRecord, parsers: dict[str, object]) -> tuple[list[SymbolRecord], list[str]]:
    if file_rec.language == "xml":
        return extract_xml_symbols(file_rec)
    # Tree-sitter parser hookup point kept for future parser-accurate ranges.
    _ = parsers.get(file_rec.language)
    return extract_kotlin_or_java_symbols_regex(file_rec)


def build_chunks(file_rec: FileRecord, symbols: list[SymbolRecord]) -> list[ChunkRecord]:
    lines = file_rec.text.splitlines()
    chunks: list[ChunkRecord] = []

    if symbols:
        for symbol in symbols:
            start_line = max(symbol.start_line, 1)
            end_line = min(symbol.end_line, len(lines))
            snippet = "\n".join(lines[start_line - 1 : end_line]).strip()
            if not snippet:
                continue
            token_count = len(tokenize(snippet))
            chunks.append(
                ChunkRecord(
                    chunk_id=stable_id(file_rec.rel_path, symbol.symbol_id, str(start_line), str(end_line)),
                    file_rel_path=file_rec.rel_path,
                    language=file_rec.language,
                    symbol_name=symbol.name,
                    symbol_type=symbol.symbol_type,
                    start_line=start_line,
                    end_line=end_line,
                    text=snippet,
                    token_count=token_count,
                    file_hash=file_rec.file_hash,
                )
            )

    if not chunks:
        window = 80
        overlap = 20
        cursor = 0
        while cursor < len(lines):
            end_idx = min(cursor + window, len(lines))
            snippet = "\n".join(lines[cursor:end_idx]).strip()
            if snippet:
                chunks.append(
                    ChunkRecord(
                        chunk_id=stable_id(file_rec.rel_path, "fallback", str(cursor), str(end_idx)),
                        file_rel_path=file_rec.rel_path,
                        language=file_rec.language,
                        symbol_name=None,
                        symbol_type=None,
                        start_line=cursor + 1,
                        end_line=end_idx,
                        text=snippet,
                        token_count=len(tokenize(snippet)),
                        file_hash=file_rec.file_hash,
                    )
                )
            if end_idx == len(lines):
                break
            cursor += max(window - overlap, 1)
    return chunks


def write_index_db(
    db_path: Path,
    file_records: list[FileRecord],
    symbols: list[SymbolRecord],
    imports: list[tuple[str, str]],
    chunks: list[ChunkRecord],
) -> None:
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    with conn:
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM symbols")
        conn.execute("DELETE FROM imports")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM lexical_postings")

        conn.executemany(
            """
            INSERT INTO files(rel_path, abs_path, language, file_hash, mtime, size_bytes)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            [(f.rel_path, f.path, f.language, f.file_hash, f.mtime, f.size_bytes) for f in file_records],
        )

        conn.executemany(
            """
            INSERT INTO symbols(symbol_id, file_rel_path, language, symbol_type, name, start_line, end_line, signature, parent_symbol)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.symbol_id,
                    s.file_rel_path,
                    s.language,
                    s.symbol_type,
                    s.name,
                    s.start_line,
                    s.end_line,
                    s.signature,
                    s.parent_symbol,
                )
                for s in symbols
            ],
        )

        conn.executemany(
            "INSERT OR IGNORE INTO imports(file_rel_path, imported_symbol) VALUES(?, ?)",
            imports,
        )

        conn.executemany(
            """
            INSERT INTO chunks(chunk_id, file_rel_path, language, symbol_name, symbol_type, start_line, end_line, text, token_count, file_hash)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.chunk_id,
                    c.file_rel_path,
                    c.language,
                    c.symbol_name,
                    c.symbol_type,
                    c.start_line,
                    c.end_line,
                    c.text,
                    c.token_count,
                    c.file_hash,
                )
                for c in chunks
            ],
        )

        postings: list[tuple[str, str, float]] = []
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            if not tokens:
                continue
            counts = Counter(tokens)
            total = max(len(tokens), 1)
            for token, count in counts.items():
                postings.append((token, chunk.chunk_id, count / total))
        conn.executemany(
            "INSERT INTO lexical_postings(token, chunk_id, tf) VALUES(?, ?, ?)",
            postings,
        )
    conn.close()


def ensure_unique_symbol_ids(symbols: list[SymbolRecord]) -> list[SymbolRecord]:
    seen: dict[str, int] = defaultdict(int)
    normalized: list[SymbolRecord] = []
    for symbol in symbols:
        seen[symbol.symbol_id] += 1
        if seen[symbol.symbol_id] == 1:
            normalized.append(symbol)
            continue
        suffix = seen[symbol.symbol_id] - 1
        normalized.append(
            SymbolRecord(
                symbol_id=f"{symbol.symbol_id}_{suffix}",
                file_rel_path=symbol.file_rel_path,
                language=symbol.language,
                symbol_type=symbol.symbol_type,
                name=symbol.name,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                signature=symbol.signature,
                parent_symbol=symbol.parent_symbol,
            )
        )
    return normalized


def ensure_unique_chunk_ids(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    seen: dict[str, int] = defaultdict(int)
    normalized: list[ChunkRecord] = []
    for chunk in chunks:
        seen[chunk.chunk_id] += 1
        if seen[chunk.chunk_id] == 1:
            normalized.append(chunk)
            continue
        suffix = seen[chunk.chunk_id] - 1
        normalized.append(
            ChunkRecord(
                chunk_id=f"{chunk.chunk_id}_{suffix}",
                file_rel_path=chunk.file_rel_path,
                language=chunk.language,
                symbol_name=chunk.symbol_name,
                symbol_type=chunk.symbol_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                token_count=chunk.token_count,
                file_hash=chunk.file_hash,
            )
        )
    return normalized


def try_embed_chunks(
    chunks: list[ChunkRecord], model_name: str, batch_size: int
) -> tuple[list[list[float]], str | None]:
    if not chunks:
        return [], None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        return [], f"semantic embedding disabled (sentence-transformers unavailable): {exc}"
    try:
        model = SentenceTransformer(model_name)
        vectors = model.encode([c.text for c in chunks], batch_size=batch_size, normalize_embeddings=True)
        return vectors.tolist(), None
    except Exception as exc:
        return [], f"semantic embedding disabled (model load/encode failed): {exc}"


def persist_vectors(index_paths: dict[str, Path], chunks: list[ChunkRecord], vectors: list[list[float]]) -> str | None:
    if not vectors:
        return "No vectors generated. Semantic retrieval will be skipped."
    chunk_rows = [asdict(c) for c in chunks]
    index_paths["chunks_jsonl_path"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in chunk_rows),
        encoding="utf-8",
    )
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return f"FAISS unavailable ({exc}). Install dependencies from tools/requirements.txt."

    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_paths["faiss_path"]))
    return None


def main() -> int:
    args = parse_args()
    target_folder = Path(args.folder)
    if not target_folder.exists() or not target_folder.is_dir():
        print(f"[error] Folder does not exist or is not a directory: {target_folder}")
        return 2

    index_dir = Path(args.index_dir) if args.index_dir else None
    index_paths = ensure_index_paths(target_folder=target_folder, index_dir=index_dir)

    if args.rebuild and index_paths["index_dir"].exists():
        shutil.rmtree(index_paths["index_dir"], ignore_errors=True)
        index_paths = ensure_index_paths(target_folder=target_folder, index_dir=index_dir)

    prior_config = read_config(index_paths["config_path"])
    discovered_files = collect_files(index_paths["target_folder"])
    file_records = [
        rec
        for rec in (
            read_file_record(path=file_path, root_folder=index_paths["target_folder"]) for file_path in discovered_files
        )
        if rec is not None
    ]
    parser_map = try_build_tree_sitter_parsers()
    if not parser_map:
        print("[warn] Tree-sitter parsers unavailable. Using regex/XML fallback extraction.")

    all_symbols: list[SymbolRecord] = []
    all_imports: list[tuple[str, str]] = []
    all_chunks: list[ChunkRecord] = []
    for file_rec in file_records:
        symbols, imports = extract_symbols_and_imports(file_rec, parser_map)
        chunks = build_chunks(file_rec, symbols)
        all_symbols.extend(symbols)
        all_imports.extend((file_rec.rel_path, imp) for imp in imports)
        all_chunks.extend(chunks)
    all_symbols = ensure_unique_symbol_ids(all_symbols)
    all_chunks = ensure_unique_chunk_ids(all_chunks)

    write_index_db(
        db_path=index_paths["db_path"],
        file_records=file_records,
        symbols=all_symbols,
        imports=all_imports,
        chunks=all_chunks,
    )

    vectors, semantic_error = try_embed_chunks(
        chunks=all_chunks, model_name=args.embedding_model, batch_size=max(args.batch_size, 1)
    )
    persist_error = persist_vectors(index_paths=index_paths, chunks=all_chunks, vectors=vectors)
    warning_msgs = [msg for msg in (semantic_error, persist_error) if msg]

    current_config = {
        "index_version": INDEX_VERSION,
        "target_folder": str(index_paths["target_folder"]),
        "files_indexed": len(file_records),
        "symbols_indexed": len(all_symbols),
        "chunks_indexed": len(all_chunks),
        "semantic_enabled": not warning_msgs,
        "embedding_model": args.embedding_model,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }
    write_config(index_paths["config_path"], current_config)

    print(
        json.dumps(
            {
                "status": "ok",
                "index_dir": str(index_paths["index_dir"]),
                "db_path": str(index_paths["db_path"]),
                "files_indexed": len(file_records),
                "symbols_indexed": len(all_symbols),
                "chunks_indexed": len(all_chunks),
                "warnings": warning_msgs,
                "previous_config_found": bool(prior_config),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
