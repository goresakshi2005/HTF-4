from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from index_schema import DEFAULT_EMBEDDING_MODEL, ensure_index_paths, read_config, tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local hybrid code index.")
    parser.add_argument("--folder", required=True, help="Indexed folder root.")
    parser.add_argument("--index-dir", default=None, help="Optional explicit index directory.")
    parser.add_argument("--query", required=True, help="Natural language/code query.")
    parser.add_argument("--top-k", type=int, default=8, help="Final number of results.")
    parser.add_argument("--lexical-k", type=int, default=20)
    parser.add_argument("--symbol-k", type=int, default=20)
    parser.add_argument("--semantic-k", type=int, default=20)
    parser.add_argument(
        "--skip-symbol",
        action="store_true",
        help="Skip symbol-based retrieval (prevents overly complex SQL in symbol_retrieval).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def load_sqlite_rows(db_path: Path, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def lexical_retrieval(db_path: Path, query_text: str, top_k: int) -> list[dict[str, Any]]:
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return []
    placeholders = ",".join("?" for _ in query_tokens)
    sql = f"""
    SELECT lp.chunk_id, SUM(lp.tf) AS lexical_score
    FROM lexical_postings lp
    WHERE lp.token IN ({placeholders})
    GROUP BY lp.chunk_id
    ORDER BY lexical_score DESC, lp.chunk_id ASC
    LIMIT ?
    """
    rows = load_sqlite_rows(db_path, sql, tuple(query_tokens + [top_k]))
    return [{"chunk_id": row["chunk_id"], "score": float(row["lexical_score"])} for row in rows]


def symbol_retrieval(db_path: Path, query_text: str, top_k: int) -> list[dict[str, Any]]:
    tokens = tokenize(query_text)
    if not tokens:
        return []
    like_clauses = " OR ".join("LOWER(s.name) LIKE ?" for _ in tokens)
    sql = f"""
    SELECT c.chunk_id, COUNT(*) AS symbol_score
    FROM symbols s
    JOIN chunks c
      ON c.file_rel_path = s.file_rel_path
     AND c.start_line <= s.start_line
     AND c.end_line >= s.start_line
    WHERE {like_clauses}
    GROUP BY c.chunk_id
    ORDER BY symbol_score DESC, c.chunk_id ASC
    LIMIT ?
    """
    params = tuple([f"%{tok.lower()}%" for tok in tokens] + [top_k])
    rows = load_sqlite_rows(db_path, sql, params)
    return [{"chunk_id": row["chunk_id"], "score": float(row["symbol_score"])} for row in rows]


def semantic_retrieval(db_path: Path, index_paths: dict[str, Path], query_text: str, top_k: int, embedding_model: str) -> list[dict[str, Any]]:
    if not index_paths["faiss_path"].exists():
        return []
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        return []

    chunk_rows = load_sqlite_rows(db_path, "SELECT chunk_id FROM chunks ORDER BY rowid ASC")
    if not chunk_rows:
        return []
    chunk_ids = [r["chunk_id"] for r in chunk_rows]
    try:
        model = SentenceTransformer(embedding_model)
        vec = model.encode([query_text], normalize_embeddings=True)
        query_vector = np.asarray(vec, dtype="float32")
        index = faiss.read_index(str(index_paths["faiss_path"]))
        distances, indices = index.search(query_vector, top_k)
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(chunk_ids):
            continue
        results.append({"chunk_id": chunk_ids[idx], "score": float(score)})
    return results


def merge_scores(
    lexical: list[dict[str, Any]],
    symbol: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    weighted: dict[str, dict[str, float]] = defaultdict(lambda: {"lexical": 0.0, "symbol": 0.0, "semantic": 0.0, "total": 0.0})
    for item in lexical:
        weighted[item["chunk_id"]]["lexical"] = item["score"]
    for item in symbol:
        weighted[item["chunk_id"]]["symbol"] = item["score"]
    for item in semantic:
        weighted[item["chunk_id"]]["semantic"] = item["score"]

    for chunk_id, score_map in weighted.items():
        score_map["total"] = 0.45 * score_map["lexical"] + 0.25 * score_map["symbol"] + 0.30 * score_map["semantic"]

    ranked = sorted(
        [{"chunk_id": cid, **scores} for cid, scores in weighted.items()],
        key=lambda item: (-item["total"], -item["lexical"], -item["semantic"], item["chunk_id"]),
    )
    return ranked[:top_k]


def fetch_chunks(db_path: Path, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    sql = f"""
    SELECT chunk_id, file_rel_path, language, symbol_name, symbol_type, start_line, end_line, text
    FROM chunks
    WHERE chunk_id IN ({placeholders})
    """
    rows = load_sqlite_rows(db_path, sql, tuple(chunk_ids))
    return {row["chunk_id"]: row for row in rows}


def fetch_context_expansion(db_path: Path, file_rel_path: str, start_line: int, end_line: int, max_items: int = 3) -> list[dict[str, Any]]:
    sql_adjacent = """
    SELECT chunk_id, file_rel_path, symbol_name, symbol_type, start_line, end_line, text
    FROM chunks
    WHERE file_rel_path = ?
      AND (
        (end_line >= ? AND start_line <= ?)
        OR ABS(start_line - ?) <= 25
        OR ABS(end_line - ?) <= 25
      )
    ORDER BY start_line ASC
    LIMIT ?
    """
    adjacent_rows = load_sqlite_rows(db_path, sql_adjacent, (file_rel_path, start_line, end_line, start_line, end_line, max_items))

    sql_imports = """
    SELECT c.chunk_id, c.file_rel_path, c.symbol_name, c.symbol_type, c.start_line, c.end_line, c.text
    FROM imports i
    JOIN symbols s ON s.file_rel_path = i.file_rel_path
    JOIN chunks c ON c.file_rel_path = s.file_rel_path
    WHERE i.file_rel_path = ?
      AND c.symbol_name LIKE '%' || s.name || '%'
    LIMIT ?
    """
    import_rows = load_sqlite_rows(db_path, sql_imports, (file_rel_path, max_items))

    seen = set()
    out: list[dict[str, Any]] = []
    for row in list(adjacent_rows) + list(import_rows):
        if row["chunk_id"] in seen:
            continue
        seen.add(row["chunk_id"])
        out.append(
            {
                "chunk_id": row["chunk_id"],
                "file_path": row["file_rel_path"],
                "symbol_name": row["symbol_name"],
                "symbol_type": row["symbol_type"],
                "line_range": [row["start_line"], row["end_line"]],
                "snippet": row["text"],
            }
        )
        if len(out) >= max_items:
            break
    return out


def render_text_output(query: str, index_dir: Path, ranked_results: list[dict[str, Any]]) -> str:
    lines = [
        f"query: {query}",
        f"index_dir: {index_dir}",
        f"results: {len(ranked_results)}",
        "-" * 80,
    ]
    for idx, item in enumerate(ranked_results, start=1):
        score = item["score_breakdown"]
        lines.extend(
            [
                f"[{idx}] {item['file_path']}:{item['line_range'][0]}-{item['line_range'][1]}",
                f"    symbol: {item.get('symbol_name') or '<none>'} ({item.get('symbol_type') or 'chunk'})",
                f"    score: total={score['total']:.4f} lexical={score['lexical']:.4f} symbol={score['symbol']:.4f} semantic={score['semantic']:.4f}",
                "    snippet:",
            ]
        )
        for snippet_line in item["snippet"].splitlines()[:12]:
            lines.append(f"      {snippet_line}")
        if len(item["snippet"].splitlines()) > 12:
            lines.append("      ...")
        if item["context_expansion"]:
            lines.append("    context_expansion:")
            for ctx in item["context_expansion"]:
                lines.append(
                    f"      - {ctx['file_path']}:{ctx['line_range'][0]}-{ctx['line_range'][1]} "
                    f"{ctx.get('symbol_name') or '<none>'}"
                )
        lines.append("-" * 80)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"[error] Folder does not exist or is not a directory: {folder}")
        return 2

    index_dir = Path(args.index_dir).resolve() if args.index_dir else None
    index_paths = ensure_index_paths(folder, index_dir)
    db_path = index_paths["db_path"]
    if not db_path.exists():
        print(f"[error] Index database not found at {db_path}. Run index_code.py first.")
        return 2

    config = read_config(index_paths["config_path"])
    embedding_model = config.get("embedding_model", args.embedding_model)

    lexical_hits = lexical_retrieval(db_path, args.query, args.lexical_k)
    symbol_hits = [] if args.skip_symbol else symbol_retrieval(db_path, args.query, args.symbol_k)
    semantic_hits = semantic_retrieval(
        db_path=db_path,
        index_paths=index_paths,
        query_text=args.query,
        top_k=args.semantic_k,
        embedding_model=embedding_model,
    )
    merged = merge_scores(lexical_hits, symbol_hits, semantic_hits, args.top_k)
    chunk_rows = fetch_chunks(db_path, [item["chunk_id"] for item in merged])

    final_results: list[dict[str, Any]] = []
    for entry in merged:
        chunk = chunk_rows.get(entry["chunk_id"])
        if chunk is None:
            continue
        context = fetch_context_expansion(
            db_path=db_path,
            file_rel_path=chunk["file_rel_path"],
            start_line=int(chunk["start_line"]),
            end_line=int(chunk["end_line"]),
            max_items=3,
        )
        final_results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "file_path": chunk["file_rel_path"],
                "language": chunk["language"],
                "symbol_name": chunk["symbol_name"],
                "symbol_type": chunk["symbol_type"],
                "line_range": [int(chunk["start_line"]), int(chunk["end_line"])],
                "snippet": chunk["text"],
                "score_breakdown": {
                    "lexical": entry["lexical"],
                    "symbol": entry["symbol"],
                    "semantic": entry["semantic"],
                    "total": entry["total"],
                },
                "context_expansion": context,
            }
        )

    payload = {
        "query": args.query,
        "index_dir": str(index_paths["index_dir"]),
        "result_count": len(final_results),
        "results": final_results,
        "retrieval_stats": {
            "lexical_candidates": len(lexical_hits),
            "symbol_candidates": len(symbol_hits),
            "semantic_candidates": len(semantic_hits),
            "semantic_enabled": bool(semantic_hits),
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_text_output(args.query, index_paths["index_dir"], final_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
