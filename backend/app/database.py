import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/rag.db")

CHUNK_METADATA_COLUMNS = {
    "source_type": "TEXT",
    "page_number": "INTEGER",
    "chunk_index": "INTEGER"
}


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name TEXT NOT NULL,
        source_type TEXT,
        page_number INTEGER,
        chunk_index INTEGER,
        chunk_text TEXT NOT NULL,
        embedding TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS source_manifest (
        source_name TEXT PRIMARY KEY,
        source_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        sha256 TEXT NOT NULL
    )
    """)

    ensure_chunk_metadata_columns(cursor)

    conn.commit()
    conn.close()


def ensure_chunk_metadata_columns(cursor):
    cursor.execute("PRAGMA table_info(chunks)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in CHUNK_METADATA_COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE chunks ADD COLUMN {column_name} {column_type}"
            )


def clear_chunks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chunks")

    conn.commit()
    conn.close()


def insert_chunk(
    source_name,
    chunk_text,
    embedding,
    source_type=None,
    page_number=None,
    chunk_index=None
):
    embedding_json = json.dumps(embedding)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO chunks (
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding_json
    ))

    conn.commit()
    conn.close()


def replace_chunks(chunks, source_manifest=None):
    conn = get_connection()

    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks")
        cursor.executemany("""
        INSERT INTO chunks (
            source_name,
            source_type,
            page_number,
            chunk_index,
            chunk_text,
            embedding
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (
                chunk["source_name"],
                chunk.get("source_type"),
                chunk.get("page_number"),
                chunk.get("chunk_index"),
                chunk["chunk_text"],
                json.dumps(chunk["embedding"]),
            )
            for chunk in chunks
        ])

        if source_manifest is not None:
            cursor.execute("DELETE FROM source_manifest")
            cursor.executemany("""
            INSERT INTO source_manifest (
                source_name,
                source_type,
                file_size,
                sha256
            )
            VALUES (?, ?, ?, ?)
            """, [
                (
                    source["source_name"],
                    source["source_type"],
                    source["file_size"],
                    source["sha256"],
                )
                for source in source_manifest
            ])

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_all_chunks(source_name=None):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        id,
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding
    FROM chunks
    """
    parameters = ()

    if source_name is not None:
        query += " WHERE source_name = ?"
        parameters = (source_name,)

    query += " ORDER BY id"
    cursor.execute(query, parameters)
    rows = cursor.fetchall()

    conn.close()

    chunks = []

    for row in rows:
        chunks.append({
            "id": row[0],
            "source_name": row[1],
            "source_type": row[2],
            "page_number": row[3],
            "chunk_index": row[4],
            "chunk_text": row[5],
            "embedding": json.loads(row[6])
        })

    return chunks


def get_chunk_by_id(chunk_id):
    if not DB_PATH.exists():
        return None

    init_db()
    conn = get_connection()

    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT
            id,
            source_name,
            source_type,
            page_number,
            chunk_index,
            chunk_text
        FROM chunks
        WHERE id = ?
        """, (chunk_id,))
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "source_name": row[1],
        "source_type": row[2],
        "page_number": row[3],
        "chunk_index": row[4],
        "chunk_text": row[5],
    }


def get_source_manifest(db_path=None):
    path = Path(db_path) if db_path is not None else DB_PATH

    if not path.exists():
        return []

    conn = sqlite3.connect(path)

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='source_manifest'"
        )

        if cursor.fetchone() is None:
            return []

        cursor.execute("""
        SELECT source_name, source_type, file_size, sha256
        FROM source_manifest
        ORDER BY source_name
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "source_name": row[0],
            "source_type": row[1],
            "file_size": row[2],
            "sha256": row[3],
        }
        for row in rows
    ]


def get_chunk_stats():
    if not DB_PATH.exists():
        return {
            "db_path": str(DB_PATH),
            "total_chunks": 0,
            "source_count": 0
        }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT source_name) FROM chunks")
    source_count = cursor.fetchone()[0]

    conn.close()

    return {
        "db_path": str(DB_PATH),
        "total_chunks": total_chunks,
        "source_count": source_count
    }


def get_indexed_sources():

    if not DB_PATH.exists():
        return []

    init_db()

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT
            source_name,
            MAX(source_type) AS source_type,
            COUNT(*) AS chunk_count,
            COUNT(DISTINCT page_number) AS page_count
        FROM chunks
        GROUP BY source_name
        ORDER BY source_name
        """)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "source_name": row[0],
            "source_type": row[1],
            "chunk_count": row[2],
            "page_count": row[3],
        }
        for row in rows
    ]
