"""High-performance O(1) target entity hydration directly from indexed SQLite database."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def hydrate_records_from_db(
    db_path: str | Path,
    entity_ids: Iterable[str],
    batch_chunk_size: int = 500,
) -> dict[str, dict[str, Any]]:
    """Hydrate target records by entity_id in O(1) time using SQLite index.

    Raises:
        KeyError: If any requested entity_id is not found in the database.
    """
    db_path = Path(db_path).resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}")

    id_list = list(set(entity_ids))
    if not id_list:
        return {}

    hydrated: dict[str, dict[str, Any]] = {}

    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
        for i in range(0, len(id_list), batch_chunk_size):
            chunk = id_list[i : i + batch_chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cursor = conn.execute(
                f"SELECT entity_id, source, country, name, addr FROM documents WHERE entity_id IN ({placeholders})",
                chunk,
            )
            for eid, src, ctry, name, addr in cursor:
                hydrated[eid] = {
                    "id": eid,
                    "source": src or "",
                    "country": ctry or "",
                    "name": name or "",
                    "addr": addr or "",
                }

    # Verify that NO requested entities are missing
    if len(hydrated) != len(id_list):
        missing = set(id_list) - set(hydrated.keys())
        sample = list(missing)[:5]
        err_msg = (
            f"Hydration failure: {len(missing)} requested target entities were not found in {db_path}! "
            f"Sample missing IDs: {sample}"
        )
        logging.error(err_msg)
        raise KeyError(err_msg)

    return hydrated
