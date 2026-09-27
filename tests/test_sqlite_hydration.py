"""Unit tests for SQLite entity hydration."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.src.indexing.sqlite_hydration import hydrate_records_from_db


class TestSQLiteHydration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "CREATE TABLE documents "
                "(doc_id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, addr TEXT)"
            )
            conn.execute("CREATE UNIQUE INDEX idx_documents_entity_id ON documents(entity_id)")
            conn.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (1, "S2-001", "source2", "us", "Acme Corp", "123 Main St"),
                    (2, "S3-002", "source3", "france", "Bistro Paris", "45 Rue de Rivoli"),
                    (3, "S2-003", "source2", "de", "Kaffee Haus", "10 Berliner Str"),
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_successful_hydration(self):
        records = hydrate_records_from_db(self.db_path, ["S2-001", "S3-002"])
        self.assertEqual(len(records), 2)
        self.assertEqual(records["S2-001"]["name"], "Acme Corp")
        self.assertEqual(records["S2-001"]["addr"], "123 Main St")
        self.assertEqual(records["S2-001"]["country"], "us")
        self.assertEqual(records["S3-002"]["name"], "Bistro Paris")
        self.assertEqual(records["S3-002"]["country"], "france")

    def test_missing_entity_raises(self):
        with self.assertRaises(KeyError) as ctx:
            hydrate_records_from_db(self.db_path, ["S2-001", "S2-UNKNOWN"])
        self.assertIn("Hydration failure", str(ctx.exception))
        self.assertIn("S2-UNKNOWN", str(ctx.exception))

    def test_empty_input(self):
        records = hydrate_records_from_db(self.db_path, [])
        self.assertEqual(records, {})


if __name__ == "__main__":
    unittest.main()
