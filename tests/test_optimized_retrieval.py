import sqlite3
import tempfile
import unittest
from array import array
from contextlib import closing
from pathlib import Path

from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)


class TestOptimizedRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "index.db"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "CREATE TABLE documents "
                "(doc_id INTEGER PRIMARY KEY, entity_id TEXT, source TEXT, country TEXT)"
            )
            for table in (
                "name_tokens",
                "name_ngrams",
                "address_tokens",
                "address_ngrams",
            ):
                conn.execute(
                    f"CREATE TABLE {table} (token TEXT PRIMARY KEY, postings BLOB)"
                )
            conn.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                [
                    (1, "S2-1", "source2", "us"),
                    (2, "S3-2", "source3", "france"),
                    (3, "S2-3", "source2", "missing"),
                ],
            )
            conn.execute(
                "INSERT INTO name_tokens VALUES (?, ?)",
                ("apple", array("I", [1, 2]).tobytes()),
            )
            conn.execute(
                "INSERT INTO name_tokens VALUES (?, ?)",
                ("banana", array("I", [3]).tobytes()),
            )
            conn.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_candidates_and_cache(self):
        with OptimizedCandidateGenerator(
            self.db_path,
            posting_cache_mb=1,
            sqlite_cache_mb=1,
            mmap_mb=1,
        ) as generator:
            first = generator.get_candidates("apple", "", "us", 10)
            second = generator.get_candidates("apple", "", "us", 10)
            profile = generator.profile_snapshot()
        self.assertEqual(first, ["S2-1"])
        self.assertEqual(second, first)
        self.assertGreater(profile["cache_hits"], 0)


if __name__ == "__main__":
    unittest.main()
