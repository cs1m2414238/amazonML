import math
import sqlite3
import tempfile
import unittest
from array import array
from contextlib import closing
from pathlib import Path

from business_entity_resolution.src.retrieval.parallel_resumable_retrieval import (
    ParallelResumableRetriever,
    RunSettings,
    WorkerSettings,
    make_record_batches,
)


class TestParallelResumableRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "index.db"
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

        self.records = [
            ("S1-1", "apple", "", "us"),
            ("S1-2", "apple", "", "france"),
            ("S1-3", "banana", "", "us"),
        ]
        worker = WorkerSettings(
            db_path=str(self.db_path),
            posting_cache_mb=1,
            sqlite_cache_mb=1,
            mmap_mb=1,
        )
        self.settings = RunSettings(
            budget=10,
            workers=2,
            batch_size=1,
            dataset_signature="unit-test-records",
            total_records=len(self.records),
            total_batches=math.ceil(len(self.records)),
            worker=worker,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parallel_results_and_resume(self):
        runner = ParallelResumableRetriever(self.root / "run")
        first = runner.run(make_record_batches(self.records, 1), self.settings)
        results = list(runner.iter_results())

        self.assertEqual(first["processed_records"], 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["candidates"], ["S2-1"])
        self.assertEqual(results[1]["candidates"], ["S3-2"])
        self.assertEqual(results[2]["candidates"], ["S2-3"])

        second = runner.run(make_record_batches(self.records, 1), self.settings)
        self.assertEqual(second["processed_records"], 0)
        self.assertEqual(second["resumed_batches"], 3)
        self.assertEqual(list(runner.iter_results()), results)

        export_path = self.root / "candidate_pairs.tsv"
        export = runner.export_candidate_pairs_tsv(export_path)
        self.assertEqual(export["rows"], 3)
        self.assertEqual(
            export_path.read_text(encoding="utf-8").splitlines(),
            [
                "source1_entity_id\tcandidate_entity_ids",
                "S1-1\tS2-1",
                "S1-2\tS3-2",
                "S1-3\tS2-3",
            ],
        )

    def test_mismatched_settings_rejected(self):
        runner = ParallelResumableRetriever(self.root / "run")
        runner.run(make_record_batches(self.records, 1), self.settings)
        changed = RunSettings(
            **{**self.settings.__dict__, "budget": 9}
        )
        with self.assertRaises(ValueError):
            runner.run(make_record_batches(self.records, 1), changed)


if __name__ == "__main__":
    unittest.main()
