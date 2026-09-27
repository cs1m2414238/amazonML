"""Unit tests for the integrated EndToEndPipeline."""

import sqlite3
import tempfile
import unittest
from array import array
from pathlib import Path

import pandas as pd

from business_entity_resolution.src.models.matcher_inference import BatchMatcher
from business_entity_resolution.src.pipeline.inference_pipeline import (
    EndToEndPipeline,
    stream_s1_batches,
    verify_index_isolation,
)


class TestInferencePipeline(unittest.TestCase):
    def test_pipeline_fallback_resolution(self):
        """Pipeline loads fallback model when requested."""
        model_path = Path("output/models/lightgbm_matcher.txt")
        if not model_path.exists():
            self.skipTest("Trained model artifact not present in local repo")

        pipeline_primary = EndToEndPipeline(
            model_path=model_path,
            threshold=0.60,
            use_fallback=False,
        )
        self.assertIsNotNone(pipeline_primary.matcher)
        self.assertEqual(pipeline_primary.threshold, 0.60)
        pipeline_primary.close()

        fallback_path = Path("output/models/lightgbm_matcher_fallback.txt")
        if fallback_path.exists():
            pipeline_fallback = EndToEndPipeline(
                model_path=model_path,
                threshold=0.70,
                use_fallback=True,
            )
            self.assertIsNotNone(pipeline_fallback.matcher)
            self.assertEqual(pipeline_fallback.threshold, 0.70)
            pipeline_fallback.close()

    def test_format_tsv_output(self):
        """BatchMatcher formats output TSV correctly with header and row format."""
        preds = {
            "S1-001": ["S2-100", "S3-200"],
            "S1-002": [],
            "S1-003": ["S3-300"],
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "test_out.tsv"
            BatchMatcher.format_tsv(preds, out_file)

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [l.rstrip("\r\n") for l in f if l.rstrip("\r\n")]

            self.assertEqual(len(lines), 4)
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertEqual(lines[1], "S1-001\tS2-100,S3-200")
            self.assertEqual(lines[2], "S1-002\t")
            self.assertEqual(lines[3], "S1-003\tS3-300")

    def test_index_isolation(self):
        """verify_index_isolation blocks test inference from accessing the training index."""
        with self.assertRaises(ValueError) as ctx:
            verify_index_isolation("test", r"H:\Amazon_ML_Work\output\step5\indices.db")
        self.assertIn("CRITICAL SAFETY VIOLATION", str(ctx.exception))

        with self.assertRaises(FileNotFoundError):
            verify_index_isolation("test", r"H:\Amazon_ML_Work\output\nonexistent\indices.db")

    def test_stream_s1_batches(self):
        """stream_s1_batches streams records in bounded batches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            s1_file = Path(tmp_dir) / "test_s1.tsv"
            with open(s1_file, "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                for i in range(10):
                    f.write(f"S1-{i}\tBusiness {i}\tAddress {i}\tus\n")

            batches = list(stream_s1_batches(s1_file, batch_size=4))
            self.assertEqual(len(batches), 3)
            self.assertEqual(len(batches[0][1]), 4)
            self.assertEqual(len(batches[1][1]), 4)
            self.assertEqual(len(batches[2][1]), 2)

            # Test limit
            limited = list(stream_s1_batches(s1_file, batch_size=4, limit=5))
            total_items = sum(len(b[1]) for b in limited)
            self.assertEqual(total_items, 5)

    def test_chunked_end_to_end_execution(self):
        """End-to-end chunked pipeline runs retrieval, hydration, scoring, and output generation."""
        model_path = Path("output/models/lightgbm_matcher.txt")
        if not model_path.exists():
            self.skipTest("Trained model artifact not present in local repo")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            db_path = root / "test_indices.db"

            # Create test database with documents and index tables
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE documents "
                    "(doc_id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE, source TEXT, country TEXT, name TEXT, addr TEXT)"
                )
                conn.execute("CREATE UNIQUE INDEX idx_documents_entity_id ON documents(entity_id)")
                for tbl in ("name_tokens", "name_ngrams", "address_tokens", "address_ngrams"):
                    conn.execute(f"CREATE TABLE {tbl} (token TEXT PRIMARY KEY, postings BLOB)")

                conn.executemany(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (1, "S2-101", "source2", "us", "Starbucks Coffee", "100 Broadway"),
                        (2, "S3-202", "source3", "us", "Starbucks Cafe", "100 Broadway New York"),
                        (3, "S2-303", "source2", "us", "Peets Coffee", "500 Market St"),
                    ],
                )
                conn.execute(
                    "INSERT INTO name_tokens VALUES (?, ?)",
                    ("starbucks", array("I", [1, 2]).tobytes()),
                )
                conn.execute(
                    "INSERT INTO name_tokens VALUES (?, ?)",
                    ("coffee", array("I", [1, 3]).tobytes()),
                )
                conn.commit()
            finally:
                conn.close()

            # Create test S1 TSV
            s1_file = root / "test_s1.tsv"
            with open(s1_file, "w", encoding="utf-8") as f:
                f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                f.write("S1-001\tStarbucks\t100 Broadway\tus\n")
                f.write("S1-002\tPeets\t500 Market\tus\n")
                f.write("S1-003\tUnknown Business\t999 Nowhere\tus\n")

            cand_tsv = root / "candidate_pairs.tsv"
            match_tsv = root / "matching_results.tsv"

            pipeline = EndToEndPipeline(
                model_path=model_path,
                threshold=0.60,
                budget=10,
                db_path=db_path,
                workers=2,
                run_dir=root / "run",
            )

            stats = pipeline.run_chunked_inference(
                s1_path=s1_file,
                total_records=3,
                batch_size=2,
                candidate_tsv=cand_tsv,
                matching_tsv=match_tsv,
                source="test",
            )

            self.assertEqual(stats["validation_status"], "PASS")
            self.assertEqual(stats["query_count"], 3)
            self.assertTrue(cand_tsv.exists())
            self.assertTrue(match_tsv.exists())

            # Verify files content
            with open(cand_tsv, "r", encoding="utf-8") as f:
                cand_lines = [l.strip() for l in f if l.strip()]
            with open(match_tsv, "r", encoding="utf-8") as f:
                match_lines = [l.strip() for l in f if l.strip()]

            self.assertEqual(len(cand_lines), 4)  # 1 header + 3 rows
            self.assertEqual(len(match_lines), 4)


if __name__ == "__main__":
    unittest.main()
