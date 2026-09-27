"""Unit tests for output_validator."""

import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.src.validation.output_validator import validate_tsv_files


class TestOutputValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cand_path = Path(self.temp_dir.name) / "candidate_pairs.tsv"
        self.match_path = Path(self.temp_dir.name) / "matching_results.tsv"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_files(self):
        with open(self.cand_path, "w", encoding="utf-8") as f_c, open(self.match_path, "w", encoding="utf-8") as f_m:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            f_m.write("source1_entity_id\tmatched_entity_ids\n")

            f_c.write("S1_001\tS2_101,S3_201\n")
            f_m.write("S1_001\tS2_101\n")

            f_c.write("S1_002\tS2_102\n")
            f_m.write("S1_002\t\n")  # singleton

            f_c.write("S1_003\t\n")
            f_m.write("S1_003\t\n")  # empty candidates

        report = validate_tsv_files(
            candidate_tsv_path=self.cand_path,
            matching_tsv_path=self.match_path,
            expected_count=3,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["rows_validated"], 3)
        self.assertEqual(report["total_candidates"], 3)
        self.assertEqual(report["total_matches"], 1)
        self.assertEqual(report["empty_matches_count (singletons)"], 2)

    def test_subset_violation(self):
        with open(self.cand_path, "w", encoding="utf-8") as f_c, open(self.match_path, "w", encoding="utf-8") as f_m:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            f_m.write("source1_entity_id\tmatched_entity_ids\n")

            f_c.write("S1_001\tS2_101\n")
            f_m.write("S1_001\tS2_999\n")  # S2_999 not in candidate list!

        with self.assertRaises(ValueError) as ctx:
            validate_tsv_files(self.cand_path, self.match_path, expected_count=1)
        self.assertIn("Violation of strict subset invariant", str(ctx.exception))

    def test_duplicate_candidates_violation(self):
        with open(self.cand_path, "w", encoding="utf-8") as f_c, open(self.match_path, "w", encoding="utf-8") as f_m:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            f_m.write("source1_entity_id\tmatched_entity_ids\n")

            f_c.write("S1_001\tS2_101,S2_101\n")  # duplicate candidate
            f_m.write("S1_001\tS2_101\n")

        with self.assertRaises(ValueError) as ctx:
            validate_tsv_files(self.cand_path, self.match_path, expected_count=1)
        self.assertIn("Duplicate candidates found", str(ctx.exception))

    def test_duplicate_s1_violation(self):
        with open(self.cand_path, "w", encoding="utf-8") as f_c, open(self.match_path, "w", encoding="utf-8") as f_m:
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")
            f_m.write("source1_entity_id\tmatched_entity_ids\n")

            f_c.write("S1_001\tS2_101\n")
            f_m.write("S1_001\tS2_101\n")
            f_c.write("S1_001\tS2_102\n")  # duplicate S1
            f_m.write("S1_001\tS2_102\n")

        with self.assertRaises(ValueError) as ctx:
            validate_tsv_files(self.cand_path, self.match_path, expected_count=2)
        self.assertIn("Duplicate S1 ID", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
