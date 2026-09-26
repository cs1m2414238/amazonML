"""Unit tests for the integrated EndToEndPipeline."""

import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.src.models.matcher_inference import BatchMatcher
from business_entity_resolution.src.pipeline.inference_pipeline import EndToEndPipeline


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


if __name__ == "__main__":
    unittest.main()
