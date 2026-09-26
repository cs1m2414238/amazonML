"""Unit tests for LightGBM business entity matcher and batch inference."""

import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np

from business_entity_resolution.src.features.pair_features import PairFeatureGenerator
from business_entity_resolution.src.models.matcher_inference import BatchMatcher


class TestMatcherModel(unittest.TestCase):

    def setUp(self):
        self.feat_gen = PairFeatureGenerator()

    def test_feature_consistency(self):
        """Feature extraction is strictly deterministic across repeated calls."""
        s1 = {"id": "S1-1", "name": "Google LLC", "addr": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"}
        s2 = {"id": "S2-1", "name": "Google", "addr": "1600 Amphitheatre Parkway", "country": "United States"}

        feat1 = self.feat_gen.compute_features(s1, s2, retrieval_rank=1, target_source="source2")
        feat2 = self.feat_gen.compute_features(s1, s2, retrieval_rank=1, target_source="source2")

        self.assertEqual(feat1.keys(), feat2.keys())
        for k in feat1:
            self.assertEqual(feat1[k], feat2[k], f"Mismatch for feature {k}")

    def test_missing_address_handling(self):
        """Missing addresses produce valid numeric features without NaNs or crashes."""
        s1 = {"id": "S1-1", "name": "Starbucks Coffee", "addr": "", "country": "US"}
        s2 = {"id": "S2-1", "name": "Starbucks", "addr": None, "country": "US"}

        feat = self.feat_gen.compute_features(s1, s2, retrieval_rank=2)
        self.assertEqual(feat["s1_missing_address"], 1.0)
        self.assertEqual(feat["s2_missing_address"], 1.0)
        self.assertEqual(feat["both_missing_address"], 1.0)
        self.assertEqual(feat["address_exact_match"], 0.0)
        self.assertEqual(feat["address_rapidfuzz_ratio"], 0.0)
        for k, v in feat.items():
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")

    def test_open_set_country(self):
        """Unseen country strings (e.g. France, Germany) are handled gracefully."""
        # Exact match on France
        s1 = {"id": "S1-1", "name": "Boulangerie Paul", "addr": "Paris", "country": "France"}
        s2 = {"id": "S2-1", "name": "Boulangerie Paul", "addr": "Paris", "country": "france"}
        feat_match = self.feat_gen.compute_features(s1, s2)
        self.assertEqual(feat_match["country_match"], 1.0)
        self.assertEqual(feat_match["is_us"], 0.0)
        self.assertEqual(feat_match["is_india"], 0.0)

        # Mismatch on France vs Germany
        s3 = {"id": "S3-1", "name": "Boulangerie Paul", "addr": "Berlin", "country": "Germany"}
        feat_mismatch = self.feat_gen.compute_features(s1, s3)
        self.assertEqual(feat_mismatch["country_match"], 0.0)

    def test_empty_candidate_pool(self):
        """Entities with zero candidates return empty prediction list cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "dummy_model.txt"
            schema_path = Path(tmpdir) / "feature_schema.json"

            # Train a tiny 2-sample model to test interface
            X = np.array([[1.0, 0.5], [0.0, 0.1]], dtype=np.float32)
            y = np.array([1, 0], dtype=np.int32)
            clf = lgb.LGBMClassifier(n_estimators=2, max_depth=2, verbose=-1)
            clf.fit(X, y)
            clf.booster_.save_model(str(model_path))

            import json
            with open(schema_path, "w") as f:
                json.dump({"features": ["name_exact_match", "address_rapidfuzz_ratio"]}, f)

            matcher = BatchMatcher(model_path=model_path, feature_schema_path=schema_path, threshold=0.5)

            s1_records = [{"id": "S1-ZERO", "name": "Singleton Shop", "addr": "None", "country": "US"}]
            candidate_records = {}
            candidates_by_s1 = {"S1-ZERO": []}

            preds = matcher.predict_batch(s1_records, candidate_records, candidates_by_s1)
            self.assertEqual(preds, {"S1-ZERO": []})

    def test_one_to_many_matches(self):
        """Preserves one-to-many matches when multiple candidates exceed threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "dummy_model.txt"
            schema_path = Path(tmpdir) / "feature_schema.json"

            # Model that predicts high probability when name_exact_match == 1.0
            X = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32)
            y = np.array([1, 0, 1, 0], dtype=np.int32)
            clf = lgb.LGBMClassifier(n_estimators=10, min_child_samples=1, max_depth=2, verbose=-1)
            clf.fit(X, y)
            clf.booster_.save_model(str(model_path))

            import json
            with open(schema_path, "w") as f:
                json.dump({"features": ["name_exact_match"]}, f)

            matcher = BatchMatcher(model_path=model_path, feature_schema_path=schema_path, threshold=0.5)

            s1 = [{"id": "S1-MULTI", "name": "Apex Corp", "addr": "100 Broadway", "country": "US"}]
            cands = {
                "S2-1": {"id": "S2-1", "name": "Apex Corp", "addr": "100 Broadway St", "country": "US"},
                "S2-2": {"id": "S2-2", "name": "Apex Corp", "addr": "100 Broadway Ave", "country": "US"},
                "S3-1": {"id": "S3-1", "name": "Random Corp", "addr": "500 5th Ave", "country": "US"},
            }
            cand_by_s1 = {"S1-MULTI": ["S2-1", "S2-2", "S3-1"]}

            preds = matcher.predict_batch(s1, cands, cand_by_s1)
            # S2-1 and S2-2 both match name exactly and should both be accepted
            self.assertIn("S2-1", preds["S1-MULTI"])
            self.assertIn("S2-2", preds["S1-MULTI"])
            self.assertNotIn("S3-1", preds["S1-MULTI"])
            self.assertEqual(len(preds["S1-MULTI"]), 2)

    def test_tsv_formatting(self):
        """TSV output matches official competition format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_file = Path(tmpdir) / "matching_results.tsv"
            preds = {
                "S1-1": ["S2-10", "S3-20"],
                "S1-2": [],  # Singleton
                "S1-3": ["S2-30"],
            }
            BatchMatcher.format_tsv(preds, tsv_file)

            lines = tsv_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertEqual(lines[1], "S1-1\tS2-10,S3-20")
            self.assertEqual(lines[2], "S1-2\t")
            self.assertEqual(lines[3], "S1-3\tS2-30")


if __name__ == "__main__":
    unittest.main()
