"""Comprehensive unit tests for validation sampling, blocking metrics, and Macro-F0.5 evaluation."""

import unittest
from business_entity_resolution.src.evaluation.candidate_recall import CandidateRecallEvaluator
from business_entity_resolution.src.evaluation.f05 import MacroF05Evaluator, compute_entity_f05
from business_entity_resolution.src.evaluation.sampling import classify_strata
from business_entity_resolution.src.evaluation.threshold_search import search_best_threshold


class TestValidationMetrics(unittest.TestCase):

    def setUp(self):
        self.synthetic_gt = {
            "S1-100": [],  # Singleton
            "S1-101": ["S2-201"],  # 1 match (S2 only)
            "S1-102": ["S2-202", "S3-302"],  # 2 matches (both sources)
            "S1-103": ["S3-303A", "S3-303B", "S3-303C"],  # 3 matches (S3 only)
            "S1-104": [
                f"S2-40{i}" for i in range(6)
            ] + [
                f"S3-50{i}" for i in range(5)
            ],  # 11 matches (max cardinality)
        }
        self.evaluator = MacroF05Evaluator(gt_map=self.synthetic_gt)
        self.recall_evaluator = CandidateRecallEvaluator(gt_map=self.synthetic_gt)

    def test_singleton_correctly_empty(self):
        """Empty prediction on a true singleton yields exactly 1.0."""
        score = compute_entity_f05(gt_targets=set(), pred_targets=set())
        self.assertEqual(score, 1.0)

        preds = {"S1-100": []}
        res = MacroF05Evaluator(gt_map={"S1-100": []}).evaluate(predictions=preds)
        rep = res["representative_metrics"]
        self.assertEqual(rep["macro_f05"], 1.0)
        self.assertEqual(rep["singleton_accuracy"], 1.0)
        self.assertEqual(rep["singleton_queries"], 1)

    def test_singleton_false_positive(self):
        """Non-empty prediction on a true singleton yields exactly 0.0."""
        score = compute_entity_f05(gt_targets=set(), pred_targets={"S2-999"})
        self.assertEqual(score, 0.0)

        preds = {"S1-100": ["S2-999"]}
        res = MacroF05Evaluator(gt_map={"S1-100": []}).evaluate(predictions=preds)
        rep = res["representative_metrics"]
        self.assertEqual(rep["macro_f05"], 0.0)
        self.assertEqual(rep["singleton_accuracy"], 0.0)
        self.assertEqual(rep["total_fp"], 1)

    def test_multi_match_complete(self):
        """Complete match on multi-target entity yields exactly 1.0."""
        gt = {"S2-202", "S3-302"}
        pred = {"S2-202", "S3-302"}
        score = compute_entity_f05(gt_targets=gt, pred_targets=pred)
        self.assertEqual(score, 1.0)

    def test_multi_match_partial(self):
        """Partial match on multi-target entity verifies formula 5*TP / (5*TP + 4*FP + FN)."""
        # S1-102 has 2 GT targets. If we predict only 1: TP=1, FP=0, FN=1.
        # F0.5 = (5 * 1) / (5 * 1 + 4 * 0 + 1) = 5 / 6 = 0.833333...
        gt = {"S2-202", "S3-302"}
        pred = {"S2-202"}
        score = compute_entity_f05(gt_targets=gt, pred_targets=pred)
        self.assertAlmostEqual(score, 5.0 / 6.0, places=4)

    def test_false_merges(self):
        """False merges (FP) are penalized heavily with weight 4 in denominator."""
        # GT = {"S2-201"}, Pred = {"S2-201", "S2-999"}
        # TP=1, FP=1, FN=0.
        # F0.5 = (5 * 1) / (5 * 1 + 4 * 1 + 0) = 5 / 9 = 0.555555...
        gt = {"S2-201"}
        pred = {"S2-201", "S2-999"}
        score = compute_entity_f05(gt_targets=gt, pred_targets=pred)
        self.assertAlmostEqual(score, 5.0 / 9.0, places=4)

    def test_empty_candidate_sets(self):
        """Empty candidate sets produce zero recall and are handled safely without crash."""
        candidates = {
            "S1-100": [],
            "S1-101": [],
            "S1-102": [],
            "S1-103": [],
            "S1-104": [],
        }
        res = self.recall_evaluator.evaluate(retrieved_candidates=candidates)
        rep = res["representative_metrics"]
        self.assertEqual(rep["pair_level_recall"], 0.0)
        self.assertEqual(rep["complete_match_set_recall"], 0.0)
        self.assertEqual(rep["candidate_counts"]["total_retrieved"], 0)
        self.assertEqual(rep["singleton_behavior"]["zero_candidate_rate"], 1.0)

    def test_invalid_ids(self):
        """Malformed target IDs and S1 IDs in predictions are detected and reported."""
        preds = {
            "S1-101": ["X1-INVALID", "S1-SELF", "S2-201"],
        }
        res = self.evaluator.evaluate(predictions=preds)
        rep = res["representative_metrics"]
        self.assertEqual(rep["validation_checks"]["invalid_id_format_count"], 2)

    def test_duplicate_ids(self):
        """Duplicate predictions are cleanly deduplicated without double-counting TP/FP."""
        preds = {
            "S1-101": ["S2-201", "S2-201", "S2-201"],
        }
        res = self.evaluator.evaluate(predictions=preds)
        rep = res["representative_metrics"]
        self.assertEqual(rep["validation_checks"]["duplicate_predictions_handled"], 2)
        self.assertEqual(rep["total_tp"], 1)
        self.assertEqual(rep["total_fp"], 0)
        self.assertEqual(rep["macro_f05"], 1.0)

    def test_cardinality_11(self):
        """Entities with maximum cardinality (11 matches) score correctly."""
        gt_11 = self.synthetic_gt["S1-104"]
        self.assertEqual(len(gt_11), 11)

        # Complete prediction
        score_complete = compute_entity_f05(set(gt_11), set(gt_11))
        self.assertEqual(score_complete, 1.0)

        # 8 out of 11 predicted (TP=8, FP=0, FN=3)
        # F0.5 = (5 * 8) / (5 * 8 + 4 * 0 + 3) = 40 / 43 = 0.93023...
        pred_8 = set(gt_11[:8])
        score_partial = compute_entity_f05(set(gt_11), pred_8)
        self.assertAlmostEqual(score_partial, 40.0 / 43.0, places=4)

    def test_candidate_subset_violations(self):
        """Predictions outside the retrieved candidate pool are flagged."""
        candidates = {
            "S1-101": ["S2-201"],
            "S1-102": ["S2-202"],  # Note S3-302 is missing from candidate pool
        }
        predictions = {
            "S1-101": ["S2-201"],
            "S1-102": ["S2-202", "S3-302"],  # S3-302 was not in candidate pool
        }
        res = self.evaluator.evaluate(predictions=predictions, candidates=candidates)
        checks = res["representative_metrics"]["validation_checks"]
        self.assertEqual(checks["candidate_subset_violations_count"], 1)
        self.assertEqual(checks["candidate_subset_violation_rate"], 0.5)

    def test_strata_classification(self):
        """Overlapping diagnostic strata are assigned correctly to entities."""
        # Singleton, US, valid address
        s1 = classify_strata("Alpha Store", "123 Main St", "US", [])
        self.assertIn("singleton", s1)
        self.assertIn("country_us", s1)
        self.assertNotIn("missing_address", s1)

        # 1 match, S2 only, India, missing address
        s2 = classify_strata("Beta Services", "", "India", ["S2-100"])
        self.assertIn("one_match", s2)
        self.assertIn("country_india", s2)
        self.assertIn("s2_only", s2)
        self.assertIn("missing_address", s2)

        # 3 matches, S3 only, US
        s3 = classify_strata("Gamma Co", "456 Oak Rd", "US", ["S3-1", "S3-2", "S3-3"])
        self.assertIn("multi_2_4", s3)
        self.assertIn("s3_only", s3)

        # 6 matches, Both sources, India
        s4 = classify_strata("Delta LLC", "789 Pine Ave", "IN", ["S2-1", "S3-1", "S3-2", "S3-3", "S3-4", "S3-5"])
        self.assertIn("multi_5_plus", s4)
        self.assertIn("both_sources", s4)
        self.assertIn("country_india", s4)

    def test_blocking_metrics_evaluator(self):
        """Candidate recall evaluator measures pair recall, complete match recall, and S2/S3 recall."""
        # S1-100: GT=[], Retr=[]
        # S1-101: GT=[S2-201], Retr=[S2-201] -> TP=1
        # S1-102: GT=[S2-202, S3-302], Retr=[S2-202] -> TP=1, FN=1
        # Total true pairs for S1-101 and S1-102 = 1 + 2 = 3. Retrieved = 2.
        # Pair recall = 2 / 3 = 0.6667.
        # S1-101 complete match = Yes. S1-102 complete match = No.
        # Complete match recall = 1 / 2 = 0.5.
        candidates = {
            "S1-100": [],
            "S1-101": ["S2-201"],
            "S1-102": ["S2-202"],
        }
        sub_gt = {
            "S1-100": [],
            "S1-101": ["S2-201"],
            "S1-102": ["S2-202", "S3-302"],
        }
        evaluator = CandidateRecallEvaluator(gt_map=sub_gt)
        res = evaluator.evaluate(retrieved_candidates=candidates, runtime_seconds=1.0)
        rep = res["representative_metrics"]
        self.assertAlmostEqual(rep["pair_level_recall"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(rep["complete_match_set_recall"], 0.5, places=4)
        self.assertEqual(rep["source2_recall"], 1.0)  # S2-201 and S2-202 both found
        self.assertEqual(rep["source3_recall"], 0.0)  # S3-302 missed

    def test_threshold_search(self):
        """Threshold tuning finds threshold maximizing Macro-F0.5."""
        # S1-101: GT = ["S2-201"]
        # Scored pair: ("S1-101", "S2-201", 0.85), ("S1-101", "S2-999", 0.40)
        # S1-100 (singleton): ("S1-100", "S2-888", 0.30)
        # At threshold 0.5: S1-101 predicts [S2-201] (Score=1.0), S1-100 predicts [] (Score=1.0) -> Macro-F0.5 = 1.0
        # At threshold 0.2: S1-101 predicts [S2-201, S2-999], S1-100 predicts [S2-888] -> Much lower score
        scored_pairs = [
            ("S1-101", "S2-201", 0.85),
            ("S1-101", "S2-999", 0.40),
            ("S1-100", "S2-888", 0.30),
        ]
        gt_map = {
            "S1-100": [],
            "S1-101": ["S2-201"],
        }
        search_res = search_best_threshold(
            scored_pairs=scored_pairs,
            gt_map=gt_map,
            thresholds=[0.2, 0.5, 0.9],
        )
        self.assertEqual(search_res["best_threshold"], 0.5)
        self.assertEqual(search_res["best_macro_f05"], 1.0)


if __name__ == "__main__":
    unittest.main()
